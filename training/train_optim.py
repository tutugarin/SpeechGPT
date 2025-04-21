#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import argparse
import functools
import evaluate
from tqdm import tqdm
from psutil import virtual_memory

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import HfFolder
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from unified_dataset import UnifiedSpeechDataset
from adapters import FCAdapter, TransformerAdapter

os.environ["XLA_DISABLE_PLUGIN_LOAD"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"  # Prevent memory fragmentation

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.set_float32_matmul_precision('high')

MAX_DURATION = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for speech model with adapter")

    parser.add_argument("--dataset_subset", type=str, default="",
                        help="Subset of dataset to use")
    parser.add_argument("--dataset_lang", type=str, default="ru",
                        help="Language of the dataset")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--max_text_length", type=int, default=512,
                        help="Maximum text length")
    parser.add_argument("--adapter_type", type=str, default="fc", choices=["fc", "transformer"],
                        help="Type of adapter to use (fc or transformer)")
    parser.add_argument("--fc_layer_dim", type=int, default=11264,
                        help="Dimension of FC layer")
    parser.add_argument("--transformer_dim", type=int, default=1024,
                        help="Dimension of transformer")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--feed_forward_dim", type=int, default=2048,
                        help="Dimension of feed forward network")
    parser.add_argument("--num_transformer_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--transformer_dropout", type=float, default=0.1,
                        help="Dropout rate for transformer")
    parser.add_argument("--token", type=str, required=True,
                        help="Token for UnifiedSpeechDataset")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--asr_model_name", type=str, default="openai/whisper-large-v3",
                        help="ASR model name")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2-0.5B",
                        help="LLM model name")
    parser.add_argument("--eval_steps", type=int, default=5,
                        help="Steps between evaluations")
    parser.add_argument("--llm_hidden_size", type=int, default=896,
                        help="LLM hidden state size")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints",
                        help="Path to save the final model")
    parser.add_argument(
        "--local_files_only", action='store_true', default=False,
        help="Use local files or not"
    )
    # parser.add_argument(
    #     "--cache_dir", type=str, default="./cache",
    #     help="Directory to use as cache"
    # )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--fp16", action='store_true', default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=1,
        help="Frequency of saving checkpoints (in epochs)"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=4,
        help="Number of batches loaded in advance by each worker"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of data loading workers (default: 2*GPU count)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--profile", action='store_true', default=False,
        help="Enable profiling for performance analysis"
    )

    return parser.parse_args()


def get_adapter(args):
    if args.adapter_type == 'fc':
        return FCAdapter(
            fc_layer_dim=args.fc_layer_dim,
            llm_hidden_size=args.llm_hidden_size
        )
    elif args.adapter_type == 'transformer':
        return TransformerAdapter(
            transform_dim=args.transformer_dim,
            num_heads=args.num_heads,
            ff_dim=args.feed_forward_dim,
            num_layers=args.num_transformer_layers,
            dropout=args.transformer_dropout,
            llm_hidden_size=args.llm_hidden_size
        )
    else:
        raise Exception(f"Некорректный тип адаптера: {args.adapter_type}")


# Optimized collate function for better CPU utilization
def collate_fn(batch, asr_processor, llm_tokenizer, max_duration, max_text_length):
    if len(batch) == 0:
        return None
    
    # Process audio in parallel for better CPU utilization
    audio = [x['speech_input'] for x in batch]
    audio_inputs = asr_processor(
        audio=audio,
        sampling_rate=16000,
        max_length=int(max_duration * 16000),
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Process text inputs in parallel
    input_text = [f"{asr_processor.tokenizer.eos_token}{x['text_prompt']}" for x in batch]
    text_inputs = llm_tokenizer(
        input_text,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Process text outputs in parallel
    output_text = [f"{asr_processor.tokenizer.eos_token}{x['text_response']}" for x in batch]
    text_outputs = llm_tokenizer(
        output_text,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "audio_features": audio_inputs.input_features,
        "text_inputs": text_inputs.input_ids,
        "text_outputs": text_outputs.input_ids
    }


class ModelWrapper(nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()
        self.device = device
        self.using_cuda = torch.cuda.is_available()
        
        # Load ASR encoder with optimized settings
        asr_config = AutoConfig.from_pretrained(
            args.asr_model_name,
            # cache_dir=args.cache_dir,
            local_files_only=args.local_files_only
        )
        asr_model = WhisperForConditionalGeneration.from_pretrained(
            args.asr_model_name,
            config=asr_config,
            # cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            torch_dtype=torch.float16 if args.fp16 and self.using_cuda else torch.float32
        )
        
        # Optimize LLM config for better performance
        llm_cfg = AutoConfig.from_pretrained(
            args.llm_model_name,
            # cache_dir=args.cache_dir,
            local_files_only=args.local_files_only
        )
        llm_cfg.use_cache = False  # Disable KV-cache to save memory
        llm_cfg.gradient_checkpointing = True  # Enable gradient checkpointing
        llm_cfg.use_sliding_window = True  # Enable sliding window attention
        llm_cfg.sliding_window = 256  # Set sliding window size
        
        # Load LLM with optimized settings
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_name,
            config=llm_cfg,
            # cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            attn_implementation="sdpa",
            torch_dtype=torch.float16 if args.fp16 and self.using_cuda else torch.float32
        )

        # Store pad token ID
        if not hasattr(args, 'pad_token_id') or args.pad_token_id is None:
            raise ValueError("pad_token_id cannot be None")
        self.pad_token_id = args.pad_token_id

        # Extract only needed components
        self.asr_encoder = asr_model.model.encoder
        self.llm_embed_layer = self.llm_model.model.embed_tokens
        self.llm_model.model.embed_tokens = torch.nn.Identity()

        # Clear memory
        del asr_model
        if self.using_cuda:
            torch.cuda.empty_cache()

        # Initialize adapter with proper device placement
        self.adapter = get_adapter(args)
        print(f"Adapter parameter count: {sum(p.numel() for p in self.adapter.parameters())}")

        # Move components to device efficiently
        self._move_to_device()

        # Freeze layers
        self._freeze_parameters()

    def _move_to_device(self):
        """Move model components to device efficiently"""
        if self.using_cuda:
            # Move components one by one with memory clearing between
            self.asr_encoder = self.asr_encoder.to(self.device)
            torch.cuda.empty_cache()
            
            self.llm_model = self.llm_model.to(self.device)
            torch.cuda.empty_cache()
            
            self.adapter = self.adapter.to(self.device)
            torch.cuda.empty_cache()
        else:
            self.asr_encoder = self.asr_encoder.to(self.device)
            self.llm_model = self.llm_model.to(self.device)
            self.adapter = self.adapter.to(self.device)

    def _freeze_parameters(self):
        """Freeze all parameters except adapter"""
        # Freeze ASR encoder
        for param in self.asr_encoder.parameters():
            param.requires_grad = False
        self.asr_encoder.eval()
        
        # Freeze LLM embedding layer
        for param in self.llm_embed_layer.parameters():
            param.requires_grad = False
        
        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        
        # Ensure adapter is trainable
        for param in self.adapter.parameters():
            param.requires_grad = True

    def _process_encoder_outputs(self, audio_features):
        """Process encoder outputs with memory optimization"""
        with torch.no_grad():
            # Use non-blocking transfer for better GPU utilization
            if self.using_cuda:
                audio_features = audio_features.to(self.device, non_blocking=True)
                
            encoder_output = self.asr_encoder(audio_features)
            encoder_embed = encoder_output.last_hidden_state.detach()
            
        return encoder_embed

    def _process_input_embeddings(self, text_inputs):
        """Process input embeddings with memory optimization"""
        with torch.no_grad():
            if self.using_cuda:
                text_inputs = text_inputs.to(self.device, non_blocking=True)
                
            return self.llm_embed_layer(text_inputs).detach()

    def forward(self, batch):
        """Forward pass with optimized memory usage"""
        if self.pad_token_id is None:
            raise ValueError("Use set_pad_token_id method")

        # Non-blocking transfer for better GPU utilization
        audio_features = batch["audio_features"].to(self.device, non_blocking=True).float()
        text_inputs = batch["text_inputs"].to(self.device, non_blocking=True).long()
        text_outputs = batch["text_outputs"].to(self.device, non_blocking=True).long()
        
        return self._forward_impl(audio_features, text_inputs, text_outputs)

    def _forward_impl(self, audio_features, text_inputs, text_outputs):
        """Optimized implementation of forward pass"""
        # Process encoder outputs
        encoder_embed = self._process_encoder_outputs(audio_features)
        
        # Process input embeddings
        llm_first_embed = self._process_input_embeddings(text_inputs)
        
        # Clear variables to save memory
        del audio_features, text_inputs
        if self.using_cuda:
            torch.cuda.empty_cache()

        # Process through adapter (trainable part)
        adapter_out = self.adapter(encoder_embed)
        del encoder_embed
        if self.using_cuda:
            torch.cuda.empty_cache()

        # Combine embeddings efficiently
        combined_embeds = torch.cat([adapter_out, llm_first_embed], dim=1)
        del llm_first_embed, adapter_out
        if self.using_cuda:
            torch.cuda.empty_cache()

        # Create attention mask efficiently
        combined_seq_length = combined_embeds.size(1)
        attention_mask = torch.ones(
            (combined_embeds.size(0), combined_seq_length), 
            device=self.device, 
            dtype=torch.long
        )

        # Prepare labels efficiently
        labels = text_outputs[:, 1:combined_seq_length + 1].clone()
        if labels.size(1) != combined_seq_length:
            pad_size = combined_seq_length - labels.size(1)
            labels = torch.nn.functional.pad(labels, (0, pad_size), value=self.pad_token_id)
        
        # Forward pass through LLM
        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        # Clean up memory
        del combined_embeds, attention_mask
        if self.using_cuda:
            torch.cuda.empty_cache()
        
        return outputs, labels


class ThroughputTracker:
    """Track training throughput for monitoring performance"""
    def __init__(self, batch_size, world_size, log_interval=10):
        self.batch_size = batch_size
        self.world_size = world_size
        self.log_interval = log_interval
        self.start_time = time.time()
        self.sample_count = 0
        self.step_count = 0
        
    def update(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self.sample_count += batch_size * self.world_size
        self.step_count += 1
        
    def get_throughput(self):
        """Calculate samples per second"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.sample_count / elapsed
    
    def should_log(self):
        return self.step_count % self.log_interval == 0
    
    def reset(self):
        self.start_time = time.time()
        self.sample_count = 0
        self.step_count = 0


def load_adapter(model, rank, checkpoint_path):
    """Load adapter weights from checkpoint"""
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.adapter.load_state_dict(state_dict["adapter_state_dict"])
    return model


def setup(rank, world_size):
    """Set up distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23551'

    env_dict = {
        "MASTER_ADDR": os.environ["MASTER_ADDR"],
        "MASTER_PORT": os.environ["MASTER_PORT"],
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size)
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set optimal process-level settings
    if torch.cuda.is_available():
        
        # Optimize thread allocation
        torch.set_num_threads(4)  # Limit CPU threads per process

    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )


def train_model(rank, world_size, args):
    """Main training function with optimized performance"""
    # Setup distributed training
    setup(rank, world_size)
    
    # Set up profiling if requested
    if args.profile and rank == 0:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.log_dir, 'profiler')
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    else:
        prof = None

    # Set up mixed precision training
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    # Set up tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir) if rank == 0 else None
    
    # Initialize throughput tracker
    throughput_tracker = ThroughputTracker(
        batch_size=args.batch_size, 
        world_size=world_size,
        log_interval=10
    )

    # Set up dataset with better caching options
    dataset = UnifiedSpeechDataset(
        token=args.token,
        lang=args.dataset_lang,
        split="train",
        subset=args.dataset_subset,
        batch_size=args.batch_size,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )

    # Load tokenizers and processors
    asr_processor = AutoProcessor.from_pretrained(
        args.asr_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )
    
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )
    
    # Add pad token if needed
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    args.pad_token_id = llm_tokenizer.pad_token_id

    # Create optimized collate function
    custom_collate = functools.partial(
        collate_fn,
        asr_processor=asr_processor,
        llm_tokenizer=llm_tokenizer,
        max_duration=MAX_DURATION,
        max_text_length=args.max_text_length
    )

    # Set up optimized data loading
    num_workers = args.num_workers if args.num_workers is not None else (8 if torch.cuda.is_available() else 4)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        sampler=sampler,
        pin_memory=True,  # Enable pinned memory for faster transfers
        num_workers=num_workers,  # Optimize worker count
        prefetch_factor=args.prefetch_factor,  # Load batches in advance
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=False
    )
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Set up device
    device = rank if torch.cuda.is_available() else 'cpu'
    model = ModelWrapper(args, device=device)

    # Set up DDP with optimized settings
    if torch.cuda.is_available():
        model = DistributedDataParallel(
            model, 
            device_ids=[rank], 
            find_unused_parameters=True,
            static_graph=False,  # Set to True if possible for better performance
            bucket_cap_mb=25  # Optimize communication bucket size
        )
    else:
        model = DistributedDataParallel(model)

    # Set up optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # Start training with optimized flow
    print("Starting training")
    global_step = 0
    for epoch in range(args.epochs):
        # Set epoch for sampler
        model.train()
        sampler.set_epoch(epoch)
        
        # Reset throughput tracker
        throughput_tracker.reset()
        
        # Initialize progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        loss_value = None
        
        # Clear optimizer state
        optimizer.zero_grad(set_to_none=True)
        
        # Track accumulated loss for gradient accumulation
        accumulated_loss = 0

        # Main training loop
        for batch_idx, batch in enumerate(progress_bar):
            # Skip empty batches
            if batch is None:
                continue
                
            # Clear memory at start of batch
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update throughput tracker  
            throughput_tracker.update()
            
            try:
                # Forward pass with mixed precision
                if scaler is not None:
                    with autocast():
                        outputs, labels = model(batch)
                        loss = outputs.loss / args.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        accumulated_loss += loss.item() * args.gradient_accumulation_steps
                else:
                    outputs, labels = model(batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.item() * args.gradient_accumulation_steps

                # Only store logits for evaluation
                logits = None
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    logits = outputs.logits.detach()
                
                # Clean up memory
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Update weights on gradient accumulation step
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(progress_bar) - 1:
                    # Apply gradient clipping
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Zero gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Log metrics
                    if rank == 0 and writer is not None:
                        writer.add_scalar("train/loss", accumulated_loss, global_step)
                        if throughput_tracker.should_log():
                            throughput = throughput_tracker.get_throughput()
                            writer.add_scalar("performance/throughput", throughput, global_step)
                    
                    # Update loss value
                    loss_value = accumulated_loss
                    accumulated_loss = 0
                    
                    # Update progress bar
                    progress_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        throughput=f"{throughput_tracker.get_throughput():.1f} samples/s",
                        mem=f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                    )

                    # Evaluate if needed
                    if global_step % args.eval_steps == 0 and logits is not None and global_step > 0:
                        with torch.no_grad():
                            # Update profiler if enabled
                            if prof is not None and rank == 0:
                                prof.step()
                                
                            # Compute WER
                            preds = torch.argmax(logits, dim=-1)
                            pred_str = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
                            label_str = llm_tokenizer.batch_decode(labels.detach(), skip_special_tokens=True)
                            wer = evaluate.load("wer").compute(
                                predictions=pred_str,
                                references=label_str
                            )

                            if device != 'cpu':
                                all_wer = [torch.tensor([wer], dtype=torch.float32).to(rank)]
                                dist.all_gather(all_wer, all_wer[0])
                                
                                if rank == 0:
                                    all_wer = torch.cat(all_wer, dim=0)
                                    avg_wer = all_wer.mean().item()
                                    print(f"Average WER on step {global_step}: {avg_wer}")
                                    if writer is not None:
                                        writer.add_scalar("eval/avg_wer", avg_wer, global_step)
                            else:
                                print(f"WER on step {global_step}: {wer}")
                                if rank == 0 and writer is not None:
                                    writer.add_scalar("eval/wer", wer, global_step)
                            
                            # Clean up
                            del preds, pred_str, label_str
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    global_step += 1

                # Clean up batch resources
                if logits is not None:
                    del logits
                del labels, batch
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM error on batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

        # Save checkpoint at end of epoch
        if rank == 0 and epoch % args.checkpoint_freq == 0:
            os.makedirs(args.model_save_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'adapter_state_dict': model.module.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }
            
            torch.save(
                checkpoint,
                os.path.join(args.model_save_dir, f'checkpoint_{args.adapter_type}_epoch_{epoch}.pt')
            )
        dist.barrier()

    # Save final model
    if rank == 0:
        torch.save({
                'adapter_state_dict': model.module.adapter.state_dict(),
            },
            os.path.join(args.model_save_dir, "adapter_final.pt")
        )
    dist.barrier()

    # Clean up resources
    if writer is not None:
        writer.close()
    if prof is not None:
        prof.stop()
        
    print(f"Model saved to {args.model_save_dir}")

    dist.destroy_process_group()
    print("Training complete")


def main():
    '''
    based on:
        https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
        https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    may also be useful:
        https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
        https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py
    '''
    args = parse_args()

    # Set environment variables for better performance
    # os.environ["HF_HOME"] = args.cache_dir
    # os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["OMP_NUM_THREADS"] = str(min(16, os.cpu_count()))
    os.environ["MKL_NUM_THREADS"] = str(min(16, os.cpu_count()))
    
    # Set offline mode
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        
    # Save token for authentication
    HfFolder.save_token(args.token)

    print(f"Use local files only: {args.local_files_only}")

    # Create directories
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.cache_dir, exist_ok=True)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")

    # Set up multi-GPU training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Running on {world_size} GPUs")

    # Run with process spawning for multi-GPU or directly for single GPU/CPU
    if world_size > 1:
        mp.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        train_model(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    # Set start method for multiprocessing - 'spawn' is more compatible across platforms
    mp.set_start_method('spawn', force=True)
    main()