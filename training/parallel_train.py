#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)

from unified_dataset import UnifiedSpeechDataset
from adapters import FCAdapter, TransformerAdapter


MAX_DURATION = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for speech model with adapter")

    parser.add_argument("--dataset_subset", type=str, default="[:32]",
                        help="Subset of dataset to use")
    parser.add_argument("--dataset_lang", type=str, default="ru",
                        help="Language of the dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8,
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
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--eval_steps", type=int, default=5,
                        help="Steps between evaluations")
    parser.add_argument("--llm_hidden_size", type=int, default=896,
                        help="LLM hidden state size")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints",
                        help="Path to save the final model")

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


def collate_fn(batch, asr_processor, llm_tokenizer, max_duration, max_text_length):
    audio = [x['speech_input'] for x in batch]
    audio_inputs = asr_processor(
        audio=audio,
        sampling_rate=16000,
        max_length=int(max_duration * 16000),
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_text = [f"{asr_processor.tokenizer.eos_token}{x['text_prompt']}" for x in batch]
    text_inputs = llm_tokenizer(
        input_text,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

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
    def __init__(self, args):
        super().__init__()
        asr_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name)

        # Костыль
        try:
            if args.pad_token_id is None:
                raise ValueError("pad_token_id cannot be None")
            self.pad_token_id = args.pad_token_id
        except:
            raise ValueError("You must pass pad_token_id in args argument")

        self.asr_encoder = asr_model.model.encoder
        self.llm_embed_layer = self.llm_model.model.embed_tokens
        self.llm_model.model.embed_tokens = torch.nn.Identity()

        self.adapter = get_adapter(args.adapter_type, args)
        print(f"Число параметров адаптера: {sum(p.numel() for p in self.adapter.parameters())}")

        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = args.device
            self.asr_encoder = self.asr_encoder.to(args.device)
            self.llm_model = self.llm_model.to(args.device)
            self.adapter = self.adapter.to(args.device)
        else:
            self.device = "cpu"

        # freeze layers
        for model in [self.asr_encoder, self.llm_embed_layer, self.llm_model]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        for param in self.adapter.parameters():
            param.requires_grad = True


    def forward(self, batch):
        if self.pad_token_id is None:
            raise ValueError("User set_pad_token_id method")

        audio_features = batch["audio_features"].to(self.device).float()
        text_inputs = batch["text_inputs"].to(self.device).long()
        text_outputs = batch["text_outputs"].to(self.device).long()

        encoder_output = self.asr_encoder(audio_features)
        encoder_embed = encoder_output.last_hidden_state.detach().clone()
        del encoder_output, audio_features

        llm_first_embed = self.llm_embed_layer(text_inputs).detach().clone()
        del text_inputs

        adapter_out = self.adapter(encoder_embed)

        combined_embeds = torch.cat([
            adapter_out,
            llm_first_embed
        ], dim=1)

        del encoder_embed, llm_first_embed, adapter_out

        combined_seq_length = combined_embeds.size(1)
        attention_mask = torch.ones((combined_embeds.size(0), combined_seq_length), device=self.device)

        labels = text_outputs[:, 1:combined_seq_length + 1].to(self.device)

        if labels.size(1) != combined_seq_length:
            pad_size = combined_seq_length - labels.size(1)
            labels = torch.nn.functional.pad(labels, (0, pad_size), value=self.pad_token_id)

        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        del combined_embeds, attention_mask
        return outputs, labels


def load_adapter(model, rank, checkpoint_path):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.adapter.load_state_dict(state_dict["adapter_state_dict"])
    return model


def setup(rank, world_size):
    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )


def train_model(rank, world_size, args):
    setup(rank, world_size)

    writer = SummaryWriter(log_dir=args.log_dir)

    dataset = UnifiedSpeechDataset(
        token=args.token,
        lang=args.dataset_lang,
        split="train",
        subset=args.dataset_subset
    )

    asr_processor = AutoProcessor.from_pretrained(args.asr_model_name)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    args.pad_token_id = llm_tokenizer.pad_token_id

    custom_collate = functools.partial(
        collate_fn,
        asr_processor=asr_processor,
        llm_tokenizer=llm_tokenizer,
        max_duration=MAX_DURATION,
        max_text_length=args.max_text_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        sampler=DistributedSampler(dataset),
    )

    model = ModelWrapper(args)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        loss_value = None
        for batch_idx, batch in enumerate(progress_bar):
            gc.collect()
            used_memory = virtual_memory().used / (1024 ** 3)
            progress_bar.set_postfix(mem=f"{used_memory:.1f}GB")

            optimizer.zero_grad(set_to_none=True)
            # ===== forward pass ======
            outputs, labels = model(batch)

            logits = outputs.logits
            loss = outputs.loss
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            del outputs

            progress_bar.set_postfix(
                loss=loss.item(),
                mem=f"{virtual_memory().used / (1024 ** 3):.1f}GB"
            )

            writer.add_scalar("train/loss", loss.item(), global_step)

            if global_step % args.eval_steps == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    pred_str = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
                    label_str = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)
                    wer = evaluate.load("wer").compute(
                        predictions=pred_str,
                        references=label_str
                    )
                    print(f"WER on step {global_step}: {wer}")
                    writer.add_scalar("eval/wer", wer, global_step)

            global_step += 1
            progress_bar.set_postfix(loss=loss.item())

            del logits, labels, loss
            gc.collect()

        if rank not in [-1, 0]:
            torch.save({
                    'epoch': epoch,
                    'adapter_state_dict': model.adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                },
                os.path.join(args.model_save_dir, f'checkpoint_{args.adapter_type}_epoch_{epoch}.pt')
            )
        dist.barrier()

    if rank == 0:
        torch.save({
                'adapter_state_dict': model.adapter.state_dict(),
            },
            os.path.join(args.model_save_path, "adapter_final.pt")
        )
    dist.barrier()

    writer.close()
    print(f"Модель сохранена в {args.model_save_path}")

    dist.destroy_process_group()


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
    os.makedirs(args.model_save_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Unable to run on GPU. Check your torch version via pip show torch.")

    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}."
    print(f"Run on {world_size} GPUs")

    mp.spawn(train_model,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
