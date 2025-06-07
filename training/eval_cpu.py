import argparse
import functools
import gc
import pickle
from contextlib import nullcontext

import evaluate
import torch
from psutil import virtual_memory
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)

from adapters import FCAdapter, TransformerAdapter
from unified_dataset import UnifiedSpeechDataset

MAX_DURATION = 30


def parse_args():
    parser = argparse.ArgumentParser(description="CPU-only testing script for speech model with adapter")

    parser.add_argument("--dataset_subset", type=str, default="[:32]",
                        help="Subset of dataset to use")
    parser.add_argument("--dataset_lang", type=str, default="ru",
                        help="Language of the dataset")

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
    parser.add_argument("--asr_model_name", type=str, default="openai/whisper-large-v3",
                        help="ASR model name")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2-0.5B",
                        help="LLM model name")
    parser.add_argument("--llm_hidden_size", type=int, default=896,
                        help="LLM hidden state size")
    parser.add_argument("--model_path", type=str, default="./adapter_final.pt",
                        help="Path to trained adapter checkpoint")

    parser.add_argument("--task", type=str, default="",
                        help="Task name inside dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation (CPU-friendly)")
    parser.add_argument("--loss_history_path", type=str, default="./loss_history.pkl",
                        help="Path to save WER history")
    parser.add_argument("--max_text_length", type=int, default=512,
                        help="Maximum text length")

    return parser.parse_args()


def get_adapter(adapter_type, args):
    if adapter_type == 'fc':
        return FCAdapter(
            fc_layer_dim=args.fc_layer_dim,
            llm_hidden_size=args.llm_hidden_size
        )
    elif adapter_type == 'transformer':
        return TransformerAdapter(
            transform_dim=args.transformer_dim,
            num_heads=args.num_heads,
            ff_dim=args.feed_forward_dim,
            num_layers=args.num_transformer_layers,
            dropout=args.transformer_dropout,
            llm_hidden_size=args.llm_hidden_size
        )
    raise ValueError(f"Unsupported adapter type: {adapter_type}")


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


def main():
    args = parse_args()
    device = torch.device("cpu") 

    dataset = UnifiedSpeechDataset(
        token=args.token,
        lang=args.dataset_lang,
        split="test",
        subset=args.dataset_subset
    )

    asr_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model_name)
    asr_processor = AutoProcessor.from_pretrained(args.asr_model_name)

    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    adapter = get_adapter(args.adapter_type, args)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])

    asr_encoder = asr_model.model.encoder
    llm_embed_layer = llm_model.model.embed_tokens      
    llm_model.model.embed_tokens = torch.nn.Identity()  

    custom_collate = functools.partial(
        collate_fn,
        asr_processor=asr_processor,
        llm_tokenizer=llm_tokenizer,
        max_duration=MAX_DURATION,
        max_text_length=args.max_text_length
    )

    dataloader = DataLoader(
        dataset.task_dataset(args.task),
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        shuffle=False
    )

    for model in (asr_encoder, llm_embed_layer, llm_model, adapter):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    metric_wer = evaluate.load("wer")

    wer_values = []
    progress_bar = tqdm(dataloader, desc="Evaluating (CPU)")

    for batch in progress_bar:
        gc.collect()

        audio_features = batch["audio_features"].to(device).float()
        text_inputs = batch["text_inputs"].to(device).long()
        text_outputs = batch["text_outputs"].to(device).long()

        with torch.no_grad(), nullcontext():
            encoder_output = asr_encoder(audio_features)
            encoder_embed = encoder_output.last_hidden_state.detach()

            llm_first_embed = llm_embed_layer(text_inputs).detach()
            adapter_out = adapter(encoder_embed)

            combined_embeds = torch.cat([adapter_out, llm_first_embed], dim=1)

            seq_len = combined_embeds.size(1)
            attention_mask = torch.ones((combined_embeds.size(0), seq_len), device=device)

            labels = text_outputs[:, 1:seq_len + 1]
            if labels.size(1) != seq_len:
                pad = seq_len - labels.size(1)
                labels = torch.nn.functional.pad(
                    labels, (0, pad), value=llm_tokenizer.pad_token_id
                )

            logits = llm_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                labels=labels
            ).logits

        preds = logits.argmax(dim=-1)
        pred_str = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer = metric_wer.compute(predictions=pred_str, references=label_str)
        wer_values.append(wer)

        progress_bar.set_postfix(
            wer=wer,
            mem=f"{virtual_memory().used / (1024 ** 3):.1f} GB"
        )

    with open(args.loss_history_path, 'wb') as f:
        pickle.dump(wer_values, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Оценка завершена. Средний WER:", sum(wer_values) / len(wer_values))


if __name__ == "__main__":
    main()
