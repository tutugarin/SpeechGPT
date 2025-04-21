#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init.py
-------
Скрипт для предварительной загрузки моделей и датасета в локальный кеш.
"""
import os
import argparse
from unified_dataset import UnifiedSpeechDataset
from huggingface_hub import HfFolder


def main():
    parser = argparse.ArgumentParser(
        description="Preload datasets and models to local cache"
    )
    # parser.add_argument(
    #     "--cache_dir", type=str, default="./cache",
    #     help="Directory to use as cache"
    # )
    parser.add_argument(
        "--local_files_only", action='store_true', default=False,
        help="Use local files or not"
    )
    parser.add_argument(
        "--dataset_subset", type=str, default="",
        help="Subset of dataset to use"
    )
    parser.add_argument(
        "--dataset_lang", type=str, default="ru",
        help="Language of the dataset"
    )
    parser.add_argument(
        "--token", type=str, required=True,
        help="Token for UnifiedSpeechDataset"
    )
    parser.add_argument(
        "--asr_model_name", type=str, default="openai/whisper-large-v3",
        help="ASR model name"
    )
    parser.add_argument(
        "--llm_model_name", type=str, default="Qwen/Qwen2-0.5B",
        help="LLM model name"
    )
    args = parser.parse_args()

    # os.environ["HF_HOME"] = args.cache_dir
    # os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    HfFolder.save_token(args.token)

    print(f"local_files_only: {args.local_files_only}")

    # os.makedirs(args.cache_dir, exist_ok=True)

    from transformers import WhisperForConditionalGeneration, AutoProcessor
    print(f"Loading ASR model {args.asr_model_name}...")
    WhisperForConditionalGeneration.from_pretrained(
        args.asr_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )
    AutoProcessor.from_pretrained(
        args.asr_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )

    print("ASR model and processor cached.")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print(f"Loading LLM model {args.llm_model_name}...")
    llm_cfg = AutoConfig.from_pretrained(
        args.llm_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )
    llm_cfg.use_sliding_window = False
    llm_cfg.sliding_window = None
    AutoModelForCausalLM.from_pretrained(
        args.llm_model_name,
        config=llm_cfg,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa"
    )
    AutoTokenizer.from_pretrained(
        args.llm_model_name,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )
    print("LLM model, config and tokenizer cached.")

    print("Loading dataset train via UnifiedSpeechDataset...")

    _ = UnifiedSpeechDataset(
        token=args.token,
        lang=args.dataset_lang,
        split="train",
        subset=args.dataset_subset,
        batch_size=1,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )

    _ = UnifiedSpeechDataset(
        token=args.token,
        lang=args.dataset_lang,
        split="test",
        subset=args.dataset_subset,
        batch_size=1,
        # cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )

    print("Dataset cached.")

    print("All resources have been preloaded into the cache.")


if __name__ == "__main__":
    main()
