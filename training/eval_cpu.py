import argparse, functools, gc, json, pickle, os
from contextlib import nullcontext
from pathlib import Path

import evaluate, torch, soundfile as sf
from psutil import virtual_memory
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (WhisperForConditionalGeneration, AutoProcessor,
                          AutoModelForCausalLM, AutoTokenizer)

from adapters import FCAdapter, TransformerAdapter
from unified_dataset import UnifiedSpeechDataset

MAX_DURATION = 30            
SR = 16_000                 


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_subset", default="[:32]")
    p.add_argument("--dataset_lang",  default="ru")
    p.add_argument("--adapter_type", choices=["fc", "transformer"], default="fc")
    p.add_argument("--fc_layer_dim", type=int, default=11264)
    p.add_argument("--transformer_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--feed_forward_dim", type=int, default=2048)
    p.add_argument("--num_transformer_layers", type=int, default=2)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    p.add_argument("--token", required=True, help="HF access token")
    p.add_argument("--asr_model_name", default="openai/whisper-large-v3")
    p.add_argument("--llm_model_name", default="Qwen/Qwen2-0.5B")
    p.add_argument("--llm_hidden_size", type=int, default=896)
    p.add_argument("--model_path", default="./adapter_final.pt")
    p.add_argument("--task",          default="")
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--max_text_length", type=int, default=512)
    p.add_argument("--pred_jsonl_path", default="predictions.jsonl")
    p.add_argument("--logits_path",      default="logits.pt")
    p.add_argument("--wer_history_path", default="wer_history.pkl")
    p.add_argument("--save_audio_dir",   default="eval_wavs",
                   help="Directory to dump input .wav files")
    return p.parse_args()


def get_adapter(kind, a):
    if kind == "fc":
        return FCAdapter(fc_layer_dim=a.fc_layer_dim,
                         llm_hidden_size=a.llm_hidden_size)
    return TransformerAdapter(transform_dim=a.transformer_dim,
                              num_heads=a.num_heads,
                              ff_dim=a.feed_forward_dim,
                              num_layers=a.num_transformer_layers,
                              dropout=a.transformer_dropout,
                              llm_hidden_size=a.llm_hidden_size)


def collate_fn(batch, asr_proc, llm_tok, max_dur, max_len):
    audio_raw = [x["speech_input"] for x in batch]             
    audio_inputs = asr_proc(audio=audio_raw, sampling_rate=SR,
                            max_length=int(max_dur * SR),
                            padding="max_length", truncation=True,
                            return_tensors="pt")

    # prompt_raw = [x['text_prompt'] for x in batch]
    prompt_raw = [f"{asr_proc.tokenizer.eos_token}{x['text_prompt']}" for x in batch]
    txt_in = llm_tok(prompt_raw, max_length=max_len,
                     padding="max_length", truncation=True,
                     return_tensors="pt")

    # ref_raw = [x['text_response'] for x in batch]
    ref_raw = [f"{asr_proc.tokenizer.eos_token}{x['text_response']}" for x in batch]
    txt_out = llm_tok(ref_raw, max_length=max_len,
                      padding="max_length", truncation=True,
                      return_tensors="pt")

    return {"audio_features": audio_inputs.input_features,
            "text_inputs": txt_in.input_ids,
            "text_outputs": txt_out.input_ids,
            "prompt_raw": prompt_raw,
            "reference_raw": ref_raw,
            "audio_raw": audio_raw}


def main():
    args = parse_args()
    device = torch.device("cpu")

    Path(args.save_audio_dir).mkdir(parents=True, exist_ok=True)

    ds = UnifiedSpeechDataset(token=args.token, lang=args.dataset_lang,
                              split="test", subset=args.dataset_subset)

    asr_model = WhisperForConditionalGeneration.from_pretrained(
        args.asr_model_name, token=args.token)
    asr_proc  = AutoProcessor.from_pretrained(
        args.asr_model_name, token=args.token)

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, token=args.token)
    llm_tok   = AutoTokenizer.from_pretrained(
        args.llm_model_name, token=args.token)
    llm_tok.add_special_tokens({"pad_token": "[PAD]"})

    adapter = get_adapter(args.adapter_type, args)
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    adapter.load_state_dict(ckpt["adapter_state_dict"])

    asr_enc = asr_model.model.encoder
    llm_emb = llm_model.model.embed_tokens
    llm_model.model.embed_tokens = torch.nn.Identity()  

    collate = functools.partial(collate_fn, asr_proc=asr_proc,
                                llm_tok=llm_tok, max_dur=MAX_DURATION,
                                max_len=args.max_text_length)
    dl = DataLoader(ds.task_dataset(args.task), batch_size=args.batch_size,
                    collate_fn=collate, shuffle=False)

    for m in (asr_enc, llm_emb, llm_model, adapter):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    metric_wer = evaluate.load("wer")

    wer_history, logits_all, records = [], [], []
    sample_id = 0

    pbar = tqdm(dl, desc="Scoring (CPU)")
    for batch in pbar:
        gc.collect()

        aud   = batch["audio_features"].to(device).float()
        tid   = batch["text_inputs"].to(device).long()
        tout  = batch["text_outputs"].to(device).long()

        with torch.no_grad(), nullcontext():
            enc_out = asr_enc(aud).last_hidden_state.detach()
            emb_llm = llm_emb(tid).detach()
            emb_cat = torch.cat([adapter(enc_out), emb_llm], dim=1)

            T = emb_cat.size(1)
            attn = torch.ones((emb_cat.size(0), T), device=device)

            labels = tout[:, 1:T + 1]
            if labels.size(1) != T:
                labels = torch.nn.functional.pad(
                    labels, (0, T - labels.size(1)),
                    value=llm_tok.pad_token_id)

            logits = llm_model(inputs_embeds=emb_cat,
                               attention_mask=attn,
                               labels=labels).logits

        preds_txt = llm_tok.batch_decode(logits.argmax(-1),
                                         skip_special_tokens=True)
        refs_txt  = llm_tok.batch_decode(labels,
                                         skip_special_tokens=True)
        wer = metric_wer.compute(predictions=preds_txt,
                                 references=refs_txt)
        wer_history.append(wer)
        pbar.set_postfix(wer=wer,
                         mem=f"{virtual_memory().used / (1024**3):.1f} GB")

        for i in range(len(preds_txt)):
            wav_path = Path(args.save_audio_dir) / f"audio_{sample_id}.wav"
            sf.write(wav_path, batch["audio_raw"][i], SR)
            records.append({
                "id": sample_id,
                "prompt":     batch["prompt_raw"][i],
                "reference":  batch["reference_raw"][i],
                "prediction": preds_txt[i],
                "wer":        wer,
                "audio_path": str(wav_path)
            })
            logits_all.append(logits[i].cpu())
            sample_id += 1

    Path(args.pred_jsonl_path).write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records),
        encoding="utf-8")

    torch.save(logits_all, args.logits_path)

    with open(args.wer_history_path, "wb") as f:
        pickle.dump(wer_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished. Mean WER:", sum(wer_history) / len(wer_history))


if __name__ == "__main__":
    main()
