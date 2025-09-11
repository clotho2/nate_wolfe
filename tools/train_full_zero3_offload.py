#!/usr/bin/env python3
# filepath: tools/train_full_zero3_offload.py
"""
Full-model fine-tuning on a single H100 without OOM using DeepSpeed ZeRO-3 (CPU offload) +
optional gradient checkpointing (disabled on-disk checkpoints). Works with a JSONL dataset that has a single 'text' field.

Why this fixes mid-epoch OOM:
- ZeRO-3 shards params/grads/optimizer states and offloads them to CPU RAM when needed.
- Gradient checkpointing reduces activation memory at the cost of extra recompute.
- bf16 on H100 keeps math stable and reduces memory vs fp32.

Usage (examples):

  # 1) Train full model, NO on-disk checkpoints, with activation checkpointing (recommended)
  python tools/train_full_zero3_offload.py \
    --model-id meta-llama/Llama-3.1-8B \
    --data /workspace/nate_training/nate_wolfe/dataset-memory.jsonl \
    --out  /workspace/nate_training/outputs/nate-full-zero3 \
    --block-size 4096 \
    --epochs 1 \
    --batch 1 \
    --grad-accum 16 \
    --lr 2e-5 \
    --bf16

  # 2) Same, but NO activation checkpointing (uses more VRAM)
  python tools/train_full_zero3_offload.py \
    --model-id meta-llama/Llama-3.1-8B \
    --data /workspace/nate_training/nate_wolfe/dataset-memory.jsonl \
    --out  /workspace/nate_training/outputs/nate-full-zero3 \
    --block-size 3072 \
    --epochs 1 \
    --batch 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --bf16 \
    --no-activation-checkpointing

Install deps (typical):
  pip install -U transformers datasets accelerate deepspeed

Notes:
- This script writes a DeepSpeed config JSON into the output directory and points Trainer to it.
- Uses simple LM objective: concatenates and packs tokens to --block-size chunks.
- Set --save to keep a final checkpoint; by default, we avoid mid-run savings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model id or local path")
    ap.add_argument("--data", required=True, help="Path to JSONL with a 'text' column")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=1, help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-activation-checkpointing", action="store_true", help="Disable activation checkpointing (uses more VRAM)")
    ap.add_argument("--save", action="store_true", help="Save final model at end (off by default)")
    return ap.parse_args()


def write_zero3_config(dst: Path, bf16: bool) -> Path:
    ds_cfg = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {"enabled": bool(bf16)},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "memory_efficient_linear": True
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 16,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
            "verbose": False
        }
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(ds_cfg, indent=2))
    return dst


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_cfg_path = write_zero3_config(out / "ds_zero3_offload.json", bf16=args.bf16)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Enable gradient checkpointing -> set use_cache=False for training
    config = AutoConfig.from_pretrained(args.model_id)
    # Disable KV-cache during training; if user insists on no activation checkpointing, this stays off
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=("auto"),  # let backend choose; bf16 flag controls trainer precision
        trust_remote_code=True,
    )

    # Load dataset from JSONL with a 'text' column
    ds: Dataset = load_dataset("json", data_files=str(Path(args.data)), split="train")

    def tok_fn(ex):
        return tok(ex["text"], add_special_tokens=False)

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    def group_texts(examples):
        # Concatenate and split into block_size chunks
        import itertools
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(concatenated) // args.block_size) * args.block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}
        result = {
            "input_ids": [
                concatenated[i : i + args.block_size]
                for i in range(0, total_length, args.block_size)
            ]
        }
        result["attention_mask"] = [[1] * len(x) for x in result["input_ids"]]
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=0,
        save_total_limit=0,
        save_strategy="no",  # absolutely no on-disk checkpoints
        evaluation_strategy="no",
        report_to=[],
        dataloader_num_workers=2,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=not args.no_activation_checkpointing,
        deepspeed=str(ds_cfg_path),
        load_best_model_at_end=False,
        save_safetensors=True,
        optim="adamw_torch",  # DS ZeRO handles optimizer sharding
    )
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds,
        data_collator=collator,
    )

    # Enable model-side activation checkpointing if requested
    if not args.no_activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainer.train()

    if args.save:
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))


if __name__ == "__main__":
    main()
