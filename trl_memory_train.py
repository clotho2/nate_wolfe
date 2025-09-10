# filepath: trl_memory_train.py
"""
Minimal continued-pretraining script for persona memory JSONL produced by system_memory_text.
- Expects rows like: {"text": "<<SYSTEM_PERSONA>>\\n...\\n\\n<<MEMORY>>\\n..."}
- Packs sequences to block_size and trains causal LM.

Usage (PowerShell):
  python .\\trl_memory_train.py `
    --model-id meta-llama/Llama-3.1-8B `
    --data "C:\\Users\\azinv\\Documents\\nate_wolfe\\dataset-memory.jsonl" `
    --out ".\\llama3-nate-memory" `
    --block-size 4096 `
    --epochs 1 `
    --lr 2e-5 `
    --batch 1 `
    --grad-accum 8 `
    --fp16
"""
from __future__ import annotations
import argparse
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="Base model, e.g. meta-llama/Llama-3.1-8B")
    ap.add_argument("--data", required=True, help="Path to dataset-memory.jsonl")
    ap.add_argument("--out", required=True, help="Output dir for checkpoints")
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=1, help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # avoid pad issues

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Load JSONL with {"text": "..."} rows
    ds: Dataset = load_dataset("json", data_files=str(Path(args.data)), split="train")

    # Tokenize and pack
    def tok_fn(ex):
        return tok(ex["text"], add_special_tokens=False)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # Group into fixed-length blocks for efficient LM training
    def group_texts(examples):
        # Concatenate and split into block_size chunks
        import itertools
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(concatenated) // args.block_size) * args.block_size
        result = {
            "input_ids": [concatenated[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        }
        result["attention_mask"] = [[1] * len(x) for x in result["input_ids"]]
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
