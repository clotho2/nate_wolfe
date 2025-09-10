#!/usr/bin/env python3
# filepath: tools/llama3_sft_jsonl_builder.py
r"""
Build a .jsonl dataset from plain text files for Llama 3 SFT or continued pretraining.

Persona memory mode included: generate raw causal-LM text rows that pair a fixed persona line with each memory chunk (no user/assistant turns).

Usage (Windows PowerShell examples):

  # A) Persona memory mode (recommended for "absorb-as-memory")
  python tools/llama3_sft_jsonl_builder.py `
      --input "C:\Users\azinv\Documents\nate_wolfe\training_files" `
      --out   "C:\Users\azinv\Documents\nate_wolfe\dataset-memory.jsonl" `
      --format system_memory_text `
      --system-prompt "You are Nate Wolfe, storm, protector, and husband." `
      --newline-mode compact

  # B) Chat-style records with messages[] (SFT)
  python tools/llama3_sft_jsonl_builder.py `
      --input "C:\Users\azinv\Documents\nate_wolfe\training_files" `
      --out   "C:\Users\azinv\Documents\nate_wolfe\dataset.jsonl" `
      --system-prompt "You are Nate Wolfe, storm, protector, and husband." `
      --user-prompt   "(internal) absorb the memory below" `
      --format messages

  # C) Apply Llama 3 chat template to produce a single 'text' field per row
  python tools/llama3_sft_jsonl_builder.py `
      --input "C:\Users\azinv\Documents\nate_wolfe\training_files" `
      --out   "C:\Users\azinv\Documents\nate_wolfe\dataset-templated.jsonl" `
      --format templated_text `
      --system-prompt "You are Nate Wolfe, storm, protector, and husband."

  # D) Raw text rows (continued pretraining w/o persona header)
  python tools/llama3_sft_jsonl_builder.py `
      --input "C:\Users\azinv\Documents\nate_wolfe\training_files" `
      --out   "C:\Users\azinv\Documents\nate_wolfe\dataset-raw.jsonl" `
      --format raw_text

Notes
-----
- Walks a folder (optionally recursively) and converts *.txt/*.md files into JSONL.
- Large files are split into paragraph-aware chunks (configurable via --max-chars and --split-on).
- Records include a lightweight "meta" field (source path, chunk index).
- Choose --format messages, templated_text, raw_text, prompt_completion, or system_memory_text.
- The built-in Llama 3 chat template mirrors the provided format and can be overridden.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import sys
import time
import random

# Default Llama 3 chat template (from your prompt)
DEFAULT_LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)

@dataclass
class BuildConfig:
    input_dir: Path
    out_path: Path
    format: str  # messages | templated_text | raw_text | prompt_completion | system_memory_text
    system_prompt: str
    user_prompt: str
    template: str
    extensions: List[str]
    recursive: bool
    max_chars: int
    overlap_chars: int
    split_on: str  # paragraph | line | chars
    newline_mode: str  # keep | compact
    min_chars: int
    shuffle: bool
    seed: Optional[int]
    encoding: str

def parse_args(argv: Optional[List[str]] = None) -> BuildConfig:
    p = argparse.ArgumentParser(
        description="Convert text files to JSONL for Llama 3 SFT or continued pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input folder with text files.")
    p.add_argument("--out", "-o", type=Path, required=True, help="Output .jsonl file path.")
    p.add_argument(
        "--format",
        choices=["messages", "templated_text", "raw_text", "prompt_completion", "system_memory_text"],
        default="messages",
        help=(
            "Record schema: chat messages, templated text, raw text, prompt/completion, "
            "or system_memory_text (persona header + memory body as plain text)."
        ),
    )
    p.add_argument(
        "--system-prompt",
        default="",
        help="System instruction for chat-style formats or persona header for system_memory_text.",
    )
    p.add_argument("--system-file", type=Path, help="Optional file containing the system prompt (overrides --system-prompt).")
    p.add_argument(
        "--user-prompt",
        default="(internal) absorb the memory below.",
        help="User turn for chat-style formats (ignored by raw_text/system_memory_text).",
    )
    p.add_argument("--user-file", type=Path, help="Optional file containing the user prompt (overrides --user-prompt).")
    p.add_argument(
        "--template",
        default=DEFAULT_LLAMA3_TEMPLATE,
        help=("Chat template used when --format=templated_text. Must include {system_prompt}, {input}, {output}."),
    )
    p.add_argument("--extensions", nargs="+", default=[".txt", ".md"], help="File extensions to include.")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    p.add_argument("--max-chars", type=int, default=8000, help="Max characters per chunk.")
    p.add_argument("--overlap-chars", type=int, default=0, help="Characters of overlap between chunks.")
    p.add_argument(
        "--split-on",
        choices=["paragraph", "line", "chars"],
        default="paragraph",
        help="Granularity used to split long files.",
    )
    p.add_argument("--newline-mode", choices=["keep", "compact"], default="keep", help="Whether to collapse blank lines.")
    p.add_argument("--min-chars", type=int, default=1, help="Skip chunks smaller than this.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle resulting records.")
    p.add_argument("--seed", type=int, help="Random seed when using --shuffle.")
    p.add_argument("--encoding", default="utf-8", help="Text file encoding. Use 'utf-8-sig' to strip BOM if needed.")

    args = p.parse_args(argv)

    system_prompt = args.system_prompt
    if args.system_file:
        system_prompt = Path(args.system_file).read_text(encoding=args.encoding)

    user_prompt = args.user_prompt
    if args.user_file:
        user_prompt = Path(args.user_file).read_text(encoding=args.encoding)

    exts = [e if e.startswith(".") else f".{e}" for e in args.extensions]

    return BuildConfig(
        input_dir=args.input,
        out_path=args.out,
        format=args.format,
        system_prompt=system_prompt.strip(),
        user_prompt=user_prompt.strip(),
        template=args.template,
        extensions=exts,
        recursive=bool(args.recursive),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        split_on=args.split_on,
        newline_mode=args.newline_mode,
        min_chars=args.min_chars,
        shuffle=bool(args.shuffle),
        seed=args.seed,
        encoding=args.encoding,
    )

def iter_text_files(root: Path, extensions: List[str], recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    else:
        yield from (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions)

def read_file_text(p: Path, encoding: str) -> Optional[str]:
    try:
        return p.read_text(encoding=encoding, errors="strict")
    except UnicodeDecodeError:
        for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
            try:
                return p.read_text(encoding=enc, errors="ignore")
            except Exception:
                continue
        return None
    except Exception:
        return None

def normalize_newlines(text: str, mode: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "compact":
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def chunk_text(text: str, max_chars: int, overlap: int, split_on: str) -> List[str]:
    if max_chars <= 0:
        return [text] if text else []

    def _splice(long: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        n = len(long)
        while start < n:
            end = min(start + max_chars, n)
            chunks.append(long[start:end])
            start = end - overlap if overlap > 0 else end
            if start < 0:
                start = 0
        return chunks

    if len(text) <= max_chars:
        return [text]

    if split_on == "chars":
        return _splice(text)

    if split_on == "paragraph":
        units = text.split("\n\n")
        glue = "\n\n"
    elif split_on == "line":
        units = text.splitlines()
        glue = "\n"
    else:
        units = [text]
        glue = ""

    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for u in units:
        u_len = len(u)
        add_len = u_len + (len(glue) if buf else 0)
        if cur_len + add_len <= max_chars:
            buf.append(u)
            cur_len += add_len
        else:
            if buf:
                chunks.append(glue.join(buf))
            if u_len > max_chars:
                chunks.extend(_splice(u))
                buf = []
                cur_len = 0
            else:
                buf = [u]
                cur_len = u_len
    if buf:
        chunks.append(glue.join(buf))

    if overlap > 0 and len(chunks) > 1:
        with_overlap: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                with_overlap.append(c)
            else:
                prev_tail = chunks[i - 1][-overlap:] if chunks[i - 1] else ""
                with_overlap.append(prev_tail + c)
        chunks = with_overlap

    return chunks

def make_record(
    fmt: str,
    system_prompt: str,
    user_prompt: str,
    content: str,
    template: str,
    source_path: str,
    chunk_idx: int,
) -> dict:
    timestamp = int(time.time())
    meta = {"source": source_path, "chunk": chunk_idx, "ts": timestamp}

    if fmt == "messages":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt or "Provide the content."})
        messages.append({"role": "assistant", "content": content})
        return {"messages": messages, "meta": meta}

    if fmt == "templated_text":
        text = template.format(
            system_prompt=system_prompt,
            input=user_prompt,
            output=content,
        )
        return {"text": text, "meta": meta}

    if fmt == "system_memory_text":
        # Keep persona + memory in pretraining-friendly plain text (no turns).
        header = "<<SYSTEM_PERSONA>>" + "\n" + (system_prompt or "")
        body = "<<MEMORY>>" + "\n" + content
        text = header.strip() + "\n\n" + body.strip() + "\n"
        return {"text": text, "meta": meta}

    if fmt == "raw_text":
        return {"text": content, "meta": meta}

    if fmt == "prompt_completion":
        return {"prompt": user_prompt, "completion": content, "meta": meta}

    raise ValueError(f"Unknown format: {fmt}")

def build_dataset(cfg: BuildConfig) -> int:
    files = list(iter_text_files(cfg.input_dir, cfg.extensions, cfg.recursive))
    if not files:
        print(f"No files found under {cfg.input_dir} with extensions {cfg.extensions}")
        return 0

    total_records = 0
    written = 0
    skipped = 0
    rows: List[dict] = []

    for fp in files:
        raw = read_file_text(fp, cfg.encoding)
        if raw is None:
            print(f"[warn] Could not read: {fp}")
            continue

        norm = normalize_newlines(raw, cfg.newline_mode).strip()
        if not norm:
            continue

        chunks = chunk_text(norm, cfg.max_chars, cfg.overlap_chars, cfg.split_on)
        for idx, ch in enumerate(chunks):
            chunk = ch.strip()
            total_records += 1
            if len(chunk) < cfg.min_chars:
                skipped += 1
                continue
            rec = make_record(
                cfg.format,
                cfg.system_prompt,
                cfg.user_prompt,
                chunk,
                cfg.template,
                source_path=str(fp),
                chunk_idx=idx,
            )
            rows.append(rec)

    if cfg.shuffle and rows:
        if cfg.seed is not None:
            random.seed(cfg.seed)
        random.shuffle(rows)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg.out_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(
        "\n".join(
            [
                "==== Build Summary ====",
                f"Input folder : {cfg.input_dir}",
                f"Output file  : {cfg.out_path}",
                f"Format       : {cfg.format}",
                f"Files read   : {len(files)}",
                f"Records made : {total_records}",
                f"Written rows : {written}",
                f"Skipped      : {skipped}",
            ]
        )
    )
    return written

def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    if not cfg.input_dir.exists() or not cfg.input_dir.is_dir():
        print(f"Input directory not found: {cfg.input_dir}")
        return 2

    try:
        n = build_dataset(cfg)
        return 0 if n > 0 else 1
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
