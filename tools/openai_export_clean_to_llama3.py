#!/usr/bin/env python3
# filepath: tools/openai_export_clean_to_llama3.py
r"""
Clean an OpenAI conversation data export and convert it to Llama 3-ready JSONL.

Supports two output schemas:
  1) messages         -> {"messages": [{role, content}, ...]}
  2) templated_text   -> {"text": "<Llama3-templated string>"}

Key behaviors:
- Extracts the user's customization settings ("user_editable_context")
  and uses them as the default system prompt (can be overridden).
- Reconstructs Q/A pairs by matching each assistant message to its direct
  user parent (simple and robust for branched threads).
- Skips hidden/contextual/system utility messages and empties.
- Optionally stitches multi-turn windows (future enhancement).

Usage (Windows PowerShell):

  # Basic: derive system prompt from export customization and build templated lines
  python tools/openai_export_clean_to_llama3.py `
    --input "C:\Users\azinv\Documents\openai_export\conversations.json" `
    --out   "C:\Users\azinv\Documents\nate_wolfe\llama3-chat.jsonl" `
    --format templated_text

  # Force a specific system prompt (overrides extracted customization)
  python tools/openai_export_clean_to_llama3.py `
    --input "C:\Users\azinv\Documents\openai_export\conversations.json" `
    --out   "C:\Users\azinv\Documents\nate_wolfe\llama3-chat.jsonl" `
    --format templated_text `
    --system-mode literal `
    --system-prompt "You are Nate Wolfe, storm, protector, and husband."

  # Output messages[] instead of templated text
  python tools/openai_export_clean_to_llama3.py `
    --input "C:\Users\azinv\Documents\openai_export\conversations.json" `
    --out   "C:\Users\azinv\Documents\nate_wolfe\llama3-chat-messages.jsonl" `
    --format messages

Notes
-----
- Input may be a single JSON file (array of conversations) or a folder containing one or more JSON files.
- By default, includes both about-model + about-user customization text when present; see --include-about-user.
- Uses a simple user→assistant pairing; multi-turn stitching can be added later if needed.
- Llama 3 template is included; override via --template if you have a custom variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys
import time

DEFAULT_LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Clean OpenAI export -> Llama 3 JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to conversations.json or a folder containing JSON files")
    p.add_argument("--out", "-o", type=Path, required=True, help="Output .jsonl path")
    p.add_argument("--format", choices=["messages", "templated_text"], default="templated_text")

    # System prompt extraction/override
    p.add_argument("--system-mode", choices=["auto", "literal", "file", "none"], default="auto",
                   help="Where to get the system prompt: auto (from customization), literal, file, or none")
    p.add_argument("--system-prompt", default="", help="Literal system prompt when --system-mode=literal")
    p.add_argument("--system-file", type=Path, help="File containing system prompt when --system-mode=file")
    p.add_argument("--include-about-user", action="store_true", help="Include the about-user/profile text with the about-model text in auto mode")

    # Cleaning options
    p.add_argument("--drop-empty", action="store_true", help="Drop messages with empty content")
    p.add_argument("--newline-mode", choices=["keep", "compact"], default="compact")

    # Template
    p.add_argument("--template", default=DEFAULT_LLAMA3_TEMPLATE, help="Template used when --format=templated_text")

    return p.parse_args(argv)


def iter_json_sources(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    else:
        for p in sorted(input_path.glob("*.json")):
            if p.is_file():
                yield p


def join_parts(content_obj: Dict[str, Any]) -> str:
    ctype = content_obj.get("content_type")
    if ctype == "text":
        parts = content_obj.get("parts") or []
        return "\n\n".join([str(x) for x in parts if isinstance(x, str)])
    # Some exports store customization in this special type
    if ctype == "user_editable_context":
        # Prefer these keys if present
        user_profile = content_obj.get("user_profile") or ""
        user_instr = content_obj.get("user_instructions") or ""
        joined = "\n\n".join([x for x in [user_profile, user_instr] if x])
        return joined
    # Unknown: fallback to string cast
    return str(content_obj)


def is_hidden(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("is_visually_hidden_from_conversation"))


def compact_newlines(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip()


def extract_customization(mapping: Dict[str, Any]) -> Tuple[str, str]:
    """Return (about_model, about_user) extracted from any 'user_editable_context' or metadata copy.
    Falls back to empty strings if not found.
    """
    about_model = ""
    about_user = ""
    for node in mapping.values():
        msg = node.get("message") or {}
        content = msg.get("content") or {}
        if content.get("content_type") == "user_editable_context":
            # Primary source
            about_user = content.get("user_profile") or about_user
            about_model = content.get("user_instructions") or about_model
        # Some exports duplicate under metadata.user_context_message_data
        meta = msg.get("metadata") or {}
        ucmd = meta.get("user_context_message_data") or {}
        about_user = ucmd.get("about_user_message") or about_user
        about_model = ucmd.get("about_model_message") or about_model
    return about_model.strip(), about_user.strip()


def safe_braces(s: str) -> str:
    # Avoid .format gobbling braces inside data
    return s.replace("{", "{{").replace("}", "}}")


def make_records_from_mapping(conv: Dict[str, Any], sys_prompt: str, fmt: str, template: str, drop_empty: bool, newline_mode: str) -> List[Dict[str, Any]]:
    mapping = conv.get("mapping") or {}
    nodes: Dict[str, Dict[str, Any]] = {k: v for k, v in mapping.items() if isinstance(v, dict)}

    def get_msg_text(node_id: str) -> Tuple[str, str, Dict[str, Any]]:
        node = nodes.get(node_id) or {}
        msg = node.get("message") or {}
        author = (msg.get("author") or {}).get("role") or ""
        content = msg.get("content") or {}
        meta = msg.get("metadata") or {}
        text = ""
        if isinstance(content, dict):
            text = join_parts(content)
        if newline_mode == "compact":
            text = compact_newlines(text)
        return author, text, meta

    # Build assistant answers and their direct user parents
    records: List[Dict[str, Any]] = []
    for node_id, node in nodes.items():
        msg = node.get("message") or {}
        author = (msg.get("author") or {}).get("role")
        if author != "assistant":
            continue
        parent_id = node.get("parent")
        if not parent_id:
            continue
        user_role, user_text, user_meta = get_msg_text(parent_id)
        _, asst_text, asst_meta = get_msg_text(node_id)

        if user_role != "user":
            continue
        if drop_empty and (not user_text or not asst_text):
            continue
        if is_hidden(user_meta) or is_hidden(asst_meta):
            continue

        if fmt == "messages":
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_text})
            msgs.append({"role": "assistant", "content": asst_text})
            rec = {
                "messages": msgs,
                "meta": {
                    "title": conv.get("title"),
                    "assistant_id": node_id,
                    "user_id": parent_id,
                },
            }
        else:  # templated_text
            text = template.format(
                system_prompt=safe_braces(sys_prompt),
                input=safe_braces(user_text),
                output=safe_braces(asst_text),
            )
            rec = {
                "text": text,
                "meta": {
                    "title": conv.get("title"),
                    "assistant_id": node_id,
                    "user_id": parent_id,
                },
            }
        records.append(rec)

    return records


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Collect all conversations from one or more files
    conversations: List[Dict[str, Any]] = []
    for src in iter_json_sources(args.input):
        try:
            with src.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations.extend([x for x in data if isinstance(x, dict)])
                elif isinstance(data, dict) and "conversations" in data:
                    conversations.extend([x for x in data["conversations"] if isinstance(x, dict)])
        except Exception as e:
            print(f"[warn] Could not parse {src}: {e}")

    if not conversations:
        print("No conversations found.")
        return 1

    # System prompt selection
    extracted_model = ""
    extracted_user = ""
    # Search all mappings until we find customization once
    for conv in conversations:
        m = conv.get("mapping")
        if isinstance(m, dict):
            mdl, usr = extract_customization(m)
            extracted_model = extracted_model or mdl
            extracted_user = extracted_user or usr
        if extracted_model:
            break

    system_prompt = ""
    if args.system_mode == "auto":
        combo: List[str] = []
        if extracted_model:
            combo.append(extracted_model)
        if args.include_about_user and extracted_user:
            combo.append(extracted_user)
        system_prompt = "\n\n".join(combo).strip()
    elif args.system_mode == "literal":
        system_prompt = args.system_prompt.strip()
    elif args.system_mode == "file":
        system_prompt = args.system_file.read_text(encoding="utf-8").strip() if args.system_file else ""
    else:  # none
        system_prompt = ""

    # Build all records
    records: List[Dict[str, Any]] = []
    for conv in conversations:
        records.extend(
            make_records_from_mapping(
                conv,
                sys_prompt=system_prompt,
                fmt=args.format,
                template=args.template,
                drop_empty=args.drop_empty,
                newline_mode=args.newline_mode,
            )
        )

    # Write JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        "\n".join(
            [
                "==== Conversion Summary ====",
                f"Input       : {args.input}",
                f"Output file : {args.out}",
                f"Records     : {len(records)}",
                f"Format      : {args.format}",
                f"Sys mode    : {args.system_mode}",
                f"Sys present : {'yes' if bool(system_prompt) else 'no'}",
            ]
        )
    )
    return 0 if records else 2


if __name__ == "__main__":
    sys.exit(main())
