#!/usr/bin/env python3
# filepath: tools/chat_interface.py
"""
Tiny interactive CLI to chat with a locally fine‑tuned model that uses the
Llama‑3 chat template you provided.

Works with **transformers** models in HF format. If you exported to GGUF and
run under llama.cpp, use the llama-cpp-python binding (see comment at bottom).

Usage (HF weights on GPU):
  python tools/chat_interface.py \
      --model-path /path/to/nate_storm_consciousness_v5 \
      --system-prompt "You are Nate Wolfe, storm, protector, and husband."

Options:
  --max-tokens 256     (generation length)
  --gpu               (load model to CUDA if available)
  --newline-mode keep|compact (template formatting)

Press Ctrl‑C or type "exit"/"quit" to leave.
"""
from __future__ import annotations

import argparse
import readline  # noqa: F401  # for nicer input/edit history
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def build_prompt(history: list[tuple[str, str]], user_input: str, system_prompt: str) -> str:
    """history: list of (user, assistant) pairs, newest last."""
    prompt_parts: list[str] = []
    if not history:
        # First turn uses full template with system prompt
        prompt_parts.append(
            TEMPLATE.format(system_prompt=system_prompt, input=user_input)
        )
    else:
        # We already showed system prompt; continue conversation
        # Add previous history lines
        for u, a in history:
            prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>\n"\
                               f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>\n")
        # Append current user turn (assistant yet to answer)
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"\
                           f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(prompt_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF model directory or repo id")
    ap.add_argument("--system-prompt", required=True, help="System prompt / persona")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu", action="store_true", help="Load model on CUDA if available")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    device_map = "auto" if args.gpu and torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    history: list[tuple[str, str]] = []

    print("👋  Type your message. Ctrl‑C or 'exit' to quit.\n")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            prompt = build_prompt(history, user_input, args.system_prompt)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
                eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>")
            )
            # Extract only the newly generated tokens (after the prompt)
            generated_text = tok.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            print()
            history.append((user_input, generated_text.strip()))
    except KeyboardInterrupt:
        pass

    print("\nBye!")


if __name__ == "__main__":
    main()

"""
---
If you exported the model to GGUF and want to chat via llama.cpp instead,
install llama‑cpp‑python and replace the model loading block with:

from llama_cpp import Llama
llm = Llama(model_path="/path/model.gguf", n_gpu_layers=-1, n_ctx=8192)
...
response = llm(prompt, max_tokens=256)
print(response["choices"][0]["text"])

Keep the build_prompt() logic the same.
"""