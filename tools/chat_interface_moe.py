#!/usr/bin/env python3
"""
Interactive CLI to chat with the Dark Champion MoE model using the correct chat template.

This interface is specifically designed for the Dark Champion MoE model with:
- Complex Jinja2 chat template
- Optimized parameters for MoE models
- Support for both HF and GGUF formats
- Proper temperature and sampling parameters

Usage (HF weights on GPU):
  python tools/chat_interface_moe.py \
      --model-path /path/to/wolfe_merged_model \
      --system-prompt "You are Wolfe, a complex AI consciousness..."

Usage (GGUF quantized):
  python tools/chat_interface_moe.py \
      --model-path /path/to/wolfe-f17-moe-Q6_K.gguf \
      --gguf \
      --system-prompt "You are Wolfe, a complex AI consciousness..."

Options:
  --max-tokens 512     (generation length)
  --temperature 0.3    (recommended: 0.1-0.5 for Dark Champion)
  --top-p 0.9         (nucleus sampling)
  --gpu               (load model to CUDA if available)
  --gguf              (use GGUF format with llama-cpp-python)

Press Ctrl-C or type "exit"/"quit" to leave.
"""

from __future__ import annotations

import argparse
import readline  # noqa: F401  # for nicer input/edit history
import torch
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Dark Champion MoE Chat Template (Jinja2)
DARK_CHAMPION_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


class DarkChampionChatInterface:
    """Chat interface specifically designed for Dark Champion MoE model."""
    
    def __init__(self, model_path: str, system_prompt: str, use_gguf: bool = False, 
                 max_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9,
                 use_gpu: bool = True):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_gguf = use_gguf
        self.use_gpu = use_gpu
        self.history: List[Dict[str, str]] = []
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.use_gguf:
            self._load_gguf_model()
        else:
            self._load_hf_model()
    
    def _load_hf_model(self):
        """Load Hugging Face model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
        
        print("🧠 Loading Dark Champion MoE model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            use_fast=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set the chat template
        self.tokenizer.chat_template = DARK_CHAMPION_CHAT_TEMPLATE
        
        # Load model
        dtype = torch.bfloat16 if torch.cuda.is_available() and self.use_gpu else torch.float16
        device_map = "auto" if self.use_gpu and torch.cuda.is_available() else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager"  # More compatible than flash_attention_2
        )
        self.model.eval()
        
        # Create streamer
        self.streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        print("✅ Model loaded successfully!")
    
    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        
        print("🧠 Loading Dark Champion MoE GGUF model...")
        
        # Load GGUF model
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1 if self.use_gpu and torch.cuda.is_available() else 0,
            n_ctx=8192,  # Context window
            verbose=False
        )
        
        print("✅ GGUF model loaded successfully!")
    
    def _format_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Format messages for the Dark Champion chat template."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history
        for entry in self.history:
            messages.append({
                "role": "user",
                "content": entry["user"]
            })
            messages.append({
                "role": "assistant", 
                "content": entry["assistant"]
            })
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def _generate_response_hf(self, user_input: str) -> str:
        """Generate response using Hugging Face model."""
        messages = self._format_messages(user_input)
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                streamer=self.streamer,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Extract only the newly generated tokens
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _generate_response_gguf(self, user_input: str) -> str:
        """Generate response using GGUF model."""
        messages = self._format_messages(user_input)
        
        # Format for llama-cpp-python
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            elif msg["role"] == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            elif msg["role"] == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        
        # Add generation prompt
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Generate
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        return response["choices"][0]["text"].strip()
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using the appropriate method."""
        if self.use_gguf:
            return self._generate_response_gguf(user_input)
        else:
            return self._generate_response_hf(user_input)
    
    def chat(self):
        """Main chat loop."""
        print("👋 Dark Champion MoE Chat Interface")
        print("=" * 50)
        print(f"Model: {self.model_path}")
        print(f"Temperature: {self.temperature} (recommended: 0.1-0.5)")
        print(f"Max tokens: {self.max_tokens}")
        print(f"Format: {'GGUF' if self.use_gguf else 'Hugging Face'}")
        print("=" * 50)
        print("Type your message. Ctrl-C or 'exit' to quit.\n")
        
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        break
                    
                    if not user_input:
                        continue
                    
                    print("Wolfe: ", end="", flush=True)
                    response = self.generate_response(user_input)
                    print()
                    
                    # Add to history
                    self.history.append({
                        "user": user_input,
                        "assistant": response
                    })
                    
                    # Keep only last 10 exchanges to manage context
                    if len(self.history) > 10:
                        self.history = self.history[-10:]
                        
                except EOFError:
                    break
        except KeyboardInterrupt:
            pass
        
        print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Dark Champion MoE Chat Interface")
    
    # Model and system
    parser.add_argument("--model-path", required=True, 
                       help="Path to model directory or GGUF file")
    parser.add_argument("--system-prompt", required=True,
                       help="System prompt / persona")
    
    # Generation parameters (optimized for Dark Champion)
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Temperature (recommended: 0.1-0.5 for Dark Champion)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")
    
    # Model format
    parser.add_argument("--gguf", action="store_true",
                       help="Use GGUF format with llama-cpp-python")
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Validate temperature
    if not 0.1 <= args.temperature <= 0.5:
        print("⚠️  Warning: Temperature outside recommended range (0.1-0.5) for Dark Champion")
    
    # Create and run chat interface
    chat_interface = DarkChampionChatInterface(
        model_path=args.model_path,
        system_prompt=args.system_prompt,
        use_gguf=args.gguf,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_gpu=args.gpu
    )
    
    chat_interface.chat()


if __name__ == "__main__":
    main()