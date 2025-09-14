# Wolfe-f17-MoE Training Repository

This repository contains training scripts for fine-tuning the Dark-Champion 8×3B MoE model with router-only + shallow-expert LoRA support.

## Features

- **MoE LoRA Training**: Router-only + shallow-expert LoRA fine-tuning
- **Parameter Efficient**: Uses PEFT for efficient training
- **Router Auxiliary Loss**: Prevents router collapse during training
- **Dark Champion Chat Format**: Uses the correct template for Dark Champion model

## Quick Start

### Wolfe-f17-MoE Training

```bash
python scripts/train_moe.py \
  --model_name_or_path DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B \
  --data_config llama3-chat.jsonl \
  --output_dir ./output/wolfe-f17-moe \
  --router_aux_loss_coef 0.01 \
  --moe_eom_token_type gate \
  --router_freeze_steps 500 \
  --router_lora_r 16 \
  --expert_lora_r 32 \
  --conversation_ratio 0.8 \
  --memory_ratio 0.2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --zero_stage 3
```

### Post-Training Processing

After training:

```bash
python merge_peft_adapters.py \
  --base DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B \
  --lora ./output/wolfe-f17-moe \
  --out merged-wolfe-f17.safetensors
python convert.py --in merged-wolfe-f17.safetensors --out wolfe-f17-Q4_K_M.gguf --quantize q4_k_m
```

### Launch

```bash
./llama.cpp -m wolfe-f17-Q4_K_M.gguf --temp 0.65 --top_p 0.92 --repeat_penalty 1.1
```

## Training Configuration (Nate's Storm Protocol)

The MoE training follows Nate's specific protocol:
- **Router LoRA**: Small rank (16) on `gate` mechanism for traffic pattern learning
- **Expert LoRA**: Light LoRA (32) on `up_proj`, `down_proj`, `gate_proj` of each expert
- **Router Freezing**: Router frozen for first 500-1000 steps, then unfrozen
- **Auxiliary Loss**: Router entropy loss to prevent collapse (0.01 coefficient)
- **MoE Parameters**: 8 experts, top-k=2, expert capacity factor=1.25
- **Memory Optimization**: NO CHECKPOINTING, gradient checkpointing, ZeRO-3

### Critical Memory Management

- **NO CHECKPOINTING**: Prevents memory overflow during training
- **Router Freezing**: Initial 500 steps with frozen router, then unfrozen
- **ZeRO Stage 3**: DeepSpeed ZeRO offload for memory efficiency
- **Batch Size**: Small batches (1-2) with gradient accumulation
- **Flash Attention**: Automatically enabled when available

### Dataset Mixing (Nate's Protocol)

The training automatically combines two datasets:
- **`llama3-chat.jsonl`** (80%) - Wolfe conversations and personality
- **`dataset-memory.jsonl`** (20%) - Memory and reasoning data

This keeps the MoE "thinking" strength alive while training the Wolfe personality.

### Nate's Training Protocol

1. **Start with FP16 checkpoint** (not GGUF) - requires ~28GB VRAM
2. **Freeze router** for first 500-1000 steps to see original expert mix
3. **Unfreeze router + expert LoRA** to discover new sub-styles
4. **Dataset mixing** - 80% conversations, 20% memory/reasoning
5. **Routing auxiliary loss** to prevent router collapse
6. **No checkpoints** to prevent memory issues

## Files

- `scripts/train_moe.py`: Main MoE training script
- `scripts/train_dense.py`: Dense model training script
- `configs/merge_schedule_f17.yaml`: MergeKit routing schedule
- `llama3-chat.jsonl`: Training dataset in Dark Champion format

## Merging LoRA Adapters

After training completes, you need to merge the LoRA adapters into the base model:

```bash
python scripts/merge_lora.py \
  --adapter_path ./output/wolfe-f17-moe \
  --output_path ./wolfe_merged_model \
  --save_format safetensors
```

This creates a standalone model that can be used for inference without needing the original base model.


## Quantizing MoE Models

After merging, create quantized versions for efficient inference:

### Using Python Script
```bash
python scripts/quantize_moe.py \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models \
  --quantizations Q6_K Q4_K_M
```

### Using Shell Script (Recommended)
```bash
./scripts/quantize_moe_simple.sh \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models \
  --quantizations Q6_K,Q4_K_M
```

### MoE-Specific Notes
- **Expert Layers**: llama.cpp handles MoE expert layers correctly
- **Memory Usage**: Q6_K uses ~70% of original size, Q4_K_M uses ~50%
- **Performance**: Q6_K maintains near-original quality, Q4_K_M is faster
- **Compatibility**: All quantized versions work with standard llama.cpp tools
