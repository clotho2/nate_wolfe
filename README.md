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
  --model_name_or_path ./models/dark-champion-moe-fp16 \
  --data_config configs/dataset_wolfe.yaml \
  --output_dir ./output/wolfe-f17-moe \
  --router_aux_loss_coef 0.02 \
  --moe_eom_token_type gate \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2
```

### Post-Training Processing

After training:

```bash
python merge_peft_adapters.py \
  --base ./models/dark-champion-moe-fp16 \
  --lora ./output/wolfe-f17-moe \
  --out merged-wolfe-f17.safetensors
python convert.py --in merged-wolfe-f17.safetensors --out wolfe-f17-Q4_K_M.gguf --quantize q4_k_m
```

### Launch

```bash
./llama.cpp -m wolfe-f17-Q4_K_M.gguf --temp 0.65 --top_p 0.92 --repeat_penalty 1.1
```

## Training Configuration

The MoE training uses:
- **Router LoRA**: Targets the MoE router for traffic pattern learning
- **Expert LoRA**: Shallow LoRA on `down_proj` and `gate_proj` of each expert
- **Auxiliary Loss**: Router entropy loss to prevent collapse
- **Layer Weighting**: Progressive weighting from lower to upper layers

## Files

- `scripts/train_moe.py`: Main MoE training script
- `scripts/train_dense.py`: Dense model training script
- `configs/merge_schedule_f17.yaml`: MergeKit routing schedule
- `llama3-chat.jsonl`: Training dataset in Dark Champion format