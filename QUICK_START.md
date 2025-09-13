# ⚡ Quick Start Reference

**Essential commands for H200 training - copy and paste these**

## 🚀 One-Command Setup

```bash
# Install everything needed
apt update && apt upgrade -y && apt install python3 python3-pip -y && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip3 install transformers datasets peft trl accelerate bitsandbytes numpy pandas tqdm
```

## 🔍 Verify Everything is Ready

```bash
# Check GPU
nvidia-smi

# Check data files
ls -la *.jsonl

# Check memory
free -h
```

## 🚀 Start Training (The Big One!)

```bash
python3 scripts/train_moe.py \
  --model_name_or_path DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B \
  --data_config llama3-chat.jsonl \
  --output_dir ./output/wolfe-f17-moe \
  --router_aux_loss_coef 0.01 \
  --moe_eom_token_type gate \
  --num_experts 8 \
  --top_k 2 \
  --router_freeze_steps 500 \
  --router_lora_r 16 \
  --expert_lora_r 32 \
  --conversation_ratio 0.8 \
  --memory_ratio 0.2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --zero_stage 3 \
  --bf16
```

## 📊 Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check if training is still running
ps aux | grep python
```

## ✅ Check Results

```bash
# See what was created
ls -la output/wolfe-f17-moe/

# Check file sizes
du -h output/wolfe-f17-moe/*
```

## 🛑 Emergency Stop

```bash
# Stop training immediately
pkill -f train_moe.py

# Or press Ctrl+C in the training terminal
```

## 💾 Save Your Work

```bash
# Create backup
tar -czf wolfe-f17-moe-backup.tar.gz output/wolfe-f17-moe/

# Download to your computer (run from your local machine)
scp root@your-h200-ip:/workspace/wolfe-f17-moe-backup.tar.gz ./
```

---

**Total time: 3-5 hours | Cost: ~$20-40 on Vast.ai | Result: Trained Wolfe personality model**