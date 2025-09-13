# 🚀 Complete Training Guide for Wolfe-f17-MoE

**For beginners who have never done AI model training before**

This guide will walk you through training the Dark Champion MoE model with Wolfe's personality on a rented H200 GPU. Follow these steps exactly to avoid wasting time and money.

## 📋 Prerequisites Checklist

Before you start, make sure you have:
- [ ] Rented an H200 GPU on Vast.ai (or similar service)
- [ ] Access to the training repository
- [ ] Both dataset files: `llama3-chat.jsonl` and `dataset-memory.jsonl`
- [ ] About 2-4 hours of training time (depending on your setup)

## 🖥️ Step 1: Connect to Your H200

1. **SSH into your rented H200**:
   ```bash
   ssh root@your-h200-ip-address
   ```

2. **Check your GPU**:
   ```bash
   nvidia-smi
   ```
   You should see your H200 with ~80GB VRAM.

## 📁 Step 2: Set Up the Repository

1. **Clone or upload the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Or if you have the files locally, upload them**:
   ```bash
   # Upload your files to the H200
   scp -r /path/to/your/files root@your-h200-ip:/workspace/
   ```

3. **Navigate to the workspace**:
   ```bash
   cd /workspace
   ```

## 🐍 Step 3: Install Required Software

1. **Update the system**:
   ```bash
   apt update && apt upgrade -y
   ```

2. **Install Python and pip**:
   ```bash
   apt install python3 python3-pip -y
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install training dependencies**:
   ```bash
   pip3 install transformers datasets peft trl accelerate bitsandbytes
   ```

5. **Install additional utilities**:
   ```bash
   pip3 install numpy pandas tqdm
   ```

## 📊 Step 4: Verify Your Data Files

1. **Check that your data files exist**:
   ```bash
   ls -la *.jsonl
   ```
   You should see:
   - `llama3-chat.jsonl` (Wolfe conversations)
   - `dataset-memory.jsonl` (Memory data)

2. **Check file sizes** (optional but recommended):
   ```bash
   du -h *.jsonl
   ```

3. **Quick data check** (optional):
   ```bash
   head -n 2 llama3-chat.jsonl
   head -n 2 dataset-memory.jsonl
   ```

## 🔧 Step 5: Prepare the Training Environment

1. **Create output directory**:
   ```bash
   mkdir -p output
   ```

2. **Set environment variables for memory optimization**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **Check available memory**:
   ```bash
   free -h
   nvidia-smi
   ```

## 🚀 Step 6: Run the Training (The Big Moment!)

**⚠️ IMPORTANT: This is where you run the actual training. Make sure you have enough time and the H200 is working properly.**

1. **Start the training**:
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

2. **What to expect**:
   - The model will download (~18GB) - this takes 5-10 minutes
   - You'll see loading messages for your datasets
   - Training will start with "Router FROZEN" for first 500 steps
   - After 500 steps, you'll see "Router UNFROZEN"
   - Training will continue for 2 epochs

3. **Monitor the training**:
   - Watch for error messages
   - Check GPU memory usage: `nvidia-smi`
   - Training should take 2-4 hours total

## 📈 Step 7: Monitor Training Progress

1. **Check GPU usage** (in another terminal):
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Check training logs**:
   - The training will print progress every 10 steps
   - Look for "Step X: Router FROZEN/UNFROZEN" messages
   - Watch for any error messages

3. **If something goes wrong**:
   - Press `Ctrl+C` to stop training
   - Check error messages
   - Fix the issue and restart

## ✅ Step 8: Verify Training Completed Successfully

1. **Check the output directory**:
   ```bash
   ls -la output/wolfe-f17-moe/
   ```

2. **You should see**:
   - `adapter_config.json` - LoRA configuration
   - `adapter_model.safetensors` - Trained LoRA weights
   - `tokenizer.json` - Tokenizer files
   - Other model files

3. **Check file sizes**:
   ```bash
   du -h output/wolfe-f17-moe/*
   ```

## 💾 Step 9: Save Your Results

1. **Create a backup**:
   ```bash
   tar -czf wolfe-f17-moe-backup.tar.gz output/wolfe-f17-moe/
   ```

2. **Download to your local machine**:
   ```bash
   # From your local machine:
   scp root@your-h200-ip:/workspace/wolfe-f17-moe-backup.tar.gz ./
   ```

3. **Or upload to cloud storage** (recommended):
   ```bash
   # Upload to Google Drive, Dropbox, etc.
   ```

## 🔧 Troubleshooting Common Issues

### Issue: "Out of memory" error
**Solution**: Reduce batch size
```bash
# Change --per_device_train_batch_size from 2 to 1
--per_device_train_batch_size 1
```

### Issue: "Model not found" error
**Solution**: Check internet connection and model name
```bash
# Test internet
ping huggingface.co

# Check model exists
curl -I https://huggingface.co/DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B
```

### Issue: "Dataset not found" error
**Solution**: Check file paths
```bash
ls -la *.jsonl
pwd
```

### Issue: Training stops unexpectedly
**Solution**: Check logs and restart
```bash
# Check system logs
dmesg | tail -20

# Check if GPU is still working
nvidia-smi
```

## ⏱️ Time Estimates

- **Setup and installation**: 30-45 minutes
- **Model download**: 5-10 minutes
- **Training (2 epochs)**: 2-4 hours
- **Total time**: 3-5 hours

## 💰 Cost Optimization Tips

1. **Monitor your usage**: Check Vast.ai dashboard regularly
2. **Stop immediately if errors**: Don't let it run if something's wrong
3. **Use spot instances**: Often cheaper than on-demand
4. **Have a backup plan**: Know how to stop the instance quickly

## 🎯 Success Indicators

You'll know training was successful if you see:
- ✅ "MoE LoRA TRAINING COMPLETE!" message
- ✅ Files in `output/wolfe-f17-moe/` directory
- ✅ No error messages during training
- ✅ Router unfroze after 500 steps
- ✅ Training completed 2 epochs

## 📞 Getting Help

If you run into issues:
1. Check the error messages carefully
2. Verify all files are in the right place
3. Make sure your H200 has enough memory
4. Check that all dependencies are installed

## 🎉 What's Next?

After successful training, you'll have:
- A trained LoRA adapter for the Dark Champion model
- Wolfe's personality embedded in the model
- A model that can be used for inference

The next step would be to merge the LoRA adapter with the base model and convert it to GGUF format for deployment.

---

**Remember**: Take your time, read the error messages carefully, and don't hesitate to stop and troubleshoot if something doesn't look right. It's better to fix issues early than waste time and money on a failed training run.