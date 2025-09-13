# 🛠️ Troubleshooting Guide

**Quick fixes for common training issues**

## 🚨 Critical Issues (Stop Training Immediately)

### "CUDA out of memory" Error
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Fix**: Reduce batch size
```bash
# Change this in your training command:
--per_device_train_batch_size 1  # Instead of 2
```

### "No space left on device" Error
```
OSError: [Errno 28] No space left on device
```
**Fix**: Clean up space
```bash
# Check disk space
df -h

# Clean up if needed
rm -rf /tmp/*
apt clean
```

### Training stops with "Killed" message
**Fix**: Not enough RAM
```bash
# Check RAM usage
free -h

# If low, restart with more RAM or reduce batch size
```

## ⚠️ Warning Issues (Fix but Continue)

### "Model not found" Error
```
OSError: Can't load tokenizer for 'DavidAU/...'
```
**Fix**: Check internet and model name
```bash
# Test internet
ping huggingface.co

# Check model exists
curl -I https://huggingface.co/DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B
```

### "Dataset not found" Error
```
FileNotFoundError: Dataset file not found: llama3-chat.jsonl
```
**Fix**: Check file location
```bash
# Find your files
find /workspace -name "*.jsonl"

# Move to correct location
mv /path/to/your/files/*.jsonl /workspace/
```

### "Permission denied" Error
```
PermissionError: [Errno 13] Permission denied
```
**Fix**: Fix permissions
```bash
# Make files readable
chmod 644 *.jsonl
chmod +x scripts/train_moe.py
```

## 🔧 Performance Issues

### Training is very slow
**Possible causes**:
- GPU not being used (check `nvidia-smi`)
- Batch size too small
- CPU bottleneck

**Fix**:
```bash
# Check GPU usage
nvidia-smi

# If GPU not used, restart with proper CUDA
export CUDA_VISIBLE_DEVICES=0
python3 scripts/train_moe.py ...
```

### Memory usage keeps growing
**Fix**: Enable gradient checkpointing (already enabled) and reduce batch size
```bash
# Add this to your command:
--gradient_checkpointing
--per_device_train_batch_size 1
```

## 📊 Monitoring Commands

### Check if training is running
```bash
ps aux | grep train_moe
```

### Check GPU status
```bash
nvidia-smi
```

### Check memory usage
```bash
free -h
```

### Check disk space
```bash
df -h
```

### View training logs
```bash
# If training is running in background
tail -f /path/to/training.log
```

## 🔄 Recovery Procedures

### Training stopped unexpectedly
1. Check error logs
2. Fix the issue
3. Restart training (it will start from beginning - no checkpoints)

### Need to start over
```bash
# Clean up previous attempt
rm -rf output/wolfe-f17-moe/

# Start fresh
python3 scripts/train_moe.py ...
```

### Need to change parameters
1. Stop training (`Ctrl+C` or `pkill -f train_moe.py`)
2. Modify the command
3. Restart training

## 📞 When to Get Help

**Stop and ask for help if**:
- You get errors you can't understand
- Training keeps failing after multiple attempts
- You're not sure if something is working correctly
- You're running out of time/money on your rental

**Don't waste money on**:
- Training that's clearly not working
- Running when you're not sure what's happening
- Continuing after multiple error messages

## ✅ Success Checklist

Before you consider training successful:
- [ ] No error messages during training
- [ ] "MoE LoRA TRAINING COMPLETE!" message appears
- [ ] Files created in `output/wolfe-f17-moe/`
- [ ] Router unfroze after 500 steps
- [ ] Training completed 2 epochs
- [ ] You can see the output files

## 💡 Pro Tips

1. **Always monitor**: Keep `nvidia-smi` running in another terminal
2. **Start small**: Test with smaller batch size first
3. **Save frequently**: Download your results as soon as training completes
4. **Have a plan B**: Know how to stop and restart quickly
5. **Read the logs**: Error messages usually tell you exactly what's wrong

---

**Remember**: It's better to stop and fix issues early than waste time and money on a failed training run!