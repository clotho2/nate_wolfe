# Quantization Guide - Fixed Scripts

The shell script had issues finding llama.cpp tools. Here are the working solutions:

## 🚀 **Method 1: Simple Python Script (Recommended)**

This is the easiest and most reliable method:

```bash
# Install llama-cpp-python (if not already installed)
pip install llama-cpp-python

# Run the simple quantization script
python scripts/quantize_simple.py \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models \
  --quantizations Q6_K Q4_K_M
```

## 🔧 **Method 2: Fixed Shell Script**

If you want to use the shell script, run it from the nate_wolfe directory:

```bash
# Make sure you're in the nate_wolfe directory
cd /path/to/nate_wolfe

# Run the fixed shell script
./scripts/quantize_moe_simple.sh \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models \
  --quantizations Q6_K,Q4_K_M
```

The script will now:
1. Check for llama.cpp tools in multiple locations
2. Install llama.cpp if not found
3. Use the correct paths for the tools

## 🐛 **Method 3: Manual llama.cpp Setup**

If both scripts fail, set up llama.cpp manually:

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc)

# Go back to nate_wolfe directory
cd ..

# Run the shell script
./scripts/quantize_moe_simple.sh \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models
```

## 📊 **Expected Output**

All methods will create:
- `wolfe-f17-moe-f16.gguf` - Full precision (18GB)
- `wolfe-f17-moe-Q6_K.gguf` - Q6_K quantized (12GB)
- `wolfe-f17-moe-Q4_K_M.gguf` - Q4_K_M quantized (9GB)

## 🚨 **Troubleshooting**

### Error: "llama.cpp not found"
- Use Method 1 (Python script) - most reliable
- Or install llama-cpp-python: `pip install llama-cpp-python`

### Error: "llama-convert not found"
- The shell script will try to install llama.cpp automatically
- Or use Method 3 to set up llama.cpp manually

### Error: "Permission denied"
- Make scripts executable: `chmod +x scripts/*.sh scripts/*.py`

### Error: "No space left on device"
- Check disk space: `df -h`
- Use a smaller quantization: `--quantizations Q4_K_M`

## 🎯 **Quick Start (Recommended)**

```bash
# 1. Install Python package
pip install llama-cpp-python

# 2. Run simple script
python scripts/quantize_simple.py \
  --model_path ./wolfe_merged_model \
  --output_dir ./quantized_models

# 3. Check results
ls -lh ./quantized_models/
```

This should work reliably without any llama.cpp installation issues! 🚀