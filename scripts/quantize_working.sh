#!/bin/bash
# Working quantization script that uses the Python scripts directly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Default values
MODEL_PATH=""
OUTPUT_DIR="./quantized_models"
QUANTIZATIONS=("Q6_K" "Q4_K_M")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quantizations)
            IFS=',' read -ra QUANTIZATIONS <<< "$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    print_error "Model path is required. Use: $0 --model_path /path/to/model"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check if llama.cpp directory exists
if [ ! -d "./llama.cpp" ]; then
    print_error "llama.cpp directory not found!"
    print_info "Please run this from the nate_wolfe directory where llama.cpp is cloned"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

print_info "🚀 Starting MoE model quantization..."
print_info "   Model: $MODEL_PATH"
print_info "   Output: $OUTPUT_DIR"
print_info "   Quantizations: ${QUANTIZATIONS[*]}"

# Convert to GGUF using Python script (only if not already exists)
if [ -f "$OUTPUT_DIR/model.gguf" ]; then
    print_info "✅ GGUF file already exists, skipping conversion"
    file_size=$(du -h "$OUTPUT_DIR/model.gguf" | cut -f1)
    print_info "   Existing file size: $file_size"
else
    print_info "🔄 Converting model to GGUF format..."
    python3 ./llama.cpp/convert_hf_to_gguf.py \
        "$MODEL_PATH" \
        --outfile "$OUTPUT_DIR/model.gguf" \
        --outtype f16

    if [ $? -eq 0 ]; then
        print_success "Model converted to GGUF"
    else
        print_error "Conversion failed!"
        exit 1
    fi
fi

# Keep full precision GGUF
cp "$OUTPUT_DIR/model.gguf" "$OUTPUT_DIR/wolfe-f17-moe-f16.gguf"
file_size=$(du -h "$OUTPUT_DIR/wolfe-f17-moe-f16.gguf" | cut -f1)
print_success "Full precision GGUF saved: $file_size"

# Quantize to different formats
for quant_type in "${QUANTIZATIONS[@]}"; do
    output_file="$OUTPUT_DIR/wolfe-f17-moe-${quant_type}.gguf"
    
    if [ -f "$output_file" ]; then
        print_info "✅ $quant_type file already exists, skipping quantization"
        file_size=$(du -h "$output_file" | cut -f1)
        print_info "   Existing file size: $file_size"
        continue
    fi
    
    print_info "🔄 Quantizing to $quant_type..."
    
    # Try different locations for llama-quantize
    if command -v llama-quantize >/dev/null 2>&1; then
        llama-quantize "$OUTPUT_DIR/model.gguf" "$output_file" "$quant_type"
    elif [ -f "./llama.cpp/llama-quantize" ]; then
        ./llama.cpp/llama-quantize "$OUTPUT_DIR/model.gguf" "$output_file" "$quant_type"
    elif [ -f "./llama.cpp/build/llama-quantize" ]; then
        ./llama.cpp/build/llama-quantize "$OUTPUT_DIR/model.gguf" "$output_file" "$quant_type"
    else
        print_error "llama-quantize not found!"
        print_info "Trying Python method instead..."
        python3 -c "
import sys
sys.path.append('./llama.cpp')
from llama_cpp import quantize
quantize('$OUTPUT_DIR/model.gguf', '$output_file', '$quant_type')
"
    fi
    
    if [ $? -eq 0 ]; then
        file_size=$(du -h "$output_file" | cut -f1)
        print_success "$quant_type quantization complete: $file_size"
    else
        print_error "$quant_type quantization failed!"
    fi
done

print_success "🎉 All quantizations completed successfully!"
print_info "Quantized models saved to: $OUTPUT_DIR"
print_info "You can now use these models with llama.cpp!"