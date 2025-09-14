#!/bin/bash
# Simple MoE quantization script using llama.cpp tools directly
# This is more reliable than the Python wrapper

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
OUTPUT_DIR="./quantized_models"
QUANTIZATIONS=("Q6_K" "Q4_K_M")

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install llama.cpp
install_llama_cpp() {
    print_info "Installing llama.cpp..."
    
    # Clone llama.cpp if it doesn't exist
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    
    # Build llama.cpp
    make -j$(nproc)
    
    # Install Python bindings
    pip install llama-cpp-python
    
    cd ..
    
    print_success "llama.cpp installed successfully"
}

# Function to convert model to GGUF
convert_to_gguf() {
    local model_path="$1"
    local output_path="$2"
    
    print_info "Converting model to GGUF format..."
    
    # Use llama.cpp convert script
    python -m llama_cpp.convert_hf_to_gguf \
        "$model_path" \
        --outfile "$output_path" \
        --outtype f16
    
    print_success "Model converted to GGUF"
}

# Function to quantize model
quantize_model() {
    local input_path="$1"
    local output_path="$2"
    local quant_type="$3"
    
    print_info "Quantizing to $quant_type..."
    
    # Use llama.cpp quantize script
    python -m llama_cpp.quantize \
        "$input_path" \
        "$output_path" \
        "$quant_type"
    
    # Get file size
    local file_size=$(du -h "$output_path" | cut -f1)
    print_success "$quant_type quantization complete: $file_size"
}

# Main function
main() {
    print_info "🚀 Starting MoE model quantization..."
    
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
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if llama.cpp is available
    if ! command_exists python || ! python -c "import llama_cpp" 2>/dev/null; then
        print_warning "llama-cpp-python not found. Installing..."
        install_llama_cpp
    fi
    
    # Convert to GGUF first
    local gguf_path="$OUTPUT_DIR/model.gguf"
    convert_to_gguf "$MODEL_PATH" "$gguf_path"
    
    # Quantize to different formats
    for quant_type in "${QUANTIZATIONS[@]}"; do
        local output_file="$OUTPUT_DIR/wolfe-f17-moe-${quant_type}.gguf"
        quantize_model "$gguf_path" "$output_file" "$quant_type"
    done
    
    print_success "🎉 All quantizations completed successfully!"
    print_info "Quantized models saved to: $OUTPUT_DIR"
    print_info "You can now use these models with llama.cpp!"
}

# Parse command line arguments
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
        --help)
            echo "Usage: $0 --model_path /path/to/model [options]"
            echo "Options:"
            echo "  --model_path PATH     Path to merged model directory"
            echo "  --output_dir PATH     Directory to save quantized models (default: ./quantized_models)"
            echo "  --quantizations LIST  Comma-separated list of quantization types (default: Q6_K,Q4_K_M)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main