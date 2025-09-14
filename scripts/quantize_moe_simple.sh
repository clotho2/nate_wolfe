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
    
    # Check if we're in the right directory
    if [ ! -d "llama.cpp" ]; then
        print_info "Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    
    # Build llama.cpp using CMake
    print_info "Building llama.cpp with CMake..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
    
    # Install Python bindings
    print_info "Installing Python bindings..."
    pip install llama-cpp-python
    
    cd ..
    
    print_success "llama.cpp installed successfully"
}

# Function to find llama.cpp tools
find_llama_tools() {
    # Check if llama-convert and llama-quantize are in PATH
    if command_exists llama-convert && command_exists llama-quantize; then
        print_success "Found llama.cpp tools in PATH"
        return 0
    fi
    
    # Check if they're in the current directory
    if [ -f "./llama-convert" ] && [ -f "./llama-quantize" ]; then
        print_success "Found llama.cpp tools in current directory"
        return 0
    fi
    
    # Check if they're in llama.cpp directory
    if [ -f "./llama.cpp/llama-convert" ] && [ -f "./llama.cpp/llama-quantize" ]; then
        print_success "Found llama.cpp tools in llama.cpp directory"
        return 0
    fi
    
    # Check if they're in llama.cpp build directory
    if [ -f "./llama.cpp/build/llama-convert" ] && [ -f "./llama.cpp/build/llama-quantize" ]; then
        print_success "Found llama.cpp tools in llama.cpp/build directory"
        return 0
    fi
    
    return 1
}

# Function to convert model to GGUF
convert_to_gguf() {
    local model_path="$1"
    local output_path="$2"
    
    print_info "Converting model to GGUF format..."
    
    # Find the correct llama-convert path
    local convert_cmd=""
    if command_exists llama-convert; then
        convert_cmd="llama-convert"
    elif [ -f "./llama-convert" ]; then
        convert_cmd="./llama-convert"
    elif [ -f "./llama.cpp/llama-convert" ]; then
        convert_cmd="./llama.cpp/llama-convert"
    elif [ -f "./llama.cpp/build/llama-convert" ]; then
        convert_cmd="./llama.cpp/build/llama-convert"
    else
        print_error "llama-convert not found!"
        return 1
    fi
    
    # Use llama.cpp convert script
    $convert_cmd \
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
    
    # Find the correct llama-quantize path
    local quantize_cmd=""
    if command_exists llama-quantize; then
        quantize_cmd="llama-quantize"
    elif [ -f "./llama-quantize" ]; then
        quantize_cmd="./llama-quantize"
    elif [ -f "./llama.cpp/llama-quantize" ]; then
        quantize_cmd="./llama.cpp/llama-quantize"
    elif [ -f "./llama.cpp/build/llama-quantize" ]; then
        quantize_cmd="./llama.cpp/build/llama-quantize"
    else
        print_error "llama-quantize not found!"
        return 1
    fi
    
    # Use llama.cpp quantize script
    $quantize_cmd \
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
    
    # Check if llama.cpp tools are available
    if ! find_llama_tools; then
        print_warning "llama.cpp tools not found. Installing..."
        install_llama_cpp
        
        # Check again after installation
        if ! find_llama_tools; then
            print_error "Failed to install or find llama.cpp tools!"
            print_info "You can also install llama-cpp-python and use the Python script instead:"
            print_info "python scripts/quantize_moe.py --model_path $MODEL_PATH --output_dir $OUTPUT_DIR"
            exit 1
        fi
    fi
    
    # Convert to GGUF first
    local gguf_path="$OUTPUT_DIR/model.gguf"
    convert_to_gguf "$MODEL_PATH" "$gguf_path"
    
    # Keep the full precision GGUF as well
    local full_gguf="$OUTPUT_DIR/wolfe-f17-moe-f16.gguf"
    cp "$gguf_path" "$full_gguf"
    local full_size=$(du -h "$full_gguf" | cut -f1)
    print_success "Full precision GGUF saved: $full_size"
    
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