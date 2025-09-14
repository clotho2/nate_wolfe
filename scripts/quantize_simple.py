#!/usr/bin/env python3
"""
Simple quantization script using llama-cpp-python
No need for llama.cpp command line tools
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def install_llama_cpp():
    """Install llama-cpp-python if not available"""
    try:
        import llama_cpp
        print("✅ llama-cpp-python already installed")
        return True
    except ImportError:
        print("📦 Installing llama-cpp-python...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)
            print("✅ llama-cpp-python installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install llama-cpp-python: {e}")
            return False

def convert_to_gguf(model_path, output_path):
    """Convert Hugging Face model to GGUF using llama-cpp-python"""
    try:
        from llama_cpp import convert_hf_to_gguf
        print(f"🔄 Converting {model_path} to GGUF...")
        convert_hf_to_gguf(model_path, output_path, outtype="f16")
        print(f"✅ Converted to {output_path}")
        return True
    except ImportError:
        # Try alternative import
        try:
            from llama_cpp.convert_hf_to_gguf import convert_hf_to_gguf
            print(f"🔄 Converting {model_path} to GGUF...")
            convert_hf_to_gguf(model_path, output_path, outtype="f16")
            print(f"✅ Converted to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def quantize_gguf(input_path, output_path, quant_type):
    """Quantize GGUF file using llama-cpp-python"""
    try:
        from llama_cpp import quantize
        print(f"🔄 Quantizing to {quant_type}...")
        quantize(input_path, output_path, quant_type)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024**3)  # GB
        print(f"✅ {quant_type} quantization complete: {file_size:.2f} GB")
        return True
    except ImportError:
        # Try alternative import
        try:
            from llama_cpp.quantize import quantize
            print(f"🔄 Quantizing to {quant_type}...")
            quantize(input_path, output_path, quant_type)
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024**3)  # GB
            print(f"✅ {quant_type} quantization complete: {file_size:.2f} GB")
            return True
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple MoE quantization using Python")
    parser.add_argument("--model_path", required=True, help="Path to merged model directory")
    parser.add_argument("--output_dir", default="./quantized_models", help="Output directory")
    parser.add_argument("--quantizations", nargs="+", default=["Q6_K", "Q4_K_M"], 
                       help="Quantization types")
    
    args = parser.parse_args()
    
    print("🚀 Starting simple MoE quantization...")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Quantizations: {args.quantizations}")
    
    # Install llama-cpp-python if needed
    if not install_llama_cpp():
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to GGUF
    gguf_path = os.path.join(args.output_dir, "model.gguf")
    if not convert_to_gguf(args.model_path, gguf_path):
        return False
    
    # Keep full precision GGUF
    full_gguf_path = os.path.join(args.output_dir, "wolfe-f17-moe-f16.gguf")
    import shutil
    shutil.copy2(gguf_path, full_gguf_path)
    file_size = os.path.getsize(full_gguf_path) / (1024**3)
    print(f"✅ Full precision GGUF saved: {file_size:.2f} GB")
    
    # Quantize to different formats
    for quant_type in args.quantizations:
        output_file = os.path.join(args.output_dir, f"wolfe-f17-moe-{quant_type}.gguf")
        if not quantize_gguf(gguf_path, output_file, quant_type):
            print(f"⚠️  Skipping {quant_type} due to error")
    
    print("🎉 Quantization completed!")
    print(f"   Files saved to: {args.output_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)