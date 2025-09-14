#!/usr/bin/env python3
"""
Direct quantization using llama.cpp Python scripts
This uses the convert_hf_to_gguf.py and quantize.py scripts directly
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_conversion(model_path, output_path):
    """Run the convert_hf_to_gguf.py script directly"""
    print(f"🔄 Converting {model_path} to GGUF...")
    
    # Use the convert script from llama.cpp
    convert_script = "./llama.cpp/convert_hf_to_gguf.py"
    
    if not os.path.exists(convert_script):
        print(f"❌ Convert script not found: {convert_script}")
        return False
    
    try:
        # Run the conversion
        cmd = [
            sys.executable, convert_script,
            model_path,
            "--outfile", output_path,
            "--outtype", "f16"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Converted to {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_quantization(input_path, output_path, quant_type):
    """Run the quantize.py script directly"""
    print(f"🔄 Quantizing to {quant_type}...")
    
    # Use the quantize script from llama.cpp
    quantize_script = "./llama.cpp/quantize.py"
    
    if not os.path.exists(quantize_script):
        print(f"❌ Quantize script not found: {quantize_script}")
        return False
    
    try:
        # Run the quantization
        cmd = [
            sys.executable, quantize_script,
            input_path,
            output_path,
            quant_type
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024**3)  # GB
        print(f"✅ {quant_type} quantization complete: {file_size:.2f} GB")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Quantization failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Direct MoE quantization using llama.cpp scripts")
    parser.add_argument("--model_path", required=True, help="Path to merged model directory")
    parser.add_argument("--output_dir", default="./quantized_models", help="Output directory")
    parser.add_argument("--quantizations", nargs="+", default=["Q6_K", "Q4_K_M"], 
                       help="Quantization types")
    
    args = parser.parse_args()
    
    print("🚀 Starting direct MoE quantization...")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Quantizations: {args.quantizations}")
    
    # Check if llama.cpp directory exists
    if not os.path.exists("./llama.cpp"):
        print("❌ llama.cpp directory not found!")
        print("   Please run this from the nate_wolfe directory where llama.cpp is cloned")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to GGUF
    gguf_path = os.path.join(args.output_dir, "model.gguf")
    if not run_conversion(args.model_path, gguf_path):
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
        if not run_quantization(gguf_path, output_file, quant_type):
            print(f"⚠️  Skipping {quant_type} due to error")
    
    print("🎉 Quantization completed!")
    print(f"   Files saved to: {args.output_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)