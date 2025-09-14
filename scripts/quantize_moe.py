#!/usr/bin/env python3
"""
Quantize MoE model to GGUF format with Q6_K and Q4_K quantization
Optimized for MoE models with proper handling of expert layers
"""

import os
import subprocess
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and handle errors"""
    logger.info(f"🔄 {description}...")
    logger.info(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"   Error: {e.stderr}")
        return False

def quantize_moe_model(model_path, output_dir, quantizations=["Q6_K", "Q4_K_M"]):
    """
    Quantize MoE model to GGUF format
    
    Args:
        model_path: Path to the merged model directory
        output_dir: Directory to save quantized models
        quantizations: List of quantization types to create
    """
    
    logger.info("🚀 Starting MoE model quantization...")
    logger.info(f"   Model path: {model_path}")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Quantizations: {quantizations}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if llama.cpp is available
    try:
        result = subprocess.run(["python", "-c", "import llama_cpp"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ llama-cpp-python not found. Installing...")
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
    except Exception as e:
        logger.error(f"❌ Error checking llama-cpp-python: {e}")
        return False
    
    # Convert to GGUF first (if not already)
    gguf_path = os.path.join(output_dir, "model.gguf")
    if not os.path.exists(gguf_path):
        logger.info("🔄 Converting model to GGUF format...")
        
        # Try llama-convert command first, then fallback to Python module
        convert_cmd = [
            "llama-convert",
            model_path,
            "--outfile", gguf_path,
            "--outtype", "f16"  # Start with f16, then quantize
        ]
        
        if not run_command(convert_cmd, "Converting to GGUF"):
            # Fallback to Python module
            logger.info("🔄 Trying Python llama_cpp module...")
            try:
                from llama_cpp import convert_hf_to_gguf
                convert_hf_to_gguf(model_path, gguf_path, outtype="f16")
                logger.info("✅ Converted using Python module")
            except Exception as e:
                logger.error(f"❌ Both convert methods failed: {e}")
                return False
    
    # Keep the full precision GGUF as well
    full_gguf_path = os.path.join(output_dir, "wolfe-f17-moe-f16.gguf")
    import shutil
    shutil.copy2(gguf_path, full_gguf_path)
    full_size = os.path.getsize(full_gguf_path) / (1024**3)  # GB
    logger.info(f"✅ Full precision GGUF saved: {full_size:.2f} GB")
    
    # Quantize to different formats
    for quant_type in quantizations:
        output_file = os.path.join(output_dir, f"wolfe-f17-moe-{quant_type}.gguf")
        
        logger.info(f"🔄 Quantizing to {quant_type}...")
        
        # Use llama.cpp quantize script
        quantize_cmd = [
            "llama-quantize",
            gguf_path,
            output_file,
            quant_type
        ]
        
        if run_command(quantize_cmd, f"Quantizing to {quant_type}"):
            # Get file size
            file_size = os.path.getsize(output_file) / (1024**3)  # GB
            logger.info(f"✅ {quant_type} quantization complete: {file_size:.2f} GB")
        else:
            logger.error(f"❌ {quant_type} quantization failed")
            return False
    
    logger.info("🎉 All quantizations completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Quantize MoE model to GGUF format")
    
    # Model paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to merged model directory")
    parser.add_argument("--output_dir", type=str, default="./quantized_models",
                       help="Directory to save quantized models")
    
    # Quantization options
    parser.add_argument("--quantizations", nargs="+", 
                       default=["Q6_K", "Q4_K_M"],
                       help="Quantization types to create")
    
    # Additional options
    parser.add_argument("--install_llama_cpp", action="store_true",
                       help="Install llama-cpp-python if not available")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model path does not exist: {args.model_path}")
        return False
    
    # Check for required files
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_path, file)):
            logger.warning(f"⚠️  Missing {file} in model directory")
    
    # Install llama-cpp-python if requested
    if args.install_llama_cpp:
        logger.info("📦 Installing llama-cpp-python...")
        try:
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
            logger.info("✅ llama-cpp-python installed successfully")
        except Exception as e:
            logger.error(f"❌ Failed to install llama-cpp-python: {e}")
            return False
    
    # Quantize the model
    success = quantize_moe_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantizations=args.quantizations
    )
    
    if success:
        logger.info("🎉 MoE model quantization completed successfully!")
        logger.info(f"   Quantized models saved to: {args.output_dir}")
        logger.info("   You can now use these models with llama.cpp!")
    else:
        logger.error("💀 MoE model quantization failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)