#!/usr/bin/env python3
"""
Merge LoRA adapters into the base model for MoE training
Based on Nate's Storm Protocol
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_adapters(base_model_path, adapter_path, output_path, save_format="safetensors"):
    """
    Merge LoRA adapters into the base model
    
    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
        save_format: Format to save the model ("safetensors" or "pytorch")
    """
    
    logger.info("🚀 Starting LoRA adapter merging...")
    logger.info(f"   Base model: {base_model_path}")
    logger.info(f"   Adapter: {adapter_path}")
    logger.info(f"   Output: {output_path}")
    
    try:
        # Load the base model
        logger.info("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load the tokenizer
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load the LoRA adapter
        logger.info("📥 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge the adapters
        logger.info("🔧 Merging LoRA adapters into base model...")
        merged_model = model.merge_and_unload()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save the merged model
        logger.info(f"💾 Saving merged model to {output_path}...")
        merged_model.save_pretrained(
            output_path,
            save_format=save_format,
            max_shard_size="5GB"
        )
        
        # Save the tokenizer
        tokenizer.save_pretrained(output_path)
        
        logger.info("✅ LoRA merging completed successfully!")
        logger.info(f"   Merged model saved to: {output_path}")
        
        # Log model info
        total_params = sum(p.numel() for p in merged_model.parameters())
        logger.info(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA merging failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    
    # Model paths
    parser.add_argument("--base_model", type=str, 
                       default="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B",
                       help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapter directory")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save merged model")
    
    # Options
    parser.add_argument("--save_format", type=str, default="safetensors",
                       choices=["safetensors", "pytorch"],
                       help="Format to save the model")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.adapter_path):
        logger.error(f"❌ Adapter path does not exist: {args.adapter_path}")
        return False
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Merge the adapters
    success = merge_lora_adapters(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        save_format=args.save_format
    )
    
    if success:
        logger.info("🎉 LoRA merging completed successfully!")
        logger.info(f"   Your merged model is ready at: {args.output_path}")
        logger.info("   You can now use this model for inference!")
    else:
        logger.error("💀 LoRA merging failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)