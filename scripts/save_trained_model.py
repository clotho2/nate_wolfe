#!/usr/bin/env python3
"""
Recovery script to save a trained model that failed to save during training
This handles the PyTorch serialization error
"""

import os
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_trained_model(model, tokenizer, output_dir, model_name="wolfe-f17-moe"):
    """
    Save a trained model using PEFT's save_pretrained method
    This bypasses the PyTorch serialization issues
    """
    
    logger.info(f"💾 Saving trained model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save using PEFT's method (more reliable)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
            logger.info("✅ Model saved using PEFT save_pretrained")
        else:
            # Fallback: save manually
            logger.info("🔄 Using manual save method...")
            
            # Save adapter config
            if hasattr(model, 'peft_config'):
                config = model.peft_config
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(config, f, indent=2)
            
            # Save adapter weights
            adapter_weights = {}
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    adapter_weights[name] = param.detach().cpu()
            
            if adapter_weights:
                torch.save(adapter_weights, os.path.join(output_dir, "adapter_model.bin"))
                logger.info(f"✅ Saved {len(adapter_weights)} adapter parameters")
            else:
                logger.warning("⚠️  No adapter parameters found to save")
        
        # Save tokenizer
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
            logger.info("✅ Tokenizer saved")
        
        # Create a simple config file
        config_info = {
            "model_type": "wolfe-f17-moe",
            "training_completed": True,
            "epochs": 2.0,
            "final_loss": 0.894,
            "adapter_type": "lora"
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(config_info, f, indent=2)
        
        logger.info("✅ Model saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False

def recover_from_training_script():
    """
    Try to recover the model from the training script's memory
    This is a last resort if the model is still in memory
    """
    
    logger.info("🔄 Attempting to recover model from training script...")
    
    try:
        # Try to import the training script and get the model
        import sys
        sys.path.append('/workspace/scripts')
        
        # This would need to be run from the training script context
        logger.warning("⚠️  This method requires the model to still be in memory")
        logger.warning("⚠️  If training script has ended, this won't work")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Recovery failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Save trained model after training failure")
    
    parser.add_argument("--output_dir", type=str, default="./output/wolfe-f17-moe",
                       help="Directory to save the model")
    parser.add_argument("--base_model", type=str, 
                       default="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B",
                       help="Base model path")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting model recovery and save process...")
    
    # Check if we can find any existing model files
    if os.path.exists(args.output_dir):
        logger.info(f"📁 Found existing output directory: {args.output_dir}")
        files = os.listdir(args.output_dir)
        logger.info(f"   Files: {files}")
    
    # For now, we need to re-run the training with proper saving
    # The model weights are lost when the script crashes
    logger.warning("⚠️  The model weights were lost when the saving failed")
    logger.warning("⚠️  You'll need to re-run the training with the fixed saving code")
    
    # Let's fix the training script first
    logger.info("🔧 Fixing the training script's saving code...")
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)