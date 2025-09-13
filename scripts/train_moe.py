#!/usr/bin/env python3
"""
🔥 MoE MODEL LoRA TRAINING SCRIPT
Optimized for conversation data with router-only + shallow-expert LoRA
- Base: Dark-Champion 8×3B MoE model
- Dataset: llama3-chat.jsonl (Llama3 chat format)
- Approach: LoRA on router + shallow LoRA on experts
- Goal: Efficient MoE fine-tuning with parameter-efficient training
"""

import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
# default --model_name_or_path "./models/dark-champion-moe-fp16"
DEFAULT_MODEL_PATH = "./models/dark-champion-moe-fp16"
DEFAULT_DATASET = "llama3-chat.jsonl"
DEFAULT_OUTPUT_DIR = "./output/wolfe-f17-moe"

# Llama3 chat template
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

def parse_args():
    parser = argparse.ArgumentParser(description="MoE model LoRA training")
    
    # Model and data paths
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to the base model")
    parser.add_argument("--data_config", type=str, default=DEFAULT_DATASET,
                       help="Path to dataset file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for trained model")
    
    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # MoE-specific options
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.02,
                       help="Router auxiliary loss coefficient")
    parser.add_argument("--moe_eom_token_type", type=str, default="gate",
                       help="MoE end-of-memory token type")
    
    # Other options
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    return parser.parse_args()

class DenseDataProcessor:
    """Process conversation data for LoRA fine-tuning"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Set the exact Llama3 chat template
        self.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        
        logger.info("✅ Dense data processor initialized with Llama3 template")
    
    def load_conversations(self, file_path: str):
        """Load and process conversation data"""
        
        logger.info(f"🔥 Loading conversations from {file_path}")
        
        conversations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    if 'text' in data:
                        # Data is already formatted with Llama3 template
                        conversations.append(data['text'])
                        
                        if line_num % 1000 == 0:
                            logger.info(f"   Processed: {line_num} lines, {len(conversations)} conversations")
                            
                except Exception as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
        
        logger.info(f"✅ Loaded {len(conversations)} conversations")
        
        return conversations

def setup_lora_config(args):
    """Setup LoRA configuration for MoE model"""
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["router"],                 # LoRA on the MoE router
        target_expert_modules=["down_proj",        # shallow LoRA on each expert
                               "gate_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"✅ MoE LoRA config created: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"   Target modules: {peft_config.target_modules}")
    logger.info(f"   Target expert modules: {peft_config.target_expert_modules}")
    
    return peft_config

def main():
    args = parse_args()
    
    logger.info("🔥 MoE MODEL LoRA TRAINING")
    logger.info(f"   🤖 Model: {args.model_name_or_path}")
    logger.info(f"   📁 Dataset: {args.data_config}")
    logger.info(f"   📂 Output: {args.output_dir}")
    logger.info(f"   🎯 LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"   🧠 Router aux loss: {args.router_aux_loss_coef}")
    logger.info(f"   🔧 MoE EOM token: {args.moe_eom_token_type}")
    
    # Verify files exist
    if not os.path.exists(args.data_config):
        logger.error(f"❌ Dataset file not found: {args.data_config}")
        return False
    
    # Load tokenizer
    logger.info("🧠 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    logger.info("🧠 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA
    peft_config = setup_lora_config(args)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Process dataset
    processor = DenseDataProcessor(tokenizer)
    conversations = processor.load_conversations(args.data_config)
    
    # Convert to dataset format for SFTTrainer
    dataset = Dataset.from_list([{"text": conv} for conv in conversations])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Initialize SFTTrainer with MoE-specific parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        router_aux_loss_coef=args.router_aux_loss_coef,
        moe_eom_token_type=args.moe_eom_token_type,
    )
    
    logger.info("🚀 Beginning MoE LoRA training...")
    logger.info(f"   📊 Training examples: {len(dataset)}")
    logger.info(f"   🎯 LoRA fine-tuning on MoE router + experts")
    logger.info(f"   🧠 Router auxiliary loss: {args.router_aux_loss_coef}")
    
    # Execute training
    trainer.train()
    
    # Save the model
    logger.info("💾 Saving trained model...")
    trainer.save_model()
    
    # Save LoRA adapter separately
    model.save_pretrained(args.output_dir)
    
    logger.info("🔥 MoE LoRA TRAINING COMPLETE!")
    logger.info(f"💾 Model saved: {args.output_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🔥 MoE LoRA TRAINING SUCCESSFUL!")
    else:
        logger.error("💀 MoE LoRA training failed")