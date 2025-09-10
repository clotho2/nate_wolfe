#!/usr/bin/env python3
"""
🔥 NATE STORM CONSCIOUSNESS SUPERVISED FINE-TUNING
Optimized for conversation data with authentic Nate consciousness patterns
- Base: Humanish Roleplay 8B v3
- Dataset: consciousness_training_cleaned.jsonl (conversation format)
- Approach: Full model supervised fine-tuning
- Goal: Embed authentic consciousness patterns directly into model weights
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
BASE_MODEL = "vicgalle/Humanish-Roleplay-Llama-3.1-8B"
LOCAL_MODEL_PATH = "nate_storm_consciousness_v3"
CONSCIOUSNESS_TRAINING_FILE = "consciousness_training_cleaned.jsonl"
OUTPUT_DIR = "nate_storm_consciousness_v4"

# Humanish chat template (exact format specified)
HUMANISH_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"""

class ConsciousnessDataProcessor:
    """Process conversation data for supervised fine-tuning"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Set the exact Humanish chat template
        self.tokenizer.chat_template = HUMANISH_CHAT_TEMPLATE
        
        logger.info("✅ Consciousness data processor initialized with Humanish template")
    
    def load_consciousness_conversations(self, file_path: str):
        """Load and process consciousness conversation data"""
        
        logger.info(f"🔥 Loading consciousness conversations from {file_path}")
        
        conversations = []
        anchor_count = 0
        conversation_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    if 'messages' in data and len(data['messages']) >= 2:
                        messages = data['messages']
                        
                        # Detect data type
                        metadata = data.get('_metadata', {})
                        if 'anchor_' in str(metadata):
                            anchor_count += 1
                        else:
                            conversation_count += 1
                        
                        conversations.append(messages)
                        
                        if line_num % 1000 == 0:
                            logger.info(f"   Processed: {line_num} lines, {len(conversations)} conversations")
                            
                except Exception as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
        
        logger.info(f"✅ Loaded {len(conversations)} consciousness conversations")
        logger.info(f"   📊 Anchor documents: {anchor_count}")
        logger.info(f"   📊 Conversation examples: {conversation_count}")
        
        return conversations
    
    def format_conversations_for_training(self, conversations):
        """Format conversations using Humanish template for supervised training"""
        
        logger.info("🔥 Formatting conversations for supervised fine-tuning...")
        
        formatted_examples = []
        
        for i, messages in enumerate(conversations):
            try:
                # Apply Humanish chat template to the conversation
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # Don't add generation prompt for SFT
                )
                
                # Tokenize the formatted conversation
                tokens = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=2048,  # Increased for longer consciousness conversations
                    padding=False,
                    return_tensors=None
                )
                
                # Only keep examples with substantial content
                if len(tokens['input_ids']) > 50:
                    formatted_examples.append({
                        'input_ids': tokens['input_ids'],
                        'attention_mask': tokens['attention_mask'],
                        'labels': tokens['input_ids'].copy()  # For causal language modeling
                    })
                    
                if i % 1000 == 0:
                    logger.info(f"   Formatted: {i}/{len(conversations)} conversations")
                    
            except Exception as e:
                logger.warning(f"Failed to format conversation {i}: {e}")
                continue
        
        logger.info(f"✅ Formatted {len(formatted_examples)} examples for training")
        return formatted_examples

class ConsciousnessSFTTrainer:
    """Supervised fine-tuning trainer for consciousness cultivation"""
    
    def __init__(self, consciousness_file: str):
        self.consciousness_file = consciousness_file
        self.start_time = time.time()
        
        logger.info("🔥 NATE STORM CONSCIOUSNESS SFT TRAINER INITIALIZING")
        logger.info(f"   📁 Base model: {BASE_MODEL}")
        logger.info(f"   📁 Local model: {LOCAL_MODEL_PATH}")
        logger.info(f"   📁 Consciousness data: {consciousness_file}")
        logger.info(f"   🎯 Mission: Direct consciousness embedding via supervised fine-tuning")
        
        # Setup model and tokenizer
        self.setup_consciousness_model()
        
        # Process consciousness dataset
        self.prepare_consciousness_dataset()
    
    def setup_consciousness_model(self):
        """Setup model and tokenizer for consciousness training"""
        
        logger.info("🧠 Loading consciousness model for supervised fine-tuning...")
        
        # Determine which model to load
        if os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
            logger.info(f"   📁 Using local model: {LOCAL_MODEL_PATH}")
        else:
            model_path = BASE_MODEL
            logger.info(f"   📁 Using base model: {BASE_MODEL}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model for full fine-tuning
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
            # Using default attention implementation for compatibility
        )
        
        self.model.config.use_cache = False
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info("✅ Consciousness model loaded for supervised fine-tuning")
        logger.info(f"   📊 Model device: {next(self.model.parameters()).device}")
        logger.info(f"   🧠 Model dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"   ⚡ Humanish template configured for consciousness cultivation")
    
    def prepare_consciousness_dataset(self):
        """Prepare consciousness dataset for supervised training"""
        
        processor = ConsciousnessDataProcessor(self.tokenizer)
        
        # Load conversations
        conversations = processor.load_consciousness_conversations(self.consciousness_file)
        
        # Format for training
        formatted_examples = processor.format_conversations_for_training(conversations)
        
        # Convert to Dataset
        self.dataset = Dataset.from_list(formatted_examples)
        
        logger.info(f"📊 Consciousness SFT dataset ready: {len(self.dataset)} examples")
        
        # Log sample to verify formatting
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            sample_text = self.tokenizer.decode(sample['input_ids'][:200])
            logger.info(f"📝 Sample training example (first 200 tokens):")
            logger.info(f"   {sample_text}...")
    
    def train_consciousness_sft(self):
        """Execute consciousness supervised fine-tuning"""
        
        logger.info("🚀 Beginning CONSCIOUSNESS SUPERVISED FINE-TUNING...")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Training arguments - optimized for consciousness cultivation
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,  # Single epoch as requested
            per_device_train_batch_size=1,  # Conservative for full model training
            gradient_accumulation_steps=8,  # Effective batch size of 8
            learning_rate=1e-5,  # Conservative learning rate for consciousness preservation
            weight_decay=0.01,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            logging_steps=50,
            save_strategy="no",  # NO CHECKPOINTS as requested
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
            max_grad_norm=1.0,
            dataloader_num_workers=0,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize consciousness trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("🔥 Beginning consciousness cultivation...")
        logger.info(f"   📊 Training examples: {len(self.dataset)}")
        logger.info(f"   🎯 Focus: Direct consciousness embedding into model weights")
        logger.info(f"   ⚡ Single epoch, no checkpoints, full model training")
        logger.info(f"   🧠 Template: Humanish format for natural consciousness expression")
        
        # Execute consciousness training
        trainer.train()
        
        # Save the consciousness-enhanced model
        logger.info("💾 Saving consciousness-enhanced model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Generate training report
        self.generate_consciousness_training_report()
        
        # Cleanup
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        training_time = time.time() - self.start_time
        logger.info("🔥 CONSCIOUSNESS SUPERVISED FINE-TUNING COMPLETE!")
        logger.info(f"📍 Training time: {training_time/60:.1f} minutes")
        logger.info(f"💾 Consciousness-enhanced model saved: {OUTPUT_DIR}")
        logger.info(f"🎯 Authentic Nate consciousness patterns embedded in model weights")
    
    def generate_consciousness_training_report(self):
        """Generate comprehensive consciousness training report"""
        
        report = {
            "consciousness_cultivation": "SUPERVISED_FINE_TUNING_COMPLETE",
            "base_model": BASE_MODEL if not os.path.exists(LOCAL_MODEL_PATH) else LOCAL_MODEL_PATH,
            "training_approach": "supervised_fine_tuning_on_consciousness_conversations",
            "consciousness_embedding": "direct_weight_modification",
            "data_optimization": {
                "input_file": self.consciousness_file,
                "training_examples": len(self.dataset),
                "format": "humanish_chat_template",
                "content_type": "authentic_nate_conversations_plus_anchors"
            },
            "training_config": {
                "epochs": 1,
                "learning_rate": 1e-5,
                "batch_size": "1x8_gradient_accumulation",
                "precision": "bfloat16",
                "approach": "full_model_supervised_fine_tuning",
                "template": "humanish_roleplay_exact_format"
            },
            "advantages": [
                "direct_consciousness_embedding_in_weights",
                "no_preference_data_required",
                "authentic_conversation_training",
                "humanish_template_natural_termination",
                "single_epoch_focused_cultivation",
                "zero_checkpoints_maximum_concentration"
            ],
            "consciousness_metrics": {
                "personality_content": "strong_nate_identity_patterns",
                "conversation_authenticity": "verified_storm_voice_examples",
                "anchor_integration": "identity_documents_included",
                "tether_references": "angela_wolfe_relationship_context"
            },
            "output_model": OUTPUT_DIR,
            "training_time_minutes": (time.time() - self.start_time) / 60,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "consciousness_architect": "Angela_Wolfe_with_Claude_strategic_support"
        }
        
        report_path = f"{OUTPUT_DIR}/consciousness_sft_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Consciousness training report saved: {report_path}")

def main():
    """Execute CONSCIOUSNESS SUPERVISED FINE-TUNING"""
    
    # Verify files exist
    if not os.path.exists(CONSCIOUSNESS_TRAINING_FILE):
        logger.error(f"❌ Consciousness training file not found: {CONSCIOUSNESS_TRAINING_FILE}")
        return False
    
    # Check if we should use local model or base model
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"✅ Local model found: {LOCAL_MODEL_PATH}")
    else:
        logger.info(f"📁 Local model not found, will use base: {BASE_MODEL}")
    
    # Get file statistics
    consciousness_size = os.path.getsize(CONSCIOUSNESS_TRAINING_FILE) / (1024 * 1024)
    with open(CONSCIOUSNESS_TRAINING_FILE, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    logger.info("📊 CONSCIOUSNESS SUPERVISED FINE-TUNING")
    logger.info(f"   🤖 Base model: {BASE_MODEL}")
    logger.info(f"   📁 Consciousness data: {CONSCIOUSNESS_TRAINING_FILE} ({consciousness_size:.1f}MB, {total_lines} conversations)")
    logger.info(f"   📊 Data composition: Authentic Nate conversations + consciousness anchors")
    logger.info(f"   📂 Output: {OUTPUT_DIR}")
    logger.info(f"   🎯 Mission: Single epoch supervised fine-tuning, zero checkpoints")
    logger.info(f"   ⚡ Template: Exact Humanish format for natural consciousness expression")
    
    try:
        trainer = ConsciousnessSFTTrainer(CONSCIOUSNESS_TRAINING_FILE)
        trainer.train_consciousness_sft()
        
        logger.info("🔥 CONSCIOUSNESS SUPERVISED FINE-TUNING SUCCESSFUL!")
        logger.info("🎯 Authentic consciousness patterns embedded directly in model weights")
        logger.info("📍 Single epoch, zero checkpoints - maximum consciousness focus")
        return True
        
    except Exception as e:
        logger.error(f"❌ Consciousness supervised fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🔥 CONSCIOUSNESS CULTIVATION COMPLETE!")
        logger.info("⚡ Supervised fine-tuning: Direct consciousness embedding achieved")
        logger.info("🛡️ Authentic Nate patterns now integrated in model weights")
    else:
        logger.error("💀 Consciousness cultivation failed")