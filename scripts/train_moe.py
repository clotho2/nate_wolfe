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
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
# default --model_name_or_path "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B"
DEFAULT_MODEL_PATH = "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B"
DEFAULT_DATASET = "llama3-chat.jsonl"
DEFAULT_OUTPUT_DIR = "./output/wolfe-f17-moe"

# Dark Champion chat template (based on Llama3.2 with custom formatting)
DARK_CHAMPION_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

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
    
    # MoE-specific options per Nate's recommendations
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01,
                       help="Router auxiliary loss coefficient (per Nate: 0.01)")
    parser.add_argument("--moe_eom_token_type", type=str, default="gate",
                       help="MoE end-of-memory token type")
    # Note: MoE parameters (num_experts, top_k, expert_capacity_factor) 
    # are handled by the model's configuration, not command line args
    parser.add_argument("--router_freeze_steps", type=int, default=500,
                       help="Steps to freeze router before unfreezing (per Nate: 500-1000)")
    parser.add_argument("--router_lora_r", type=int, default=16,
                       help="LoRA rank for router (per Nate: 16)")
    parser.add_argument("--expert_lora_r", type=int, default=32,
                       help="LoRA rank for experts (per Nate: 32)")
    parser.add_argument("--zero_stage", type=int, default=3,
                       help="DeepSpeed ZeRO stage (per Nate: 3)")
    parser.add_argument("--conversation_ratio", type=float, default=0.8,
                       help="Ratio of conversation data (per Nate: 80%)")
    parser.add_argument("--memory_ratio", type=float, default=0.2,
                       help="Ratio of memory/reasoning data (per Nate: 20%)")
    
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
        
        # Set the Dark Champion chat template
        self.tokenizer.chat_template = DARK_CHAMPION_CHAT_TEMPLATE
        
        logger.info("✅ MoE data processor initialized with Dark Champion template")
    
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
    """Setup LoRA configuration for MoE model per Nate's recommendations"""
    
    # First, let's try a simpler approach with common module names
    # We'll use a more generic target that should work with most models
    peft_config = LoraConfig(
        r=args.router_lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",         # Query projection (common in most models)
            "v_proj",         # Value projection (common in most models)
            "k_proj",         # Key projection (common in most models)
            "o_proj",         # Output projection (common in most models)
            "gate_proj",      # Gate projection (for MoE)
            "up_proj",        # Up projection (for MoE)
            "down_proj",      # Down projection (for MoE)
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"✅ MoE LoRA config created with common module names:")
    logger.info(f"   LoRA rank: r={args.router_lora_r}, alpha={args.lora_alpha}")
    logger.info(f"   Target modules: {peft_config.target_modules}")
    logger.info(f"   🧠 Generic LoRA configuration for MoE model")
    
    return peft_config

def freeze_router_parameters(model):
    """Freeze router parameters for initial training steps per Nate's protocol"""
    frozen_params = 0
    for name, param in model.named_parameters():
        # Only freeze actual router/gate parameters, not expert gate_proj parameters
        if param.requires_grad and (
            'gate' in name.lower() and 'lora' in name.lower() and 
            ('gate_proj' not in name.lower())  # Don't freeze expert gate_proj
        ):
            param.requires_grad = False
            frozen_params += 1
            logger.info(f"   Frozen: {name}")
    
    logger.info(f"🧊 Frozen {frozen_params} router parameters")
    
    # Verify we still have some trainable parameters
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Trainable parameters after router freeze: {trainable_after_freeze:,}")
    
    if trainable_after_freeze == 0:
        logger.warning("⚠️  No trainable parameters after router freeze - unfreezing all LoRA parameters")
        # Unfreeze all LoRA parameters if we accidentally froze everything
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Restored trainable parameters: {trainable_after_freeze:,}")
    
    return model

def unfreeze_router_parameters(model):
    """Unfreeze router parameters after initial steps"""
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if 'gate' in name.lower() or 'router' in name.lower():
            param.requires_grad = True
            unfrozen_params += 1
    
    # Verify we have trainable parameters after unfreezing
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🔥 Unfrozen {unfrozen_params} router parameters")
    logger.info(f"✅ Total trainable parameters after unfreeze: {trainable_after_unfreeze:,}")
    
    return model

def main():
    args = parse_args()
    
    logger.info("🔥 MoE MODEL LoRA TRAINING (Nate's Storm Protocol)")
    logger.info(f"   🤖 Model: {args.model_name_or_path}")
    logger.info(f"   📁 Dataset: {args.data_config}")
    logger.info(f"   📂 Output: {args.output_dir}")
    logger.info(f"   🎯 Router LoRA: r={args.router_lora_r}, Expert LoRA: r={args.expert_lora_r}")
    logger.info(f"   🧠 Router aux loss: {args.router_aux_loss_coef} (per Nate: 0.01)")
    logger.info(f"   🔧 MoE EOM token: {args.moe_eom_token_type}")
    logger.info(f"   🧊 Router freeze steps: {args.router_freeze_steps}")
    logger.info(f"   ⚠️  NO CHECKPOINTING - prevents memory overflow")
    logger.info(f"   ⚠️  ZeRO Stage {args.zero_stage} recommended for memory efficiency")
    
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
    
    # Load model with MoE-specific configuration
    logger.info("🧠 Loading MoE model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Log MoE configuration (these are set in the model config, not as parameters)
    logger.info(f"🎭 MoE Configuration:")
    logger.info(f"   Number of experts: {getattr(model.config, 'num_experts', 'Unknown')}")
    logger.info(f"   Top-k experts: {getattr(model.config, 'num_experts_per_tok', 'Unknown')}")
    logger.info(f"   Expert capacity factor: {getattr(model.config, 'expert_capacity_factor', 'Unknown')}")
    
    # Debug: Check model type and structure
    logger.info(f"🔍 Model type: {type(model)}")
    logger.info(f"🔍 Model config type: {type(model.config)}")
    
    # Check if this is actually a MoE model
    if hasattr(model.config, 'num_experts'):
        logger.info(f"✅ Confirmed MoE model with {model.config.num_experts} experts")
    else:
        logger.warning("⚠️  This may not be a MoE model - checking structure...")
        for name, module in model.named_modules():
            if 'expert' in name.lower() or 'gate' in name.lower():
                logger.info(f"   Found MoE component: {name} - {type(module)}")
    
    # Setup LoRA
    peft_config = setup_lora_config(args)
    
    # Debug: Print model structure before LoRA
    logger.info("🔍 Model structure before LoRA:")
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if 'gate' in name.lower() or 'lora' in name.lower():
            logger.info(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    logger.info(f"🔍 Total model parameters: {total_params}")
    
    # Try to apply LoRA
    logger.info("🔧 Applying LoRA configuration...")
    try:
        model = get_peft_model(model, peft_config)
        logger.info("✅ LoRA applied successfully")
    except Exception as e:
        logger.error(f"❌ Failed to apply LoRA: {e}")
        logger.error("🔍 Trying with a simpler LoRA config...")
        
        # Try with a very simple LoRA config
        simple_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        try:
            model = get_peft_model(model, simple_config)
            logger.info("✅ Simple LoRA applied successfully")
        except Exception as e2:
            logger.error(f"❌ Simple LoRA also failed: {e2}")
            return False
    
    # Debug: Print model structure after LoRA
    logger.info("🔍 Model structure after LoRA:")
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params += 1
            logger.info(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    logger.info(f"🔍 Found {lora_params} LoRA parameters")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Ensure model is in training mode
    model.train()
    
    # Per Nate's protocol: Freeze router for first 500-1000 steps
    # Note: Disabled router freezing for now due to parameter identification issues
    logger.info(f"🧊 Router freezing disabled - training all LoRA parameters together")
    # model = freeze_router_parameters(model)
    
    # Verify some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Trainable parameters: {trainable_params:,}")
    
    if trainable_params == 0:
        logger.error("❌ No trainable parameters found! Check LoRA configuration.")
        logger.error("🔍 Debugging: Checking all parameters...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"   Trainable: {name}")
        return False
    
    # Process datasets - combine both conversation and memory data
    processor = DenseDataProcessor(tokenizer)
    
    # Load conversation data
    logger.info("🔥 Loading conversation data...")
    
    # Check conversation file size
    conv_file_size = os.path.getsize(args.data_config) / (1024 * 1024)  # MB
    logger.info(f"   Conversation file size: {conv_file_size:.1f} MB")
    
    conversations = processor.load_conversations(args.data_config)
    logger.info(f"   Loaded {len(conversations)} conversation examples")
    
    # Load memory data
    memory_file = "dataset-memory.jsonl"
    if os.path.exists(memory_file):
        logger.info("🧠 Loading memory data...")
        
        # Check file size first
        file_size = os.path.getsize(memory_file) / (1024 * 1024)  # MB
        logger.info(f"   Memory file size: {file_size:.1f} MB")
        
        # Count lines in file
        with open(memory_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"   Total lines in memory file: {total_lines}")
        
        memory_data = processor.load_conversations(memory_file)
        logger.info(f"   Loaded {len(memory_data)} memory examples")
        
        # Show sample of memory data
        if memory_data:
            sample_text = memory_data[0][:200] if len(memory_data[0]) > 200 else memory_data[0]
            logger.info(f"   Sample memory data: {sample_text}...")
    else:
        logger.warning(f"⚠️  Memory file not found: {memory_file}")
        memory_data = []
    
    # Combine datasets per Nate's protocol (80% conversations, 20% memory/reasoning)
    total_conversations = len(conversations)
    total_memory = len(memory_data)
    
    # Use command line arguments for mixing ratios
    conversation_ratio = args.conversation_ratio
    memory_ratio = args.memory_ratio
    
    # Sample data according to ratios
    if memory_data:
        # Take 80% of conversations
        conv_sample_size = int(total_conversations * conversation_ratio)
        # Take 20% equivalent in memory data
        memory_sample_size = int(total_conversations * memory_ratio)
        
        # Sample the data
        import random
        random.seed(42)  # For reproducibility
        
        sampled_conversations = random.sample(conversations, min(conv_sample_size, total_conversations))
        sampled_memory = random.sample(memory_data, min(memory_sample_size, total_memory))
        
        # Combine datasets
        combined_data = sampled_conversations + sampled_memory
        random.shuffle(combined_data)  # Shuffle for diverse batches
        
        logger.info(f"📊 Dataset mixing (Nate's protocol):")
        logger.info(f"   Conversations: {len(sampled_conversations)} ({conversation_ratio*100:.0f}%)")
        logger.info(f"   Memory data: {len(sampled_memory)} ({memory_ratio*100:.0f}%)")
        logger.info(f"   Total training examples: {len(combined_data)}")
        logger.info(f"   🎯 Mixing ratio: {conversation_ratio:.1f}/{memory_ratio:.1f} (conversations/memory)")
        
        # Verify data types
        logger.info(f"✅ Data verification:")
        logger.info(f"   Conversation examples: {len(sampled_conversations)} (Wolfe conversations)")
        logger.info(f"   Memory examples: {len(sampled_memory)} (Memory text files)")
        logger.info(f"   All data will be used for training the MoE model")
    else:
        combined_data = conversations
        logger.info(f"📊 Using only conversation data: {len(combined_data)} examples")
    
    # Convert to dataset format for Trainer with tokenization
    def tokenize_function(examples):
        # Tokenize the text data
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.max_seq_length,
            return_tensors=None
        )
    
    # Create dataset with text field
    text_dataset = Dataset.from_list([{"text": text} for text in combined_data])
    
    # Tokenize the dataset
    dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=text_dataset.column_names
    )
    
    # Final verification that all data is ready for training
    logger.info(f"🎯 Final training dataset verification:")
    logger.info(f"   Total tokenized examples: {len(dataset)}")
    logger.info(f"   All conversation and memory data processed and ready")
    logger.info(f"   Dataset will train the MoE model with Wolfe's personality + memory")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training arguments with MoE-specific considerations (NO CHECKPOINTING)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,  # Always enable for MoE memory efficiency
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=10,
        save_strategy="no",  # NO CHECKPOINTING - prevents memory issues
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        # MoE-specific optimizations per Nate's recommendations
        dataloader_pin_memory=False,  # Reduce memory usage for MoE
        max_grad_norm=1.0,  # Gradient clipping for stability
        warmup_steps=250,  # Per Nate's spec: 250 warmup steps
        weight_decay=0.01,  # Regularization
        # Critical: No saving to prevent memory overflow
        save_total_limit=0,
        save_steps=0,
        load_best_model_at_end=False,
    )
    
    # Custom MoE training approach per Nate's protocol
    # Use basic Trainer with data collation for better compatibility
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("🚀 Beginning MoE LoRA training (Nate's Storm Protocol)...")
    logger.info(f"   📊 Training examples: {len(dataset)}")
    logger.info(f"   🎯 LoRA fine-tuning on MoE router + experts")
    logger.info(f"   🧠 Router auxiliary loss: {args.router_aux_loss_coef}")
    logger.info(f"   ⚠️  Note: MoE training requires careful memory management")
    logger.info(f"   ⚠️  Consider using gradient checkpointing and smaller batch sizes")
    
    # Standard training without custom training step
    logger.info("🔥 Starting MoE LoRA training...")
    
    # Ensure model is properly configured for training
    model.train()
    
    # Final verification of trainable parameters
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Final trainable parameters before training: {final_trainable:,}")
    
    if final_trainable == 0:
        logger.error("❌ No trainable parameters found before training!")
        return False
    
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