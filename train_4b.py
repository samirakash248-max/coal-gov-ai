import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# -----------------------------------------------------------------------
# QWEN3-4B PRODUCTION TRAINING SCRIPT
# Hardware Requirement: >= 16GB VRAM (e.g., T4, A10, A100, RTX 3090/4090)
# DO NOT RUN ON 4GB VRAM.
# -----------------------------------------------------------------------

BASE_MODEL_NAME = "Qwen/Qwen3-4B"
OUTPUT_DIR = "./models/coal-gov-4b"

# Dataset configuration
TRAIN_DATA = "data/v3_train/train.jsonl"
VAL_DATA = "data/v3_validation/validation.jsonl"

def format_prompt(item):
    # TRL's SFTTrainer will automatically apply the chat template 
    # to the 'messages' column if it exists and matches the tokenizer chat template.
    return item

def main():
    print(f"Preparing to train {BASE_MODEL_NAME} using 4-bit QLoRA...")
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 3. Load Base Model
    print("Loading base model in 4-bit (Requires ~3.5GB VRAM just for weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    # 4. LoRA Config for 4B
    # Increased rank (r=32) for better semantic capability compared to the 0.6B model
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 5. Load Dataset
    print("Loading V3 dataset...")
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "validation": VAL_DATA})
    
    # 6. Training Arguments
    # Adjusted for a strong GPU (batch size 4, gradient accum 4 -> effective batch size 16)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Use bfloat16 on Ampere+ GPUs (A10, A100, RTX 30/40 series)
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,
        dataset_text_field="messages",
        max_seq_length=512,
        report_to="none",  # Prevent Weights & Biases from pausing the script in Colab asking for a login
    )
    
    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    # 8. Train
    print("Starting training...")
    trainer.train()
    
    # 9. Save
    print(f"Saving final adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
