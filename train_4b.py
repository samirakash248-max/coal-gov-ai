import os
import torch
import argparse
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

def main(smoke_test=False):
    if smoke_test:
        print("=== RUNNING IN SMOKE TEST MODE (max_steps=5) ===")
        
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
        bnb_4bit_compute_dtype=torch.float16 # T4 uses FP16, BF16 causes slow down/crash
    )
    
    # 3. Load Base Model
    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16, # Force FP16 globally so config.json doesn't inject BF16 causing AMP crashes on T4
        trust_remote_code=True
    )
    
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    # 4. LoRA Config for 4B
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # BUG FIX: Do NOT call get_peft_model here because SFTTrainer will apply it.
    # Doing it twice causes issues in newer TRL versions.
    
    # 5. Load Dataset
    print("Loading V3 dataset...")
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "validation": VAL_DATA})
    
    # Determine steps based on mode
    max_steps = 5 if smoke_test else -1
    save_steps = 2 if smoke_test else 100
    eval_steps = 2 if smoke_test else 100

    # 6. Training Arguments
    # Adjusted for T4 VRAM (14.5GB), using bs=1, grad accum=16. 
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=1 if smoke_test else 10,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=True,   # Enabled for T4 (Turing)
        bf16=False,  # Disabled for T4 (Turing)
        max_grad_norm=0.3,
        warmup_steps=1 if smoke_test else 10, # Replaces warmup_ratio which is unsupported in this TRL version
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_steps,
        gradient_checkpointing=True,
        dataset_text_field="messages",
        max_length=128, # Replaces max_seq_length. Perfectly covers V3 dataset (max token length observed: 96)
        report_to="none",  
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
    parser = argparse.ArgumentParser(description="Qwen3-4B Production Training")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 5-step smoke test")
    args = parser.parse_args()
    
    main(smoke_test=args.smoke_test)
