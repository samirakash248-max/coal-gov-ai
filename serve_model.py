import os
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn

app = FastAPI()

# Configurable paths for switchability (0.6B development vs 4B production)
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./models/coal-gov-v2")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "coal-gov-model")
MODEL_NAME = BASE_MODEL

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Base Model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=quant_config, device_map="auto"
)

try:
    print("Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
except Exception as e:
    print(f"Failed to load adapter. Using base model. Error: {e}")
    model = base_model

model.eval()

class ChatRequest(BaseModel):
    messages: list
    max_tokens: int = 150
    temperature: float = 0.1

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Construct prompt from messages using chat template
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=req.temperature > 0
        )
        
    # Extract only the newly generated tokens
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Return OpenAI compatible schema
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": MODEL_ALIAS,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": gen_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
