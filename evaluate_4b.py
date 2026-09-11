import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import collections

MODEL_NAME = "Qwen/Qwen3-4B"
ADAPTER_DIR = "./models/coal-gov-4b"
TEST_FILE = "data/v3_test/test.jsonl"

def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except:
        pass
    return None

def run_evaluation():
    print("Loading test set...")
    test_data = load_dataset("json", data_files=TEST_FILE, split="train")
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization_config, device_map="auto"
    )
    
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        print("Loaded LoRA adapter successfully.")
    except Exception as e:
        print(f"No adapter found at {ADAPTER_DIR}, using base model. Error: {e}")
        model = base_model
        
    model.eval()
    
    results = {
        "total": len(test_data),
        "json_valid": 0,
        "schema_valid": 0,
        "event_type_acc": 0,
        "category_acc": 0,
        "severity_acc": 0,
        "needs_human_review_acc": 0,
        "overall_success": 0,
        "hallucinations": 0
    }
    
    # Track confusion for manual review
    confusion_event_type = collections.defaultdict(int)
    
    failures = []
    correct_examples = []

    print("Evaluating...")
    for i, item in enumerate(tqdm(test_data)):
        # Apply chat template safely extracting from messages array
        user_text = item["messages"][0]["content"]
        target = item["messages"][1]["content"]
        target_json = extract_json(target)
        
        messages = [{"role": "user", "content": user_text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gen_json = extract_json(gen_text)
        
        is_success = False

        if gen_json:
            results["json_valid"] += 1
            
            if target_json and isinstance(gen_json, dict) and isinstance(target_json, dict):
                if set(gen_json.keys()) == set(target_json.keys()):
                    results["schema_valid"] += 1
                    
                correct_fields = 0
                total_fields = 0

                if "event_type" in target_json:
                    total_fields += 1
                    t_val = target_json.get("event_type")
                    g_val = gen_json.get("event_type")
                    if g_val == t_val:
                        results["event_type_acc"] += 1
                        correct_fields += 1
                    confusion_event_type[f"T:{t_val} -> G:{g_val}"] += 1
                
                if "category" in target_json:
                    total_fields += 1
                    if gen_json.get("category") == target_json.get("category"):
                        results["category_acc"] += 1
                        correct_fields += 1
                        
                if "severity" in target_json:
                    total_fields += 1
                    if gen_json.get("severity") == target_json.get("severity"):
                        results["severity_acc"] += 1
                        correct_fields += 1
                        
                if "needs_human_review" in target_json:
                    total_fields += 1
                    if gen_json.get("needs_human_review") == target_json.get("needs_human_review"):
                        results["needs_human_review_acc"] += 1
                        correct_fields += 1
                
                if total_fields > 0 and correct_fields == total_fields:
                    results["overall_success"] += 1
                    is_success = True
                    if len(correct_examples) < 2:
                        correct_examples.append({"input": user_text, "output": gen_json})
                    
                # Hallucination check
                if any(k not in target_json for k in gen_json):
                    results["hallucinations"] += 1

        if not is_success and len(failures) < 5:
            failures.append({
                "input": user_text,
                "target": target_json,
                "generated": gen_json,
            })
                    
    print("\n--- FINAL EVALUATION ---")
    print(f"Total evaluated: {results['total']}")
    print(f"JSON Validity: {results['json_valid'] / results['total'] * 100:.2f}%")
    if results['json_valid'] > 0:
        print(f"Schema Validity: {results['schema_valid'] / results['json_valid'] * 100:.2f}%")
        print(f"Event Type Accuracy: {results['event_type_acc'] / results['json_valid'] * 100:.2f}%")
        print(f"Category Accuracy: {results['category_acc'] / results['json_valid'] * 100:.2f}%")
        print(f"Severity Accuracy: {results['severity_acc'] / results['json_valid'] * 100:.2f}%")
        print(f"Human Review Accuracy: {results['needs_human_review_acc'] / results['json_valid'] * 100:.2f}%")
        print(f"Hallucination Rate: {results['hallucinations'] / results['json_valid'] * 100:.2f}%")
        print(f"Overall Task Success: {results['overall_success'] / results['total'] * 100:.2f}%")

    print("\n--- EVENT TYPE CONFUSION MATRIX (Top 5) ---")
    top_confusions = sorted(confusion_event_type.items(), key=lambda x: x[1], reverse=True)[:5]
    for k, v in top_confusions:
        print(f"{k}: {v}")

    print("\n--- REPRESENTATIVE SUCCESSES ---")
    for s in correct_examples:
        print(f"OUTPUT: {s['output']}")

    print("\n--- REPRESENTATIVE FAILURES ---")
    for f in failures:
        print(f"TARGET: {f['target']}")
        print(f"GENERATED: {f['generated']}")
        print("---")

if __name__ == "__main__":
    run_evaluation()
