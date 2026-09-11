import json
import random
from pathlib import Path

random.seed(4242)

BASE_DIR = Path("data")
TRAIN_DIR = BASE_DIR / "v3_train"
VAL_DIR = BASE_DIR / "v3_validation"
TEST_DIR = BASE_DIR / "v3_test"

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# COMIBINATORIAL SEMANTIC GENERATION (V3)
# To teach the 4B model actual semantics, we combine subjects, actions, outcomes, and locations.
# ---------------------------------------------------------

subjects_contractor = ["Contractor crew", "A subcontracted electrician", "The external maintenance team", "A vendor driver"]
subjects_internal = ["Shift supervisor", "Operator", "Miner", "Mechanic", "The blasting team"]

# HAZARDS (No event happened yet, just a bad condition)
hazards_safety = [
    ("loose rock formation above the main travelway", "high"),
    ("frayed electrical cable on the continuous miner", "medium"),
    ("missing guardrail on the conveyor crossover", "medium"),
    ("puddles of oil near the generator room", "low"),
    ("unsupported roof section marked as safe", "critical")
]
hazards_env = [
    ("dust suppression sprays disabled on the transfer point", "medium"),
    ("water runoff bypassing the sediment pond", "high"),
    ("chemical drum stored without secondary containment", "low"),
    ("excessive diesel exhaust accumulating in the blind heading", "critical")
]

# NEAR MISSES (Event happened, nobody hurt, no damage)
near_misses_safety = [
    ("a 50kg rock fell 2 meters away from the crew", "critical"),
    ("the haul truck slid on the ramp and stopped inches from the berm", "high"),
    ("a high-tension cable snapped and whipped past the mechanic", "critical"),
    ("a wrench was dropped from the scaffolding, landing near a worker", "medium")
]

# INCIDENTS (Injury or damage occurred)
incidents_safety = [
    ("worker twisted ankle on uneven ground requiring first aid", "low"),
    ("contractor crushed hand in gears, requiring emergency medical evacuation", "critical"),
    ("light vehicle rolled over, driver suffered concussion", "high"),
    ("worker got grit in eye due to not wearing goggles, sent to clinic", "medium")
]

# UNSAFE CONDITIONS (General operational states)
unsafe_conditions = [
    ("ventilation fan operating at 30% capacity causing heat buildup", "operations", "high"),
    ("gas monitors out of calibration date by 3 weeks", "safety", "critical"),
    ("contractors working 14 hour shifts without recorded breaks", "labour", "medium")
]

locations = ["Sector 7G", "the main portal", "the washing plant", "Workshop 4", "Haul Road A", "the tailings dam"]

def generate_safety_example():
    category_choice = random.choice(["safety", "environment", "contractor", "operations"])
    
    # 25% hazard, 25% near_miss, 25% incident, 25% unsafe_condition
    event_type = random.choice(["hazard_observation", "near_miss", "incident", "unsafe_condition"])
    
    text = ""
    severity = "low"
    
    subject = random.choice(subjects_contractor) if category_choice == "contractor" else random.choice(subjects_internal)
    loc = random.choice(locations)
    
    if event_type == "hazard_observation":
        if category_choice == "environment":
            h, sev = random.choice(hazards_env)
        else:
            h, sev = random.choice(hazards_safety)
        text = f"{subject} reported {h} near {loc}."
        severity = sev
    elif event_type == "near_miss":
        nm, sev = random.choice(near_misses_safety)
        text = f"Near {loc}, {nm}. {subject} witnessed the event."
        severity = sev
    elif event_type == "incident":
        inc, sev = random.choice(incidents_safety)
        text = f"At {loc}, {inc}."
        if category_choice == "contractor":
            text = f"Contractor involved: {text}"
        severity = sev
    else: # unsafe condition
        cond, cat, sev = random.choice(unsafe_conditions)
        category_choice = cat # override category based on the condition
        text = f"Inspection at {loc} revealed: {cond}."
        severity = sev
        
    # Add noise
    if random.random() < 0.2:
        text = text.lower()
        
    needs_review = severity in ["high", "critical"] or event_type in ["incident", "near_miss"]
    
    output = {
        "event_type": event_type,
        "category": category_choice,
        "severity": severity,
        "needs_human_review": needs_review
    }
    
    return {
        "messages": [
            {"role": "user", "content": f"Classify this mining field report. Return STRICT JSON with keys: event_type, category, severity, needs_human_review.\n\n{text}"},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}
        ]
    }

def generate_uncertainty_example():
    ambiguous_texts = [
        "Something smells strange in the tunnel.",
        "A piece of equipment is making a weird noise.",
        "Found a puddle of unknown liquid.",
        "The sensor is blinking red but no alarm is sounding.",
        "Workers are complaining about the air."
    ]
    text = random.choice(ambiguous_texts)
    output = {
        "event_type": "hazard_observation",
        "category": "unknown",
        "severity": "unknown",
        "needs_human_review": True
    }
    return {
        "messages": [
            {"role": "user", "content": f"Classify this mining field report. If information is missing, output 'unknown' for that field. Return STRICT JSON with keys: event_type, category, severity, needs_human_review.\n\n{text}"},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}
        ]
    }

def generate_v3_data(num_samples):
    data = []
    for _ in range(int(num_samples * 0.8)):
        data.append(generate_safety_example())
    for _ in range(int(num_samples * 0.2)):
        data.append(generate_uncertainty_example())
    random.shuffle(data)
    return data

train_data = generate_v3_data(2500)
val_data = generate_v3_data(250)
test_data = generate_v3_data(250)

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(TRAIN_DIR / "train.jsonl", train_data)
save_jsonl(VAL_DIR / "validation.jsonl", val_data)
save_jsonl(TEST_DIR / "test.jsonl", test_data)

print("Generated V3 Dataset.")
print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
