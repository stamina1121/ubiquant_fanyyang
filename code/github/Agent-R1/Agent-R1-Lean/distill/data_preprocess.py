from datasets import load_dataset
import json
import random
import uuid

# Load the first dataset from Hugging Face
print("Loading first dataset (algebra_exercises)...")
dataset1 = load_dataset("pkuAI4M/algebra_exercises_v4_11_0_cleaner")

# Extract the formal_statement field and add the prefix
processed_data1 = []
for item in dataset1["train"]:
    if "formal_statement" in item and item["formal_statement"]:
        statement = f"Please finish this proof in Lean4: {item['formal_statement']}"
        processed_data1.append({
            "id": str(uuid.uuid4()),
            "query": statement
        })

# Sample 500 items from the first dataset
if len(processed_data1) > 500:
    processed_data1 = random.sample(processed_data1, 500)
print(f"Processed {len(processed_data1)} statements from the first dataset")

# Load the second dataset from Hugging Face
print("Loading second dataset (Lean-workbook-proofs)...")
dataset2 = load_dataset("Goedel-LM/Lean-workbook-proofs")

# Extract the full_proof field, take content before ":=", and add the prefix
processed_data2 = []
for item in dataset2["train"]:
    if "full_proof" in item and item["full_proof"]:
        # Split by ":=" and take the first part
        statement_parts = item["full_proof"].split(":=", 1)
        if statement_parts:
            statement = f"Please finish this proof in Lean4: {statement_parts[0].strip()}"
            processed_data2.append({
                "id": str(uuid.uuid4()),
                "query": statement
            })

# Sample 500 items from the second dataset
if len(processed_data2) > 500:
    processed_data2 = random.sample(processed_data2, 500)
print(f"Processed {len(processed_data2)} statements from the second dataset")

# Combine the two datasets
combined_data = processed_data1 + processed_data2
random.shuffle(combined_data)  # Shuffle to mix the two sources

# Save to file
with open("test_statements.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"Combined {len(combined_data)} statements and saved to test_statements.json")
