#!/bin/bash

# Create data directory
mkdir -p data/lean_ds_verl

# Process Lean theorem proving data from HuggingFace
python -m examples.data_preprocess.lean_for_verl \
    --local_dir data/lean_ds_verl \
    --hf_dataset "deepseek-ai/DeepSeek-Prover-V1" \
    --hf_key "formal_statement" \
    --train_size 10000 \
    --test_size 16 \
    --data_source lean_prover \
    --header $'import Mathlib\nimport Aesop\nopen BigOperators Real Nat Topology Rat\n'

# Alternatively, use a local JSON file of theorems
# python -m Agent-R1.examples.data_preprocess.lean \
#     --local_dir data/lean \
#     --theorems_file path/to/theorems.json \
#     --train_size 100 \
#     --test_size 10
