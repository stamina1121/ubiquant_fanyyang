# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess Lean theorem proving datasets to parquet format
"""

import os
import argparse
import json
import random
import re
from datasets import Dataset, load_dataset
from tqdm import tqdm
import pandas as pd
try:
    from verl.utils.hdfs_io import copy, makedirs
    HDFS_AVAILABLE = True
except ImportError:
    HDFS_AVAILABLE = False

# Sample theorems for fallback
SAMPLE_THEOREMS = [
    {
        "statement": "import Mathlib\ntheorem simple_add_comm (a b : ℕ) : a + b = b + a := by",
        "description": "Simple addition commutativity"
    },
    {
        "statement": "import Mathlib\ntheorem simple_mul_comm (a b : ℕ) : a * b = b * a := by",
        "description": "Simple multiplication commutativity"
    },
    {
        "statement": "import Mathlib\ntheorem simple_add_assoc (a b c : ℕ) : (a + b) + c = a + (b + c) := by",
        "description": "Simple addition associativity"
    },
    {
        "statement": "import Mathlib\ntheorem simple_mul_assoc (a b c : ℕ) : (a * b) * c = a * (b * c) := by",
        "description": "Simple multiplication associativity"
    },
    {
        "statement": "import Mathlib\ntheorem simple_distributivity (a b c : ℕ) : a * (b + c) = a * b + a * c := by",
        "description": "Simple distributivity"
    }
]

def load_theorems_from_file(filepath):
    """Load theorems from a JSON file."""
    if not os.path.exists(filepath):
        return SAMPLE_THEOREMS
        
    try:
        with open(filepath, 'r') as f:
            theorems = json.load(f)
        return theorems
    except Exception as e:
        print(f"Error loading theorems from file: {e}")
        return SAMPLE_THEOREMS

def process_hf_dataset(dataset_name, key="proof", proof_length_limit=1000, header=""):
    """Process a HuggingFace dataset to extract Lean theorem statements."""
    print(f"Loading dataset {dataset_name} from HuggingFace...")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Typically use train split
        data = dataset["train"]
        
        # Process items
        theorems = []
        for item in tqdm(data, desc="Processing HuggingFace dataset"):
            # Skip items without the key
            if key not in item:
                continue
                
            # Get full proof
            proof = item[key]
            
            # Extract formal statement
            if ":=" in proof:
                formal_statement = proof.split(":=")[0] + ":= by"
                formal_statement = header + formal_statement
            else:
                continue
                
            # Skip if too long
            if len(formal_statement) > proof_length_limit:
                continue
                
            # Extract description from lemma/theorem name if possible
            description = ""
            match = re.search(r'(theorem|lemma)\s+([a-zA-Z0-9_]+)', formal_statement)
            if match:
                description = match.group(2)
                
            # Add to result
            theorems.append({
                "statement": formal_statement,
                "description": description,
                "original_item": item
            })
        
        print(f"Extracted {len(theorems)} valid theorems from dataset")
        return theorems
        
    except Exception as e:
        print(f"Error processing HuggingFace dataset: {e}")
        print("Falling back to sample theorems")
        return SAMPLE_THEOREMS

def make_instruction_prompt(theorem):
    """Create an instruction prompt for the theorem."""
    statement = theorem["statement"]
    description = theorem.get("description", "")
    
    instruction = """
Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""
    
    question = f"Theorem: {statement}"
    if description:
        question = f"Theorem ({description}): {statement}"
        
    return instruction + question

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/lean')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=1)
    parser.add_argument('--theorems_file', type=str, default=None, 
                       help='JSON file containing theorem statements')
    parser.add_argument('--data_source', type=str, default='lean')
    
    # HuggingFace dataset options
    parser.add_argument('--hf_dataset', type=str, default=None,
                       help='HuggingFace dataset name to use instead of local theorems')
    parser.add_argument('--hf_key', type=str, default='formal_statement',
                       help='Key in HuggingFace dataset containing the proof')
    parser.add_argument('--proof_length_limit', type=int, default=1000,
                       help='Maximum length of the formal statement')
    parser.add_argument('--header', type=str, default='',
                       help='Header to prepend to each theorem statement')
    
    args = parser.parse_args()
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Load theorems
    if args.hf_dataset:
        # Load from HuggingFace
        all_theorems = process_hf_dataset(
            args.hf_dataset, 
            key=args.hf_key,
            proof_length_limit=args.proof_length_limit,
            header=args.header,
        )
    elif args.theorems_file:
        # Load from local file
        all_theorems = load_theorems_from_file(args.theorems_file)
    else:
        # Use sample theorems
        all_theorems = SAMPLE_THEOREMS
    
    # Ensure we have enough theorems
    if len(all_theorems) < args.train_size + args.test_size:
        print(f"Warning: Not enough theorems ({len(all_theorems)}) for requested sizes (train={args.train_size}, test={args.test_size})")
        # Duplicate theorems if needed
        while len(all_theorems) < args.train_size + args.test_size:
            all_theorems.extend(all_theorems[:min(len(all_theorems), args.train_size + args.test_size - len(all_theorems))])
    
    # Shuffle theorems to randomize splits
    random.seed(42)
    random.shuffle(all_theorems)
    
    train_theorems = all_theorems[:args.train_size]
    test_theorems = all_theorems[args.train_size:args.train_size + args.test_size]
    
    # Process each data item for training set
    train_data = []
    for idx, theorem in enumerate(tqdm(train_theorems, desc="Processing training data")):
        question = make_instruction_prompt(theorem)
        
        data = {
            "data_source": args.data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": """
You are a lean4 expert. Prove the following theorem in Lean4. You can use the tools provided to you to help you.
You must first conduct reasoning inside <think>...</think> but no need to think too many contents. After thinking, you need to use the tool. 
You can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.You can use the tool as many times as you want. 
`leansearch` is a tool for searching relative theroems and 'leanverify' is a tool for verifying lean4 code. When you have the final proof in Lean4 code, you can output it inside <answer>...</answer>.
""",
                },
                {
                "role": "user",
                "content": question,
            }],
            "ability": "theorem_proving",
            "reward_model": {
                "style": "rule",
                "ground_truth": theorem["statement"]
            },
            "extra_info": {
                'split': 'train',
                'index': str(idx),
                'description': theorem.get("description", ""),
                'statement': theorem["statement"],
                "tools_kwargs": {
                    "leansearch": {
                        "create_kwargs": {"ground_truth": theorem["statement"]},
                    },
                    "leanverify": {
                        "create_kwargs": {"ground_truth": theorem["statement"]},
                    }
                }
            }
        }
        
        # Add original item if available
        if "original_item" in theorem:
            for k, v in theorem["original_item"].items():
                if k not in data["extra_info"] and isinstance(v, (str, int, float, bool)):
                    data["extra_info"][k] = str(v)
                    
        train_data.append(data)
    
    # Process each data item for test set
    test_data = []
    for idx, theorem in enumerate(tqdm(test_theorems, desc="Processing test data")):
        question = make_instruction_prompt(theorem)
        
        data = {
            "data_source": args.data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": """
You are a lean4 expert. Prove the following theorem in Lean4. You can use the tools provided to you to help you.
You must first conduct reasoning inside <think>...</think> but no need to think too many contents. After thinking, you need to use the tool. 
You can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.You can use the tool as many times as you want. 
`leansearch` is a tool for searching relative theroems and 'leanverify' is a tool for verifying lean4 code. When you have the final proof in Lean4 code, you can output it inside <answer>...</answer>.
"""
                },
                {
                "role": "user",
                "content": question,
            }],
            "ability": "theorem_proving",
            "reward_model": {
                "style": "rule",
                "ground_truth": theorem["statement"]
            },
            "extra_info": {
                'split': 'test',
                'index': str(idx),
                'description': theorem.get("description", ""),
                'statement': theorem["statement"],
                "tools_kwargs": {
                    "leansearch": {
                        "create_kwargs": {"ground_truth": theorem["statement"]},
                    },
                    "leanverify": {
                        "create_kwargs": {"ground_truth": theorem["statement"]},
                    }
                }
            }
        }
        
        # Add original item if available
        if "original_item" in theorem:
            for k, v in theorem["original_item"].items():
                if k not in data["extra_info"] and isinstance(v, (str, int, float, bool)):
                    data["extra_info"][k] = str(v)
                    
        test_data.append(data)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Save datasets
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Created datasets: {len(train_dataset)} training examples, {len(test_dataset)} test examples")
    print(f"Saved to {local_dir}/train.parquet and {local_dir}/test.parquet")
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        if HDFS_AVAILABLE:
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            print(f"Copied data to HDFS: {args.hdfs_dir}")
        else:
            print("HDFS utilities not available. Skipping HDFS copy.")
