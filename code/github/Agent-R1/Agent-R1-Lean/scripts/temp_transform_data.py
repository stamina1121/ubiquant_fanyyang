import pprint
import datasets

# template_old = """Prove the following theorem in Lean4. You can use the tools provided to you to help you.
# You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.You can use the tool as many times as you want.
# When you have the final proof in Lean4 code, you can output it inside <answer>...</answer>.

# Output format for tool call:
# <think>
# ...
# </think>
# <tool_call>
# ...
# </tool_call>
# Output format for answer:
# <think>
# ...
# </think>
# <answer>
# ...
# </answer>

# {formal_statement}
# """

template = """You are an expert in Mathematics and Lean 4. Your task is to prove the following theorem in Lean 4. You can use the tools provided to you to help you.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.You can use the tool as many times as you want.
When you have the final answer, you can output it inside <answer>...</answer>. Please always wrap up your lean4 code in ```lean ... ```

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

Please prove the following theorem
```lean
{formal_statement}
```
"""

def transform(data: dict) -> dict:
    formal_statement = data['reward_model']['ground_truth']['formal_statement']
    result = {
        "data_source": 'lean_' + data['data_source'] if 'lean' not in data['data_source'] else data['data_source'],
        "prompt": [{
            "content": template.format(formal_statement=formal_statement),
            "role": "user"
        }],
        "ability": "theorem_proving",
        "reward_model": {
            "ground_truth": formal_statement,
            "style": "rule"
        }
    }
    return result

def main(path, output_path):
    orm_data = datasets.load_dataset('parquet', data_files=path, split='train')
    cols = orm_data.column_names
    orm_data = orm_data.map(transform, remove_columns=cols)
    orm_data.to_parquet(output_path)
    pprint.pp(orm_data[-1])

if __name__=='__main__':
    import sys
    # main('data/EI_proofnet_mathlib_aug_shuffle_fewshot_prompt/8689_shuffle_prompt_train.parquet',
    #      'data/EI_proofnet_mathlib_aug_shuffle_agent_prompt_new/8689_shuffle_prompt_train.parquet')
    # main('data/concate_test_set_fewshot_prompt/wkbk_dsv1_algv2_test_20250326_155349.parquet', 
    #      'data/concate_test_set_agent_prompt_new/wkbk_dsv1_algv2_test_20250326_155349.parquet')
    main(sys.argv[1], sys.argv[2])
    