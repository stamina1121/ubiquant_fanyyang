import re
import logging
import json
import time
import pprint
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple, Dict, Any
import datetime

# Constants for Lean verification
DEFAULT_LAKE_PATH = "/AI4M/users/ytwang/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/AI4M/users/ytwang/auto-proof/repl_server/lean_test_v4160"
VERIFICATION_LOG_FILE = "verification_log.jsonl"
TIMEOUT = 60
ANSWER_SCORE = 4.0

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='verification_logs.txt'
)

def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the Lean proof code from the solution string."""
    try:
        # Extract the last assistant block
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return None
        
        # Look for code directly within <answer> tags in the last assistant block
        answer_block = assistant_blocks[-1]
        answer_match = re.search(r'<answer>(.*?)</answer>', answer_block, re.DOTALL)
        
        if not answer_match:
            return None
            
        code = answer_match.group(1).replace("```lean4","").replace("```lean","").replace("```","").strip()
        # Extract the actual proof part after ":= by" or ":=by"
        if ":= by" in code:
            return ':= by'.join(code.split(":= by")[1:])
        elif ":=by" in code:
            return ':=by'.join(code.split(":=by")[1:])

        else:
            # logging.warning("Lean code found but no ':= by' pattern detected")
            return None  # Return the entire code if no by pattern is found
            
    except Exception as e:
        logging.error(f"Error extracting solution: {str(e)}")
        return None

def log_verification_attempt(proof_code: str, formal_statement: str, result: Dict[str, Any], duration: float) -> None:
    """Log verification attempt to JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "proof_code": proof_code,
        "formal_statement": formal_statement,
        "result": result,
        "duration_seconds": duration
    }
    with open(VERIFICATION_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def verify_proof(code: str, formal_statement: str, 
                lake_path: str = DEFAULT_LAKE_PATH,
                lean_workspace: str = DEFAULT_LEAN_WORKSPACE,
                timeout: int = TIMEOUT) -> bool:
    """Verify a Lean proof using the Lean toolchain."""
    if code is None:
        # logging.warning("verify_proof called with None code parameter")
        return False
        
    full_code = formal_statement.strip() + code
    command = {"cmd": full_code, "allTactics": False, "ast": False, 
              "tactics": False, "premises": False}
    message_str = json.dumps(command, ensure_ascii=False)
    
    process = None
    start_time = time.time()
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            process = subprocess.Popen(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=lean_workspace
            )
            outputs, errors = process.communicate(timeout=timeout)
        result = json.loads(outputs)
        result.update({'verify_code': full_code})
        # pprint.pp(result)
        processed_result = {
            "sorries": result.get("sorries", []),
            "errors": [m for m in result.get("messages", []) if m["severity"] == "error"],
        }
        
        duration = time.time() - start_time
        log_verification_attempt(code, formal_statement, processed_result, duration)
        return not processed_result["errors"] and not processed_result["sorries"]
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        duration = time.time() - start_time
        log_verification_attempt(code, formal_statement, {"error": "timeout"}, duration)
        logging.error(f"Verification timed out after {timeout}s")
        return False
    except Exception as e:
        if process:
            process.kill()
        duration = time.time() - start_time
        print(traceback.format_exc())
        log_verification_attempt(code, formal_statement, {"error": str(e)}, duration)
        logging.error(f"Verification failed: {str(e)}")
        return False
    finally:
        if process and process.poll() is None:
            process.kill()

def compute_score_format(solution_str: str) -> float:
    """Compute format score for the solution string."""
    if solution_str is None:
        return 0.0
    
    try:
        format_reward = 0.0
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            return 0.0
            
        # Check for think and tool_call in intermediary blocks
        for i, block in enumerate(assistant_blocks[:-1]):
            if block.count('<think>') == 1 and block.count('</think>') == 1 and block.count('<tool_call>') == 1 and block.count('</tool_call>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', block, re.DOTALL)
                if think_match:
                    format_reward += 0.5
                
        # Check last assistant block has think and answer
        if assistant_blocks:
            last_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.5
                answer_content = think_answer_match.group(2)
                if re.search(r'```lean.*?```', answer_content, re.DOTALL):
                    format_reward += 0.5
        return min(format_reward, 1.5)
    except Exception as e:
        logging.error(f"Error computing format score: {str(e)}")
        return 0.0

def compute_score_answer(solution_str: str, ground_truth: str) -> float:
    """Compute answer score based on Lean proof verification."""
    if solution_str is None:
        return 0.0
    
    try:
        # Extract proof code
        proof_code = extract_solution(solution_str)
        
        if proof_code is None:
            return 0.0
            
        # Verify the proof
        verification_result = verify_proof(proof_code, ground_truth)
        
        # Return 1.0 for successful verification, 0.0 otherwise
        return ANSWER_SCORE if verification_result else 0.0
    except Exception as e:
        print(traceback.format_exc())
        logging.error(f"Error computing answer score: {str(e)}")
        return 0.0

def compute_score_format_answer(solution_str: str, ground_truth: str) -> float:
    """The scoring function combining format and answer scores."""
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        
        return format_reward + answer_reward
        # if format_reward == 1.0:
        #     return -1.0 + format_reward + answer_reward
        # else:
        #     return -1.0 + format_reward
    except Exception as e:
        logging.error(f"[DEBUG]Error in compute_score: {str(e)}")
        return 0.0

def test_lean_single():
    test_statement = "import Mathlib\ntheorem test_add : 2 + 2 = 4 := by"
    test_proof = "\n  ring"
    verification_result = verify_proof(
        test_proof,
        test_statement,
        DEFAULT_LAKE_PATH,
        DEFAULT_LEAN_WORKSPACE
    )
    print(verification_result)
    assert verification_result, "Proof verification failed"

def test_lean_reward():
    """Test the lean reward functionality."""
    test_inputs = ["""<|im_start|>assistant
<think>
Let's analyze the problem.
We need to prove that if d^2/2 = 40, then d^2 = 80.
This is straightforward algebra.
</think>
<answer>
theorem thm_26878 (d : ℝ) (h : d > 0) (h₀ : d ^ 2 / 2 = 40) : d ^ 2 = 80 := by
  have h₁ : d ^ 2 / 2 = 40 := h₀
  have h₂ : d ^ 2 = 80 := by
    rw [← mul_right_inj' (two_ne_zero' ℝ)] at h₁
    linarith
  exact h₂
</answer>
<|im_end|>
    """, """<|im_start|>assistant
<think>
Let's analyze the problem.
We need to prove that if d^2/2 = 40, then d^2 = 80.
This is straightforward algebra.
</think>
<answer>
```lean
theorem thm_26878 (d : ℝ) (h : d > 0) (h₀ : d ^ 2 / 2 = 40) : d ^ 2 = 80 := by
  have h₁ : d ^ 2 / 2 = 40 := h₀
  have h₂ : d ^ 2 = 80 := by
    rw [← mul_right_inj' (two_ne_zero' ℝ)] at h₁
    linarith
  exact h₂
```
</answer>
<|im_end|>
    ""","""<|im_start|>assistant
<think>
Let's analyze the problem.
We need to prove that if d^2/2 = 40, then d^2 = 80.
This is straightforward algebra.
</think>
<answer>
```lean
theorem thm_26878 (d : ℝ) (h : d > 0) (h₀ : d ^ 2 / 2 = 40) : d ^ 2 = 80 := by
  admit
```
</answer>
<|im_end|>
    """]
    
    for test_input in test_inputs:
        proof_code = extract_solution(test_input)
        print(f"Extracted proof code: {proof_code}")
        assert proof_code is not None, "Failed to extract solution"
        ground_truth = "import Mathlib\ntheorem thm_26878 (d : ℝ) (h : d > 0) (h₀ : d ^ 2 / 2 = 40) : d ^ 2 = 80 := by"
        
        format_score = compute_score_format(test_input)
        print(f"Format score: {format_score}")
        assert format_score > 0, "Format score should be positive"
        
        answer_score = compute_score_answer(test_input, ground_truth)
        print(f"Answer score: {answer_score}")
        
        total_score = compute_score_format_answer(test_input, ground_truth)
        print(f"Total score: {total_score}")
        assert total_score > -1.0, "Score should be greater than -1.0 for correct format"

if __name__ == "__main__":
    test_lean_single()
    test_lean_reward()
    print("All tests passed!")
