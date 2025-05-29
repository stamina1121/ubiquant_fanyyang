"""
LeanVerify tool implementation for verifying Lean4 code
"""

import requests
from typing import Dict, List, Any, Optional
import json
import tempfile
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

from agent_r1.tool.tool_base import Tool

DEFAULT_LAKE_PATH = "/AI4M/users/ytwang/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/AI4M/users/ytwang/auto-proof/repl_server/lean_test_v4160"
TIMEOUT = 60

class LeanVerifyTool(Tool):
    """
    Tool for verifying Lean4 code
    """
    
    def __init__(self, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE):
        """
        Initialize the Lean verification tool
        
        Args:
            lake_path: Path to the lake executable
            lean_workspace: Path to the Lean workspace directory
        """
        name = "leanverify"
        description = "Verify Lean4 code and check for correctness."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Lean4 code to verify.",
                },
            },
            "required": ["code"],
        }
        
        self.lake_path = lake_path
        self.lean_workspace = lean_workspace
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute Lean4 code verification
        
        Args:
            args: Tool parameters, containing:
                - "code": Lean4 code string
                - "timeout": optional timeout in seconds
            
        Returns:
            Verification results
        """
        code = args["code"]
        # timeout = args.get("timeout", 300)
        
        try:
            result = self._verify_lean4_code(code, timeout=TIMEOUT)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "pass": False,
                "complete": False,
                "system_errors": str(e)
            })
    
    def worker(self, args):
        return self.execute(args)
        
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute batch verification using multiprocessing
        
        Args:
            args_list: List of verification parameter dictionaries
            
        Returns:
            List of verification results
        """
        max_workers = min(64, len(args_list))
        
        # Define a worker function to execute on each process
        
        
        # Execute in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.worker, args_list))
        
        return results
    
    def _verify_lean4_code(self, code: str, timeout: int = TIMEOUT):
        """
        Verify Lean4 code locally using the Lean toolchain
        
        Args:
            code: Lean4 code to verify
            timeout: Timeout in seconds
            
        Returns:
            Verification results
        """
        try:
            command = {"cmd": code, "allTactics": False, "ast": False, 
                      "tactics": False, "premises": False}
            message_str = json.dumps(command, ensure_ascii=False)
            
            process = None
            start_time = time.time()
            try:
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
                    temp_file.write(message_str + "\r\n\r\n")
                    temp_file.seek(0)
                    process = subprocess.Popen(
                        [self.lake_path, "exe", "repl"],
                        stdin=temp_file,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=self.lean_workspace
                    )
                    outputs, errors = process.communicate(timeout=timeout)
                
                # Parse the result from the process output
                result = json.loads(outputs)
                
                # Process error positions if there are errors
                code_lines = code.split('\n')
                for error in result.get('errors', []):
                    if 'pos' in error and 'endPos' in error:
                        start_line = error['pos']['line'] - 1  # 0-indexed
                        start_col = error['pos']['column']
                        end_line = error['endPos']['line'] - 1
                        end_col = error['endPos']['column']
                        
                        # Extract the error context
                        if start_line == end_line:
                            line = code_lines[start_line] if start_line < len(code_lines) else ""
                            if end_col - start_col <= 80:
                                # Show the entire segment if it's short enough
                                error_context = line[start_col:end_col]
                            else:
                                # Show 40 chars before and after
                                start_idx = max(0, start_col)
                                end_idx = min(len(line), end_col)
                                error_context = line[start_idx:start_idx+40] + "..." + line[end_idx-40:end_idx]
                            
                            # Add the error context to the error object
                            error['error_pos'] = error_context
                
                return result
                
            except subprocess.TimeoutExpired:
                if process:
                    process.kill()
                return {
                    "pass": False,
                    "complete": False,
                    "system_errors": f"Verification timeout after {timeout} seconds"
                }
                
        except Exception as e:
            return {
                "pass": False,
                "complete": False,
                "system_errors": str(e)
            }
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for verification action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Higher reward if verification passes
        result_obj = json.loads(result)
        if result_obj.get("pass", False):
            return 0.1
        # Lower reward if verification completes but has errors
        elif result_obj.get("complete", False):
            return 0.1
        # No reward if verification fails completely
        else:
            return 0.0
        # if args.get("code", "") == "":
        #     return 0.0
        # else:
        #     return 0.0

if __name__ == "__main__":
    Verifier = LeanVerifyTool()
    print(Verifier.parameters)
    print(type(Verifier.parameters))