import logging
import requests
from anthropic import Anthropic
import json
import re
import ast
import tempfile
import subprocess
import copy
import concurrent.futures
import threading

import traceback
from datetime import datetime
import os
import argparse
import colorlog
import sys
import time

# Thread-local storage for client
thread_local = threading.local()

# 配置日志格式
def setup_logger():
    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # 确保实时输出
    handler.flush = sys.stdout.flush

setup_logger()

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Claude Lean4 Formalization Assistant Batch Processor")
    parser.add_argument("--input_file", type=str, help="Path to the JSON input file containing queries")
    parser.add_argument("--output_file", type=str, default="batch_results.json", help="Path to the output JSON file")
    parser.add_argument("--max_turns", type=int, default=7, help="Maximum number of conversation turns per query")
    parser.add_argument("--api_key", type=str, help="Anthropic API key (defaults to env variable)")
    parser.add_argument("--delay", type=int, default=1, help="Delay between queries in seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for Lean4 verification in seconds")
    parser.add_argument("--test", action="store_true", help="Run in test mode to test sample queries and tools")
    parser.add_argument("--thinking_budget", type=int, default=1500, help="Budget for Claude's thinking in tokens")
    parser.add_argument("--max_workers", type=int, default=20, help="Maximum number of concurrent workers")
    parser.add_argument("--resume", action="store_true", help="Resume processing from the last run, skipping already processed IDs")
    return parser.parse_args()

args = parse_args()

# 初始化 Claude 客户端
def get_client():
    """Get or create a thread-local client instance"""
    if not hasattr(thread_local, "client"):
        thread_local.client = Anthropic(
            api_key=args.api_key or "sk-vn8SgqS78wiQ6yVSFsKtJ122vMrzywlBxDertnI4d41PnrmN",
            base_url="https://api.openai-proxy.org/anthropic"
        )
    return thread_local.client

# Lean4 验证配置
TIMEOUT = args.timeout
DEFAULT_LAKE_PATH = "/AI4M/users/qzh/rl/elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/AI4M/users/qzh/rl/Deepseek-Prover/mathlib4"

# 工具定义
tools = [
    {
        "name": "search_mathlib4",
        "description": "Search for theorems and definitions in Mathlib4 based on a query. When you intend to invoke the search function for Mathlib, your query should precisely articulate the theorem you are seeking in a  sentence. For instance, 'Rank-Nullity Theorem: For any linear transformation mapping from a vector space V to another vector space, the dimension of V equals the sum of the dimension of the transformation's image and the dimension of its kernel.'. You should not use query like 'dimension, vector space,linear transformation'",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g., 'bezout lemma'.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "The number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "verify_lean4_code",
        "description": "Verify Lean4 code using the Lean4 verification tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Lean4 code to verify. The system will automatically add any user-defined definitions before your code. You MUST use the think tool to think about the problem EACH TIME you receive the lean_verify_code results.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "think",
        "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you find some theorems, call this tool to think about if they are useful and how to use them to solve the problem.; if you find a bug in the code, call this tool to think about how to fix it or search for related theorems.",
        "input_schema": {
            "type": "object",
            "properties": {
            "thought": {
                "type": "string",
                "description": "Your thoughts."
            }
            },
            "required": ["thought"]
        }
    }
]

def verify_lean4_code(code: str, timeout: int = TIMEOUT):
    """
    Verify Lean4 code locally using the Lean toolchain
    
    Args:
        code: Lean4 code to verify
        timeout: Timeout in seconds
        
    Returns:
        Verification results
    """
    if "import Mathlib" not in code:
        code = "import Mathlib\n" + code
    print("Code:")
    print(code)     
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
                    [DEFAULT_LAKE_PATH, "exe", "repl"],
                    stdin=temp_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=DEFAULT_LEAN_WORKSPACE
                )
                outputs, errors = process.communicate(timeout=timeout)
            
            # Parse the result from the process output
            result = json.loads(outputs)
            
            result = {
                "sorries": result.get("sorries", []),
                "errors": [m for m in result.get("messages", []) if m["severity"] == "error"],
                "warnings": [m for m in result.get("messages", []) if m["severity"] == "warning"],
                "infos": [m for m in result.get("messages", []) if m["severity"] == "info"],
                "system_errors": None,
            }
            result["pass"] = not result["errors"]
            result["complete"] = result["pass"] and not result["sorries"]
            
            # Process error positions if there are errors
            code_lines = code.split('\n')
            for error in result.get("errors", []):
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
        except Exception as inner_e:
            return {
                "pass": False,
                "complete": False,
                "system_errors": f"Inner error: {str(inner_e)}"
            }
            
    except Exception as e:
        return {
            "pass": False,
            "complete": False,
            "system_errors": str(e)
        }

def is_function_call(single_message):
    """Determine whether the current system message is a function call."""
    pattern = re.compile(r'([^\n`]*?)\n({.*?})(?=\w*\n|$)', re.DOTALL)
    matches = pattern.findall(single_message)
    if not matches:
        return False

    func_name, args_str = matches[0]
    func_name = func_name.strip()
    try:
        parsed_args = json.loads(args_str)
    except json.JSONDecodeError:
        try:
            parsed_args = ast.literal_eval(args_str)
        except:
            return False
    
    return {"name": func_name, "arguments": parsed_args}

def leansearch(query):
    """
    Search for theorems in mathlib4
    
    Args:
        query: Search query string
        
    Returns:
        Search results (already parsed from JSON)
    """
    if query is None:
        return {"error": "Missing required parameter: query"}
        
    num_results = 6
    
    # Convert single string to list
    if isinstance(query, str):
        query = [query]
    
    try:
        url = 'https://console.siflow.cn/siflow/draco/ai4math/tyxu/leansearch-api-v4-16/search'
        params = {
            'query': query,
            'num': num_results
        }
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get theorems, status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# 处理工具调用
def process_tool_call(tool_name, tool_input):
    if tool_name == "search_mathlib4":
        return leansearch(tool_input["query"])
    elif tool_name == "verify_lean4_code":
        return verify_lean4_code(code=tool_input["code"])
    elif tool_name == "think":
        # Simply log the thought and return it
        logging.info(f"Thinking: {tool_input['thought']}")
        return {"result": "Thought recorded"}
    else:
        return {"error": f"Unknown tool: {tool_name}"}

# Check if the Anthropic client supports the thinking feature
def is_thinking_supported():
    try:
        import anthropic
        import pkg_resources
        version = pkg_resources.get_distribution("anthropic").version
        # Thinking is supported in anthropic>=0.19.0
        version_parts = version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        return (major > 0) or (major == 0 and minor >= 19)
    except Exception as e:
        print(f"Error checking Anthropic version: {e}")
        return False

# Modified send_message to use thread-local client
def send_message(messages, system_prompt, thinking_budget=1500):
    """Send a message to Claude with thinking enabled."""
    logging.info(f"Sending messages to model: {messages}")
    
    # Get thread-local client
    client = get_client()
    
    # 确保消息不包含 thinking 参数
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = msg.copy()
        if "thinking" in cleaned_msg:
            del cleaned_msg["thinking"]  # 删除消息中的 thinking 字段
        cleaned_messages.append(cleaned_msg)
    
    # 构建基本参数
    kwargs = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 8192,
        "tools": tools,
        "system": system_prompt,
        "messages": cleaned_messages  # 使用清理后的消息
    }
    
    # 始终启用思考功能 - 作为顶级参数
    thinking_enabled = False
    thinking_supported = is_thinking_supported()
    if thinking_supported:
        try:
            thinking_enabled = True
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        except Exception as e:
            thinking_enabled = False
            logging.warning(f"Failed to add thinking parameter: {str(e)}")
    else:
        logging.warning("Thinking feature requires Anthropic Python SDK >= 0.19.0. Feature will be disabled.")
    
    try:
        if thinking_enabled:
            logging.info("Sending request with thinking enabled")
        else:
            logging.info("Sending request without thinking")
            
        response = client.messages.create(**kwargs)
        logging.info(f"Model response received with {len(response.content)} content blocks")
        return response
    except Exception as e:
            raise

def get_from_http(query, num=10):
    """Search for theorems in Mathlib4."""
    url = 'https://console.siflow.cn/siflow/draco/ai4math/zhqin/leansearch-v1/get_relate_theorems'
    params = {
        'query': query,
        'num': num
    }
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Failed to get theorems, status code: {response.status_code}")

SYSTEM_PROMPT = '''
You are a Lean4 formalization assistant.
## Using the think tool

Before taking any action or fixing any bug after receiving tool results, you MUST use the think tool as a scratchpad to:
- Analyze the theorems you've found through search
- Plan your formalization approach step by step
- Analyze the error message
- Identify potential issues in code before verification
- Reason through mathematical concepts and their formal representations

Here are some examples of what to iterate over inside the think tool:
<think_tool_example_1>
User wants to prove that the sum of even numbers is even
- Found theorems:
  * `even_add`: a + b is even iff a and b are both even or both odd
  * `even_iff_exists_mul_two`: n is even iff n = 2k for some k
- Plan for formalization:
  1. Define two arbitrary even numbers a and b
  2. Use even_iff_exists_mul_two to express a = 2j and b = 2k
  3. Express the sum: a + b = 2j + 2k = 2(j + k)
  4. Apply even_iff_exists_mul_two backward to prove a + b is even
- Required tactics: rewrite, exists, ring
</think_tool_example_1>

<think_tool_example_2>
User's code has an error when proving properties of a group
- Error message indicates "unknown identifier 'mul_assoc'"
- Analysis of the error:
  * Could be missing import for algebraic structures
  * Could be using wrong namespace
  * Could be incorrect theorem name
- Relevant theorems:
  * `Group.mul_assoc`: Multiplication in a group is associative
  * `Monoid.mul_assoc`: Multiplication in a monoid is associative
- Fix options:
  1. Try using full notation: Group.mul_assoc
  2. Check if we're working in a group or monoid context
- Plan:
  1. Check context of the proof
  2. Search for another theorems
</think_tool_example_2>
You MUST use the think tool to think about the problem every time you receive lean_verify_code results or search results.

## Use the verify tool
Each time you fix your code or want to submit your answer, you must use the verify tool to check your answer.
'''

def process_query(query, max_turns=9, thinking_budget=1500, max_retries=3, query_id=None):
    """Process a single query with a maximum number of conversation turns."""
    logging.info(f"Processing query: {query}")
    
    # Initialize conversation data
    conversation = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "messages": [],
        "tool_calls": 0,
        "turn_details": []  # For detailed turn-by-turn information
    }
    
    # Add query ID if provided
    if query_id:
        conversation["id"] = query_id
    
    # Start with the user's query
    messages = [{"role": "user", "content": query}]
    conversation["messages"].append({"role": "user", "content": query})
    
    # Process conversation turns up to max_turns
    turn_count = 0
    while turn_count < max_turns:
        # Send message to Claude with retries
        retry_count = 0
        response = None
        last_error = None
        
        while retry_count < max_retries and response is None:
            try:
                # Try with progressively lower thinking budget if we're having issues
                current_budget = thinking_budget if retry_count == 0 else max(500, thinking_budget - retry_count * 300)
                response = send_message(
                    messages, 
                    SYSTEM_PROMPT,
                    thinking_budget=current_budget
                )
                break
            except Exception as e:
                last_error = e
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                time.sleep(2 * retry_count)  # Exponential backoff
        
        # If all retries failed, raise the last error
        if response is None:
            raise RuntimeError(f"Failed to get response after {max_retries} retries: {str(last_error)}")
        
        # Convert response blocks to serializable format
        serializable_blocks = []
        for block in response.content:
            if block.type == "text":
                serializable_blocks.append({
                    "type": "text",
                    "text": block.text
                })
            elif block.type == "tool_use":
                serializable_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
            elif block.type in ["thinking", "redacted_thinking"]:
                serializable_blocks.append({
                    "type": block.type,
                    "thinking": block.thinking if hasattr(block, "thinking") else "[Redacted thinking]"
                })
        
        # Record detailed information about this turn
        turn_detail = {
            "turn_number": turn_count + 1,
            "timestamp": datetime.now().isoformat(),
            "response_blocks": serializable_blocks
        }
        conversation["turn_details"].append(turn_detail)
        
        # Create assistant message with all blocks for API (keeping original blocks)
        assistant_message = {
            "role": "assistant",
            "content": response.content
        }
        
        # Add to conversation record (with serializable blocks)
        conversation["messages"].append({
            "role": "assistant", 
            "content": serializable_blocks,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if we've reached the end of the conversation
        if response.stop_reason != "tool_use":
            # Add to messages for API (preserve original blocks)
            messages.append(assistant_message)
            logging.info(f"Conversation complete after {turn_count + 1} turns")
            break
        
        # Handle tool use
        tool_use = next((block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"), None)
        if not tool_use:
            logging.error("Expected tool use but none found in response")
            break
            
        tool_name = tool_use.name
        tool_input = tool_use.input
        
        # Record tool call start time
        tool_call_start = datetime.now()
        
        # Process tool call with error handling
        try:
            tool_result = process_tool_call(tool_name, tool_input)
            tool_success = True
        except Exception as e:
            logging.error(f"Error processing tool call: {str(e)}")
            tool_result = {"error": f"Tool execution failed: {str(e)}"}
            tool_success = False
        
        # Record tool call completion time
        tool_call_end = datetime.now()
        
        # Create user message with tool result
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                }
            ]
        }
        
        # Record detailed tool call information
        tool_call_detail = {
            "tool_name": tool_name,
            "input": tool_input,
            "timestamp_start": tool_call_start.isoformat(),
            "timestamp_end": tool_call_end.isoformat(),
            "duration_seconds": (tool_call_end - tool_call_start).total_seconds(),
            "success": tool_success,
            "result": tool_result
        }
        conversation["turn_details"][turn_count]["tool_call"] = tool_call_detail
        
        # Add messages to both API history and conversation record
        messages.append(assistant_message)
        messages.append(user_message)
        
        # Add user message to conversation record with timestamp
        conversation["messages"].append({
            "role": "user",
            "content": user_message["content"],
            "timestamp": datetime.now().isoformat(),
            "tool_result": {
                "tool_name": tool_name,
                "input": tool_input,
                "result": tool_result,
                "success": tool_success
            }
        })
        
        # Update tool call count
        conversation["tool_calls"] += 1
        
        # Increment turn counter
        turn_count += 1
    
    # Add final summary information
    conversation["summary"] = {
        "total_turns": turn_count,
        "total_tool_calls": conversation["tool_calls"],
        "completed": response.stop_reason != "tool_use",
        "duration_seconds": (datetime.now() - datetime.fromisoformat(conversation["timestamp"])).total_seconds()
    }
    
    logging.info(f"Completed processing query with {turn_count} turns and {conversation['tool_calls']} tool calls")
    return conversation

def process_batch(input_file, output_file, max_turns=7, thinking_budget=1500, delay=1, max_retries=3, max_workers=20, resume=False):
    """Process a batch of queries from a JSON file using thread pool for concurrency."""
    # Read queries from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Load existing results if resuming
    processed_ids = set()
    results = []
    
    if resume and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                # Extract processed IDs
                for result in results:
                    if "id" in result:
                        processed_ids.add(result["id"])
            logging.info(f"Resuming from existing output file. Found {len(processed_ids)} already processed items.")
        except Exception as e:
            logging.error(f"Error loading existing results for resume: {str(e)}")
            results = []
            processed_ids = set()
    
    # Extract and filter queries from input data
    if isinstance(input_data, list):
        # Check if it's a list of dictionaries with id and query fields
        if len(input_data) > 0 and isinstance(input_data[0], dict) and "id" in input_data[0] and "query" in input_data[0]:
            # Filter out already processed items if resuming
            if resume:
                queries = [item for item in input_data if item["id"] not in processed_ids]
            else:
                queries = input_data
        else:
            # Convert simple list of strings to id/query format
            queries = [{"id": str(i), "query": q} for i, q in enumerate(input_data)]
    elif isinstance(input_data, dict) and "queries" in input_data:
        # Dictionary with "queries" key
        if isinstance(input_data["queries"][0], dict) and "id" in input_data["queries"][0]:
            if resume:
                queries = [item for item in input_data["queries"] if item["id"] not in processed_ids]
            else:
                queries = input_data["queries"]
        else:
            queries = [{"id": str(i), "query": q} for i, q in enumerate(input_data["queries"])]
    else:
        # Try to interpret other formats
        queries = [{"id": "single", "query": str(input_data)}]
    
    logging.info(f"Loaded {len(queries)} queries to process from {input_file}")
    if resume:
        logging.info(f"Skipping {len(processed_ids)} already processed items")
    
    # Initialize results lock for thread-safe updates
    results_lock = threading.Lock()
    
    # Function to process a single query and update results safely
    def process_and_save(i, query_item):
        query_idx = i + 1
        query_id = query_item["id"]
        query_text = query_item["query"]
        
        logging.info(f"Processing query {query_idx}/{len(queries)}: ID={query_id}, Text={query_text[:50]}...")
        
        try:
            # Process the query
            conversation = process_query(
                query=query_text,
                max_turns=max_turns,
                thinking_budget=thinking_budget,
                max_retries=max_retries,
                query_id=query_id
            )
            
            result = conversation
        except Exception as e:
            logging.error(f"Error processing query {query_idx}: {str(e)}")
            # Create error record
            result = {
                "id": query_id,
                "query": query_text,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
        
        # Safely update the results list and save to file
        with results_lock:
            results.append(result)
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return result
    
    # Process queries concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures
        futures = {executor.submit(process_and_save, i, query_item): (i, query_item) for i, query_item in enumerate(queries)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            i, query_item = futures[future]
            try:
                future.result()  # Just to catch and log any exceptions not caught in process_and_save
            except Exception as e:
                logging.error(f"Unexpected error in worker processing query {i+1}: {str(e)}")
    
    logging.info(f"Batch processing complete. Results saved to {output_file}")
    return results

def run_tests():
    """Run tests for the tools and sample queries."""
    logging.info("Starting tests...")
    test_results = {
        "tool_tests": {},
        "query_tests": []
    }
    
    # Test 1: Test the search_mathlib4 tool
    logging.info("Testing search_mathlib4 tool...")
    try:
        search_query = "Pythagorean theorem"
        search_result = process_tool_call("search_mathlib4", {"query": search_query})
        test_results["tool_tests"]["search_mathlib4"] = {
            "input": search_query,
            "success": True,
            "result_count": len(search_result) if isinstance(search_result, list) else 0,
            "sample_result": search_result[0] if isinstance(search_result, list) and len(search_result) > 0 else None
        }
        logging.info(f"search_mathlib4 test result: {json.dumps(test_results['tool_tests']['search_mathlib4'], ensure_ascii=False)}")
    except Exception as e:
        test_results["tool_tests"]["search_mathlib4"] = {
            "input": search_query,
            "success": False,
            "error": str(e)
        }
        logging.error(f"search_mathlib4 test failed: {str(e)}")

    # Test 2: Test the verify_lean4_code tool
    logging.info("Testing verify_lean4_code tool...")
    try:
        test_code = """
import Mathlib
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ a ih => 
    simp
    rw [Nat.add_succ, ih, Nat.succ_add]
"""
        verify_result = process_tool_call("verify_lean4_code", {"code": test_code})
        test_results["tool_tests"]["verify_lean4_code"] = {
            "input": test_code,
            "success": True,
            "passed": verify_result.get("pass", False),
            "errors": verify_result.get("errors", [])
        }
        logging.info(f"verify_lean4_code test result: {json.dumps(test_results['tool_tests']['verify_lean4_code'], ensure_ascii=False)}")
    except Exception as e:
        test_results["tool_tests"]["verify_lean4_code"] = {
            "input": test_code,
            "success": False,
            "error": str(e)
        }
        logging.error(f"verify_lean4_code test failed: {str(e)}")

    # Test 3: Test the think tool
    logging.info("Testing think tool...")
    try:
        thought = "Testing the think tool functionality."
        think_result = process_tool_call("think", {"thought": thought})
        test_results["tool_tests"]["think"] = {
            "input": thought,
            "success": True,
            "result": think_result
        }
        logging.info(f"think test result: {json.dumps(test_results['tool_tests']['think'], ensure_ascii=False)}")
    except Exception as e:
        test_results["tool_tests"]["think"] = {
            "input": thought,
            "success": False,
            "error": str(e)
        }
        logging.error(f"think test failed: {str(e)}")

    # Test 4: Test sample query 1
    logging.info("Testing sample query 1...")
    try:
        query1 = "Prove that the sum of two even numbers is even in Lean4"
        max_turns = 2  # Limit to 2 turns for testing purposes
        conversation1 = process_query(query1, max_turns=max_turns)
        test_results["query_tests"].append({
            "query": query1,
            "max_turns": max_turns,
            "success": True,
            "tool_calls": conversation1["tool_calls"],
            "turns": len(conversation1["messages"]) // 2,
            "messages": conversation1["messages"]
        })
        logging.info(f"Sample query 1 test completed with {conversation1['tool_calls']} tool calls")
    except Exception as e:
        test_results["query_tests"].append({
            "query": query1,
            "max_turns": max_turns,
            "success": False,
            "error": str(e)
        })
        logging.error(f"Sample query 1 test failed: {str(e)}")

    # Test 5: Test sample query 2
    logging.info("Testing sample query 2...")
    try:
        query2 = "Write a Lean4 function to check if a natural number is prime"
        max_turns = 2  # Limit to 2 turns for testing purposes
        conversation2 = process_query(query2, max_turns=max_turns)
        test_results["query_tests"].append({
            "query": query2,
            "max_turns": max_turns,
            "success": True,
            "tool_calls": conversation2["tool_calls"],
            "turns": len(conversation2["messages"]) // 2,
            "messages": conversation2["messages"]
        })
        logging.info(f"Sample query 2 test completed with {conversation2['tool_calls']} tool calls")
    except Exception as e:
        test_results["query_tests"].append({
            "query": query2,
            "max_turns": max_turns,
            "success": False,
            "error": str(e)
        })
        logging.error(f"Sample query 2 test failed: {str(e)}")
    
    # Save test results to file
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logging.info("All tests completed. Results saved to test_results.json")
    return test_results

def main():
    """Main entry point for the batch processing script."""
    # Get thinking budget from command line arguments
    thinking_budget = args.thinking_budget
    max_workers = args.max_workers
    resume = args.resume
    
    # Check if in test mode
    if args.test:
        logging.info("Running in test mode")
        try:
            run_tests()
        except Exception as e:
            logging.error(f"Testing failed: {str(e)}")
            sys.exit(1)
        return
    
    # Check if input file is provided
    if not args.input_file:
        logging.error("Input file is required. Use --test to run tests or provide an input file.")
        sys.exit(1)
        
    # Initialize basic parameters
    input_file = args.input_file
    output_file = args.output_file
    max_turns = args.max_turns
    delay = args.delay
    
    # Validate input file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Process the batch of queries
    try:
        logging.info(f"Starting batch processing with {max_workers} concurrent workers")
        if resume:
            logging.info("Resume mode enabled - will skip already processed items")
            
        results = process_batch(
            input_file=input_file,
            output_file=output_file,
            max_turns=max_turns,
            thinking_budget=thinking_budget,
            delay=delay,
            max_retries=3,
            max_workers=max_workers,
            resume=resume
        )
        
        # Calculate and log summary statistics
        total_queries = len(results)
        # Skip failed queries when calculating stats
        successful_results = [r for r in results if "error" not in r]
        total_turns = sum(len(conv["messages"]) // 2 for conv in successful_results if "messages" in conv)
        total_tool_calls = sum(conv["tool_calls"] for conv in successful_results if "tool_calls" in conv)
        
        logging.info(f"Batch Processing Summary:")
        logging.info(f"Total queries processed: {total_queries}")
        logging.info(f"Successful queries: {len(successful_results)}")
        logging.info(f"Total conversation turns: {total_turns}")
        logging.info(f"Total tool calls: {total_tool_calls}")
        
    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()