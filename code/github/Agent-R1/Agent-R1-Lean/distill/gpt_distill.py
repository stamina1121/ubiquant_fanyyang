import json
import re
import ast
import requests
import traceback
import tempfile
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Remove local model loading
# MODEL_PATH = "/AI4M/llm/GLM-4-9B-0414/"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# Configure OpenAI client
client = OpenAI(base_url="https://api.openai-proxy.org/v1",api_key="sk-vn8SgqS78wiQ6yVSFsKtJ122vMrzywlBxDertnI4d41PnrmN")  # Replace with your actual API key

DEFAULT_LAKE_PATH = "/AI4M/users/ytwang/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/AI4M/users/ytwang/auto-proof/repl_server/lean_test_v4160"
TIMEOUT = 60

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
        JSON string containing theorem results
    """
    if query is None:
        return json.dumps({"error": "Missing required parameter: query"})
        
    num_results = 6
    
    # Convert single string to list
    if isinstance(query, str):
        query = [query]
    
    try:
        url = 'https://console.siflow.cn/siflow/draco/ai4math/tyxu/leansearch-api-v4-16/search'
        params = {
            'query': query,
            'num_results': num_results
        }
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            results = response.json()
            return format_results(results)
        else:
            return json.dumps({"error": f"Failed to get theorems, status code: {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def format_results(results):
    """
    Format search results as a JSON string
    """
    if isinstance(results, dict) and "error" in results:
        return json.dumps({"error": results['error']})
        
    formatted_results = []
    
    # Process the nested list structure
    if isinstance(results, list):
        for result_group in results:
            for item in result_group:
                # Extract data from the result
                docstring = item['result']['docstring']
                name = '.'.join(item['result']['name'])
                signature = item['result']['signature']
                value = item['result']['value']
                informal_description = item['result'].get('informal_description', '')
                
                # Create formatted theorem text
                theorem_text = f"theorem {name} {signature} {value}"
                
                # Add to results list
                formatted_results.append({
                    "name": name,
                    "statement": theorem_text,
                    "docstring": docstring,
                    "informal_description": informal_description if False else ""
                })
    else:
        return json.dumps({"error": "Search returned unexpected format. No results found."})
            
    return json.dumps(formatted_results, indent=2, ensure_ascii=False)

# Define tools for the API - following the specified format
tools = [   
    {
        "type": "function",
        "name": "leansearch",
        "description": "Search for theorems and definitions in Mathlib4 based on a query. Query should articulate the theorem in a sentence. Include your thinking process in the think_content parameter.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Queries like 'For integers \\(a, b \\neq 0\\), there exist \\(x, y \\in \\mathbb{Z}\\) such that \\(ax + by = \\gcd(a, b)\\).'",
                },
                "think_content": {
                    "type": "string",
                    "description": "It is crucial to provide a complete output of your reasoning process before the tool_call, including your thought process for why you are searching for this theorem and what you hope to find. This is essential for improving search results quality."
                }
            },
            "required": ["query","think_content"]
        }
    },
    {
        "type": "function",
        "name": "verify_lean4_code",
        "description": "Verify Lean4 code and check for correctness. If no error is returned, then the code is correct.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Lean4 code to verify. Only support use 'import Mathlib' as header.",
                },
                "think_content": {
                    "type": "string",
                    "description": "It is crucial to provide a complete output of your reasoning process before the tool_call, including your thought process for why you are searching for this theorem and what you hope to find. This is essential for improving results quality."
                }
            },
            "required": ["code", "think_content"]
        }
    }
]

# Example queries list - you can add more queries to process
queries = [
    {"query": "Finish the proof in Lean4 code:\nimport Mathlib\n\n--Prove that a finite p-group G is simple if and only if ∣G∣ = p.\nopen Subgroup\nexample  {G : Type*} [Group G] [Fintype G] {p : ℕ} [hp : Fact (Nat.Prime p)](h : IsPGroup p G) : IsSimpleGroup G ↔ Fintype.card G = p := sorry. You must use the tool to verify your code before responding!"}
]

# Process each query
for item in queries:
    input_messages = [{"role": "user", "content": item["query"]}]
    print("\n🌟--- Processing Query ---🌟")
    print(f"🔍 **User Query:** {item['query']}")
    
    # Call the Responses API with tools enabled and allow parallel tool calls
    response = client.responses.create(
        model="o3",
        input=[
            {"role": "system", "content": "Before answering questions, you should utilize search to find appropriate theorems. You must use the tool to verify your code before responding!"
            },
            {"role": "user", "content": item["query"]}
        ],
        tools=tools,
        parallel_tool_calls=True
    )
    
    # Determine if a tool call is needed and process accordingly
    # while True:
    #     print(response.output)
    #     for block in response.output:
    #         print(block.type)
    #         if block.type =="reasoning":
    #             reasonning = block
    #             reasonning = response.output[0]
    #             if reasonning.summary:
    #                 print("Reasonning content:")
    #                 for i in reasonning.summary:
    #                     print(i)
    #         elif block.type == "function_call":
    #             tool_call = block
    #             tool_name = tool_call.name
    #             print(f"\n🔧 **Model triggered a tool call:** {tool_name}")
                
    #             if tool_name == "leansearch":
    #                 print("🔍 **Invoking leansearch tool...**")
    #                 # Parse arguments and call function
    #                 function_args = json.loads(tool_call.arguments)
    #                 print(function_args)
    #                 # Print the model's reasoning process
    #                 if 'think_content' in function_args:
    #                     print(f"\n💭 **Model Reasoning Process:**\n{function_args.get('think_content')}")
                    
    #                 result = leansearch(function_args.get("query"))
    #                 print("✅ **leansearch tool invoked successfully.**")
    #                 print(f"结果: {result}")
    #             elif tool_name == "verify_lean4_code":
    #                 print("🔍 **Invoking verify_lean4_code tool...**")
    #                 # Parse arguments and call function
    #                 function_args = json.loads(tool_call.arguments)
                    
    #                 # Print the model's reasoning process if provided
    #                 if 'think_content' in function_args:
    #                     print(f"\n💭 **Model Reasoning Process:**\n{function_args.get('think_content')}")
                    
    #                 result = verify_lean4_code(function_args.get("code"))
    #                 print("✅ **verify_lean4_code tool invoked successfully.**")
    #                 print(f"结果: {json.dumps(result)}")
    #             else:
    #                 print(f"🔍 **Unknown tool called: {tool_name}**")
    #                 result = json.dumps({"error": "Unknown tool called"})
                
    #             # Append the tool call and its output back into the conversation
    #             input_messages.extend(response.output)
                
    #             input_messages.append({
    #                 "type": "function_call_output",
    #                 "call_id": tool_call.call_id,
    #                 "output": str(result)
    #             })
    #             # print(input_messages)
    #             # Get the final answer incorporating the tool's result
    #             response = client.responses.create(
    #                 model="o4-mini",
    #                 input=input_messages,
    #                 tools=tools
    #             )
    #             continue

    #         else:
    #             print("\n💡 **Final Answer:**")
    #             print(block.content[0].text)
    #             break
            
    #     else:
    #         continue
    while True:
        print(response.output)
        for block in response.output:
            print(block.type)
            
            if block.type == "reasoning":
                if hasattr(block, "summary") and block.summary:
                    print("Reasoning content:")
                    for i in block.summary:
                        print(i)
            
            elif block.type == "function_call":
                tool_call = block
                tool_name = tool_call.name
                print(f"\n🔧 **Model triggered a tool call:** {tool_name}")

                if tool_name == "leansearch":
                    print("🔍 **Invoking leansearch tool...**")
                    # Parse arguments and call function
                    function_args = json.loads(tool_call.arguments)
                    print(function_args)
                    # Print the model's reasoning process
                    if 'think_content' in function_args:
                        print(f"\n💭 **Model Reasoning Process:**\n{function_args.get('think_content')}")
                    
                    result = leansearch(function_args.get("query"))
                    print("✅ **leansearch tool invoked successfully.**")
                    print(f"结果: {result}")
                elif tool_name == "verify_lean4_code":
                    print("🔍 **Invoking verify_lean4_code tool...**")
                    # Parse arguments and call function
                    function_args = json.loads(tool_call.arguments)
                    
                    # Print the model's reasoning process if provided
                    if 'think_content' in function_args:
                        print(f"\n💭 **Model Reasoning Process:**\n{function_args.get('think_content')}")
                    
                    result = verify_lean4_code(function_args.get("code"))
                    print("✅ **verify_lean4_code tool invoked successfully.**")
                    print(f"结果: {json.dumps(result)}")
                else:
                    print(f"🔍 **Unknown tool called: {tool_name}**")
                    result = json.dumps({"error": "Unknown tool called"})
                

                # 更新 input_messages 并获取新响应
                input_messages.extend(response.output)
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result)
                })
                
                response = client.responses.create(
                    model="o3",
                    input=input_messages,
                    tools=tools
                )
                continue  # 退出 for 循环，重新处理新 response
            
            else:
                print("\n💡 **Final Answer:**")
                print(block.content[0].text)
                break  # 退出 for 循环
        
        else:
            continue  # 如果 for 循环未 break，继续 while 循环
        
        break  # 退出 while 循环
