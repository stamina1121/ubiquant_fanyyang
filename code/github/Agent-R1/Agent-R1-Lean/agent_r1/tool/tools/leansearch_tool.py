"""
LeanSearch tool implementation for searching Lean theorems
"""

import requests
import json
import pprint
from typing import Dict, List, Any, Optional

from agent_r1.tool.tool_base import Tool

from .utils import CodeUtil

NUM_RESULTS = 5
INFORMAL = True
DEBUG = False
class LeanSearchTool(Tool):
    """
    Tool for searching Lean theorems in mathlib4
    """
    def __init__(self):
        """
        Initialize the Lean search tool
        """
        name = "leansearch"
        description = "Search for theorems and definitions in Mathlib4 based on a query. Query should articulate the theorem in a sentence. Do not use query like 'dimension, vector space,linear transformation'"
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Queries like 'For integers \\(a, b \\neq 0\\), there exist \\(x, y \\in \\mathbb{Z}\\) such that \\(ax + by = \\gcd(a, b)\\).'",
                },
            },
            "required": ["query"],
        }
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute Lean theorem search
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "num_results": optional int to limit number of results
            
        Returns:
            Search results as JSON string
        """
        query = args.get("query")
        if query is None:
            return str({"error": "Missing required parameter: query)"})
            
        num_results = NUM_RESULTS
        
        # Handle both string and list formats for query
        if isinstance(query, str):
            query = [query]  # Convert single string to list
        
        try:
            results = self._search_mathlib4(query, num_results)
            formatted = self._format_results(results)
            return formatted
        except Exception as e:
            return str({"error": str(e)})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute batch search queries
        
        Args:
            args_list: List of query parameter dictionaries
            
        Returns:
            List of search results
        """
        # Collect all queries together
        queries = []
        for args in args_list:
            query = args.get("query","")
            if isinstance(query, str):
                queries.append(query)
            else:
                queries.extend(query)
        
        # Execute batch query
        if DEBUG:
            print("[DEBUG]Search_queries:")
            for query in queries:
                print(query,end="    ")
        results = self._search_mathlib4(queries, num_results=NUM_RESULTS)
        
        # Format and return results
        if isinstance(results, list):
            formatted_results = [self._format_results([result_group]) for result_group in results]
            # if DEBUG:
            #     for f, r in zip(formatted_results, results):
            #         print("[DEBUG] RESULTS", r, '[DEBUG] FORMATTEDRESULTS', f)
            return formatted_results
        else:
            # If error occurred, return the same error for all queries
            error_message = self._format_results(results)
            return [error_message] * len(args_list)
    
    def _search_mathlib4(self, query: List[str], num_results = NUM_RESULTS) -> Dict[str, Any]:
        """
        Search for theorems in mathlib4
        
        Args:
            query: List of search keywords
            num_results: Number of results to return
            
        Returns:
            List of lists of theorem dictionaries, or Dict with 'error' key
        """
        # url = 'https://console.siflow.cn/siflow/draco/ai4math/tyxu/leansearch-api-v4-16/search'
        url = 'http://leansearch-api-v4-16.t-ai4math-tyxu.svc.cluster.local/search'
        params = {
            'query': query,
            'num_results': num_results
        }
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get theorems, status code: {response.status_code}, response: {response.text}"}

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Return positive reward if results are returned successfully
        if "query" in args:
            if 'error' in result:
                return 0.0
            if result == '[]':
                return 0.0
            return 0.5
        else:
            return 0.0

    def _format_results(self, results: Any) -> str:
        """
        Format search results as a JSON list
        
        Args:
            results: Search result from the API - either a list of lists of theorem dicts,
                    or a dict with an 'error' key
            
        Returns:
            JSON string containing theorem results
        """
        if isinstance(results, dict) and "error" in results:
            return json.dumps({"error": results['error']})
            
        formatted_results = []
        # print("[TO BE FORMAT]", results)
        # Process the nested list structure
        if isinstance(results, list):
            for result_group in results:
                for item in result_group:
                    # Extract data from the result
                    # print('[ITEM INFO]',item)
                    docstring = item['result']['docstring']
                    name = '.'.join(item['result']['name'])
                    signature = item['result']['signature']
                    value = item['result']['value']
                    informal_description = item['result'].get('informal_description', '')
                    kind = item['result']['kind']
                    
                    # Create formatted theorem text
                    # theorem_text = f"theorem {name} {signature} {value}"
                    if kind in ['theorem', 'lemma', 'instance', 'structure', 'class']:
                        theorem_text = f"{kind} {name} {signature}"
                    elif kind == 'definition' and "Lean.ParserDescr" not in signature:
                        theorem_text = f"def {name} {signature} {value}"
                    else:
                        continue
                    theorem_text = CodeUtil.remove_comment(theorem_text)
                    # print('[THEOREM TEXT]', theorem_text)
                    if docstring is not None and len(docstring) > 128:
                        docstring = docstring[:128]
                    if informal_description is not None and len(informal_description) > 128:
                        informal_description = informal_description[:128]
                    # Add to results list
                    formatted_results.append({
                        "name": name,
                        "statement": theorem_text,
                        "docstring": docstring,
                        "informal_description": informal_description if INFORMAL else ""
                    })
        else:
            return json.dumps({"error": "Search returned unexpected format. No results found."})
        # return ""      
        return json.dumps(formatted_results,indent=2,ensure_ascii=False)

if __name__=='__main__':
    from pprint import pp
    tool = LeanSearchTool()
    arg_list = [{'query': 'Group product in Lean 4'}, {'query': 'Bezout lemma in Lean'}]
    result = tool.batch_execute(arg_list)
    pp(result[0])