"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.leansearch_tool import LeanSearchTool
from agent_r1.tool.tools.leanverify_tool import LeanVerifyTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'LeanSearchTool',
    'LeanVerifyTool',
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'lean_search':
        return [LeanSearchTool()]
    elif env == 'lean_verify':
        return [LeanVerifyTool()]
    elif env == 'lean_full':
        return [LeanSearchTool(), LeanVerifyTool()]   
    else:
        return []
