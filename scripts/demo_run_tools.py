"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the tools will be operation the serena repo itself.
"""

import json
from pprint import pprint

from serena.agent import SerenaAgent
from serena.constants import REPO_ROOT
from serena.tools import FindReferencingSymbolsTool, GetSymbolsOverviewTool, JetBrainsFindSymbolTool

if __name__ == "__main__":
    agent = SerenaAgent(project=REPO_ROOT)

    # apply a tool
    find_symbol_tool = agent.get_tool(JetBrainsFindSymbolTool)
    find_refs_tool = agent.get_tool(FindReferencingSymbolsTool)
    overview_tool = agent.get_tool(GetSymbolsOverviewTool)

    result = agent.execute_task(
        lambda: find_symbol_tool.apply("displayBasicStats"),
    )
    pprint(json.loads(result))
    # input("Press Enter to continue...")
