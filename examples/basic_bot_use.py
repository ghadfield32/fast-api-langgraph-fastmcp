"""
Basic bot use example

This example shows how to use the bot to answer a question.

location: examples/basic_bot_use.py
"""

import pathlib

# ---  Add parent directory to PYTHONPATH (local-run convenience) ---
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools import all_tools  # exported in __init__.py

agent = LangGraphAgent(tools=all_tools)            # instantiates & compiles
reply = agent.get_response(
    thread_id="local-demo",
    messages=[{"role": "user", "content": "Summarise LangGraph in 1 line"}],
)
print(reply[-1]["content"])
