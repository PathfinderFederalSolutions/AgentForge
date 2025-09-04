# tests/test_imports.py (expand)
import pytest
import forge_types  # Custom types
import router  # New
import orchestrator
import agents
from router import DynamicRouter

def test_no_circular_imports():
    router = DynamicRouter()
    assert router
    from agents import Agent
    agent = Agent(forge_types.AgentContract(name="test", capabilities=[], memory_scopes=[], tools=[], budget=1000))
    assert agent.router  # Uses DynamicRouter without circle
    