import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests  # Should import without error
from forge_types import Task, AgentContract, MemoryScope
from router import DynamicRouter
import orchestrator
import agents

def test_stdlib_types():
    assert Task
    assert AgentContract
    assert MemoryScope

def test_custom_types():
    t = Task(id="1", description="desc", memory_scopes=[], budget=10, tools=[], priority=1)
    assert t.description == "desc"

def test_requests_installed():
    resp = requests.models.Response()
    assert isinstance(resp, requests.models.Response)

def test_no_circular_imports():
    router = DynamicRouter()
    assert router
    from agents import Agent
    agent = Agent(AgentContract(name="test", capabilities=[], memory_scopes=[], tools=[], budget=1000))
    assert agent.router  # Uses DynamicRouter without circle

def test_router_imports():
    from router import DynamicRouter
    r = DynamicRouter()
    assert r