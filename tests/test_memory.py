import pytest
from agentforge.memory import EvoMemory  # Assume import path

@pytest.fixture
def memory():
    return EvoMemory()  # Test init

def test_semantic_search_min_score(memory):
    memory.add("test_key", "vector data", scopes=["short_term"])
    results = memory.semantic_search("vector data", min_score=0.6, scopes=["short_term"])
    assert len(results) > 0
    results_low = memory.semantic_search("vector data", min_score=0.99, scopes=["short_term"])
    assert len(results_low) == 0  # With noise and hashing, 0.99 should filter out

def test_semantic_search_scopes(memory):
    memory.add("key1", "data one", scopes=["short_term"])
    memory.add("key2", "data two", scopes=["long_term"])
    results_short = memory.semantic_search("data", min_score=0.3, scopes=["short_term"])
    assert any("data one" in str(r) for r in results_short)
    results_long = memory.semantic_search("data", min_score=0.3, scopes=["long_term"])
    assert any("data two" in str(r) for r in results_long)