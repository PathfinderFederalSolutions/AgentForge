# Ensure this test is optional and offline-safe
import importlib.util
import pytest

spec = importlib.util.find_spec("anthropic")
anthropic = pytest.importorskip("anthropic", reason="anthropic not installed")

def test_anthropic_client_constructs():
    # Construct client without any API calls
    client = anthropic.Anthropic(api_key="test_key")
    assert client is not None