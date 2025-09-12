import importlib.util
import pytest

spec = importlib.util.find_spec("langchain_huggingface")
langchain_huggingface = pytest.importorskip("langchain_huggingface", reason="optional dependency not installed")

def test_hf_import():
    assert hasattr(langchain_huggingface, "HuggingFaceEndpoint")