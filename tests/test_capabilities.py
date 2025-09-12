import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swarm.capabilities.registry import registry
from swarm.capabilities import fusion_caps  # noqa: F401

def test_registry_and_caps():
    cap = registry.get("bayesian_fusion")
    assert cap is not None
    out = cap.func(eo=[1,2,3,4], ir=[2,3,4,5])
    assert "mu" in out and "var" in out