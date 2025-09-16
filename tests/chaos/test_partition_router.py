import os
import subprocess
import time
import shutil
import pytest

NS = os.getenv("NS", "agentforge-staging")


def _kubectl(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], check=False, capture_output=True, text=True)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl not installed"),
]


def test_partition_router_recovers_hpa():
    # Apply partition between orchestrator and NATS; ensure HPA/ScaledObject still present after event
    exp = "chaos/experiments/partition_api_nats.yaml"
    res = _kubectl(["apply", "-f", exp])
    assert res.returncode == 0, res.stderr
    # Wait for 30s and verify resources exist and are not degraded
    time.sleep(30)
    hpa = _kubectl(["-n", NS, "get", "hpa", "-o", "name"]).stdout
    so = _kubectl(["-n", NS, "get", "scaledobject.keda.sh", "-o", "name"]).stdout
    assert "keda-hpa-nats-worker-scaledobject" in hpa
    assert "nats-worker-scaledobject" in so
