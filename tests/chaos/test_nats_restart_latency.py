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


@pytest.mark.skipif(os.getenv("SKIP_CHAOS", "1") == "1", reason="Skipping chaos tests in local/CI runs")
@pytest.mark.timeout(10)
def test_nats_restart_latency():
    # Apply the nats_restart chaos and verify the cluster recovers within a bounded time window.
    exp = "chaos/experiments/nats_restart.yaml"
    start = time.time()
    res = _kubectl(["apply", "-f", exp])
    assert res.returncode == 0, res.stderr
    # Wait up to 180s for NATS pod to be Running and ready
    for _ in range(60):
        pods = _kubectl(["-n", NS, "get", "pods", "-l", "app=nats", "-o", "json"]).stdout
        ready = "Running" in pods and "1/1" in pods
        if ready:
            break
        time.sleep(3)
    elapsed = time.time() - start
    # Gate: recovery under 120s
    assert elapsed < 120, f"NATS restart recovery exceeded target: {elapsed:.1f}s"
