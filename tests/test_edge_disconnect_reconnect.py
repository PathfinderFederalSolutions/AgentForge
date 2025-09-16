import asyncio
import os
import tempfile
from pathlib import Path
import sys

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class DummyNATS:
    def __init__(self):
        self.published = []

    async def publish(self, subj, data, headers=None):  # bare publish
        self.published.append((subj, data, headers or {}))

    def jetstream(self):
        class _JS:
            def __init__(self, outer):
                self.outer = outer
            async def publish(self, subj, data, headers=None):
                self.outer.published.append((subj, data, headers or {}))
        return _JS(self)


@pytest.fixture(autouse=True)
async def patch_nats(monkeypatch):
    # Patch nats module used by syncdaemon
    from types import SimpleNamespace
    dummy = DummyNATS()

    async def _connect(**kwargs):
        return dummy

    mod = SimpleNamespace(connect=_connect)
    monkeypatch.setitem(sys.modules, 'nats', mod)
    yield


def _extract_metric_value(text: str, metric: str, labels: dict) -> float | None:
    # Build the exact metric label string, e.g., replay_queue_depth{site="edge-test"}
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    target = f"{metric}{{{label_str}}}"
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        if line.startswith(target):
            parts = line.split()
            if parts:
                try:
                    return float(parts[-1])
                except Exception:
                    return None
    return None


async def test_replay_queue_depth_depletes(monkeypatch):
    # Use a temp directory as queue
    with tempfile.TemporaryDirectory() as tmpd:
        q = Path(tmpd)
        logs = q / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        # Create sample files
        for i in range(5):
            (q / f"msg_{i}.json").write_text(f"{{\"i\": {i}}}")

        os.environ['SYNC_QUEUE_DIR'] = str(q)
        os.environ['SYNC_LEDGER_DIR'] = str(logs)
        os.environ['SYNC_SUBJECT'] = 'swarm.results.edge'
        os.environ['SITE'] = 'edge-test'

        # Import app and explicit lifecycle handlers after env set
        from services.syncdaemon.app.main import app, on_startup, on_shutdown
        import httpx

        # Explicitly start the app lifecycle to run background worker
        await on_startup()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                # Poll until queue drains
                drained = False
                for _ in range(100):
                    r = await client.get('/metrics')
                    assert r.status_code == 200
                    txt = r.text
                    # Parse numeric value to avoid format assumptions (0 vs 0.0)
                    val = _extract_metric_value(txt, 'replay_queue_depth', {'site': 'edge-test'})
                    if val is not None and val == 0.0:
                        drained = True
                        break
                    await asyncio.sleep(0.1)
                assert drained, "Queue did not drain to zero in time"
                # Ensure ledger written
                # Give a brief settle time for writes
                await asyncio.sleep(0.05)
                log_files = list((logs).glob('*.jsonl'))
                assert log_files, "ledger jsonl should exist"
        finally:
            # Ensure graceful shutdown to avoid pending task warnings
            await on_shutdown()
