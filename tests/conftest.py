import os
import sys
import pathlib
import pytest

# Enable Python's faulthandler for stack traces on timeout
try:
    import faulthandler
    faulthandler.enable()
except Exception:
    pass

# Asyncio tweaks for better debug output on hangs
os.environ.setdefault("PYTHONASYNCIODEBUG", "1")

# Default: treat external services as unavailable unless explicitly enabled
ENABLE_INTEGRATION = os.getenv("ENABLE_INTEGRATION", "0") == "1"

# Ensure project root (parent of tests directory) is on sys.path for package imports like 'swarm'
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Disable auto-loading external pytest plugins to avoid unexpected hangs in CI
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

# Ensure offline-friendly defaults for libraries that may phone home
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Dummy API keys to satisfy libs that require them merely for construction
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", "test_key")

def pytest_collection_modifyitems(config, items):
    if ENABLE_INTEGRATION:
        return
    skip_integration = pytest.mark.skip(reason="Integration disabled (set ENABLE_INTEGRATION=1 to enable)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

@pytest.fixture(autouse=True)
def _tighten_timeouts(monkeypatch):
    # Keep network calls from stalling indefinitely
    for env_key, default in [
        ("NATS_REQUEST_TIMEOUT_S", "3"),
        ("HTTP_CLIENT_TIMEOUT_S", "5"),
        ("PROM_QUERY_TIMEOUT_S", "5"),
    ]:
        os.environ.setdefault(env_key, default)

@pytest.fixture(scope="session", autouse=True)
def _isolate_env():
    # Ensure tests don't leak state across the repo
    os.environ.setdefault("EDGE_MODE", "false")
    os.environ.setdefault("AF_JOBS_SUBJECT", "swarm.jobs.test")
    os.environ.setdefault("AF_RESULTS_SUBJECT", "swarm.results.test")
    # If integration disabled, force stubs/mocks downstream
    if not ENABLE_INTEGRATION:
        os.environ.setdefault("USE_NATS_STUBS", "1")

# Compatibility shim for nats-py enum names across versions used by tests
try:  # pragma: no cover - simple aliasing
    from nats.js.api import RetentionPolicy, StorageType
    if not hasattr(RetentionPolicy, "WorkQueue") and hasattr(RetentionPolicy, "WORK_QUEUE"):
        setattr(RetentionPolicy, "WorkQueue", RetentionPolicy.WORK_QUEUE)
    if not hasattr(RetentionPolicy, "Limits") and hasattr(RetentionPolicy, "LIMITS"):
        setattr(RetentionPolicy, "Limits", RetentionPolicy.LIMITS)
    if not hasattr(StorageType, "File") and hasattr(StorageType, "FILE"):
        setattr(StorageType, "File", StorageType.FILE)
    if not hasattr(StorageType, "Memory") and hasattr(StorageType, "MEMORY"):
        setattr(StorageType, "Memory", StorageType.MEMORY)
except Exception:
    pass
