import os
import sys
import pathlib

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
