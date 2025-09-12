# Package marker for capability registry
# Ensure fusion capabilities register themselves at import time
try:
    from . import fusion_caps  # noqa: F401
except Exception:
    # Safe no-op if optional deps not installed during minimal test runs
    pass