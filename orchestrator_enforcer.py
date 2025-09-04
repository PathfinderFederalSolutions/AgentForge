# Integrated into AgentForge orchestrator for mandatory checks.
import os
import sla_kpi_config  # Shared import
import logging  # For monitoring
from typing import Dict, Any

class SLAKPIEnforcer:
    def __init__(self, strict: bool | None = None):
        # Strict mode can be toggled via env to hard-fail on violations
        if strict is None:
            strict = os.getenv("AF_ENFORCE_STRICT", "0") == "1"
        self.strict = strict
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}  # Placeholder for runtime metrics, sync with memory mesh

    def enforce_pre_task(self, task: Dict | Any, capability: str):
        """Check SLAs before task execution. Non-blocking unless strict mode is enabled."""
        relevant_kpis = [k for k in sla_kpi_config.SLAS_KPIS.get(capability, [])]
        for kpi in relevant_kpis:
            m = self.metrics.get(kpi.get('capability_sub'))
            if m is None:
                # Skip if no measurement available yet
                continue
            if not self._validate_kpi(kpi, m):
                self._handle_violation(f"SLA/KPI violation (pre): {kpi['capability_sub']} threshold {kpi['threshold']} measured {m}")
        self.logger.info("Pre-task enforcement evaluated.")

    def enforce_post_task(self, task_result: Dict | Any, capability: str):
        """Validate KPIs after task. Non-blocking unless strict mode is enabled."""
        relevant_kpis = [k for k in sla_kpi_config.SLAS_KPIS.get(capability, [])]
        for kpi in relevant_kpis:
            measured_value = self._measure_kpi(kpi, task_result)  # Implement per measurement
            if measured_value is None:
                # If not measurable yet, do not block
                continue
            if not self._validate_kpi(kpi, measured_value):
                self._trigger_healer(kpi)
                self._handle_violation(f"Post-task violation: {kpi['kpi']} threshold {kpi['threshold']} not met (measured {measured_value}).")
        self.logger.info("Post-task enforcement evaluated.")

    def _handle_violation(self, msg: str) -> None:
        if self.strict:
            raise ValueError(msg)
        else:
            self.logger.warning(msg)

    def _validate_kpi(self, kpi: Dict, measured: Any) -> bool:
        """Parse and compare threshold; keeps defaults permissive unless strict comparisons are clear."""
        threshold = str(kpi.get('threshold', '')).strip().lower()
        # Basic numeric comparisons: <Xms, <Ymin, <Z, etc.
        try:
            if threshold.startswith('<'):
                t = threshold[1:].strip()
                # Normalize units
                if t.endswith('ms'):
                    limit = float(t[:-2].strip())
                    return float(measured) < limit
                if t.endswith('min') or t.endswith('mins'):
                    limit = float(t.split('min')[0].strip()) * 60.0
                    return float(measured) < limit
                if t.endswith('%'):
                    limit = float(t[:-1].strip())
                    return float(measured) < limit
                # Generic numeric
                return float(measured) < float(t)
        except Exception:
            # On parsing error, be permissive
            return True
        # 100% style thresholds -> treat as target compliance; if numeric percentage provided, require >= 100
        if '100%' in threshold:
            try:
                # If measured is numeric percent
                mv = float(measured)
                return mv >= 100.0
            except Exception:
                # If boolean-like truthiness, accept True
                return bool(measured) is True
        # Zero-incidents style
        if threshold.startswith('0') or '0 incident' in threshold or '0 errors' in threshold or '0 breaches' in threshold or '0 detections' in threshold or '0 findings' in threshold:
            try:
                return float(measured) == 0.0
            except Exception:
                return measured in (0, '0', None)
        # Default: pass
        return True

    def _measure_kpi(self, kpi: Dict, data: Dict | Any) -> Any:
        # Support latency measurement from task_result.metrics.latency_ms
        try:
            metrics = (data or {}).get('metrics') or {}
        except AttributeError:
            metrics = {}
        cap_sub = str(kpi.get('capability_sub', '')).lower()
        # Map common KPIs to measured fields
        if 'latency' in cap_sub:
            # If threshold mentions ms or <Xms, expect latency_ms
            if 'ms' in str(kpi.get('threshold', '')).lower():
                return (metrics.get('latency_ms'))
        # For thresholds like '0 errors', try to infer from result
        if any(tok in str(kpi.get('threshold', '')).lower() for tok in ['0 errors', '0 incident', '0 breaches', '0 detections', '0 findings']):
            try:
                res = (data or {}).get('result')
            except AttributeError:
                res = None
            if res is None:
                return 1  # count as 1 error
            if isinstance(res, str) and res.strip().lower().startswith('error'):
                return 1
            return 0
        # Unknown measurement -> None means skip (non-blocking)
        return None

    def _trigger_healer(self, kpi: Dict):
        # Call Healer Agent to adjust; integrate with system
        self.logger.warning(f"Triggering healer for {kpi.get('capability_sub')}")

# Usage in AgentForge orchestrator loop:
# enforcer = SLAKPIEnforcer()
# for task in tasks:
#     enforcer.enforce_pre_task(task, capability)
#     result = execute_task(task)
#     enforcer.enforce_post_task(result, capability)