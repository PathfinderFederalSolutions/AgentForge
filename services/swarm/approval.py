"""
Lightweight approval/HITL gating integrated with orchestrator flow.
Defaults to no-op unless enabled via env variables.

Env flags:
- AF_APPROVAL_ENABLE: when "1", evaluate approval gates per subtask
- AF_APPROVAL_REQUIRE_STRICT: when "1", block on non-approved tasks by
  raising ValueError
- AF_HITL_AUTOAPPROVE: when "1" (default), auto-approve escalations for
  CI/tests to avoid hangs

Persist adjudications by publishing to the Memory Mesh via orchestrator
(caller provides publisher).
"""
from __future__ import annotations
from typing import Dict, Any
import os

IMPACT_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}
CONF_LEVELS = {"low": 0, "medium": 1, "high": 2}


class ApprovalManager:
    def __init__(
        self,
        enabled: bool | None = None,
        strict: bool | None = None,
        autoapprove: bool | None = None,
    ):
        if enabled is None:
            enabled = os.getenv("AF_APPROVAL_ENABLE", "0") == "1"
        if strict is None:
            strict = os.getenv("AF_APPROVAL_REQUIRE_STRICT", "0") == "1"
        if autoapprove is None:
            autoapprove = os.getenv("AF_HITL_AUTOAPPROVE", "1") == "1"
        self.enabled = enabled
        self.strict = strict
        self.autoapprove = autoapprove

    def _infer_impact(self, task: Dict[str, Any]) -> str:
        # Prefer explicit metadata; fall back to keywords in description
        md = (task or {}).get("metadata", {}) or {}
        imp = str(md.get("impact", "")).lower()
        if imp in IMPACT_LEVELS:
            return imp
        desc = (task or {}).get("description", "").lower()
        if any(
            w in desc for w in ["prod", "payment", "security", "pii", "deploy", "infra"]
        ):
            return "high"
        if any(w in desc for w in ["schema", "api", "auth", "encryption"]):
            return "medium"
        return "low"

    def _infer_confidence(self, result: Any) -> str:
        # Simple heuristic: longer non-error strings => higher confidence
        try:
            if not isinstance(result, str):
                return "medium"
            r = result.strip()
            if not r or r.lower().startswith("error"):
                return "low"
            if len(r) > 800:
                return "high"
            if len(r) > 200:
                return "medium"
            return "low"
        except Exception:
            return "medium"

    def check_and_gate(
        self,
        task: Dict[str, Any],
        result: Any,
        capability: str,
        publisher: callable | None = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with fields:
        - approved: bool
        - escalated: bool
        - reason: str
        """
        if not self.enabled:
            return {"approved": True, "escalated": False, "reason": "disabled"}
        impact = self._infer_impact(task)
        confidence = self._infer_confidence(result)
        risk_score = IMPACT_LEVELS.get(impact, 0) - CONF_LEVELS.get(confidence, 1)
        needs_hitl = risk_score >= 2  # escalate when high impact and low confidence
        if needs_hitl:
            if publisher:
                publisher(
                    "approval.requested",
                    {
                        "task_id": task.get("id"),
                        "capability": capability,
                        "impact": impact,
                        "confidence": confidence,
                        "risk_score": risk_score,
                    },
                )
            if self.autoapprove:
                if publisher:
                    publisher(
                        "approval.auto",
                        {
                            "task_id": task.get("id"),
                            "approved": True,
                            "reason": "autoapprove",
                        },
                    )
                return {"approved": True, "escalated": True, "reason": "autoapproved"}
            if self.strict:
                raise ValueError(
                    "Approval required and strict: task "
                    f"{task.get('id')} capability {capability}"
                )
            return {"approved": False, "escalated": True, "reason": "needs_hitl"}
        return {"approved": True, "escalated": False, "reason": "low_risk"}
