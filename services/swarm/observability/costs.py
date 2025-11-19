from __future__ import annotations
import time
import contextvars
from prometheus_client import Counter, Summary

# Context variables for dynamic label enrichment
_mission_ctx = contextvars.ContextVar("mission_id", default="unknown")
_task_ctx = contextvars.ContextVar("task_id", default="unknown")

def set_observability_context(mission_id: str | None = None, task_id: str | None = None):
    if mission_id:
        _mission_ctx.set(mission_id)
    if task_id:
        _task_ctx.set(task_id)

LLM_TOKENS_TOTAL = Counter("llm_tokens_total", "Total LLM tokens", ["provider", "model", "kind", "mission_id", "task_id"])
LLM_COST_USD_TOTAL = Counter("llm_cost_usd_total", "Total LLM cost (USD)", ["provider", "model", "mission_id", "task_id"])
LLM_LATENCY_SECONDS = Summary("llm_latency_seconds", "LLM call latency", ["provider", "model", "mission_id", "task_id"])

def wrap_llm_call(provider: str, model: str, fn):
    def _wrapped(*args, **kwargs):
        mission_id = _mission_ctx.get()
        task_id = _task_ctx.get()
        start = time.perf_counter()
        resp = fn(*args, **kwargs)
        dur = time.perf_counter() - start
        prompt_toks = getattr(resp, "prompt_tokens", 0) or 0
        comp_toks = getattr(resp, "completion_tokens", 0) or 0
        cost = getattr(resp, "cost_usd", 0.0) or 0.0
        LLM_LATENCY_SECONDS.labels(provider, model, mission_id, task_id).observe(dur)
        LLM_TOKENS_TOTAL.labels(provider, model, "prompt", mission_id, task_id).inc(prompt_toks)
        LLM_TOKENS_TOTAL.labels(provider, model, "completion", mission_id, task_id).inc(comp_toks)
        LLM_COST_USD_TOTAL.labels(provider, model, mission_id, task_id).inc(cost)
        return resp
    return _wrapped