from __future__ import annotations
import os
from typing import Optional, Dict, Any

_DEFAULT_ATTRIBUTES: Dict[str, Any] = {}

# Public symbols list
__all__ = [
    'set_default_observable_context',
    'init_tracing',
    'tag_span',
    'set_dag_hash'
]

def set_default_observable_context(mission_id: Optional[str] = None, task_id: Optional[str] = None):
    if mission_id:
        _DEFAULT_ATTRIBUTES["mission.id"] = mission_id
    if task_id:
        _DEFAULT_ATTRIBUTES["task.id"] = task_id

def init_tracing(service_name: str = "agentforge", service_version: Optional[str] = None, environment: Optional[str] = None):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    _ = (TracerProvider, BatchSpanProcessor, OTLPSpanExporter)

    attrs = {
        "service.name": service_name or "agentforge",
    }
    attrs.update(_DEFAULT_ATTRIBUTES)
    if service_version:
        attrs["service.version"] = service_version
    if environment:
        attrs["deployment.environment"] = environment

    resource = Resource.create(attrs)
    tracer_provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()

    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(service_name)
    if not tracer:
        raise RuntimeError("tracer not initialized")
    return tracer

try:
    from opentelemetry import trace as _trace_mod
except Exception:
    _trace_mod = None

def tag_span(mission_id: Optional[str] = None, task_id: Optional[str] = None):
    if _trace_mod is None:
        return
    span = _trace_mod.get_current_span()
    if not span or not hasattr(span, "set_attribute"):
        return
    if mission_id:
        span.set_attribute("mission.id", mission_id)
    if task_id:
        span.set_attribute("task.id", task_id)

try:
    def set_dag_hash(dag_hash: str):  # type: ignore
        span = _trace_mod.get_current_span()  # type: ignore
        if span and hasattr(span, 'set_attribute'):
            span.set_attribute('dag.hash', dag_hash)  # type: ignore
except Exception:  # pragma: no cover
    def set_dag_hash(dag_hash: str):  # type: ignore
        pass