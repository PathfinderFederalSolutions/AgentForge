# router.py
import os
import time
import logging
from typing import List, Dict, Any, Union, Tuple, Optional
from dotenv import load_dotenv
import anthropic

# Optional OTEL tracing
try:
    from opentelemetry import trace  # type: ignore
    from contextlib import nullcontext  # type: ignore
    _TRACER = trace.get_tracer("agentforge.router")
except Exception:  # pragma: no cover
    trace = None  # type: ignore
    nullcontext = None  # type: ignore
    _TRACER = None  # type: ignore

# Optional lineage recording
try:
    from swarm import lineage  # type: ignore
except Exception:  # pragma: no cover
    lineage = None  # type: ignore

from observability import (
    token_counter,
    latency_hist,
    cost_counter,
    error_counter,
    log_with_id,
)  # Assume absolute; update if relative

load_dotenv()

# Allow Anthropic model control without hardcoding deprecated names.
# Set ANTHROPIC_MODEL to a valid model (e.g., "claude-opus-4-1-20250805" per latest docs).
ANTHROPIC_FALLBACK_MODELS = [
    # Only used if ANTHROPIC_MODEL and payload overrides are not provided.
    # Keep conservative, modern defaults; do not use date-stamped names that 404.
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]


def _getenv(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _anthropic_split_system_and_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:  # noqa: C901
    """
    Anthropic Messages API:
      - top-level `system` string (concatenate all prior system messages)
      - messages: list[{role: "user"|"assistant", content: [{type:"text", text:"..."}]}]
    """
    system_parts: List[str] = []
    norm: List[Dict[str, Any]] = []

    def _to_text_blocks(content: Any) -> List[Dict[str, str]]:
        if isinstance(content, list):
            blocks: List[Dict[str, str]] = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text" and "text" in c:
                    blocks.append({"type": "text", "text": str(c["text"])})
                else:
                    blocks.append({"type": "text", "text": str(c)})
            return blocks
        return [{"type": "text", "text": str(content)}]

    for msg in messages or []:
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        system_parts.append(str(c["text"]))
                    else:
                        system_parts.append(str(c))
            else:
                system_parts.append(str(content))
            continue
        if role not in ("user", "assistant"):
            role = "user"
        norm.append({"role": role, "content": _to_text_blocks(content)})

    system = "\n".join([s for s in system_parts if s]).strip() or None
    return system, norm


def _anthropic_model_candidates(payload: Dict[str, Any]) -> List[str]:
    """
    Model selection priority:
    1) payload["anthropic_model"] or payload["target_model"] if provided
    2) env ANTHROPIC_MODEL
    3) conservative fallbacks
    """
    candidates: List[str] = []
    for key in ("anthropic_model", "target_model"):
        v = str(payload.get(key, "")).strip()
        if v:
            candidates.append(v)
    env_m = os.getenv("ANTHROPIC_MODEL", "").strip()
    if env_m:
        candidates.append(env_m)
    candidates.extend(ANTHROPIC_FALLBACK_MODELS)
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _anthropic_call(client, payload: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    system, a_messages = _anthropic_split_system_and_messages(payload.get("messages") or [])
    # Prefer env override to reduce truncation; fallback to payload or default.
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", payload.get("max_tokens") or payload.get("maxTokens") or 2048))
    temperature = float(payload.get("temperature", 0.2))
    models = _anthropic_model_candidates(payload)

    last_exc: Optional[Exception] = None
    for model_name in models:
        if not model_name:
            continue
        try:
            resp = client.messages.create(
                model=model_name,
                system=system,
                messages=a_messages if a_messages else [
                    {"role": "user", "content": [{"type": "text", "text": str(payload.get("prompt") or "")}]}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text_out = ""
            try:
                for blk in getattr(resp, "content", []) or []:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text_out += blk.get("text", "")
                    elif hasattr(blk, "type") and getattr(blk, "type") == "text":
                        text_out += getattr(blk, "text", "")
            except Exception:
                text_out = getattr(resp, "content", "") or str(resp)
            # Keep meta available for callers that need it, but primary output is text_out
            return {"text": text_out, "raw": resp, "model_used": model_name}
        except Exception as e:
            logging.info("Anthropic model failed (%s), trying next: %s", model_name, e)
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Anthropic: no models available to try")


def _anthropic_payload_from_locals(_locals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a payload dict from whatever the caller has in scope.
    Avoids NameError when 'payload' isn't defined.
    """
    for key in ("payload", "body", "request", "params", "data"):
        val = _locals.get(key)
        if isinstance(val, dict):
            return val
    pd: Dict[str, Any] = {}
    if isinstance(_locals.get("messages"), list):
        pd["messages"] = _locals["messages"]
    if "prompt" in _locals and "messages" not in pd:
        pd["prompt"] = _locals["prompt"]
    return pd


class ChatGrok:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        # ... (existing implementation)


class MockLLM:
    def __init__(self, model: str = "mock-model"):
        self.model = model

    def invoke(self, messages):
        return type(
            'Response',
            (),
            {
                'content': f"[MOCK:{self.model}] "
                + (messages[-1].get('content') if messages else ''),
            },
        )


class DynamicRouter:
    def __init__(self):  # noqa: C901
        self.llms: dict[str, dict] = {}
        self.benchmarks: dict[str, dict] = {}

        force_mock = os.getenv("AF_FORCE_MOCK", "0") == "1"

        if not force_mock:
            # Build provider clients only when keys exist; import lazily to avoid heavy deps on import time
            oai_key = _getenv("OPENAI_API_KEY", "OPENAI_KEY")
            if oai_key:
                try:
                    from langchain_openai import ChatOpenAI
                    self.llms["gpt-4o-mini"] = {
                        "client": ChatOpenAI(model="gpt-4o-mini", api_key=oai_key),
                        "healthy": True,
                        "quota": 100000,
                        "cost_per_token": 0.000002,
                    }
                    self.benchmarks["gpt-4o-mini"] = {
                        "cost": 0.002,
                        "speed": 0.85,
                        "accuracy": 0.90,
                    }
                except Exception:
                    pass

            anth_key = _getenv("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")
            if anth_key:
                try:
                    self.llms["claude-3-5"] = {
                        "client": anthropic.Anthropic(api_key=anth_key),
                        "healthy": True,
                        "quota": 50000,
                        "cost_per_token": 0.000015,
                    }
                    self.benchmarks["claude-3-5"] = {
                        "cost": 0.015,
                        "speed": 0.65,
                        "accuracy": 0.93,
                    }
                except Exception:
                    pass

            google_key = _getenv("GOOGLE_API_KEY", "GOOGLE_KEY")
            if google_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llms["gemini-1-5"] = {
                        "client": ChatGoogleGenerativeAI(
                            model="gemini-1.5-pro", api_key=google_key
                        ),
                        "healthy": True,
                        "quota": 30000,
                        "cost_per_token": 0.000018,
                    }
                    self.benchmarks["gemini-1-5"] = {
                        "cost": 0.018,
                        "speed": 0.75,
                        "accuracy": 0.91,
                    }
                except Exception:
                    pass

            mistral_key = _getenv("MISTRAL_API_KEY", "MISTRAL_KEY")
            if mistral_key:
                try:
                    from langchain_mistralai import ChatMistralAI
                    self.llms["mistral-large"] = {
                        "client": ChatMistralAI(
                            model="mistral-large-latest", api_key=mistral_key
                        ),
                        "healthy": True,
                        "quota": 20000,
                        "cost_per_token": 0.00001,
                    }
                    self.benchmarks["mistral-large"] = {
                        "cost": 0.01,
                        "speed": 0.8,
                        "accuracy": 0.89,
                    }
                except Exception:
                    pass

            cohere_key = _getenv("COHERE_API_KEY", "CO_API_KEY")
            if cohere_key:
                try:
                    from langchain_cohere import ChatCohere
                    self.llms["cohere-command"] = {
                        "client": ChatCohere(
                            model="command-r-plus", api_key=cohere_key
                        ),
                        "healthy": True,
                        "quota": 40000,
                        "cost_per_token": 0.000012,
                    }
                    self.benchmarks["cohere-command"] = {
                        "cost": 0.012,
                        "speed": 0.7,
                        "accuracy": 0.90,
                    }
                except Exception:
                    pass

            xai_key = _getenv("XAI_API_KEY", "XAI_KEY")
            if xai_key:
                try:
                    # Local lightweight wrapper defined below
                    self.llms["grok-4"] = {
                        "client": ChatGrok(model="grok-4", api_key=xai_key),
                        "healthy": True,
                        "quota": 50000,
                        "cost_per_token": 0.00002,
                    }
                    self.benchmarks["grok-4"] = {
                        "cost": 0.02,
                        "speed": 0.6,
                        "accuracy": 0.94,
                    }
                except Exception:
                    pass

        if force_mock or not self.llms:
            # Fallback mock for local/dev
            self.llms["mock"] = {
                "client": MockLLM(),
                "healthy": True,
                "quota": 10_000_000,
                "cost_per_token": 0.0,
            }
            self.benchmarks["mock"] = {"cost": 0.0, "speed": 1.0, "accuracy": 0.50}

        self.rate_limits = {
            k: {"requests": 0, "last_reset": time.time(), "max_rpm": 500}
            for k in self.llms.keys()
        }
        self.fallback_order = list(self.llms.keys())
        self.skip_health = os.getenv("AF_SKIP_HEALTHCHECK", "1") == "1"

    def health_check(self, model: str):
        if self.skip_health:
            self.llms[model]["healthy"] = True
            return
        try:
            client = self.llms[model]["client"]
            if model == "claude-3-5":
                client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "ping"}],
                )
            else:
                # For langchain clients and mock, this is a no-op or fast
                _ = client.invoke([{"role": "user", "content": "ping"}])
            self.llms[model]["healthy"] = True
        except Exception:
            self.llms[model]["healthy"] = False

    def call(
        self,
        model: str,
        messages: List[Dict],
        tools: List[Dict] | None = None,
        task_id: str = "unknown",
    ) -> Any:  # noqa: C901
        start_time = time.time()
        force_mock = os.getenv("AF_FORCE_MOCK", "0") == "1"
        if force_mock and "mock" in self.llms:
            model = "mock"
        if model not in self.llms:
            model = next(iter(self.llms.keys()))
        self.health_check(model)
        if not self.llms[model]["healthy"] or self.llms[model]["quota"] <= 0:
            for fallback in self.fallback_order:
                if (
                    fallback != model
                    and self.llms[fallback]["healthy"]
                    and self.llms[fallback]["quota"] > 0
                ):
                    model = fallback
                    break
        if time.time() - self.rate_limits[model]["last_reset"] > 60:
            self.rate_limits[model]["requests"] = 0
            self.rate_limits[model]["last_reset"] = time.time()
        if self.rate_limits[model]["requests"] >= self.rate_limits[model]["max_rpm"]:
            time.sleep(1)
        client = self.llms[model]["client"]

        # Set up optional tracing span
        span_ctx = (_TRACER.start_as_current_span("llm.call") if _TRACER else (nullcontext() if nullcontext else None))
        if span_ctx is None:
            # Fallback do-nothing context manager
            class _Noop:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False
            span_ctx = _Noop()

        with span_ctx as span:
            if span is not None and trace is not None:
                try:
                    span.set_attribute("llm.provider", model)
                    span.set_attribute("task.id", task_id)
                except Exception:
                    pass
            try:
                if model == "claude-3-5":
                    try:
                        payload_dict = _anthropic_payload_from_locals(locals())
                        result = _anthropic_call(self.llms["claude-3-5"]["client"], payload_dict)
                        # Emulate a LangChain-like response for downstream handling
                        resp = type("Resp", (), {"content": result.get("text", ""), "_raw": result})
                    except Exception as e:
                        logging.info("Anthropic call failed, falling back: %s", e)
                        raise
                else:
                    resp = client.invoke(messages)  # Langchain/Mock invoke
                usage = len(str(getattr(resp, 'content', resp))) // 4  # Approx
                cost = usage * self.llms[model]["cost_per_token"]
                self.llms[model]["quota"] -= usage
                token_counter.labels(provider=model, agent="router").inc(usage)
                cost_counter.labels(provider=model, agent="router").inc(cost)
                log_with_id(task_id, f"Called {model} successfully")
                latency_hist.labels(provider=model, agent="router").observe(
                    time.time() - start_time
                )
                self.rate_limits[model]["requests"] += 1
                # Record lineage and span attributes for cost/tokens
                if lineage:
                    try:
                        lineage.record_event(
                            "llm_call",
                            {
                                "provider": model,
                                "tokens": int(usage),
                                "cost": float(cost),
                            },
                            job_id=task_id,
                        )
                    except Exception:
                        pass
                if span is not None and trace is not None:
                    try:
                        span.set_attribute("llm.tokens", int(usage))
                        span.set_attribute("llm.cost_usd", float(cost))
                    except Exception:
                        pass
                return resp
            except Exception as e:
                error_counter.labels(provider=model, agent="router").inc()
                log_with_id(task_id, f"Failed {model}: {str(e)}")
                # Record failure in lineage
                if lineage:
                    try:
                        lineage.record_event(
                            "llm_call_error",
                            {"provider": model, "error": str(e)},
                            job_id=task_id,
                        )
                    except Exception:
                        pass
                raise

    def _normalize_response(self, resp: Any, model: str) -> Any:
        if model == "claude-3-5":
            return type('Resp', (), {'content': resp.content[0].text if resp.content else str(resp)})
        return resp

    def route(self, task: Union[str, Any]) -> str:  # noqa: C901
        # Accept Task or raw description string
        description = task if isinstance(task, str) else getattr(task, "description", "")
        desc_l = (description or "").lower()

        # Keep your existing routing logic, but return keys present in llms
        if "code" in desc_l or "implement" in desc_l or "bug" in desc_l:
            return "gpt-5"
        if "analyze" in desc_l or "critique" in desc_l or "review" in desc_l:
            return "claude-3-5"
        if "search" in desc_l or "web" in desc_l or "browse" in desc_l:
            return "gemini-1-5"
        return "gpt-5"
