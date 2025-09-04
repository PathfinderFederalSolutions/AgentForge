# router.py
import os
import time
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import anthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI

try:
    from forge_types import Task
except Exception:
    Task = None  # allow import without types at import time

from observability import token_counter, latency_hist, cost_counter, error_counter, log_with_id  # Assume absolute; update if relative

load_dotenv()

def _getenv(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None

class ChatGrok:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        # ... (existing implementation)

class MockLLM:
    def __init__(self, model: str = "mock-model"):
        self.model = model
    def invoke(self, messages):
        return type('Response', (), {'content': f"[MOCK:{self.model}] " + (messages[-1].get('content') if messages else '')})

class DynamicRouter:
    def __init__(self):
        self.llms: dict[str, dict] = {}
        self.benchmarks: dict[str, dict] = {}

        # Build provider clients only when keys exist; else add a mock
        oai_key = _getenv("OPENAI_API_KEY", "OPENAI_KEY")
        if oai_key:
            self.llms["gpt-4o-mini"] = {"client": ChatOpenAI(model="gpt-4o-mini", api_key=oai_key), "healthy": True, "quota": 100000, "cost_per_token": 0.000002}
            self.benchmarks["gpt-4o-mini"] = {"cost": 0.002, "speed": 0.85, "accuracy": 0.90}

        anth_key = _getenv("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")
        if anth_key:
            self.llms["claude-3-5"] = {"client": anthropic.Anthropic(api_key=anth_key), "healthy": True, "quota": 50000, "cost_per_token": 0.000015}
            self.benchmarks["claude-3-5"] = {"cost": 0.015, "speed": 0.65, "accuracy": 0.93}

        google_key = _getenv("GOOGLE_API_KEY", "GOOGLE_KEY")
        if google_key:
            self.llms["gemini-1-5"] = {"client": ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=google_key), "healthy": True, "quota": 30000, "cost_per_token": 0.000018}
            self.benchmarks["gemini-1-5"] = {"cost": 0.018, "speed": 0.75, "accuracy": 0.91}

        mistral_key = _getenv("MISTRAL_API_KEY", "MISTRAL_KEY")
        if mistral_key:
            self.llms["mistral-large"] = {"client": ChatMistralAI(model="mistral-large-latest", api_key=mistral_key), "healthy": True, "quota": 20000, "cost_per_token": 0.00001}
            self.benchmarks["mistral-large"] = {"cost": 0.01, "speed": 0.8, "accuracy": 0.89}

        cohere_key = _getenv("COHERE_API_KEY", "CO_API_KEY")
        if cohere_key:
            self.llms["cohere-command"] = {"client": ChatCohere(model="command-r-plus", api_key=cohere_key), "healthy": True, "quota": 40000, "cost_per_token": 0.000012}
            self.benchmarks["cohere-command"] = {"cost": 0.012, "speed": 0.7, "accuracy": 0.90}

        xai_key = _getenv("XAI_API_KEY", "XAI_KEY")
        if xai_key:
            self.llms["grok-4"] = {"client": ChatGrok(model="grok-4", api_key=xai_key), "healthy": True, "quota": 50000, "cost_per_token": 0.00002}
            self.benchmarks["grok-4"] = {"cost": 0.02, "speed": 0.6, "accuracy": 0.94}

        if not self.llms:
            # Fallback mock for local/dev
            self.llms["mock"] = {"client": MockLLM(), "healthy": True, "quota": 10_000_000, "cost_per_token": 0.0}
            self.benchmarks["mock"] = {"cost": 0.0, "speed": 1.0, "accuracy": 0.50}

        self.rate_limits = {k: {"requests": 0, "last_reset": time.time(), "max_rpm": 500} for k in self.llms.keys()}
        self.fallback_order = list(self.llms.keys())

    def health_check(self, model: str):
        try:
            client = self.llms[model]["client"]
            if model == "claude-3-5":
                client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=1, messages=[{"role": "user", "content": "ping"}])
            else:
                # For langchain clients and mock, this is a no-op or fast
                _ = client.invoke([{"role": "user", "content": "ping"}])
            self.llms[model]["healthy"] = True
        except Exception:
            self.llms[model]["healthy"] = False

    def call(self, model: str, messages: List[Dict], tools: List[Dict] = None, task_id: str = "unknown") -> Any:
        start_time = time.time()
        if model not in self.llms:
            model = next(iter(self.llms.keys()))
        self.health_check(model)
        if not self.llms[model]["healthy"] or self.llms[model]["quota"] <= 0:
            for fallback in self.fallback_order:
                if fallback != model and self.llms[fallback]["healthy"] and self.llms[fallback]["quota"] > 0:
                    model = fallback
                    break
        if time.time() - self.rate_limits[model]["last_reset"] > 60:
            self.rate_limits[model]["requests"] = 0
            self.rate_limits[model]["last_reset"] = time.time()
        if self.rate_limits[model]["requests"] >= self.rate_limits[model]["max_rpm"]:
            time.sleep(1)
        client = self.llms[model]["client"]
        try:
            if model == "claude-3-5":
                resp = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=1024, messages=messages, tools=tools)
                resp = self._normalize_response(resp, model)
            else:
                resp = client.invoke(messages)  # Langchain/Mock invoke
            usage = len(str(getattr(resp, 'content', resp))) // 4  # Approx
            cost = usage * self.llms[model]["cost_per_token"]
            self.llms[model]["quota"] -= usage
            token_counter.labels(provider=model, agent="router").inc(usage)
            cost_counter.labels(provider=model, agent="router").inc(cost)
            log_with_id(task_id, f"Called {model} successfully")
            latency_hist.labels(provider=model, agent="router").observe(time.time() - start_time)
            self.rate_limits[model]["requests"] += 1
            return resp
        except Exception as e:
            error_counter.labels(provider=model, agent="router").inc()
            log_with_id(task_id, f"Failed {model}: {str(e)}")
            raise

    def _normalize_response(self, resp: Any, model: str) -> Any:
        if model == "claude-3-5":
            return type('Resp', (), {'content': resp.content[0].text if resp.content else str(resp)})
        return resp

    def route(self, task: Union["Task", str]) -> str:
        # Accept Task or raw description string
        description = task if isinstance(task, str) else getattr(task, "description", "")
        desc_l = (description or "").lower()

        # Keep your existing routing logic, but use desc_l
        if "code" in desc_l or "implement" in desc_l or "bug" in desc_l:
            return "gpt-5"
        if "analyze" in desc_l or "critique" in desc_l or "review" in desc_l:
            return "claude-3-5"
        if "search" in desc_l or "web" in desc_l or "browse" in desc_l:
            return "gemini-1.5-pro"
        return "gpt-5"