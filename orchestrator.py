from __future__ import annotations
import hashlib
from typing import Any, Dict, List
from swarm.planner import Planner, Executor, PlanStep

# Optional vector service
try:
    from swarm.vector import service as vector_service  # type: ignore
except ImportError:
    vector_service = None  # type: ignore

# Optional pgvector store and embedding provider
try:
    from swarm.memory.pgvector_store import PGVectorStore, EmbeddingProviderFactory  # type: ignore
except ImportError:
    PGVectorStore = None  # type: ignore
    EmbeddingProviderFactory = None  # type: ignore

import asyncio
import time
from forge_types import Task
from agents import Agent, AgentSwarm
from router_v2 import MoERouter, Provider
from orchestrator_enforcer import SLAKPIEnforcer
from approval import ApprovalManager
from swarm.lineage import persist_dag

# Optional OpenTelemetry tracer
try:
    from opentelemetry import trace  # type: ignore
    _tracer = trace.get_tracer("swarm.orchestrator")
except Exception:  # pragma: no cover
    _tracer = None

# Optional observability helper
try:
    from swarm.observability.otel import set_dag_hash as _set_dag_hash  # type: ignore
except Exception:  # pragma: no cover
    def _set_dag_hash(_x: str):  # type: ignore
        pass

try:
    from swarm.observability.task_latency import record_dag_hash  # type: ignore
except Exception:  # pragma: no cover
    def record_dag_hash(_x: str):
        pass

def _scope_for_goal(goal: str) -> str:
    h = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]
    return f"job:{h}"

def build_orchestrator(num_agents: int = 3) -> Orchestrator:
    return Orchestrator(num_agents=num_agents)

class Orchestrator:
    def __init__(self, num_agents: int = 3) -> None:
        self.num_agents = max(1, num_agents)
        self.planner = Planner()
        self.swarm = AgentSwarm(num_agents=self.num_agents)
        self.mesh = self.swarm.mesh  # shared mesh
        self.router = MoERouter(epsilon=0.1)
        self.enforcer = SLAKPIEnforcer()  # Soft enforcement by default
        self.approval = ApprovalManager()  # HITL/approval gating

        # Memory / embedding backend (lazy init to avoid startup cost if unused)
        self._pgvector_store = None
        self._embed_provider = None

        # Register providers based on available LLM clients
        caps_by_key = {
            "gpt-5": {"general", "code"},
            "claude-3-5": {"general", "analysis"},
            "gemini-1-5": {"general", "search"},
            "mistral-large": {"general", "code"},
            "cohere-command": {"general", "writing"},
            "grok-4": {"general"},
            "mock": {"general"},
        }
        for key in getattr(self.swarm, 'llms', {}).keys():
            self.router.register(
                Provider(
                    key=key,
                    model=key,
                    capabilities=caps_by_key.get(key, {"general"}),
                )
            )

    def _sla_cap_for_desc(self, desc: str) -> str:
        d = (desc or "").lower()
        if any(w in d for w in ["spawn", "agent", "scale", "dispatch"]):
            return "Dynamic Agent Lifecycle"
        if any(w in d for w in ["memory", "mesh", "provenance", "scope"]):
            return "Memory Mesh"
        if any(w in d for w in ["latency", "throughput", "performance", "p95", "p99"]):
            return "Scalability and Performance"
        if any(w in d for w in ["heal", "fix", "critic", "regression"]):
            return "Self-Healing Loop"
        return "Dynamic Agent Lifecycle"

    def _seed_args(self, scope: str, goal: str, step: PlanStep) -> Dict[str, Any]:
        args: Dict[str, Any] = dict(step.args)
        
        # Always provide safe defaults for known capabilities
        if step.capability == "bayesian_fusion":
            args.setdefault("eo", [1, 2, 3, 4, 5])
            args.setdefault("ir", [2, 3, 4, 5, 6])
        elif step.capability == "fuse_and_persist_track":
            args.setdefault("eo", [0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6])
            args.setdefault("ir", [0.05, 0.15, 0.22, 0.28, 0.52, 0.58, 0.62])
            args.setdefault("alpha", 0.1)
        elif step.capability == "conformal_validate":
            args.setdefault("residuals", [0.1, -0.2, 0.05, 0.0])
            args.setdefault("alpha", 0.1)
        
        # Vector-assisted retrieval to seed capability inputs
        if vector_service:
            try:
                hits = vector_service.search(scope=scope, query=goal, top_k=5)
                # Stash top snippets for LLM-style capabilities
                try:
                    args.setdefault("context_snippets", [str(h)[:256] for h in list(hits)[:3]])
                except Exception:
                    pass
            except Exception:
                pass
        return args

    def _select_embedding_provider(self, task_desc: str):
        """Heuristic selection of embedding provider per task.
        Priority order (when available): OpenAI > Cohere > SentenceTransformers > Hash.
        May later adapt based on length/domain or cost constraints.
        """
        if self._embed_provider is not None:
            return self._embed_provider
        if EmbeddingProviderFactory:
            try:
                self._embed_provider = EmbeddingProviderFactory.best_available()
                return self._embed_provider
            except Exception:
                pass
        self._embed_provider = None
        return None

    async def _ensure_vector_store(self):
        if self._pgvector_store or not PGVectorStore:
            return self._pgvector_store
        try:
            store = PGVectorStore()
            await store.init()
            self._pgvector_store = store
        except Exception:
            self._pgvector_store = None
        return self._pgvector_store

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        provider = self._select_embedding_provider(";".join(texts))
        if provider is None:
            return []
        try:
            return await provider.embed(texts)
        except Exception:
            return []

    async def run_goal(self, goal: str) -> List[Dict]:
        """Execute a goal asynchronously across the swarm.
        Refactored to reduce cyclomatic complexity by delegating DAG prep to _prepare_dag
        and subtask execution kept in nested run_one for shared closure variables.
        """
        await self._ensure_vector_store()  # best-effort embeddings backend
        dag, dag_path, subtasks = self._prepare_dag(goal)

        async def run_one(i: int, st: Dict) -> Dict:
            desc = st["desc"]
            pkey = self.router.route(desc)
            cap = self._sla_cap_for_desc(desc)
            agent: Agent = self.swarm.agents[i % len(self.swarm.agents)]
            task = Task(
                id=st["id"],
                description=desc,
                metadata={"provider": pkey},
            )
            # Pre-task SLA/KPI (non-blocking unless strict)
            try:
                self.enforcer.enforce_pre_task(task.model_dump(), cap)
            except Exception:
                pass

            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            if _tracer:
                with _tracer.start_as_current_span("orchestrator.run_one") as span:  # type: ignore
                    span.set_attribute("task.capability", cap)  # type: ignore
                    span.set_attribute("task.provider", pkey)  # type: ignore
                    result = await loop.run_in_executor(None, agent.process, task)
            else:
                result = await loop.run_in_executor(None, agent.process, task)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # trivial reward: length/quality heuristic
            reward = 1.0 if result and isinstance(result, str) and len(result) > 0 else 0.0
            self.router.feedback(pkey, reward)

            # Post-task SLA/KPI validations for both task capability and performance
            for post_cap in {cap, "Scalability and Performance"}:
                try:
                    self.enforcer.enforce_post_task(
                        {"result": result, "id": st["id"], "metrics": {"latency_ms": latency_ms}},
                        post_cap,
                    )
                except Exception:
                    pass

            # Approval/HITL gating (non-blocking by default, strict via env)
            try:
                decision = self.approval.check_and_gate(
                    task.model_dump(),
                    result,
                    cap,
                    publisher=self.mesh.publish,
                )
            except Exception:
                decision = {"approved": False, "escalated": True, "reason": "exception"}

            self.mesh.publish(
                "subtask.done",
                {
                    "id": st["id"],
                    "provider": pkey,
                    "result": (result or "")[:500],
                    "approval": decision,
                    "metrics": {"latency_ms": latency_ms},
                },
            )
            return {
                "id": st["id"],
                "provider": pkey,
                "result": result,
                "approval": decision,
                "metrics": {"latency_ms": latency_ms},
            }

        if _tracer:
            with _tracer.start_as_current_span("orchestrator.run_goal") as span:  # type: ignore
                span.set_attribute("goal.length", len(goal))  # type: ignore
                results = await asyncio.gather(
                    *[run_one(i, st) for i, st in enumerate(subtasks)],
                    return_exceptions=False,
                )
        else:
            results = await asyncio.gather(
                *[run_one(i, st) for i, st in enumerate(subtasks)],
                return_exceptions=False,
            )
        # Optionally store summary embedding for goal (fire-and-forget)
        try:
            if self._pgvector_store and results:
                summary = f"Goal: {goal}\nResults: " + " | ".join(str(r.get('result',''))[:200] for r in results)
                await self._pgvector_store.upsert_batch('global', [("goal", summary, {"type":"goal_summary"})])
        except Exception:
            pass
        self.mesh.publish("goal.done", {"goal": goal, "count": len(results), "dag_hash": dag.hash, "path": dag_path})
        return results

    def _prepare_dag(self, goal: str):
        """Create & persist deterministic DAG, emit lineage & metrics, and return subtasks.
        Returns (dag, dag_path, subtasks)."""
        seed = int(hashlib.sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
        dag = self.planner.make_dag(goal, seed=seed)
        dag_path = persist_dag(dag)
        if _set_dag_hash:
            try:
                _set_dag_hash(dag.hash)
            except Exception:
                pass
        try:
            record_dag_hash(dag.hash)
        except Exception:
            pass
        self.mesh.publish("dag.created", {"goal": goal, "hash": dag.hash, "path": dag_path})
        subtasks = self.planner.make_plan(goal)
        return dag, dag_path, subtasks

    def run_goal_sync(self, goal: str) -> List[Dict[str, Any]]:
        scope = _scope_for_goal(goal)
        # Deterministic DAG for sync path
        seed = int(hashlib.sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
        dag = self.planner.make_dag(goal, seed=seed)
        persist_dag(dag)
        if _set_dag_hash:
            try:
                _set_dag_hash(dag.hash)
            except Exception:
                pass
        try:
            record_dag_hash(dag.hash)
        except Exception:
            pass
        plan = self.planner.make_plan(goal)
        execu = Executor(scope=scope)
        # Seed plan args via vector retrieval
        enriched = []
        for step in plan:
            step.args = self._seed_args(scope, goal, step)
            enriched.append(step)
        out = execu.run(enriched, agents=self.num_agents)
        # Flatten for enforcer consumption; include capability tag
        flat: List[Dict[str, Any]] = []
        for cap_name, res in zip(out["steps"], out["results"]):
            item = dict(res)
            item["capability"] = cap_name
            flat.append(item)
        return flat