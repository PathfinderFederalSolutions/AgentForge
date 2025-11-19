"""
Legacy System Bridge - Integration with existing AgentForge components
Preserves useful functionality from legacy orchestrator and quantum-scheduler
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

# Legacy imports (optional)
try:
    from services.swarm.planner import Planner, Executor, PlanStep
    from services.swarm.forge_types import Task
    from core.agents import Agent, AgentSwarm
    from router_v2 import MoERouter, Provider
    from services.orchestrator.app.orchestrator_enforcer import SLAKPIEnforcer
    from approval import ApprovalManager
    from services.swarm.lineage import persist_dag
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False

# Optional vector service integration
try:
    from services.swarm.vector import service as vector_service
    from services.swarm.app.memory.pgvector_store import PGVectorStore, EmbeddingProviderFactory
    VECTOR_SERVICE_AVAILABLE = True
except ImportError:
    VECTOR_SERVICE_AVAILABLE = False

# Optional observability integration
try:
    from opentelemetry import trace
    from services.swarm.observability.otel import set_dag_hash
    from services.swarm.observability.task_latency import record_dag_hash
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

log = logging.getLogger("legacy-bridge")

@dataclass
class LegacyTaskSpec:
    """Legacy task specification preserved from orchestrator models"""
    id: str
    type: str  # gather, analyze, synthesize, review, map, reduce, custom
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    review: bool = False

@dataclass
class LegacySwarmJob:
    """Legacy swarm job preserved from orchestrator models"""
    request_id: str
    goal: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    tools_allowed: List[str] = field(default_factory=list)
    plan: List[LegacyTaskSpec] = field(default_factory=list)
    reply_subject: Optional[str] = None

class LegacyToolRegistry:
    """Preserved tool registry functionality"""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        
        # Register baseline tools
        baseline_tools = [
            "browser.fetch", "browser.render", "code.exec", 
            "file.ingest", "rag.search", "review.validate"
        ]
        for tool in baseline_tools:
            self.register(tool, kind="remote")
    
    def register(self, name: str, kind: str, validator: Optional[Callable] = None):
        """Register a tool"""
        self._tools[name] = {"kind": kind, "validator": validator}
    
    def list(self) -> List[str]:
        """List all registered tools"""
        return list(self._tools.keys())
    
    def allowed_for(self, job: LegacySwarmJob) -> List[str]:
        """Get tools allowed for a specific job"""
        if not job.tools_allowed:
            return self.list()
        return [t for t in self.list() if t in job.tools_allowed]

class LegacyIntegrationBridge:
    """
    Bridge to integrate useful legacy components with unified orchestrator
    """
    
    def __init__(self, enable_legacy_features: bool = True):
        self.enable_legacy_features = enable_legacy_features
        
        # Legacy components (if available)
        self.planner: Optional[Planner] = None
        self.agent_swarm: Optional[AgentSwarm] = None
        self.router: Optional[MoERouter] = None
        self.sla_enforcer: Optional[SLAKPIEnforcer] = None
        self.approval_manager: Optional[ApprovalManager] = None
        
        # Tool registry
        self.tool_registry = LegacyToolRegistry()
        
        # Vector service integration
        self._pgvector_store = None
        self._embed_provider = None
        
        # Observability
        self._tracer = None
        
        if self.enable_legacy_features and LEGACY_COMPONENTS_AVAILABLE:
            self._initialize_legacy_components()
        
        if OBSERVABILITY_AVAILABLE:
            self._tracer = trace.get_tracer("unified-orchestrator.legacy-bridge")
    
    def _initialize_legacy_components(self):
        """Initialize legacy components"""
        try:
            self.planner = Planner()
            self.agent_swarm = AgentSwarm(num_agents=3)
            self.router = MoERouter(epsilon=0.1)
            self.sla_enforcer = SLAKPIEnforcer()
            self.approval_manager = ApprovalManager()
            
            # Register LLM providers
            caps_by_key = {
                "gpt-5": {"general", "code"},
                "claude-3-5": {"general", "analysis"},
                "gemini-1-5": {"general", "search"},
                "mistral-large": {"general", "code"},
                "cohere-command": {"general", "writing"},
                "grok-4": {"general"},
                "mock": {"general"},
            }
            
            if self.router and hasattr(self.agent_swarm, 'llms'):
                for key in getattr(self.agent_swarm, 'llms', {}).keys():
                    self.router.register(
                        Provider(
                            key=key,
                            model=key,
                            capabilities=caps_by_key.get(key, {"general"}),
                        )
                    )
            
            log.info("Legacy components initialized successfully")
            
        except Exception as e:
            log.warning(f"Failed to initialize some legacy components: {e}")
    
    def scope_for_goal(self, goal: str) -> str:
        """Generate scope hash for goal (preserved from legacy)"""
        h = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]
        return f"job:{h}"
    
    def sla_capability_for_description(self, description: str) -> str:
        """Map description to SLA capability (preserved from legacy)"""
        d = (description or "").lower()
        if any(w in d for w in ["spawn", "agent", "scale", "dispatch"]):
            return "Dynamic Agent Lifecycle"
        if any(w in d for w in ["memory", "mesh", "provenance", "scope"]):
            return "Memory Mesh"
        if any(w in d for w in ["latency", "throughput", "performance", "p95", "p99"]):
            return "Scalability and Performance"
        if any(w in d for w in ["heal", "fix", "critic", "regression"]):
            return "Self-Healing Loop"
        return "Dynamic Agent Lifecycle"
    
    async def seed_task_args(self, scope: str, goal: str, step: PlanStep) -> Dict[str, Any]:
        """Seed task arguments using vector retrieval (preserved from legacy)"""
        args: Dict[str, Any] = dict(step.args) if hasattr(step, 'args') else {}
        
        # Provide safe defaults for known capabilities
        if hasattr(step, 'capability'):
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
        if VECTOR_SERVICE_AVAILABLE and vector_service:
            try:
                hits = vector_service.search(scope=scope, query=goal, top_k=5)
                args.setdefault("context_snippets", [str(h)[:256] for h in list(hits)[:3]])
            except Exception:
                pass
        
        return args
    
    async def ensure_vector_store(self):
        """Ensure vector store is initialized"""
        if self._pgvector_store or not VECTOR_SERVICE_AVAILABLE or not PGVectorStore:
            return self._pgvector_store
        
        try:
            store = PGVectorStore()
            await store.init()
            self._pgvector_store = store
        except Exception:
            self._pgvector_store = None
        
        return self._pgvector_store
    
    def select_embedding_provider(self, task_desc: str):
        """Select embedding provider for task"""
        if self._embed_provider is not None:
            return self._embed_provider
        
        if VECTOR_SERVICE_AVAILABLE and EmbeddingProviderFactory:
            try:
                self._embed_provider = EmbeddingProviderFactory.best_available()
                return self._embed_provider
            except Exception:
                pass
        
        self._embed_provider = None
        return None
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using provider"""
        provider = self.select_embedding_provider(";".join(texts))
        if provider is None:
            return []
        
        try:
            return await provider.embed(texts)
        except Exception:
            return []
    
    async def run_legacy_goal(self, goal: str, num_agents: int = 3) -> List[Dict[str, Any]]:
        """Run goal using legacy orchestrator logic"""
        if not self.enable_legacy_features or not LEGACY_COMPONENTS_AVAILABLE:
            raise RuntimeError("Legacy features not available")
        
        scope = self.scope_for_goal(goal)
        
        # Deterministic DAG
        seed = int(hashlib.sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
        dag = self.planner.make_dag(goal, seed=seed)
        
        # Persist DAG
        dag_path = persist_dag(dag)
        
        # Record observability
        if OBSERVABILITY_AVAILABLE:
            try:
                set_dag_hash(dag.hash)
                record_dag_hash(dag.hash)
            except Exception:
                pass
        
        # Make plan
        plan = self.planner.make_plan(goal)
        execu = Executor(scope=scope)
        
        # Seed plan args via vector retrieval
        enriched = []
        for step in plan:
            step.args = await self.seed_task_args(scope, goal, step)
            enriched.append(step)
        
        # Execute plan
        out = execu.run(enriched, agents=num_agents)
        
        # Flatten for consumption
        flat: List[Dict[str, Any]] = []
        for cap_name, res in zip(out["steps"], out["results"]):
            item = dict(res)
            item["capability"] = cap_name
            flat.append(item)
        
        return flat
    
    async def enforce_sla_pre_task(self, task: Dict[str, Any], capability: str):
        """Enforce SLA before task execution"""
        if self.sla_enforcer:
            try:
                self.sla_enforcer.enforce_pre_task(task, capability)
            except Exception as e:
                log.warning(f"SLA pre-task enforcement failed: {e}")
    
    async def enforce_sla_post_task(self, task_result: Dict[str, Any], capability: str):
        """Enforce SLA after task execution"""
        if self.sla_enforcer:
            try:
                self.sla_enforcer.enforce_post_task(task_result, capability)
            except Exception as e:
                log.warning(f"SLA post-task enforcement failed: {e}")
    
    async def check_approval(self, task: Dict[str, Any], result: Any, capability: str,
                           publisher: Optional[Callable] = None) -> Dict[str, Any]:
        """Check approval for task result"""
        if self.approval_manager:
            try:
                return self.approval_manager.check_and_gate(
                    task, result, capability, publisher=publisher
                )
            except Exception as e:
                log.warning(f"Approval check failed: {e}")
                return {"approved": False, "escalated": True, "reason": str(e)}
        
        return {"approved": True, "escalated": False, "reason": "no approval manager"}
    
    def route_task(self, description: str) -> str:
        """Route task to appropriate provider"""
        if self.router:
            try:
                return self.router.route(description)
            except Exception as e:
                log.warning(f"Task routing failed: {e}")
        
        return "mock"  # Default provider
    
    def provide_router_feedback(self, provider_key: str, reward: float):
        """Provide feedback to router"""
        if self.router:
            try:
                self.router.feedback(provider_key, reward)
            except Exception as e:
                log.warning(f"Router feedback failed: {e}")
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status and component health"""
        return {
            "legacy_features_enabled": self.enable_legacy_features,
            "components_available": LEGACY_COMPONENTS_AVAILABLE,
            "vector_service_available": VECTOR_SERVICE_AVAILABLE,
            "observability_available": OBSERVABILITY_AVAILABLE,
            "components": {
                "planner": self.planner is not None,
                "agent_swarm": self.agent_swarm is not None,
                "router": self.router is not None,
                "sla_enforcer": self.sla_enforcer is not None,
                "approval_manager": self.approval_manager is not None,
                "vector_store": self._pgvector_store is not None,
                "embedding_provider": self._embed_provider is not None
            },
            "tool_registry": {
                "total_tools": len(self.tool_registry.list()),
                "registered_tools": self.tool_registry.list()
            }
        }

# Alias for backward compatibility
LegacyBridge = LegacyIntegrationBridge
