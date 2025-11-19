"""DEPRECATED: Legacy Orchestrator - Redirects to Unified Quantum Orchestrator

This file is maintained for backward compatibility only.
All new development should use services.unified_orchestrator.

Migration Guide: /UNIFIED_ORCHESTRATOR_MIGRATION_GUIDE.md
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import warnings
from typing import Any, Dict, List

# Issue deprecation warning
warnings.warn(
    "orchestrator.py is deprecated. Use services.unified_orchestrator.UnifiedQuantumOrchestrator instead. "
    "See UNIFIED_ORCHESTRATOR_MIGRATION_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

log = logging.getLogger("legacy-orchestrator")

# Import unified orchestrator
try:
    from services.unified_orchestrator import UnifiedQuantumOrchestrator, TaskPriority, SecurityLevel
    UNIFIED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    UNIFIED_ORCHESTRATOR_AVAILABLE = False
    log.error("Unified orchestrator not available - legacy fallback will be used")

def _scope_for_goal(goal: str) -> str:
    """Legacy function preserved for compatibility"""
    h = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]
    return f"job:{h}"

def build_orchestrator(num_agents: int = 3) -> 'LegacyOrchestratorWrapper':
    """Build orchestrator - now returns unified orchestrator wrapper"""
    log.warning("build_orchestrator() is deprecated. Use UnifiedQuantumOrchestrator directly.")
    return LegacyOrchestratorWrapper(num_agents=num_agents)

class LegacyOrchestratorWrapper:
    """Legacy orchestrator wrapper that delegates to unified orchestrator"""
    
    def __init__(self, num_agents: int = 3) -> None:
        log.warning("LegacyOrchestratorWrapper is deprecated. Migrate to UnifiedQuantumOrchestrator.")
        
        self.num_agents = max(1, num_agents)
        
        if UNIFIED_ORCHESTRATOR_AVAILABLE:
            # Use unified orchestrator
            self._unified_orchestrator = None
            log.info("Delegating to UnifiedQuantumOrchestrator")
        else:
            # Fallback to legacy implementation (not recommended)
            log.error("UnifiedQuantumOrchestrator not available - using legacy fallback")
            self._setup_legacy_fallback()
    
    def _setup_legacy_fallback(self):
        """Setup legacy fallback (not recommended for production)"""
        try:
            from services.swarm.planner import Planner
            from core.agents import AgentSwarm
            from router_v2 import MoERouter
            
            self.planner = Planner()
            self.swarm = AgentSwarm(num_agents=self.num_agents)
            self.mesh = self.swarm.mesh
            self.router = MoERouter(epsilon=0.1)
            
            log.warning("Legacy fallback initialized - migrate to unified orchestrator ASAP")
        except ImportError as e:
            log.error(f"Legacy fallback failed: {e}")
            raise RuntimeError("Cannot initialize orchestrator - unified orchestrator not available and legacy components missing")
    
    async def _ensure_unified_orchestrator(self):
        """Ensure unified orchestrator is initialized"""
        if not UNIFIED_ORCHESTRATOR_AVAILABLE:
            raise RuntimeError("UnifiedQuantumOrchestrator not available")
        
        if self._unified_orchestrator is None:
            self._unified_orchestrator = UnifiedQuantumOrchestrator(
                node_id=f"legacy-wrapper-{id(self)}",
                max_agents=min(self.num_agents * 1000, 100000),  # Scale up capacity
                enable_security=False  # Disable for legacy compatibility
            )
            await self._unified_orchestrator.start()
            
            # Register agents based on num_agents
            for i in range(self.num_agents):
                await self._unified_orchestrator.register_agent(
                    agent_id=f"legacy-agent-{i}",
                    capabilities={"general", "legacy"},
                    security_clearance=SecurityLevel.UNCLASSIFIED
                )
    
    async def run_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Run goal using unified orchestrator (async version)"""
        log.warning("run_goal() is deprecated. Use UnifiedQuantumOrchestrator.submit_task() instead.")
        
        if UNIFIED_ORCHESTRATOR_AVAILABLE:
            await self._ensure_unified_orchestrator()
            
            # Submit task to unified orchestrator
            task_id = await self._unified_orchestrator.submit_task(
                task_description=goal,
                priority=TaskPriority.NORMAL,
                required_agents=self.num_agents,
                required_capabilities={"general"},
                classification=SecurityLevel.UNCLASSIFIED
            )
            
            # Wait for completion (simplified)
            import time
            start_time = time.time()
            timeout = 300  # 5 minutes
            
            while time.time() - start_time < timeout:
                status = self._unified_orchestrator.get_system_status()
                
                # Check if task completed
                completed_tasks = status["tasks"]["completed"]
                if completed_tasks > 0:
                    # Return mock result for compatibility
                    return [{
                        "id": task_id,
                        "result": f"Completed via unified orchestrator: {goal}",
                        "provider": "unified-quantum",
                        "capability": "general"
                    }]
                
                await asyncio.sleep(1.0)
            
            # Timeout
            return [{
                "id": task_id,
                "result": "Task timeout in unified orchestrator",
                "provider": "unified-quantum", 
                "capability": "general",
                "error": "timeout"
            }]
        else:
            # Use legacy fallback
            return self.run_goal_sync(goal)
    
    def run_goal_sync(self, goal: str) -> List[Dict[str, Any]]:
        """Synchronous goal execution (deprecated)"""
        log.warning("run_goal_sync() is deprecated. Use async run_goal() or UnifiedQuantumOrchestrator.")
        
        if not UNIFIED_ORCHESTRATOR_AVAILABLE:
            # Use legacy implementation if available
            try:
                return self._run_legacy_goal_sync(goal)
            except Exception as e:
                log.error(f"Legacy goal execution failed: {e}")
                return []
        
        # For unified orchestrator, we need to run async in sync context
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we can't use run_until_complete
                # Return a placeholder and log warning
                log.warning("Cannot run sync goal in async context - use run_goal() instead")
                return [{
                    "id": "sync-placeholder",
                    "result": "Use async run_goal() method",
                    "provider": "unified-quantum",
                    "capability": "general",
                    "error": "sync_in_async_context"
                }]
            else:
                # Run async method in sync context
                return loop.run_until_complete(self.run_goal(goal))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.run_goal(goal))
    
    def _run_legacy_goal_sync(self, goal: str) -> List[Dict[str, Any]]:
        """Legacy synchronous goal execution"""
        if not hasattr(self, 'planner'):
            raise RuntimeError("Legacy components not available")
        
        scope = _scope_for_goal(goal)
        
        # Use legacy implementation
        try:
            from services.swarm.planner import Executor
            
            seed = int(hashlib.sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
            dag = self.planner.make_dag(goal, seed=seed)
            
            plan = self.planner.make_plan(goal)
            execu = Executor(scope=scope)
            
            # Execute with legacy agents
            out = execu.run(plan, agents=self.num_agents)
            
            # Format results
            flat: List[Dict[str, Any]] = []
            for cap_name, res in zip(out["steps"], out["results"]):
                item = dict(res)
                item["capability"] = cap_name
                flat.append(item)
            
            return flat
            
        except Exception as e:
            log.error(f"Legacy goal execution failed: {e}")
            return []
    
    # Legacy compatibility methods (deprecated)
    def _sla_cap_for_desc(self, desc: str) -> str:
        """Legacy SLA capability mapping (deprecated)"""
        log.warning("_sla_cap_for_desc() is deprecated")
        return "general"
    
    def _seed_args(self, scope: str, goal: str, step) -> Dict[str, Any]:
        """Legacy seed args method (deprecated)"""
        log.warning("_seed_args() is deprecated")
        return {}
    
    def _select_embedding_provider(self, task_desc: str):
        """Legacy embedding provider selection (deprecated)"""
        log.warning("_select_embedding_provider() is deprecated")
        return None
    
    async def _ensure_vector_store(self):
        """Legacy vector store initialization (deprecated)"""
        log.warning("_ensure_vector_store() is deprecated")
        return None
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Legacy text embedding (deprecated)"""
        log.warning("embed_texts() is deprecated")
        return []
    
    def _prepare_dag(self, goal: str):
        """Legacy DAG preparation (deprecated)"""
        log.warning("_prepare_dag() is deprecated")
        return None, None, []

# Backward compatibility alias
Orchestrator = LegacyOrchestratorWrapper