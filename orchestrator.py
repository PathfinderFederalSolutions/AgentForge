from typing import List, Dict
import asyncio
from forge_types import Task
from agents import Agent, AgentSwarm
from router_v2 import MoERouter, Provider
from planner import TaskPlanner
from orchestrator_enforcer import SLAKPIEnforcer
from approval import ApprovalManager


def build_orchestrator(num_agents: int = 2):
    return Orchestrator(num_agents=num_agents)


class Orchestrator:
    def __init__(self, num_agents: int = 2):
        self.swarm = AgentSwarm(num_agents=num_agents)
        self.mesh = self.swarm.mesh  # shared mesh
        self.planner = TaskPlanner(self.mesh, self.swarm.factory)
        self.router = MoERouter(epsilon=0.1)
        self.enforcer = SLAKPIEnforcer()  # Soft enforcement by default
        self.approval = ApprovalManager()  # HITL/approval gating

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

    async def run_goal(self, goal: str) -> List[Dict]:
        subtasks = self.planner.plan(goal)

        async def run_one(i: int, st: Dict) -> Dict:
            desc = st["desc"]
            pkey = self.router.route(desc)
            cap = self._sla_cap_for_desc(desc)
            # simple round-robin agent pick
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
                # Already logged by enforcer in strict mode
                pass

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, agent.process, task)

            # trivial reward: length/quality heuristic
            reward = 1.0 if result and isinstance(result, str) and len(result) > 0 else 0.0
            self.router.feedback(pkey, reward)

            # Post-task SLA/KPI validations for both task capability and performance
            for post_cap in {cap, "Scalability and Performance"}:
                try:
                    self.enforcer.enforce_post_task(
                        {"result": result, "id": st["id"]},
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
                # In strict and non-approved cases this raises; minimal info
                decision = {"approved": False, "escalated": True, "reason": "exception"}

            self.mesh.publish(
                "subtask.done",
                {
                    "id": st["id"],
                    "provider": pkey,
                    "result": (result or "")[:500],
                    "approval": decision,
                },
            )
            return {
                "id": st["id"],
                "provider": pkey,
                "result": result,
                "approval": decision,
            }

        results = await asyncio.gather(
            *[run_one(i, st) for i, st in enumerate(subtasks)],
            return_exceptions=False,
        )
        self.mesh.publish("goal.done", {"goal": goal, "count": len(results)})
        return results

    def run_goal_sync(self, goal: str) -> List[Dict]:
        return asyncio.run(self.run_goal(goal))