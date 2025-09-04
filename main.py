from dotenv import load_dotenv
load_dotenv()

import os, sys, json
from orchestrator import build_orchestrator
try:
    from enforcement_bridge import load_sla, make_enforcer
except Exception:
    load_sla = lambda: {"thresholds": {"error_rate": 0.0, "completeness": 0.95}, "policies": {"strict_unknown_kpis": True, "require_hitl_on_violation": True}}
    class _Noop: 
        def enforce(self, goal, results): 
            return {"approved": True, "action": "approve", "reason": "noop"}
    def make_enforcer(_): return _Noop()

def main():
    goal = " ".join(sys.argv[1:]) or os.getenv("AF_GOAL") or "Design UI. Build backend API. Create DB schema. Implement code. Write unit tests."
    num_agents = int(os.getenv("AF_AGENTS", "2"))
    orch = build_orchestrator(num_agents=num_agents)

    results = orch.run_goal_sync(goal)

    sla = load_sla()
    enforcer = make_enforcer(sla)
    decision = enforcer.enforce(goal=goal, results=results)

    print(json.dumps({"goal": goal, "results": results, "decision": decision}, indent=2))

    # Optional exit gating for CI
    if os.getenv("AF_EXIT_ON_REJECT", "0") == "1" and not decision.get("approved", False):
        sys.exit(2)

if __name__ == "__main__":
    main()
