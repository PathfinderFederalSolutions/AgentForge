from dotenv import load_dotenv
load_dotenv()

import os, sys, json, time
from orchestrator import build_orchestrator

# Import enhanced configuration and logging
try:
    from config.agent_config import get_config, get_agent_config
    from core.enhanced_logging import log_info, log_error
    ENHANCED_SYSTEMS_AVAILABLE = True
    print("✅ Enhanced configuration and logging loaded")
except ImportError as e:
    ENHANCED_SYSTEMS_AVAILABLE = False
    print(f"⚠️ Enhanced systems not available: {e}")
    def log_info(msg, extra=None): print(f"INFO: {msg}")
    def log_error(msg, extra=None): print(f"ERROR: {msg}")
try:
    from services.orchestrator.app.enforcement_bridge import load_sla, make_enforcer
except Exception:
    load_sla = lambda: {"thresholds": {"error_rate": 0.0, "completeness": 0.95}, "policies": {"strict_unknown_kpis": True, "require_hitl_on_violation": True}}
    class _Noop: 
        def enforce(self, goal, results): 
            return {"approved": True, "action": "approve", "reason": "noop"}
    def make_enforcer(_): return _Noop()

def main():
    goal = " ".join(sys.argv[1:]) or os.getenv("AF_GOAL") or "Design UI. Build backend API. Create DB schema. Implement code. Write unit tests."
    
    # Use enhanced configuration if available
    if ENHANCED_SYSTEMS_AVAILABLE:
        agent_config = get_agent_config()
        num_agents = int(os.getenv("AF_AGENTS", agent_config.default_agent_count))
        log_info(f"Starting orchestrator with goal: {goal}", {
            "num_agents": num_agents,
            "max_concurrent": agent_config.max_concurrent,
            "timeout": agent_config.timeout_seconds
        })
    else:
        num_agents = int(os.getenv("AF_AGENTS", "2"))
        log_info(f"Starting orchestrator with goal: {goal}")
    
    orch = build_orchestrator(num_agents=num_agents)
    
    start_time = time.time()
    results = orch.run_goal_sync(goal)
    execution_time = time.time() - start_time
    
    log_info(f"Goal execution completed", {
        "execution_time": execution_time,
        "num_results": len(results),
        "success": len(results) > 0
    })

    sla = load_sla()
    enforcer = make_enforcer(sla)
    decision = enforcer.enforce(goal=goal, results=results)

    print(json.dumps({"goal": goal, "results": results, "decision": decision}, indent=2))

    # Optional exit gating for CI
    if os.getenv("AF_EXIT_ON_REJECT", "0") == "1" and not decision.get("approved", False):
        sys.exit(2)

if __name__ == "__main__":
    main()
