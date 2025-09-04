from dotenv import load_dotenv
load_dotenv()

import os, sys, json
from orchestrator import build_orchestrator

def main():
    goal = " ".join(sys.argv[1:]) or os.getenv("AF_GOAL") or "Design UI. Build backend API. Create DB schema. Implement code. Write unit tests."
    num_agents = int(os.getenv("AF_AGENTS", "2"))
    orch = build_orchestrator(num_agents=num_agents)
    results = orch.run_goal_sync(goal)
    print(json.dumps({"goal": goal, "results": results}, indent=2))

if __name__ == "__main__":
    main()
