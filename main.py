from orchestrator import build_orchestrator
from agents import AgentSwarm, MetaLearner

graph, router, healer = build_orchestrator()
swarm = AgentSwarm(num_agents=2)
learner = MetaLearner(num_agents=2)

# Test Orchestrator
task = "Write Python code for a web scraper"
routed_model = router.route(task)
print(f"Routed to: {routed_model}")

# Test Healing
sample_output = "print('hello"  # Buggy
fixed, trace = healer.heal(sample_output, task, routed_model)
print(f"Fixed: {fixed}\nTrace: {trace}")

# Test Swarm
tasks = [task, "Review code for errors"]
results, mem_usage = swarm.parallel_process(tasks)
print(f"Swarm results: {results}\nMemory usage: {mem_usage}")

# Test Meta-Learning
new_weights = learner.learn(results, task)
print(f"Updated weights: {new_weights}")