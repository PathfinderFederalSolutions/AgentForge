from swarm.planner import Planner
from swarm.capabilities import fusion_caps  # noqa: F401
from swarm.lineage import persist_dag, list_job_events
from swarm.lineage import init_db  # ensure DB
from swarm.protocol.messages import DAGSpec
import os

# Reference to avoid linter removal (ensures capabilities registered)
_ = fusion_caps

def test_planner_makes_fusion_plan():
    p = Planner()
    plan = p.make_plan("Perform EO/IR fusion with conformal calibration")
    names = [s.capability for s in plan]
    assert "bayesian_fusion" in names or "conformal_validate" in names

def test_dag_determinism_and_persistence(tmp_path):
    init_db()
    p = Planner()
    seed = 42
    dag1 = p.make_dag("Perform EO/IR fusion with conformal calibration", seed=seed, budget_ms=1500)
    dag2 = p.make_dag("Perform EO/IR fusion with conformal calibration", seed=seed, budget_ms=1500)
    assert dag1.hash == dag2.hash
    assert len(dag1.nodes) == len(dag2.nodes)
    path = persist_dag(dag1)
    assert os.path.exists(path)
    # Validate file name includes hash
    assert dag1.hash in os.path.basename(path)

def test_dag_hash_stability():
    p = Planner()
    h1 = p.make_dag('Goal A', seed=123).hash
    h2 = p.make_dag('Goal A', seed=123).hash
    assert h1 == h2