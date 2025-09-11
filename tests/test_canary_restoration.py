from __future__ import annotations
import os, sys, json, tempfile, pathlib
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
from swarm.canary import router as canary  # type: ignore
from swarm.bkg import store as bkg_store  # type: ignore


def test_canary_restoration(monkeypatch):
    # Setup temporary policy path
    tmpdir = tempfile.TemporaryDirectory()
    policy_path = pathlib.Path(tmpdir.name) / 'router_policies.json'
    monkeypatch.setenv('ROUTER_POLICY_PATH', str(policy_path))
    # Write initial base policy
    base_policy = [{"name": "base", "task_regex": ".*", "priority": 0}]
    policy_path.write_text(json.dumps(base_policy), encoding='utf-8')
    # Persist BKG record with a different policy (simulating previous stable)
    prev_policy = [{"name":"stable","task_regex":"stable","priority":10}]
    bkg_store.update('router_policy', {"policy_json": json.dumps(prev_policy)}, results=['base_policy'])
    # Start canary and induce regression to force rollback / restoration
    canary.start_canary(target_fraction=0.1)
    for _ in range(60):
        canary.record_observation(False, 50, False, 0.99)
        canary.record_observation(True, 120, True, 0.70)
    res = canary.maybe_progress()
    # ensure rollback path executed eventually
    if res['phase'] != 'rollback':
        for _ in range(5):
            res = canary.maybe_progress()
            if res['phase'] == 'rollback':
                break
    assert res['phase'] == 'rollback'
    # Policy file should now contain stable policy
    restored = json.loads(policy_path.read_text(encoding='utf-8'))
    assert restored[0]['name'] == 'stable'
