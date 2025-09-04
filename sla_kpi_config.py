# Shared configuration for SLAs and KPIs in AgentForge. All agents import this for access.
# Orchestrator enforces via validation hooks.

SLAS_KPIS = {
    'Memory Mesh': [
        {'capability_sub': 'Hot Memory Availability', 'sla': '99.999% uptime', 'kpi': 'Downtime incidents', 'measurement': 'Monitoring logs', 'threshold': '0 incidents per mission'},
        {'capability_sub': 'Read/Write Latency', 'sla': 'p99 < 10ms', 'kpi': 'Average ops/sec', 'measurement': 'Prometheus metrics', 'threshold': '100% compliance'},
        {'capability_sub': 'Data Provenance Integrity', 'sla': '100% tamper-evident', 'kpi': 'Audit delta matches', 'measurement': 'Vector clock checks', 'threshold': '100% match rate'},
        {'capability_sub': 'Classification Conformance', 'sla': '100% per-record enforcement', 'kpi': 'Misclassification errors', 'measurement': 'Automated scans', 'threshold': '0 errors'},
        # Add iterated additions: Provenance chain completeness, Time-travel replay accuracy, etc.
        {'capability_sub': 'Provenance Chain Completeness', 'sla': '100% linked to sources', 'kpi': 'Link checks', 'measurement': 'Audit scans', 'threshold': '100% completeness'},
        {'capability_sub': 'Time-Travel Replay Accuracy', 'sla': '100% deterministic', 'kpi': 'Reproduction matches', 'measurement': 'Replay tests', 'threshold': '100% match'},
    ],
    'Security and Governance': [
        {'capability_sub': 'Access Control Enforcement', 'sla': '100% ABAC/MAC compliance', 'kpi': 'Breach incidents', 'measurement': 'SIEM logs', 'threshold': '0 breaches'},
        {'capability_sub': 'Audit Log Integrity', 'sla': '100% tamper-evident', 'kpi': 'Log verification pass rate', 'measurement': 'Cryptographic hashes', 'threshold': '100% pass'},
        {'capability_sub': 'PII Minimization', 'sla': '100% scrubbing/redaction', 'kpi': 'Residual PII detections', 'measurement': 'Automated DLP scans', 'threshold': '0 detections'},
        {'capability_sub': 'Compliance Mapping', 'sla': '100% RMF/NIST controls', 'kpi': 'Audit findings', 'measurement': 'Continuous monitoring', 'threshold': '0 findings per cycle'},
        # Iterated: CDS pathway latency
        {'capability_sub': 'CDS Pathway Latency', 'sla': '<100ms', 'kpi': 'Flow conformance', 'measurement': 'Latency logs', 'threshold': '100% conformance'},
    ],
    'Scalability and Performance': [
        {'capability_sub': 'System Availability', 'sla': '99.99%', 'kpi': 'Uptime percentage', 'measurement': 'HA/DR tests', 'threshold': '100% during missions'},
        {'capability_sub': 'Hypothesis Throughput', 'sla': '1M+/sec sustained', 'kpi': 'Processed/sec', 'measurement': 'Ray/K8s metrics', 'threshold': 'No throttling events'},
        {'capability_sub': 'Ingest-to-Decision Latency', 'sla': 'p95 < 500ms', 'kpi': 'End-to-end time', 'measurement': 'Tracing (Jaeger)', 'threshold': '100% under threshold'},
        {'capability_sub': 'Replay Determinism', 'sla': '100% identical outputs', 'kpi': 'Seed/snapshot matches', 'measurement': 'Regression runs', 'threshold': '100% match'},
        # Iterated: Resource utilization
        {'capability_sub': 'Resource Utilization', 'sla': '<80% peak', 'kpi': 'Utilization levels', 'measurement': 'Kubernetes autoscaler logs', 'threshold': 'No overages'},
    ],
    'Probabilistic Rigor': [
        {'capability_sub': 'Calibration Error', 'sla': 'Brier/NLL/ACE < 0.01', 'kpi': 'Score per batch', 'measurement': 'Continuous monitoring', 'threshold': '0 deviations'},
        {'capability_sub': 'Coverage Guarantee', 'sla': '95% conformal', 'kpi': 'Abstention rate', 'measurement': 'Validation sets', 'threshold': '<5% abstentions'},
        {'capability_sub': 'Fusion Accuracy', 'sla': 'JPDA/MHT FPR@TPR < 0.001', 'kpi': 'ROC/DET curves', 'measurement': 'Simulated scenarios', 'threshold': '100% target hit'},
        {'capability_sub': 'Uncertainty Propagation', 'sla': '100% tracked', 'kpi': 'Variance checks', 'measurement': 'Propagation audits', 'threshold': '0 untracked vars'},
        # Iterated: Adversarial robustness
        {'capability_sub': 'Adversarial Robustness', 'sla': '<1% drop under spoofing', 'kpi': 'Performance delta', 'measurement': 'Chaos tests', 'threshold': '<1% drop'},
    ],
    'Dynamic Agent Lifecycle': [
        {'capability_sub': 'Spawn Latency', 'sla': '<100ms', 'kpi': 'Average time', 'measurement': 'Factory logs', 'threshold': '100% under'},
        {'capability_sub': 'Context Inheritance', 'sla': '100% accurate', 'kpi': 'Validation checks', 'measurement': 'Clock/ACL matches', 'threshold': '0 errors'},
        {'capability_sub': 'Task Idempotency', 'sla': '100% re-entrant', 'kpi': 'Retry success', 'measurement': 'Replay tests', 'threshold': '100%'},
        {'capability_sub': 'Autoscaling Efficiency', 'sla': 'Quota adherence 100%', 'kpi': 'Scaling events', 'measurement': 'K8s metrics', 'threshold': 'No violations'},
        # Iterated: Work stealing
        {'capability_sub': 'Work Stealing', 'sla': 'Backlog drain <1min', 'kpi': 'Drain time', 'measurement': 'Queue metrics', 'threshold': '<1min'},
    ],
    'Self-Healing Loop': [
        {'capability_sub': 'Violation Detection', 'sla': 'Latency <1s', 'kpi': 'Response time', 'measurement': 'Critic alerts', 'threshold': '100% timely'},
        {'capability_sub': 'Auto-Fix Success', 'sla': '99.9%', 'kpi': 'Resolution rate', 'measurement': 'Healer logs', 'threshold': '0 unresolved'},
        {'capability_sub': 'Regression Prevention', 'sla': '100% best-known-good', 'kpi': 'Canary pass rate', 'measurement': 'A/B tests', 'threshold': '100%'},
        {'capability_sub': 'Contradiction Resolution', 'sla': '100%', 'kpi': 'Check pass rate', 'measurement': 'Policy rules', 'threshold': '0 open issues'},
        # Iterated: Hardening against shifts
        {'capability_sub': 'Drift Hardening', 'sla': '<5% impact post-healing', 'kpi': 'Drift delta', 'measurement': 'Post-heal tests', 'threshold': '<5%'},
    ],
    'Human-in-the-Loop': [
        {'capability_sub': 'Gate Compliance', 'sla': '100% for thresholds', 'kpi': 'Invocation rate', 'measurement': 'Policy logs', 'threshold': '0 misses'},
        {'capability_sub': 'Evidence Bundle Quality', 'sla': '100% complete', 'kpi': 'Review scores', 'measurement': 'HITL feedback', 'threshold': '100% acceptance'},
        {'capability_sub': 'Feedback Integration', 'sla': '100% write-back', 'kpi': 'Update success', 'measurement': 'Mesh deltas', 'threshold': '0 failures'},
        {'capability_sub': 'Autonomy Rate', 'sla': '>95% when KPIs met', 'kpi': 'Auto-commit percentage', 'measurement': 'Mission stats', 'threshold': 'Maximize w/o errors'},
        # Iterated: Counterfactual inclusion
        {'capability_sub': 'Counterfactual Inclusion', 'sla': '100% for high-impact', 'kpi': 'Bundle checks', 'measurement': 'Audit', 'threshold': '100%'},
    ],
    'Synchronization and Visibility': [
        {'capability_sub': 'Propagation Latency', 'sla': '<50ms', 'kpi': 'Pub/sub time', 'measurement': 'Event logs', 'threshold': '100% under'},
        {'capability_sub': 'Consistency Guarantee', 'sla': '100% causal', 'kpi': 'Clock violations', 'measurement': 'Vector checks', 'threshold': '0 violations'},
        {'capability_sub': 'Visibility Coverage', 'sla': '100% assertions visible', 'kpi': 'Subscription tests', 'measurement': 'Agent polls', 'threshold': 'Full coverage'},
        {'capability_sub': 'Conflict Resolution', 'sla': '100% automated', 'kpi': 'Resolution rate', 'measurement': 'CRDT ops', 'threshold': '0 manual interventions'},
        # Iterated: Multi-mission isolation
        {'capability_sub': 'Multi-Mission Isolation', 'sla': '0% cross-scope leaks', 'kpi': 'Leak detections', 'measurement': 'Scope audits', 'threshold': '0 leaks'},
    ],
    'Hypothesis Parallelism': [
        {'capability_sub': 'Parallel Throughput', 'sla': '1M+/batch', 'kpi': 'Processed/batch', 'measurement': 'Ray metrics', 'threshold': 'No bottlenecks'},
        {'capability_sub': 'Scoring Accuracy', 'sla': '>99% posterior', 'kpi': 'Validation error', 'measurement': 'Ground truth sims', 'threshold': '<1% error'},
        {'capability_sub': 'Pruning Efficiency', 'sla': 'Utility threshold met 100%', 'kpi': 'Beam search yield', 'measurement': 'Prune logs', 'threshold': 'Optimal retention'},
        {'capability_sub': 'Branch Management', 'sla': '100% merge/split', 'kpi': 'Operation success', 'measurement': 'Graph audits', 'threshold': '0 orphans'},
        # Iterated: Hypothesis diversity
        {'capability_sub': 'Hypothesis Diversity', 'sla': 'Index >0.9', 'kpi': 'Diversity score', 'measurement': 'Index calc', 'threshold': '>0.9'},
    ],
}

def get_sla_kpi(capability, sub_capability):
    """Retrieve specific SLA/KPI by capability and sub."""
    for item in SLAS_KPIS.get(capability, []):
        if item['capability_sub'] == sub_capability:
            return item
    return None