# Fusion ROC / DET & EER Runbook

Purpose: Monitor calibration quality of fusion confidence via ROC / DET curves and Equal Error Rate (EER).

Key Metrics:
- fusion_roc_eer_bucket: Histogram of observed EER values (lower is better).
- fusion_latency_budget_violations_total: Count of fusion pipeline latency budget breaches.
- fusion_pipeline_latency_ms: Latency distribution of fusion pipeline.

Alert: FusionHighEqualErrorRate triggers when median (p50) EER > 0.3 over 15m.

Operational Actions:
1. Confirm data freshness: ensure fused tracks being produced (fusion_fused_tracks_total increasing).
2. If EER elevated:
   - Inspect distribution: scrape /metrics and compute p90. If >0.4 escalate.
   - Retrieve sample residuals (enable include_residuals flag in test or diagnostic path) and compute variance; high variance + high EER may indicate underfitting calibration.
   - Validate conformal alpha configuration. If alpha too large, intervals too wide causing misclassification threshold shift.
3. Mitigations:
   - Adjust calibration window size (increase sample count for residuals).
   - Recompute empirical residual distribution excluding obvious outliers.
   - Tune alpha downward (e.g., 0.1 -> 0.05) to tighten intervals if over-confident, or upward to reduce false positives.
4. Post-mitigation: watch EER histogram for downward trend (<0.25 target) over next 30m.

Latency Budget:
- Set env FUSION_LATENCY_BUDGET_MS (e.g., 150) to enforce soft cap. Violations increment fusion_latency_budget_violations_total.
- Investigate CPU throttling or I/O if violations spike.

Runbook Last Updated: 2025-09-10
