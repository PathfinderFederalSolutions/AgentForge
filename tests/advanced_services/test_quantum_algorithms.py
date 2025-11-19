#!/usr/bin/env python3
import numpy as np


def test_grover_amplify_choice_prefers_top_score():
    from services.unified_orchestrator.quantum.algorithms import grover_amplify_choice
    scores = [0.1, 0.2, 0.3, 0.9, 0.05, 0.07]
    idx = grover_amplify_choice(scores, iterations=5, deterministic=True)
    assert idx == int(np.argmax(scores))


def test_qaoa_inspired_select_returns_k_indices():
    from services.unified_orchestrator.quantum.algorithms import qaoa_inspired_select
    rng = np.random.default_rng(123)
    scores = rng.random(20)
    k = 5
    sel = qaoa_inspired_select(scores, k=k, p=2, maxiter=20, seed=42)
    assert len(sel) == k
    assert len(set(sel)) == k
    assert all(0 <= i < len(scores) for i in sel)


