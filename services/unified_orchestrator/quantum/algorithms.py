"""Advanced quantum-inspired optimization algorithms for AgentForge orchestration."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from .mathematical_foundations import (
    QuantumErrorMitigation,
    QuantumNoiseModel,
    QuantumStateVector,
)

log = logging.getLogger("quantum-algorithms")


@dataclass
class QuantumAssignmentMetadata:
    """Diagnostics collected during quantum-inspired assignment optimization."""

    path: str
    candidate_index: int
    iterations: int
    probabilities: np.ndarray
    score_distribution: np.ndarray
    refinement_steps: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "candidate_index": self.candidate_index,
            "iterations": self.iterations,
            "probabilities": self.probabilities.tolist(),
            "score_distribution": self.score_distribution.tolist(),
            "refinement_steps": self.refinement_steps,
            "diagnostics": self.diagnostics,
        }


class QuantumOptimizationSuite:
    """Collection of reusable quantum-inspired optimization utilities."""

    def __init__(
        self,
        max_candidates: int = 48,
        max_combinations: int = 256,
        amplification_bias: float = 0.25,
    ):
        self.max_candidates = max_candidates
        self.max_combinations = max_combinations
        self.amplification_bias = float(np.clip(amplification_bias, 0.05, 0.5))
        self.random_state = np.random.default_rng()
        self.background_noise = QuantumNoiseModel(
            depolarizing_prob=0.015,
            dephasing_prob=0.02,
            amplitude_damping_prob=0.01,
            name="scheduler-background-noise",
        )

    def score_execution_paths(
        self,
        execution_paths: Sequence[str],
        path_context: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        scores = []
        for path in execution_paths:
            context = path_context.get(path, {})
            base = context.get("capacity", 1.0)
            historical = context.get("historical_success", 1.0)
            coherence = context.get("coherence", 1.0)
            workload = context.get("workload_pressure", 1.0)
            entanglement = context.get("entanglement_support", 1.0)

            score = base
            score *= 0.5 + 0.5 * historical
            score *= 0.4 + 0.6 * coherence
            score *= 0.6 + 0.4 * workload
            score *= 0.5 + 0.5 * entanglement

            scores.append(max(score, 1e-9))

        return np.array(scores, dtype=float)

    def build_superposition(
        self,
        execution_paths: Sequence[str],
        path_scores: np.ndarray,
    ) -> np.ndarray:
        if len(execution_paths) != len(path_scores):
            raise ValueError("Execution paths and scores must align")
        probabilities = self._normalize(path_scores)
        amplitudes = np.sqrt(probabilities)

        marked = self._select_marked_indices(probabilities)
        iterations = self._amplification_iterations(len(amplitudes), len(marked))
        amplified = self.amplitude_amplification(amplitudes, marked, iterations)
        refined = self.variational_refinement(amplified, path_scores, steps=2)
        refined /= np.linalg.norm(refined)
        return refined

    def optimize_assignment(
        self,
        candidate_agents: Sequence[Any],
        team_size: int,
        scoring_callback: Callable[[Sequence[Any]], float],
        path: str,
    ) -> Tuple[List[Any], Optional[QuantumAssignmentMetadata]]:
        if team_size <= 0:
            return [], None

        candidate_agents = list(candidate_agents)
        if len(candidate_agents) <= team_size:
            return list(candidate_agents), None

        combinations = self._generate_candidate_combinations(candidate_agents, team_size)
        if not combinations:
            return [], None

        raw_scores = np.array(
            [max(scoring_callback(combo), 1e-9) for combo in combinations], dtype=float
        )
        probabilities = self._normalize(raw_scores)
        amplitudes = np.sqrt(probabilities)

        marked = self._select_marked_indices(probabilities)
        iterations = self._amplification_iterations(len(combinations), len(marked))
        amplified = self.amplitude_amplification(amplitudes, marked, iterations)
        refined = self.variational_refinement(amplified, raw_scores, steps=min(3, len(combinations)))

        qstate = QuantumStateVector(refined, [f"candidate_{i}" for i in range(len(combinations))])
        qstate = self.background_noise.apply(qstate)
        qstate = QuantumErrorMitigation.stabilize_state(qstate, minimum_purity=0.35)
        final_probabilities = qstate.get_probabilities()

        best_index = int(np.argmax(final_probabilities))
        metadata = QuantumAssignmentMetadata(
            path=path,
            candidate_index=best_index,
            iterations=iterations,
            probabilities=final_probabilities,
            score_distribution=raw_scores,
            refinement_steps=min(3, len(combinations)),
            diagnostics={
                "candidate_count": len(combinations),
                "marked_count": len(marked),
                "max_score": float(np.max(raw_scores)),
                "min_score": float(np.min(raw_scores)),
            },
        )

        return list(combinations[best_index]), metadata

    # --- Quantum primitives -------------------------------------------------

    def amplitude_amplification(
        self,
        amplitudes: np.ndarray,
        marked_indices: Sequence[int],
        iterations: Optional[int] = None,
    ) -> np.ndarray:
        state = np.array(amplitudes, dtype=complex)
        marked_indices = tuple(sorted(set(marked_indices)))
        if not marked_indices:
            return state / np.linalg.norm(state)

        iterations = iterations or self._amplification_iterations(len(state), len(marked_indices))
        if iterations <= 0:
            return state / np.linalg.norm(state)

        for _ in range(iterations):
            oracle = np.ones_like(state)
            oracle[list(marked_indices)] *= -1
            state = oracle * state
            mean = np.mean(state)
            state = 2 * mean - state

        norm = np.linalg.norm(state)
        if norm <= 0:
            return np.ones_like(state) / math.sqrt(len(state))
        return state / norm

    def variational_refinement(
        self,
        amplitudes: np.ndarray,
        scores: np.ndarray,
        steps: int = 2,
    ) -> np.ndarray:
        if steps <= 0:
            return amplitudes

        diag = np.diag(scores / (np.linalg.norm(scores) + 1e-9))
        time_step = 0.45
        unitary = scipy.linalg.expm(-1j * diag * time_step)
        state = amplitudes
        for _ in range(steps):
            state = unitary @ state
            state = state / (np.linalg.norm(state) + 1e-12)
        return state

    # --- Helper methods -----------------------------------------------------

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.array(values, dtype=float)
        values = np.clip(values, 1e-12, None)
        total = np.sum(values)
        if total <= 0:
            return np.ones_like(values) / len(values)
        return values / total

    def _select_marked_indices(self, probabilities: np.ndarray) -> List[int]:
        count = max(1, int(math.ceil(len(probabilities) * self.amplification_bias)))
        ranked = np.argsort(probabilities)[::-1]
        return list(ranked[:count])

    def _amplification_iterations(self, num_candidates: int, num_marked: int) -> int:
        if num_candidates <= 0 or num_marked <= 0:
            return 0
        theta = math.asin(math.sqrt(num_marked / num_candidates))
        return max(1, int(round((math.pi / (4 * theta)) - 0.5)))

    def _generate_candidate_combinations(
        self,
        agents: Sequence[Any],
        team_size: int,
    ) -> List[Tuple[Any, ...]]:
        agents = list(agents)
        if len(agents) < team_size:
            return []

        scored = sorted(
            agents,
            key=lambda agent: getattr(agent, "success_rate", 1.0) * getattr(agent, "coherence_level", 1.0),
            reverse=True,
        )
        truncated = scored[: min(len(scored), self.max_candidates)]
        combinations = list(itertools.combinations(truncated, team_size))

        if len(combinations) > self.max_combinations:
            selected_indices = self.random_state.choice(
                len(combinations), size=self.max_combinations, replace=False
            )
            combinations = [combinations[i] for i in selected_indices]

        return combinations

