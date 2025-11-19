"""
Quantum Mathematical Foundations - Rigorous Implementation
Proper quantum mechanical formulations for AGI orchestration
"""

from __future__ import annotations
import math
import numpy as np
import scipy.linalg
from scipy.stats import entropy
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import logging
import time

log = logging.getLogger("quantum-math")


def is_unitary_matrix(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check whether a matrix is unitary within numerical tolerance."""
    matrix = np.array(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0], dtype=complex)
    product = matrix @ matrix.conj().T
    return np.allclose(product, identity, atol=tolerance)

@dataclass
class QuantumStateVector:
    """
    Proper quantum state vector implementation using complex probability amplitudes
    
    Based on quantum mechanical principles:
    - State vector |ψ⟩ = Σᵢ αᵢ|i⟩ where |αᵢ|² = probability of state i
    - Normalization: Σᵢ |αᵢ|² = 1
    - Complex amplitudes allow for quantum interference
    """
    amplitudes: np.ndarray  # Complex probability amplitudes
    basis_states: List[str]  # Basis state labels
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate and normalize quantum state vector"""
        if len(self.amplitudes) != len(self.basis_states):
            raise ValueError("Amplitudes and basis states must have same length")
        
        # Ensure complex dtype
        self.amplitudes = np.array(self.amplitudes, dtype=complex)
        
        # Normalize the state vector
        self._normalize()
        
    def _normalize(self):
        """Normalize quantum state vector to unit probability"""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-10:  # Avoid division by zero
            self.amplitudes /= norm
        else:
            # If all amplitudes are zero, create equal superposition
            n_states = len(self.amplitudes)
            self.amplitudes = np.ones(n_states, dtype=complex) / np.sqrt(n_states)
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities |αᵢ|²"""
        return np.abs(self.amplitudes) ** 2
    
    def get_phases(self) -> np.ndarray:
        """Get quantum phases arg(αᵢ)"""
        return np.angle(self.amplitudes)

    def dimension(self) -> int:
        """Return Hilbert space dimension for this state."""
        return len(self.amplitudes)

    def clone(self) -> 'QuantumStateVector':
        """Return a defensive copy of the quantum state."""
        return QuantumStateVector(self.amplitudes.copy(), self.basis_states.copy())

    def apply_unitary(self, unitary: np.ndarray):
        """Apply a unitary matrix directly to the state vector."""
        unitary = np.array(unitary, dtype=complex)
        if unitary.shape != (self.dimension(), self.dimension()):
            raise ValueError("Unitary matrix dimension does not align with state dimension")
        if not is_unitary_matrix(unitary):
            log.warning("Applied matrix is not strictly unitary; results may be unstable")
        self.amplitudes = unitary @ self.amplitudes
        self._normalize()

    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute ⟨ψ|O|ψ⟩ for a given observable."""
        observable = np.array(observable, dtype=complex)
        if observable.shape != (self.dimension(), self.dimension()):
            raise ValueError("Observable dimension does not match quantum state")
        return np.vdot(self.amplitudes, observable @ self.amplitudes)

    def purity(self) -> float:
        """Return purity Tr(ρ²) for the associated density matrix."""
        density = self.to_density_matrix()
        return float(np.real(np.trace(density.matrix @ density.matrix)))

    def to_density_matrix(self) -> 'QuantumDensityMatrix':
        """Construct the density matrix ρ = |ψ⟩⟨ψ|."""
        outer = np.outer(self.amplitudes, self.amplitudes.conj())
        return QuantumDensityMatrix(outer, self.basis_states)

    def mix_with(self, other: 'QuantumStateVector', weight: float) -> 'QuantumStateVector':
        """Blend this state with another state vector by convex combination."""
        if len(self.amplitudes) != len(other.amplitudes):
            raise ValueError("Quantum states must have the same dimension to mix")
        weight = float(np.clip(weight, 0.0, 1.0))
        mixed_amplitudes = (1 - weight) * self.amplitudes + weight * other.amplitudes
        mixed_state = QuantumStateVector(mixed_amplitudes, self.basis_states)
        return mixed_state
    
    def measure(self, measurement_operator: 'QuantumMeasurement') -> Tuple[str, float]:
        """
        Perform quantum measurement and collapse state
        
        Returns:
            (measured_state, probability)
        """
        probabilities = self.get_probabilities()
        
        # Apply measurement operator
        measured_probs = measurement_operator.apply(probabilities, self.amplitudes)
        
        # Quantum measurement - probabilistic collapse
        measured_index = np.random.choice(len(self.basis_states), p=measured_probs)
        measured_state = self.basis_states[measured_index]
        probability = measured_probs[measured_index]
        
        # Collapse state vector to measured state
        self.amplitudes = np.zeros(len(self.amplitudes), dtype=complex)
        self.amplitudes[measured_index] = 1.0
        
        return measured_state, probability
    
    def evolve(self, unitary_operator: 'UnitaryTransformation', time_step: float = 1.0):
        """
        Evolve quantum state using unitary transformation
        |ψ(t+dt)⟩ = U(dt)|ψ(t)⟩
        """
        evolution_matrix = unitary_operator.get_evolution_matrix(time_step)
        self.amplitudes = evolution_matrix @ self.amplitudes
        
    def entangle_with(self, other: 'QuantumStateVector') -> 'QuantumStateVector':
        """
        Create entangled state with another quantum system
        |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ (tensor product)
        """
        # Tensor product of amplitudes
        entangled_amplitudes = np.kron(self.amplitudes, other.amplitudes)
        
        # Tensor product of basis states
        entangled_basis = [
            f"{state1}⊗{state2}" 
            for state1 in self.basis_states 
            for state2 in other.basis_states
        ]
        
        return QuantumStateVector(
            amplitudes=entangled_amplitudes,
            basis_states=entangled_basis
        )
    
    def get_von_neumann_entropy(self) -> float:
        """
        Calculate von Neumann entropy S = -Tr(ρ log ρ)
        Measures quantum state purity
        """
        probabilities = self.get_probabilities()
        # Avoid log(0) by adding small epsilon
        probabilities = probabilities + 1e-12
        return -np.sum(probabilities * np.log2(probabilities))
    
    def fidelity(self, other: 'QuantumStateVector') -> float:
        """
        Calculate quantum fidelity F = |⟨ψ₁|ψ₂⟩|²
        Measures similarity between quantum states
        """
        if len(self.amplitudes) != len(other.amplitudes):
            return 0.0
        
        inner_product = np.vdot(self.amplitudes, other.amplitudes)
        return abs(inner_product) ** 2

class UnitaryTransformation:
    """
    Unitary transformation matrix for quantum state evolution
    
    Properties:
    - U†U = UU† = I (unitary condition)
    - det(U) = 1 (preserves probability)
    - Represents reversible quantum evolution
    """
    
    def __init__(self, hamiltonian: np.ndarray, name: str = ""):
        """
        Initialize from Hamiltonian matrix H
        Evolution: U(t) = exp(-iHt/ℏ) (ℏ = 1 in natural units)
        """
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self.name = name
        
        # Verify Hamiltonian is Hermitian
        if not np.allclose(self.hamiltonian, self.hamiltonian.conj().T):
            log.warning(f"Hamiltonian {name} is not Hermitian - may not preserve probability")
    
    def get_evolution_matrix(self, time_step: float) -> np.ndarray:
        """
        Get unitary evolution matrix U(t) = exp(-iHt)
        """
        # Matrix exponential of -i * H * t
        evolution_matrix = scipy.linalg.expm(-1j * self.hamiltonian * time_step)
        
        # Verify unitarity (for numerical stability)
        if not self._is_unitary(evolution_matrix):
            log.warning(f"Evolution matrix for {self.name} is not unitary")
        
        return evolution_matrix
    
    @staticmethod
    def _is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if matrix is unitary within tolerance."""
        return is_unitary_matrix(matrix, tolerance)

    @classmethod
    def from_unitary(cls, unitary: np.ndarray, name: str = "custom_unitary") -> 'UnitaryTransformation':
        """Construct a transformation directly from a unitary matrix."""
        unitary = np.array(unitary, dtype=complex)
        if not is_unitary_matrix(unitary):
            raise ValueError("Provided matrix is not unitary")
        hamiltonian = scipy.linalg.logm(unitary) / (-1j)
        hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2  # enforce Hermiticity
        return cls(hamiltonian, name)

    def apply(self, state: QuantumStateVector, time_step: float = 1.0) -> QuantumStateVector:
        """Apply the transformation to a quantum state vector."""
        evolution_matrix = self.get_evolution_matrix(time_step)
        state.apply_unitary(evolution_matrix)
        return state
    
    @classmethod
    def pauli_x_rotation(cls, angle: float) -> 'UnitaryTransformation':
        """Create X-rotation (bit flip) transformation"""
        hamiltonian = angle * np.array([[0, 1], [1, 0]], dtype=complex)
        return cls(hamiltonian, f"X_rotation({angle})")
    
    @classmethod
    def pauli_z_rotation(cls, angle: float) -> 'UnitaryTransformation':
        """Create Z-rotation (phase flip) transformation"""
        hamiltonian = angle * np.array([[1, 0], [0, -1]], dtype=complex)
        return cls(hamiltonian, f"Z_rotation({angle})")
    
    @classmethod
    def controlled_operation(cls, control_unitary: np.ndarray, target_unitary: np.ndarray) -> 'UnitaryTransformation':
        """Create controlled quantum operation"""
        # Controlled-U gate implementation
        n_control = control_unitary.shape[0]
        n_target = target_unitary.shape[0]
        
        # Create controlled Hamiltonian
        identity_target = np.eye(n_target)
        controlled_hamiltonian = np.kron(control_unitary, target_unitary) - np.kron(np.eye(n_control), identity_target)
        
        return cls(controlled_hamiltonian, "controlled_operation")

class QuantumMeasurement:
    """
    Quantum measurement operator implementing Born rule
    
    Measurement outcomes follow Born rule: P(outcome) = |⟨outcome|ψ⟩|²
    """
    
    def __init__(self, measurement_basis: List[np.ndarray], outcome_labels: List[str]):
        """
        Initialize measurement in given basis
        
        Args:
            measurement_basis: List of orthonormal basis vectors
            outcome_labels: Labels for measurement outcomes
        """
        self.measurement_basis = [np.array(basis, dtype=complex) for basis in measurement_basis]
        self.outcome_labels = outcome_labels
        
        # Verify orthonormality
        self._verify_orthonormal()
    
    def _verify_orthonormal(self):
        """Verify measurement basis is orthonormal"""
        n_basis = len(self.measurement_basis)
        
        for i in range(n_basis):
            # Check normalization
            norm = np.linalg.norm(self.measurement_basis[i])
            if not np.isclose(norm, 1.0):
                log.warning(f"Basis vector {i} is not normalized (norm={norm})")
            
            # Check orthogonality
            for j in range(i + 1, n_basis):
                overlap = np.vdot(self.measurement_basis[i], self.measurement_basis[j])
                if not np.isclose(overlap, 0.0):
                    log.warning(f"Basis vectors {i} and {j} are not orthogonal (overlap={overlap})")
    
    def apply(self, state_probabilities: np.ndarray, amplitudes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply measurement operator to get outcome probabilities
        
        For computational basis measurement, this is identity operation
        For general measurement, need to transform probabilities
        """
        if amplitudes is not None:
            try:
                transformed = []
                for basis_vector in self.measurement_basis:
                    if len(basis_vector) != len(amplitudes):
                        raise ValueError("Measurement basis and state amplitudes mismatch")
                    projected = np.vdot(basis_vector, amplitudes)
                    transformed.append(abs(projected) ** 2)
                probabilities = np.array(transformed, dtype=float)
            except Exception as exc:  # pragma: no cover - defensive fallback
                log.warning(f"Falling back to classical measurement due to error: {exc}")
                probabilities = state_probabilities
        else:
            probabilities = state_probabilities

        total = np.sum(probabilities)
        if total <= 0:
            n = len(probabilities)
            return np.ones(n) / n
        return probabilities / total
    
    @classmethod
    def computational_basis(cls, n_qubits: int) -> 'QuantumMeasurement':
        """Create computational basis measurement (|0⟩, |1⟩, ...)"""
        n_states = 2 ** n_qubits
        basis_vectors = []
        labels = []
        
        for i in range(n_states):
            # Create computational basis state |i⟩
            basis_vector = np.zeros(n_states, dtype=complex)
            basis_vector[i] = 1.0
            basis_vectors.append(basis_vector)
            
            # Binary representation label
            binary_label = format(i, f'0{n_qubits}b')
            labels.append(f"|{binary_label}⟩")
        
        return cls(basis_vectors, labels)

class EntanglementMatrix:
    """
    Quantum entanglement correlation matrix
    
    Tracks quantum correlations between agents using:
    - Mutual information I(A:B)
    - Entanglement entropy S(A|B)
    - Correlation coefficients
    """
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        
        # Initialize correlation matrices
        self.mutual_information = np.zeros((self.n_agents, self.n_agents))
        self.entanglement_entropy = np.zeros((self.n_agents, self.n_agents))
        self.correlation_coefficients = np.eye(self.n_agents)  # Start with identity
        
        # Track interaction history
        self.interaction_history: Dict[Tuple[str, str], List[float]] = {}
        self.last_updated = time.time()
    
    def update_correlation(self, agent1: str, agent2: str, correlation_strength: float):
        """Update correlation between two agents"""
        try:
            idx1 = self.agent_ids.index(agent1)
            idx2 = self.agent_ids.index(agent2)
        except ValueError:
            log.warning(f"Unknown agents in correlation update: {agent1}, {agent2}")
            return
        
        # Update correlation coefficient (symmetric)
        self.correlation_coefficients[idx1, idx2] = correlation_strength
        self.correlation_coefficients[idx2, idx1] = correlation_strength
        
        # Track interaction history
        pair_key = tuple(sorted([agent1, agent2]))
        if pair_key not in self.interaction_history:
            self.interaction_history[pair_key] = []
        
        self.interaction_history[pair_key].append(correlation_strength)
        
        # Keep only recent history (sliding window)
        if len(self.interaction_history[pair_key]) > 100:
            self.interaction_history[pair_key] = self.interaction_history[pair_key][-100:]
        
        # Calculate mutual information
        self._update_mutual_information(idx1, idx2)
        
        # Calculate entanglement entropy
        self._update_entanglement_entropy(idx1, idx2)
        
        self.last_updated = time.time()
    
    def _update_mutual_information(self, idx1: int, idx2: int):
        """Calculate mutual information I(A:B) = H(A) + H(B) - H(A,B)"""
        # Get correlation history for this pair
        agent1, agent2 = self.agent_ids[idx1], self.agent_ids[idx2]
        pair_key = tuple(sorted([agent1, agent2]))
        
        if pair_key not in self.interaction_history:
            return
        
        correlations = np.array(self.interaction_history[pair_key])
        if len(correlations) < 2:
            return
        
        # Convert correlations to probability distributions
        # Discretize correlation values into bins
        n_bins = 10
        hist1, _ = np.histogram(correlations, bins=n_bins, density=True)
        hist2, _ = np.histogram(correlations, bins=n_bins, density=True)
        
        # Joint distribution (simplified - assume independence for now)
        joint_hist = np.outer(hist1, hist2)
        joint_hist = joint_hist / np.sum(joint_hist)  # Normalize
        
        # Calculate entropies
        h_a = entropy(hist1 + 1e-12)  # Add small epsilon to avoid log(0)
        h_b = entropy(hist2 + 1e-12)
        h_ab = entropy(joint_hist.flatten() + 1e-12)
        
        # Mutual information
        mutual_info = h_a + h_b - h_ab
        self.mutual_information[idx1, idx2] = mutual_info
        self.mutual_information[idx2, idx1] = mutual_info
    
    def _update_entanglement_entropy(self, idx1: int, idx2: int):
        """Calculate entanglement entropy between two agents"""
        correlation = self.correlation_coefficients[idx1, idx2]
        
        # Convert correlation to entanglement entropy
        # For two-qubit system: S = -p log₂(p) - (1-p) log₂(1-p)
        # where p is derived from correlation strength
        p = (1 + abs(correlation)) / 2  # Map [-1,1] to [0,1]
        p = max(min(p, 1-1e-12), 1e-12)  # Avoid log(0)
        
        entanglement_entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        self.entanglement_entropy[idx1, idx2] = entanglement_entropy
        self.entanglement_entropy[idx2, idx1] = entanglement_entropy
    
    def get_strongly_entangled_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs of agents with strong entanglement"""
        strong_pairs = []
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                correlation = self.correlation_coefficients[i, j]
                if abs(correlation) > threshold:
                    strong_pairs.append((
                        self.agent_ids[i],
                        self.agent_ids[j],
                        correlation
                    ))
        
        return sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def get_entanglement_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive entanglement network statistics"""
        # Eigenvalue decomposition for network analysis
        eigenvalues, _ = np.linalg.eigh(self.correlation_coefficients)
        
        return {
            "n_agents": self.n_agents,
            "total_correlations": np.sum(np.abs(self.correlation_coefficients)),
            "max_correlation": np.max(np.abs(self.correlation_coefficients - np.eye(self.n_agents))),
            "mean_correlation": np.mean(np.abs(self.correlation_coefficients - np.eye(self.n_agents))),
            "spectral_radius": np.max(np.abs(eigenvalues)),
            "network_connectivity": np.sum(np.abs(self.correlation_coefficients) > 0.1) / (self.n_agents * (self.n_agents - 1)),
            "total_mutual_information": np.sum(self.mutual_information) / 2,  # Divide by 2 for symmetry
            "mean_entanglement_entropy": np.mean(self.entanglement_entropy),
            "last_updated": self.last_updated
        }

class QuantumCoherenceTracker:
    """
    Track and maintain quantum coherence across distributed agent network
    
    Implements proper decoherence modeling based on:
    - Environmental interaction
    - Information leakage
    - System-bath coupling
    """
    
    def __init__(self, decoherence_time: float = 100.0):
        """
        Initialize coherence tracker
        
        Args:
            decoherence_time: Characteristic decoherence time T₂ (seconds)
        """
        self.decoherence_time = decoherence_time
        self.coherence_states: Dict[str, QuantumStateVector] = {}
        self.environmental_coupling: Dict[str, float] = {}
        
        # Decoherence models
        self.decoherence_models = {
            "exponential": self._exponential_decoherence,
            "gaussian": self._gaussian_decoherence,
            "power_law": self._power_law_decoherence
        }
        
        self.start_time = time.time()
    
    def add_quantum_system(self, system_id: str, initial_state: QuantumStateVector, 
                          coupling_strength: float = 0.01):
        """Add quantum system to coherence tracking"""
        self.coherence_states[system_id] = initial_state
        self.environmental_coupling[system_id] = coupling_strength
    
    def update_coherence(self, dt: float = 1.0, model: str = "exponential"):
        """Update quantum coherence for all tracked systems"""
        current_time = time.time() - self.start_time
        
        for system_id, state in self.coherence_states.items():
            coupling = self.environmental_coupling[system_id]
            
            # Apply decoherence model
            decoherence_factor = self.decoherence_models[model](current_time, coupling)
            
            # Modify quantum state amplitudes
            self._apply_decoherence(state, decoherence_factor)
    
    def _exponential_decoherence(self, time: float, coupling: float) -> float:
        """Exponential decoherence: exp(-t/T₂)"""
        effective_decoherence_time = self.decoherence_time / (1 + coupling)
        return np.exp(-time / effective_decoherence_time)
    
    def _gaussian_decoherence(self, time: float, coupling: float) -> float:
        """Gaussian decoherence: exp(-(t/T₂)²)"""
        effective_decoherence_time = self.decoherence_time / (1 + coupling)
        return np.exp(-(time / effective_decoherence_time) ** 2)
    
    def _power_law_decoherence(self, time: float, coupling: float) -> float:
        """Power law decoherence: (1 + t/T₂)^(-α)"""
        alpha = 2.0  # Decoherence exponent
        effective_decoherence_time = self.decoherence_time / (1 + coupling)
        return (1 + time / effective_decoherence_time) ** (-alpha)
    
    def _apply_decoherence(self, state: QuantumStateVector, decoherence_factor: float):
        """Apply decoherence to quantum state"""
        # Decoherence reduces off-diagonal elements of density matrix
        # For pure states, this means reducing coherence between basis states
        
        # Get current phases
        phases = state.get_phases()
        
        # Apply phase decoherence
        random_phases = np.random.normal(0, 1 - decoherence_factor, len(phases))
        new_phases = phases + random_phases
        
        # Update amplitudes with new phases
        magnitudes = np.abs(state.amplitudes)
        state.amplitudes = magnitudes * np.exp(1j * new_phases)
        
        # Renormalize
        state._normalize()
    
    def get_global_coherence(self) -> float:
        """Calculate global coherence across all quantum systems"""
        if not self.coherence_states:
            return 0.0
        
        total_coherence = 0.0
        for state in self.coherence_states.values():
            # Coherence measure: 1 - von Neumann entropy / log(d)
            entropy = state.get_von_neumann_entropy()
            max_entropy = np.log2(len(state.amplitudes))
            coherence = 1 - entropy / max_entropy if max_entropy > 0 else 0
            total_coherence += coherence
        
        return total_coherence / len(self.coherence_states)


class QuantumDensityMatrix:
    """Density matrix formalism for mixed quantum states."""

    def __init__(self, matrix: np.ndarray, basis_states: Sequence[str]):
        self.matrix = np.array(matrix, dtype=complex)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Density matrix must be square")
        if self.matrix.shape[0] != len(basis_states):
            raise ValueError("Basis size must match density matrix dimension")
        self.basis_states = list(basis_states)
        self._enforce_trace_one()

    def _enforce_trace_one(self):
        trace = np.trace(self.matrix)
        if abs(trace) < 1e-12:
            dim = self.matrix.shape[0]
            self.matrix = np.eye(dim, dtype=complex) / dim
        else:
            self.matrix = self.matrix / trace

    def apply_unitary(self, unitary: np.ndarray):
        unitary = np.array(unitary, dtype=complex)
        if unitary.shape != self.matrix.shape:
            raise ValueError("Unitary dimension does not match density matrix")
        self.matrix = unitary @ self.matrix @ unitary.conj().T
        self._enforce_trace_one()

    def apply_kraus(self, kraus_operators: Sequence[np.ndarray]):
        new_matrix = np.zeros_like(self.matrix)
        for operator in kraus_operators:
            op = np.array(operator, dtype=complex)
            if op.shape != self.matrix.shape:
                raise ValueError("Kraus operator dimension mismatch")
            new_matrix += op @ self.matrix @ op.conj().T
        self.matrix = new_matrix
        self._enforce_trace_one()

    def get_purity(self) -> float:
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def von_neumann_entropy(self) -> float:
        eigenvalues = np.clip(np.real(np.linalg.eigvals(self.matrix)), 1e-12, None)
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def as_array(self) -> np.ndarray:
        return self.matrix

    def to_state_vector(self) -> QuantumStateVector:
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        dominant_index = int(np.argmax(eigenvalues))
        dominant_state = eigenvectors[:, dominant_index]
        return QuantumStateVector(dominant_state, self.basis_states)


class QuantumNoiseModel:
    """Parameterized noise model for simulated quantum dynamics."""

    def __init__(self, depolarizing_prob: float = 0.0, dephasing_prob: float = 0.0,
                 amplitude_damping_prob: float = 0.0, name: str = "custom_noise"):
        self.depolarizing_prob = float(np.clip(depolarizing_prob, 0.0, 1.0))
        self.dephasing_prob = float(np.clip(dephasing_prob, 0.0, 1.0))
        self.amplitude_damping_prob = float(np.clip(amplitude_damping_prob, 0.0, 1.0))
        self.name = name

    def apply_to_density(self, density: QuantumDensityMatrix):
        dim = density.matrix.shape[0]
        identity = np.eye(dim, dtype=complex) / dim

        if self.depolarizing_prob > 0:
            density.matrix = ((1 - self.depolarizing_prob) * density.matrix +
                              self.depolarizing_prob * identity)

        if self.dephasing_prob > 0:
            diagonal = np.diag(np.diag(density.matrix))
            density.matrix = ((1 - self.dephasing_prob) * density.matrix +
                              self.dephasing_prob * diagonal)

        if self.amplitude_damping_prob > 0 and dim > 1:
            damping = self.amplitude_damping_prob
            damped = density.matrix.copy()
            for i in range(1, dim):
                damped[0, 0] += damping * density.matrix[i, i]
                damped[i, i] *= (1 - damping)
                damped[0, i] *= math.sqrt(1 - damping)
                damped[i, 0] *= math.sqrt(1 - damping)
            density.matrix = damped

        density._enforce_trace_one()

    def apply(self, state: QuantumStateVector) -> QuantumStateVector:
        density = state.to_density_matrix()
        self.apply_to_density(density)
        return density.to_state_vector()


class QuantumCircuit:
    """Composable unitary circuit with optional noise layers for simulation."""

    def __init__(self, dimension: int, label: str = ""):
        self.dimension = dimension
        self.label = label or f"circuit-{int(time.time() * 1000)}"
        self.operations: List[Tuple[str, Any]] = []

    def add_unitary(self, unitary: np.ndarray, description: str = ""):
        unitary = np.array(unitary, dtype=complex)
        if unitary.shape != (self.dimension, self.dimension):
            raise ValueError("Unitary dimension does not match circuit dimension")
        self.operations.append(("unitary", unitary, description))

    def add_noise(self, noise_model: QuantumNoiseModel):
        self.operations.append(("noise", noise_model, noise_model.name))

    def run(self, state: QuantumStateVector) -> QuantumStateVector:
        if state.dimension() != self.dimension:
            raise ValueError("State dimension mismatch for circuit execution")
        for op_type, payload, _ in self.operations:
            if op_type == "unitary":
                state.apply_unitary(payload)
            elif op_type == "noise":
                state = payload.apply(state)
        return state


class QuantumGateLibrary:
    """Collection of reusable quantum gates for circuit construction."""

    @staticmethod
    def hadamard() -> np.ndarray:
        return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

    @staticmethod
    def phase(theta: float) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

    @staticmethod
    def pauli_x() -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def pauli_y() -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def pauli_z() -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def rotation(axis: str, angle: float) -> np.ndarray:
        if axis.lower() == "x":
            return QuantumGateLibrary.pauli_x_rotation(angle)
        if axis.lower() == "y":
            return QuantumGateLibrary.pauli_y_rotation(angle)
        if axis.lower() == "z":
            return QuantumGateLibrary.pauli_z_rotation(angle)
        raise ValueError(f"Unknown rotation axis: {axis}")

    @staticmethod
    def pauli_x_rotation(angle: float) -> np.ndarray:
        return scipy.linalg.expm(-1j * angle / 2 * QuantumGateLibrary.pauli_x())

    @staticmethod
    def pauli_y_rotation(angle: float) -> np.ndarray:
        return scipy.linalg.expm(-1j * angle / 2 * QuantumGateLibrary.pauli_y())

    @staticmethod
    def pauli_z_rotation(angle: float) -> np.ndarray:
        return scipy.linalg.expm(-1j * angle / 2 * QuantumGateLibrary.pauli_z())


class QuantumErrorMitigation:
    """Simple error mitigation utilities for simulated quantum states."""

    @staticmethod
    def stabilize_state(state: QuantumStateVector, minimum_purity: float = 0.4) -> QuantumStateVector:
        density = state.to_density_matrix()
        purity = density.get_purity()
        if purity < minimum_purity:
            blend = np.clip(minimum_purity - purity, 0.0, 0.5)
            dim = density.matrix.shape[0]
            identity = np.eye(dim, dtype=complex) / dim
            density.matrix = (1 - blend) * density.matrix + blend * identity
            return density.to_state_vector()
        return state

    @staticmethod
    def mitigate_readout_errors(probabilities: np.ndarray, confusion_matrix: np.ndarray) -> np.ndarray:
        confusion_matrix = np.array(confusion_matrix, dtype=float)
        probabilities = np.array(probabilities, dtype=float)
        if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
            raise ValueError("Confusion matrix must be square")
        try:
            corrected = np.linalg.pinv(confusion_matrix) @ probabilities
        except np.linalg.LinAlgError:
            log.warning("Failed to invert confusion matrix; returning original probabilities")
            corrected = probabilities
        corrected = np.clip(corrected, 0.0, None)
        total = corrected.sum()
        if total <= 0:
            return np.ones_like(corrected) / len(corrected)
        return corrected / total
