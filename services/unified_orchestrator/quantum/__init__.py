"""
Quantum Computing Module for Agent Orchestration
Revolutionary quantum algorithms for next-generation AI coordination

Complete Quantum Algorithm Suite:
- Grover's Search: O(√N) agent selection
- Quantum Annealing: Global optimization
- QAOA: Combinatorial optimization
- VQE: Ground state finding
- Quantum Machine Learning: QNN, QPCA, QRL, QBM
- Quantum Walks: Graph traversal
- QFT: Pattern recognition
- Quantum Error Correction: Fault tolerance
- Amplitude Amplification: Probability boost
- Quantum Gradient Descent: Optimized training
"""

from .mathematical_foundations import (
    QuantumStateVector,
    UnitaryTransformation,
    QuantumMeasurement,
    EntanglementMatrix,
    QuantumCoherenceTracker,
    QuantumDensityMatrix,
    QuantumNoiseModel,
    QuantumCircuit,
    QuantumGateLibrary,
    QuantumErrorMitigation
)
from .algorithms import QuantumOptimizationSuite, QuantumAssignmentMetadata

from .advanced_algorithms import (
    GroverSearchAlgorithm,
    QuantumAnnealingOptimizer,
    QuantumApproximateOptimization,
    VariationalQuantumEigensolver,
    QuantumErrorCorrection
)

from .quantum_machine_learning import (
    QuantumNeuralNetwork,
    QuantumPrincipalComponentAnalysis,
    QuantumReinforcementLearning,
    QuantumBoltzmannMachine
)

from .quantum_walks_and_optimization import (
    QuantumWalk,
    QuantumFourierTransform,
    AmplitudeAmplification,
    QuantumGradientDescent,
    QuantumCounting
)

from .quantum_advantage_benchmark import (
    QuantumAdvantageBenchmark,
    BenchmarkResult
)

from .quantum_integration_layer import (
    QuantumIntegrationLayer,
    QuantumCapabilities
)

__all__ = [
    # Mathematical Foundations
    "QuantumStateVector",
    "UnitaryTransformation",
    "QuantumMeasurement",
    "EntanglementMatrix",
    "QuantumCoherenceTracker",
    "QuantumDensityMatrix",
    "QuantumNoiseModel",
    "QuantumCircuit",
    "QuantumGateLibrary",
    "QuantumErrorMitigation",
    "QuantumOptimizationSuite",
    "QuantumAssignmentMetadata",
    
    # Advanced Algorithms
    "GroverSearchAlgorithm",
    "QuantumAnnealingOptimizer",
    "QuantumApproximateOptimization",
    "VariationalQuantumEigensolver",
    "QuantumErrorCorrection",
    
    # Machine Learning
    "QuantumNeuralNetwork",
    "QuantumPrincipalComponentAnalysis",
    "QuantumReinforcementLearning",
    "QuantumBoltzmannMachine",
    
    # Walks and Optimization
    "QuantumWalk",
    "QuantumFourierTransform",
    "AmplitudeAmplification",
    "QuantumGradientDescent",
    "QuantumCounting",
    
    # Benchmarking
    "QuantumAdvantageBenchmark",
    "BenchmarkResult",
    
    # Integration
    "QuantumIntegrationLayer",
    "QuantumCapabilities"
]

__version__ = "2.0.0"
__author__ = "AgentForge Quantum Team"
__description__ = "Production-ready quantum algorithms for AI agent orchestration"
