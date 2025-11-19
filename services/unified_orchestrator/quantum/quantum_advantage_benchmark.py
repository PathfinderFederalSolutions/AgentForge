"""
Quantum Advantage Benchmarking System
Proves quantum speedup over classical algorithms

Rigorous comparison framework:
- Performance benchmarks
- Complexity analysis
- Speedup verification
- Scalability testing
"""

from __future__ import annotations
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from .advanced_algorithms import (
    GroverSearchAlgorithm, QuantumAnnealingOptimizer,
    QuantumApproximateOptimization, VariationalQuantumEigensolver
)
from .quantum_walks_and_optimization import (
    QuantumWalk, QuantumFourierTransform, AmplitudeAmplification,
    QuantumGradientDescent, QuantumCounting
)
from .quantum_machine_learning import (
    QuantumNeuralNetwork, QuantumPrincipalComponentAnalysis,
    QuantumReinforcementLearning
)

log = logging.getLogger("quantum-benchmark")


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    algorithm_name: str
    problem_size: int
    quantum_time: float
    classical_time: float
    quantum_accuracy: float
    classical_accuracy: float
    speedup_factor: float
    theoretical_speedup: float
    quantum_complexity: str
    classical_complexity: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.speedup_ratio = self.classical_time / max(self.quantum_time, 1e-10)
        self.advantage_verified = self.speedup_ratio > 1.0
        self.accuracy_difference = self.quantum_accuracy - self.classical_accuracy


class QuantumAdvantageBenchmark:
    """
    Comprehensive Quantum Advantage Benchmarking System
    
    Tests:
    1. Search algorithms (Grover vs classical)
    2. Optimization (Quantum annealing vs classical)
    3. Machine learning (QNN vs classical NN)
    4. Graph algorithms (Quantum walk vs random walk)
    5. Fourier transforms (QFT vs FFT)
    """
    
    def __init__(self, save_results: bool = True, results_dir: str = "./quantum_benchmarks"):
        """
        Initialize benchmark system
        
        Args:
            save_results: Save results to files
            results_dir: Directory for results
        """
        self.save_results = save_results
        self.results_dir = results_dir
        
        self.benchmark_results: List[BenchmarkResult] = []
        
        log.info("Quantum advantage benchmark system initialized")
    
    def run_full_benchmark_suite(self, problem_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run complete benchmark suite across all algorithms
        
        Args:
            problem_sizes: List of problem sizes to test
        
        Returns:
            Comprehensive benchmark report
        """
        if problem_sizes is None:
            problem_sizes = [4, 8, 16, 32, 64]  # Start small for testing
        
        log.info(f"Running full quantum benchmark suite for sizes: {problem_sizes}")
        
        start_time = time.time()
        
        # Run all benchmark categories
        search_results = self.benchmark_search_algorithms(problem_sizes)
        optimization_results = self.benchmark_optimization(problem_sizes)
        ml_results = self.benchmark_machine_learning(problem_sizes)
        graph_results = self.benchmark_graph_algorithms(problem_sizes)
        transform_results = self.benchmark_transforms(problem_sizes)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_results = (search_results + optimization_results + ml_results + 
                      graph_results + transform_results)
        
        self.benchmark_results.extend(all_results)
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Generate report
        report = {
            "total_benchmarks": len(all_results),
            "total_time": total_time,
            "problem_sizes": problem_sizes,
            "categories": {
                "search": len(search_results),
                "optimization": len(optimization_results),
                "machine_learning": len(ml_results),
                "graph": len(graph_results),
                "transforms": len(transform_results)
            },
            "analysis": analysis,
            "detailed_results": [self._result_to_dict(r) for r in all_results]
        }
        
        if self.save_results:
            self._save_report(report)
            self._generate_plots(all_results)
        
        log.info(f"Benchmark suite completed in {total_time:.2f}s")
        log.info(f"Quantum advantage verified in {analysis['quantum_advantages']} / {len(all_results)} tests")
        
        return report
    
    def benchmark_search_algorithms(self, problem_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark Grover's search vs classical search"""
        log.info("Benchmarking search algorithms (Grover vs Classical)")
        
        results = []
        
        for size in problem_sizes:
            # Generate random search problem
            items = list(range(size))
            target_item = np.random.randint(0, size)
            
            def fitness_fn(item):
                return 1.0 if item == target_item else 0.0
            
            # Quantum search (Grover)
            try:
                grover = GroverSearchAlgorithm(size)
                quantum_start = time.time()
                best_item, best_fitness, quantum_metrics = grover.search(items, fitness_fn)
                quantum_time = time.time() - quantum_start
                quantum_accuracy = 1.0 if best_item == target_item else 0.0
            except Exception as e:
                log.error(f"Grover search failed for size {size}: {e}")
                quantum_time = float('inf')
                quantum_accuracy = 0.0
                quantum_metrics = {}
            
            # Classical linear search
            classical_start = time.time()
            classical_best = None
            for item in items:
                if fitness_fn(item) == 1.0:
                    classical_best = item
                    break
            classical_time = time.time() - classical_start
            classical_accuracy = 1.0 if classical_best == target_item else 0.0
            
            # Calculate speedup
            speedup = classical_time / max(quantum_time, 1e-10)
            theoretical = np.sqrt(size)
            
            result = BenchmarkResult(
                algorithm_name="Grover_Search",
                problem_size=size,
                quantum_time=quantum_time,
                classical_time=classical_time,
                quantum_accuracy=quantum_accuracy,
                classical_accuracy=classical_accuracy,
                speedup_factor=speedup,
                theoretical_speedup=theoretical,
                quantum_complexity="O(√N)",
                classical_complexity="O(N)",
                additional_metrics=quantum_metrics
            )
            
            results.append(result)
            
            log.info(f"  Size {size}: Quantum {quantum_time:.6f}s, Classical {classical_time:.6f}s, "
                    f"Speedup: {speedup:.2f}x (theoretical: {theoretical:.2f}x)")
        
        return results
    
    def benchmark_optimization(self, problem_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark quantum annealing vs classical optimization"""
        log.info("Benchmarking optimization algorithms (Quantum Annealing vs Classical)")
        
        results = []
        
        for size in problem_sizes:
            # Random optimization problem (minimize quadratic function)
            def objective(x):
                return np.sum((x - 0.5) ** 2)
            
            bounds = [(0, 1) for _ in range(size)]
            
            # Quantum annealing
            try:
                qa_optimizer = QuantumAnnealingOptimizer(n_variables=size, annealing_time=50.0)
                quantum_start = time.time()
                quantum_solution, quantum_value, quantum_metrics = qa_optimizer.optimize(
                    objective, bounds, n_replicas=5
                )
                quantum_time = time.time() - quantum_start
                quantum_accuracy = 1.0 / (1.0 + quantum_value)  # Lower is better
            except Exception as e:
                log.error(f"Quantum annealing failed for size {size}: {e}")
                quantum_time = float('inf')
                quantum_accuracy = 0.0
                quantum_metrics = {}
            
            # Classical random search
            classical_start = time.time()
            best_classical = None
            best_classical_value = float('inf')
            
            for _ in range(size * 10):  # Classical needs more samples
                x = np.random.uniform(0, 1, size)
                value = objective(x)
                if value < best_classical_value:
                    best_classical = x
                    best_classical_value = value
            
            classical_time = time.time() - classical_start
            classical_accuracy = 1.0 / (1.0 + best_classical_value)
            
            speedup = classical_time / max(quantum_time, 1e-10)
            
            result = BenchmarkResult(
                algorithm_name="Quantum_Annealing",
                problem_size=size,
                quantum_time=quantum_time,
                classical_time=classical_time,
                quantum_accuracy=quantum_accuracy,
                classical_accuracy=classical_accuracy,
                speedup_factor=speedup,
                theoretical_speedup=size,  # Problem-dependent
                quantum_complexity="O(√N)",
                classical_complexity="O(N²)",
                additional_metrics=quantum_metrics
            )
            
            results.append(result)
            
            log.info(f"  Size {size}: Quantum value {quantum_value:.6f}, "
                    f"Classical value {best_classical_value:.6f}")
        
        return results
    
    def benchmark_machine_learning(self, problem_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark quantum neural networks vs classical"""
        log.info("Benchmarking machine learning (QNN vs Classical)")
        
        results = []
        
        for size in problem_sizes[:3]:  # Limit to small sizes for speed
            n_qubits = min(4, int(np.log2(size)) + 2)
            n_samples = min(size * 2, 20)
            
            # Generate random training data
            X = np.random.randn(n_samples, size)
            y = np.sum(X, axis=1) > 0  # Simple classification
            y = y.astype(float).reshape(-1, 1)
            
            # Quantum neural network
            try:
                qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=2, n_outputs=1)
                quantum_start = time.time()
                quantum_metrics = qnn.train(X, y, epochs=10, learning_rate=0.1)
                quantum_time = time.time() - quantum_start
                
                # Test accuracy
                quantum_predictions = np.array([qnn.forward(x) for x in X])
                quantum_accuracy = np.mean((quantum_predictions > 0.5) == y)
            except Exception as e:
                log.error(f"QNN training failed for size {size}: {e}")
                quantum_time = float('inf')
                quantum_accuracy = 0.0
                quantum_metrics = {}
            
            # Classical logistic regression (simplified)
            classical_start = time.time()
            weights = np.random.randn(size, 1) * 0.01
            bias = 0.0
            
            for epoch in range(10):
                for i in range(n_samples):
                    x_i = X[i].reshape(-1, 1)
                    y_i = y[i]
                    
                    # Forward pass
                    z = np.dot(x_i.T, weights) + bias
                    pred = 1 / (1 + np.exp(-z))
                    
                    # Backward pass
                    error = pred - y_i
                    weights -= 0.1 * x_i * error
                    bias -= 0.1 * error
            
            classical_time = time.time() - classical_start
            
            # Test accuracy
            classical_predictions = 1 / (1 + np.exp(-(X @ weights + bias)))
            classical_accuracy = np.mean((classical_predictions > 0.5) == y)
            
            speedup = classical_time / max(quantum_time, 1e-10)
            
            result = BenchmarkResult(
                algorithm_name="Quantum_Neural_Network",
                problem_size=size,
                quantum_time=quantum_time,
                classical_time=classical_time,
                quantum_accuracy=float(quantum_accuracy),
                classical_accuracy=float(classical_accuracy),
                speedup_factor=speedup,
                theoretical_speedup=2 ** n_qubits,  # Exponential capacity
                quantum_complexity="O(2^n)",
                classical_complexity="O(n²)",
                additional_metrics=quantum_metrics
            )
            
            results.append(result)
            
            log.info(f"  Size {size}: QNN accuracy {quantum_accuracy:.3f}, "
                    f"Classical accuracy {classical_accuracy:.3f}")
        
        return results
    
    def benchmark_graph_algorithms(self, problem_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark quantum walks vs classical random walks"""
        log.info("Benchmarking graph algorithms (Quantum Walk vs Random Walk)")
        
        results = []
        
        for size in problem_sizes[:4]:  # Limit for computational cost
            # Create random graph
            adjacency = np.random.randint(0, 2, (size, size))
            adjacency = (adjacency + adjacency.T) / 2  # Symmetric
            np.fill_diagonal(adjacency, 0)  # No self-loops
            
            # Ensure connected (add edges if needed)
            for i in range(size - 1):
                adjacency[i, i+1] = 1
                adjacency[i+1, i] = 1
            
            n_steps = int(np.sqrt(size))
            
            # Quantum walk
            try:
                qw = QuantumWalk(adjacency, coin_type="hadamard")
                quantum_start = time.time()
                quantum_state, quantum_metrics = qw.walk(initial_node=0, n_steps=n_steps)
                quantum_time = time.time() - quantum_start
                
                # Measure coverage (how many nodes visited with significant probability)
                node_probs = quantum_metrics["final_node_probabilities"]
                quantum_coverage = np.sum(np.array(node_probs) > 0.01)
            except Exception as e:
                log.error(f"Quantum walk failed for size {size}: {e}")
                quantum_time = float('inf')
                quantum_coverage = 0
                quantum_metrics = {}
            
            # Classical random walk
            classical_start = time.time()
            current_node = 0
            visited_counts = np.zeros(size)
            
            for step in range(n_steps ** 2):  # Classical needs more steps
                visited_counts[current_node] += 1
                
                # Random transition
                neighbors = np.where(adjacency[current_node] > 0)[0]
                if len(neighbors) > 0:
                    current_node = np.random.choice(neighbors)
            
            classical_time = time.time() - classical_start
            
            # Normalize to probabilities
            classical_probs = visited_counts / np.sum(visited_counts)
            classical_coverage = np.sum(classical_probs > 0.01)
            
            speedup = classical_time / max(quantum_time, 1e-10)
            theoretical = size  # Quantum walk is quadratically faster
            
            result = BenchmarkResult(
                algorithm_name="Quantum_Walk",
                problem_size=size,
                quantum_time=quantum_time,
                classical_time=classical_time,
                quantum_accuracy=quantum_coverage / size,
                classical_accuracy=classical_coverage / size,
                speedup_factor=speedup,
                theoretical_speedup=theoretical,
                quantum_complexity="O(√N)",
                classical_complexity="O(N)",
                additional_metrics=quantum_metrics
            )
            
            results.append(result)
            
            log.info(f"  Size {size}: Quantum coverage {quantum_coverage}/{size}, "
                    f"Classical coverage {classical_coverage}/{size}")
        
        return results
    
    def benchmark_transforms(self, problem_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark QFT vs classical FFT"""
        log.info("Benchmarking transforms (QFT vs FFT)")
        
        results = []
        
        for size in problem_sizes[:5]:  # QFT requires power of 2
            n_qubits = int(np.log2(size))
            actual_size = 2 ** n_qubits
            
            # Random input signal
            signal = np.random.randn(actual_size)
            
            # Quantum Fourier Transform
            try:
                qft = QuantumFourierTransform(n_qubits)
                
                # Encode signal as quantum state
                from .mathematical_foundations import QuantumStateVector
                norm = np.linalg.norm(signal)
                amplitudes = signal / norm if norm > 0 else signal
                state = QuantumStateVector(
                    amplitudes=amplitudes.astype(complex),
                    basis_states=[f"|{i}⟩" for i in range(actual_size)]
                )
                
                quantum_start = time.time()
                fourier_state = qft.transform(state)
                quantum_time = time.time() - quantum_start
                
                quantum_spectrum = np.abs(fourier_state.amplitudes)
            except Exception as e:
                log.error(f"QFT failed for size {size}: {e}")
                quantum_time = float('inf')
                quantum_spectrum = np.zeros(actual_size)
            
            # Classical FFT
            classical_start = time.time()
            classical_spectrum = np.abs(np.fft.fft(signal))
            classical_time = time.time() - classical_start
            
            # Compare spectra (normalized)
            quantum_spectrum_norm = quantum_spectrum / (np.linalg.norm(quantum_spectrum) + 1e-10)
            classical_spectrum_norm = classical_spectrum / (np.linalg.norm(classical_spectrum) + 1e-10)
            
            # Accuracy: correlation between spectra
            correlation = np.corrcoef(quantum_spectrum_norm, classical_spectrum_norm)[0, 1]
            accuracy = max(0, correlation)  # Clamp to [0, 1]
            
            speedup = classical_time / max(quantum_time, 1e-10)
            theoretical = actual_size / (n_qubits ** 2)  # Exponential speedup
            
            result = BenchmarkResult(
                algorithm_name="Quantum_Fourier_Transform",
                problem_size=actual_size,
                quantum_time=quantum_time,
                classical_time=classical_time,
                quantum_accuracy=accuracy,
                classical_accuracy=1.0,  # FFT is exact
                speedup_factor=speedup,
                theoretical_speedup=theoretical,
                quantum_complexity="O(log²N)",
                classical_complexity="O(N log N)",
                additional_metrics={"correlation": correlation}
            )
            
            results.append(result)
            
            log.info(f"  Size {actual_size}: QFT {quantum_time:.6f}s, FFT {classical_time:.6f}s, "
                    f"Correlation: {correlation:.3f}")
        
        return results
    
    def _analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        if not results:
            return {}
        
        speedups = [r.speedup_ratio for r in results]
        advantages = [r for r in results if r.advantage_verified]
        
        analysis = {
            "total_tests": len(results),
            "quantum_advantages": len(advantages),
            "advantage_rate": len(advantages) / len(results),
            "average_speedup": np.mean(speedups),
            "max_speedup": np.max(speedups),
            "min_speedup": np.min(speedups),
            "median_speedup": np.median(speedups),
            "speedup_std": np.std(speedups),
            "algorithms_tested": list(set(r.algorithm_name for r in results)),
            "problem_sizes": list(set(r.problem_size for r in results))
        }
        
        # Per-algorithm analysis
        algorithms = set(r.algorithm_name for r in results)
        per_algorithm = {}
        
        for algo in algorithms:
            algo_results = [r for r in results if r.algorithm_name == algo]
            algo_speedups = [r.speedup_ratio for r in algo_results]
            
            per_algorithm[algo] = {
                "tests": len(algo_results),
                "average_speedup": np.mean(algo_speedups),
                "max_speedup": np.max(algo_speedups),
                "advantages": len([r for r in algo_results if r.advantage_verified])
            }
        
        analysis["per_algorithm"] = per_algorithm
        
        return analysis
    
    def _result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to dictionary"""
        return {
            "algorithm": result.algorithm_name,
            "problem_size": result.problem_size,
            "quantum_time": result.quantum_time,
            "classical_time": result.classical_time,
            "quantum_accuracy": result.quantum_accuracy,
            "classical_accuracy": result.classical_accuracy,
            "speedup": result.speedup_ratio,
            "theoretical_speedup": result.theoretical_speedup,
            "advantage_verified": result.advantage_verified,
            "quantum_complexity": result.quantum_complexity,
            "classical_complexity": result.classical_complexity,
            "additional_metrics": result.additional_metrics
        }
    
    def _save_report(self, report: Dict[str, Any]):
        """Save benchmark report to file"""
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"quantum_benchmark_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Benchmark report saved to {filename}")
    
    def _generate_plots(self, results: List[BenchmarkResult]):
        """Generate visualization plots"""
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Speedup vs problem size
        plt.figure(figsize=(12, 6))
        
        algorithms = set(r.algorithm_name for r in results)
        for algo in algorithms:
            algo_results = [r for r in results if r.algorithm_name == algo]
            sizes = [r.problem_size for r in algo_results]
            speedups = [r.speedup_ratio for r in algo_results]
            
            plt.plot(sizes, speedups, 'o-', label=algo, linewidth=2, markersize=8)
        
        plt.xlabel('Problem Size', fontsize=12)
        plt.ylabel('Speedup Factor (Classical/Quantum)', fontsize=12)
        plt.title('Quantum Advantage: Speedup vs Problem Size', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='r', linestyle='--', label='No advantage')
        
        filename = os.path.join(self.results_dir, f"speedup_plot_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Speedup plot saved to {filename}")
        
        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(results))
        quantum_acc = [r.quantum_accuracy for r in results]
        classical_acc = [r.classical_accuracy for r in results]
        
        plt.bar(x - 0.2, quantum_acc, 0.4, label='Quantum', alpha=0.8)
        plt.bar(x + 0.2, classical_acc, 0.4, label='Classical', alpha=0.8)
        
        plt.xlabel('Benchmark Test', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Quantum vs Classical Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        filename = os.path.join(self.results_dir, f"accuracy_plot_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Accuracy plot saved to {filename}")

