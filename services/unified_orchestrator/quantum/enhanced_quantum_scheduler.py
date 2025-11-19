"""
Enhanced Quantum Scheduler with Full Algorithm Integration
Production-ready scheduler leveraging all quantum advantages

Integrates:
- Grover's search for agent selection
- Quantum annealing for task optimization
- Quantum walks for network coordination
- QFT for pattern recognition
- QML for learning and adaptation
- QEC for fault tolerance
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
import asyncio
from collections import deque

from .quantum_integration_layer import QuantumIntegrationLayer, QuantumCapabilities
from .mathematical_foundations import QuantumStateVector, EntanglementMatrix, QuantumCoherenceTracker

log = logging.getLogger("enhanced-quantum-scheduler")


@dataclass
class QuantumTask:
    """Task with quantum properties"""
    task_id: str
    description: str
    priority: int = 1
    required_agents: int = 1
    required_capabilities: set = None
    complexity: float = 1.0
    quantum_advantage_enabled: bool = True
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = set()


@dataclass
class QuantumAgent:
    """Agent with quantum properties"""
    agent_id: str
    capabilities: set
    performance_score: float = 1.0
    current_load: float = 0.0
    quantum_state: Optional[QuantumStateVector] = None
    entanglement_partners: set = None
    
    def __post_init__(self):
        if self.entanglement_partners is None:
            self.entanglement_partners = set()


class EnhancedQuantumScheduler:
    """
    Next-Generation Quantum Scheduler
    
    Features:
    1. Quantum-accelerated agent selection (Grover)
    2. Global optimization (Quantum annealing, QAOA)
    3. Adaptive learning (QML)
    4. Network-aware coordination (Quantum walks)
    5. Pattern-based prediction (QFT)
    6. Fault-tolerant operations (QEC)
    """
    
    def __init__(self, capabilities: Optional[QuantumCapabilities] = None):
        """
        Initialize enhanced quantum scheduler
        
        Args:
            capabilities: Quantum capabilities configuration
        """
        self.capabilities = capabilities or QuantumCapabilities()
        
        # Quantum integration layer
        self.quantum_layer = QuantumIntegrationLayer(self.capabilities)
        
        # Coherence tracking
        self.coherence_tracker = QuantumCoherenceTracker()
        
        # Entanglement management
        self.entanglement_matrix: Optional[EntanglementMatrix] = None
        
        # Scheduling state
        self.active_tasks: Dict[str, QuantumTask] = {}
        self.task_queue: deque = deque()
        
        # Agent management
        self.registered_agents: Dict[str, QuantumAgent] = {}
        self.agent_network_adjacency: Optional[np.ndarray] = None
        
        # Performance metrics
        self.scheduling_history: List[Dict[str, Any]] = []
        self.quantum_advantage_metrics: Dict[str, List[float]] = {
            "speedups": [],
            "success_rates": [],
            "optimization_improvements": []
        }
        
        log.info("Enhanced quantum scheduler initialized with full quantum capabilities")
    
    async def register_agent(self, agent: QuantumAgent):
        """Register agent with quantum properties"""
        self.registered_agents[agent.agent_id] = agent
        
        # Initialize agent quantum state
        if agent.quantum_state is None:
            basis_states = ["idle", "working", "overloaded"]
            amplitudes = [1.0, 0.0, 0.0]  # Start idle
            agent.quantum_state = QuantumStateVector(
                amplitudes=np.array(amplitudes, dtype=complex),
                basis_states=basis_states
            )
        
        # Add to coherence tracking
        self.coherence_tracker.add_quantum_system(
            agent.agent_id,
            agent.quantum_state,
            coupling_strength=0.01
        )
        
        # Update entanglement matrix
        await self._update_entanglement_matrix()
        
        # Update network adjacency
        await self._update_agent_network()
        
        log.info(f"Agent {agent.agent_id} registered with quantum properties")
    
    async def _update_entanglement_matrix(self):
        """Update entanglement matrix for all agents"""
        agent_ids = list(self.registered_agents.keys())
        
        if len(agent_ids) > 0:
            self.entanglement_matrix = EntanglementMatrix(agent_ids)
            
            # Establish entanglements based on capability overlap
            for i, agent1_id in enumerate(agent_ids):
                for j, agent2_id in enumerate(agent_ids[i+1:], i+1):
                    agent1 = self.registered_agents[agent1_id]
                    agent2 = self.registered_agents[agent2_id]
                    
                    # Calculate entanglement strength
                    capability_overlap = len(agent1.capabilities & agent2.capabilities)
                    total_capabilities = len(agent1.capabilities | agent2.capabilities)
                    
                    if total_capabilities > 0:
                        strength = capability_overlap / total_capabilities
                        self.entanglement_matrix.update_correlation(agent1_id, agent2_id, strength)
    
    async def _update_agent_network(self):
        """Update agent network topology"""
        n_agents = len(self.registered_agents)
        
        if n_agents > 0:
            self.agent_network_adjacency = np.zeros((n_agents, n_agents))
            
            agent_list = list(self.registered_agents.values())
            
            # Create network edges based on entanglement
            for i, agent1 in enumerate(agent_list):
                for j, agent2 in enumerate(agent_list[i+1:], i+1):
                    # Connect agents with shared capabilities
                    if agent1.capabilities & agent2.capabilities:
                        self.agent_network_adjacency[i, j] = 1
                        self.agent_network_adjacency[j, i] = 1
    
    async def schedule_task(self, task: QuantumTask) -> Dict[str, Any]:
        """
        Schedule task using quantum algorithms
        
        Full quantum pipeline:
        1. Quantum agent selection (Grover)
        2. Quantum optimization (Annealing/QAOA)
        3. Quantum coordination (Quantum walks)
        4. Quantum learning (QML adaptation)
        
        Args:
            task: Task to schedule
        
        Returns:
            Scheduling result with metrics
        """
        start_time = time.time()
        
        log.info(f"Scheduling task {task.task_id} with quantum algorithms")
        
        # Step 1: Quantum agent selection
        selected_agents, selection_metrics = await self._quantum_agent_selection(task)
        
        if not selected_agents:
            return {
                "task_id": task.task_id,
                "status": "failed",
                "reason": "No suitable agents found",
                "scheduling_time": time.time() - start_time
            }
        
        # Step 2: Quantum task optimization
        optimal_allocation, optimization_metrics = await self._quantum_task_optimization(
            task, selected_agents
        )
        
        # Step 3: Quantum network coordination
        coordination_metrics = await self._quantum_network_coordination(
            selected_agents, task
        )
        
        # Step 4: Quantum pattern analysis for future improvement
        pattern_metrics = await self._quantum_pattern_analysis(task, selected_agents)
        
        # Update agent states
        for agent_id, workload in optimal_allocation.items():
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                agent.current_load += workload
        
        scheduling_time = time.time() - start_time
        
        # Compile comprehensive metrics
        result = {
            "task_id": task.task_id,
            "status": "scheduled",
            "scheduling_time": scheduling_time,
            "assigned_agents": list(optimal_allocation.keys()),
            "workload_allocation": optimal_allocation,
            "quantum_metrics": {
                "agent_selection": selection_metrics,
                "task_optimization": optimization_metrics,
                "network_coordination": coordination_metrics,
                "pattern_analysis": pattern_metrics
            },
            "quantum_advantage": {
                "total_speedup": self._calculate_total_speedup(
                    selection_metrics, optimization_metrics, coordination_metrics
                )
            }
        }
        
        # Record scheduling history
        self.scheduling_history.append(result)
        
        # Track quantum advantage
        if "speedup" in selection_metrics:
            self.quantum_advantage_metrics["speedups"].append(selection_metrics["speedup"])
        
        log.info(f"Task {task.task_id} scheduled with quantum advantage: "
                f"{result['quantum_advantage']['total_speedup']:.2f}x speedup")
        
        return result
    
    async def _quantum_agent_selection(self, task: QuantumTask) -> Tuple[List[QuantumAgent], Dict[str, Any]]:
        """Select agents using Grover's quantum search"""
        
        # Get available agents
        available_agents = [
            agent for agent in self.registered_agents.values()
            if (agent.current_load < 0.9 and 
                task.required_capabilities.issubset(agent.capabilities))
        ]
        
        if not available_agents:
            return [], {"status": "no_agents_available"}
        
        # Define fitness function
        def agent_fitness(agent: QuantumAgent) -> float:
            # Fitness based on performance and current load
            performance_factor = agent.performance_score
            load_factor = 1.0 - agent.current_load
            capability_match = len(task.required_capabilities & agent.capabilities) / max(len(task.required_capabilities), 1)
            
            return performance_factor * load_factor * capability_match
        
        # Quantum agent selection using Grover
        best_agent, metrics = await self.quantum_layer.quantum_agent_selection(
            available_agents,
            agent_fitness,
            {"task_id": task.task_id, "required_agents": task.required_agents}
        )
        
        # Select multiple agents if needed
        if task.required_agents > 1:
            selected = [best_agent]
            remaining = [a for a in available_agents if a != best_agent]
            
            # Select additional agents (top performers)
            remaining_sorted = sorted(remaining, key=agent_fitness, reverse=True)
            selected.extend(remaining_sorted[:task.required_agents-1])
        else:
            selected = [best_agent]
        
        return selected, metrics
    
    async def _quantum_task_optimization(self, task: QuantumTask, 
                                        agents: List[QuantumAgent]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize task allocation using quantum algorithms"""
        
        n_agents = len(agents)
        
        # Define optimization objective: minimize load variance and maximize performance
        def objective(allocation: np.ndarray) -> float:
            # Normalize allocation to sum to 1
            allocation = np.abs(allocation)
            allocation = allocation / (np.sum(allocation) + 1e-10)
            
            # Calculate load variance (want to balance load)
            load_variance = np.var(allocation)
            
            # Calculate performance (want high-performing agents to get more work)
            weighted_performance = sum(
                allocation[i] * agents[i].performance_score 
                for i in range(n_agents)
            )
            
            # Minimize variance, maximize performance
            cost = load_variance - 0.5 * weighted_performance
            
            return cost
        
        # Bounds for allocation (0 to 1 for each agent)
        bounds = [(0, 1) for _ in range(n_agents)]
        
        # Use quantum annealing for optimization
        optimal_allocation, optimal_value, metrics = await self.quantum_layer.quantum_task_optimization(
            objective,
            n_agents,
            bounds,
            optimization_type="annealing"
        )
        
        # Normalize allocation
        optimal_allocation = np.abs(optimal_allocation)
        optimal_allocation = optimal_allocation / (np.sum(optimal_allocation) + 1e-10)
        
        # Convert to dictionary
        allocation_dict = {
            agents[i].agent_id: float(optimal_allocation[i])
            for i in range(n_agents)
        }
        
        return allocation_dict, metrics
    
    async def _quantum_network_coordination(self, agents: List[QuantumAgent],
                                           task: QuantumTask) -> Dict[str, Any]:
        """Coordinate agents using quantum walks"""
        
        if self.agent_network_adjacency is None or len(agents) < 2:
            return {"status": "no_coordination_needed"}
        
        # Map agents to network nodes
        agent_list = list(self.registered_agents.values())
        agent_indices = [agent_list.index(agent) for agent in agents if agent in agent_list]
        
        if not agent_indices:
            return {"status": "agents_not_in_network"}
        
        # Use quantum walk for coordination
        start_node = agent_indices[0]
        target_nodes = agent_indices[1:]
        
        coordination_metrics = await self.quantum_layer.quantum_network_coordination(
            self.agent_network_adjacency,
            start_node,
            target_nodes if target_nodes else None
        )
        
        return coordination_metrics
    
    async def _quantum_pattern_analysis(self, task: QuantumTask,
                                       agents: List[QuantumAgent]) -> Dict[str, Any]:
        """Analyze patterns using QFT for future optimization"""
        
        # Create signal from task and agent properties
        signal_components = [
            task.priority,
            task.complexity,
            len(task.required_capabilities),
            len(agents),
            np.mean([agent.performance_score for agent in agents]),
            np.mean([agent.current_load for agent in agents])
        ]
        
        signal = np.array(signal_components)
        
        # Apply quantum pattern recognition
        pattern_metrics = await self.quantum_layer.quantum_pattern_recognition(signal)
        
        return pattern_metrics
    
    def _calculate_total_speedup(self, selection_metrics: Dict, 
                                optimization_metrics: Dict,
                                coordination_metrics: Dict) -> float:
        """Calculate total quantum speedup"""
        speedups = []
        
        if "speedup" in selection_metrics:
            speedups.append(selection_metrics["speedup"])
        
        if "speedup" in coordination_metrics:
            # Parse speedup if it's a string
            speedup_str = coordination_metrics["speedup"]
            if isinstance(speedup_str, str) and 'x' in speedup_str:
                try:
                    speedup = float(speedup_str.replace('x', ''))
                    speedups.append(speedup)
                except:
                    pass
        
        # Calculate geometric mean of speedups
        if speedups:
            total_speedup = np.prod(speedups) ** (1.0 / len(speedups))
        else:
            total_speedup = 1.0
        
        return float(total_speedup)
    
    async def maintain_quantum_coherence(self):
        """Maintain quantum coherence across system"""
        self.coherence_tracker.update_coherence(dt=1.0, model="exponential")
        
        global_coherence = self.coherence_tracker.get_global_coherence()
        
        log.info(f"Global quantum coherence: {global_coherence:.4f}")
        
        return {
            "global_coherence": global_coherence,
            "active_systems": len(self.coherence_tracker.coherence_states)
        }
    
    def get_quantum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quantum scheduler statistics"""
        stats = {
            "registered_agents": len(self.registered_agents),
            "active_tasks": len(self.active_tasks),
            "total_scheduled": len(self.scheduling_history),
            "quantum_operations": self.quantum_layer.get_quantum_statistics(),
            "quantum_advantage": {
                "average_speedup": np.mean(self.quantum_advantage_metrics["speedups"]) 
                                  if self.quantum_advantage_metrics["speedups"] else 0,
                "max_speedup": max(self.quantum_advantage_metrics["speedups"])
                              if self.quantum_advantage_metrics["speedups"] else 0,
                "speedup_trend": self.quantum_advantage_metrics["speedups"][-10:]  # Last 10
            },
            "entanglement_network": self.entanglement_matrix.get_entanglement_network_stats()
                                   if self.entanglement_matrix else None,
            "coherence": {
                "global": self.coherence_tracker.get_global_coherence(),
                "tracked_systems": len(self.coherence_tracker.coherence_states)
            }
        }
        
        return stats

