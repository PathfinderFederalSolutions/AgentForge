"""
Agent Lifecycle Manager - Integrated into Unified Orchestrator
Manages agent spawning, lifecycle, and termination with quantum coordination
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .quantum_orchestrator import QuantumAgent, UnifiedTask

log = logging.getLogger("agent-lifecycle-manager")

class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"

class LifecyclePolicy(Enum):
    """Agent lifecycle management policies"""
    PERSISTENT = "persistent"
    AUTO_TERMINATE = "auto_terminate"
    IDLE_TIMEOUT = "idle_timeout"
    RESOURCE_BASED = "resource_based"

class SpawningStrategy(Enum):
    """Agent spawning strategies"""
    ON_DEMAND = "on_demand"
    PREDICTIVE = "predictive"
    POOL_BASED = "pool_based"
    ADAPTIVE = "adaptive"

@dataclass
class AgentLifecycleConfig:
    """Configuration for agent lifecycle management"""
    max_idle_time: float = 300.0  # 5 minutes
    max_agents: int = 1000000
    min_pool_size: int = 10
    max_pool_size: int = 100
    spawn_threshold: float = 0.8  # Spawn when 80% of agents are busy
    terminate_threshold: float = 0.3  # Terminate when 30% are idle
    resource_limit_cpu: float = 0.8
    resource_limit_memory: float = 0.8

class AgentLifecycleManager:
    """Advanced agent lifecycle management with quantum coordination"""
    
    def __init__(self, config: AgentLifecycleConfig = None):
        self.config = config or AgentLifecycleConfig()
        self.agents: Dict[str, QuantumAgent] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_pools: Dict[str, List[str]] = {}  # capability -> agent_ids
        self.spawning_strategy = SpawningStrategy.ADAPTIVE
        self.lifecycle_policies: Dict[str, LifecyclePolicy] = {}
        
        # Metrics
        self.spawn_count = 0
        self.terminate_count = 0
        self.active_tasks: Dict[str, str] = {}  # agent_id -> task_id
        
        # Background tasks
        self.running = False
        self.lifecycle_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the lifecycle manager"""
        self.running = True
        
        # Start background tasks
        self.lifecycle_tasks = [
            asyncio.create_task(self._lifecycle_monitor()),
            asyncio.create_task(self._pool_manager()),
            asyncio.create_task(self._resource_monitor())
        ]
        
        log.info("Agent Lifecycle Manager started")
    
    async def stop(self):
        """Stop the lifecycle manager"""
        self.running = False
        
        # Cancel background tasks
        for task in self.lifecycle_tasks:
            task.cancel()
        
        # Terminate all agents gracefully
        await self._terminate_all_agents()
        
        log.info("Agent Lifecycle Manager stopped")
    
    async def spawn_agent(self, capabilities: Set[str], task_id: str = None, 
                         lifecycle_policy: LifecyclePolicy = LifecyclePolicy.AUTO_TERMINATE) -> str:
        """Spawn a new agent with specified capabilities"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Create quantum agent
        agent = QuantumAgent(
            agent_id=agent_id,
            capabilities=capabilities
        )
        
        # Initialize agent state
        self.agents[agent_id] = agent
        self.agent_states[agent_id] = AgentState.INITIALIZING
        self.lifecycle_policies[agent_id] = lifecycle_policy
        
        if task_id:
            self.active_tasks[agent_id] = task_id
        
        # Add to appropriate pools
        for capability in capabilities:
            if capability not in self.agent_pools:
                self.agent_pools[capability] = []
            self.agent_pools[capability].append(agent_id)
        
        # Simulate agent initialization
        await asyncio.sleep(0.1)
        self.agent_states[agent_id] = AgentState.ACTIVE
        
        self.spawn_count += 1
        log.info(f"Spawned agent {agent_id} with capabilities {capabilities}")
        
        return agent_id
    
    async def terminate_agent(self, agent_id: str, reason: str = "manual"):
        """Terminate an agent gracefully"""
        if agent_id not in self.agents:
            return
        
        self.agent_states[agent_id] = AgentState.TERMINATING
        
        # Remove from pools
        agent = self.agents[agent_id]
        for capability in agent.capabilities:
            if capability in self.agent_pools and agent_id in self.agent_pools[capability]:
                self.agent_pools[capability].remove(agent_id)
        
        # Clean up
        del self.agents[agent_id]
        del self.agent_states[agent_id]
        if agent_id in self.lifecycle_policies:
            del self.lifecycle_policies[agent_id]
        if agent_id in self.active_tasks:
            del self.active_tasks[agent_id]
        
        self.terminate_count += 1
        log.info(f"Terminated agent {agent_id} - {reason}")
    
    async def assign_task(self, agent_id: str, task: UnifiedTask):
        """Assign a task to an agent"""
        if agent_id in self.agents:
            self.agent_states[agent_id] = AgentState.BUSY
            self.active_tasks[agent_id] = task.task_id
    
    async def complete_task(self, agent_id: str, task_id: str):
        """Mark task as completed for an agent"""
        if agent_id in self.agents:
            self.agent_states[agent_id] = AgentState.IDLE
            if agent_id in self.active_tasks:
                del self.active_tasks[agent_id]
            
            # Check if agent should be terminated based on policy
            policy = self.lifecycle_policies.get(agent_id, LifecyclePolicy.AUTO_TERMINATE)
            if policy == LifecyclePolicy.AUTO_TERMINATE:
                await self.terminate_agent(agent_id, "task_completed")
    
    async def get_available_agents(self, capabilities: Set[str]) -> List[str]:
        """Get available agents with specified capabilities"""
        available = []
        
        for agent_id, agent in self.agents.items():
            if (self.agent_states[agent_id] in [AgentState.ACTIVE, AgentState.IDLE] and
                capabilities.issubset(agent.capabilities)):
                available.append(agent_id)
        
        return available
    
    async def _lifecycle_monitor(self):
        """Monitor agent lifecycles and apply policies"""
        while self.running:
            try:
                current_time = time.time()
                agents_to_terminate = []
                
                for agent_id, state in self.agent_states.items():
                    policy = self.lifecycle_policies.get(agent_id, LifecyclePolicy.AUTO_TERMINATE)
                    agent = self.agents[agent_id]
                    
                    # Check idle timeout
                    if (policy == LifecyclePolicy.IDLE_TIMEOUT and 
                        state == AgentState.IDLE and
                        current_time - agent.last_heartbeat > self.config.max_idle_time):
                        agents_to_terminate.append((agent_id, "idle_timeout"))
                    
                    # Check resource usage
                    elif (policy == LifecyclePolicy.RESOURCE_BASED and
                          (agent.cpu_allocation > self.config.resource_limit_cpu or
                           agent.memory_allocation > self.config.resource_limit_memory)):
                        agents_to_terminate.append((agent_id, "resource_limit"))
                
                # Terminate agents that meet termination criteria
                for agent_id, reason in agents_to_terminate:
                    await self.terminate_agent(agent_id, reason)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Error in lifecycle monitor: {e}")
                await asyncio.sleep(30)
    
    async def _pool_manager(self):
        """Manage agent pools based on demand"""
        while self.running:
            try:
                # Analyze current load and adjust pools
                for capability, agent_ids in self.agent_pools.items():
                    active_count = sum(1 for aid in agent_ids 
                                     if self.agent_states.get(aid) == AgentState.BUSY)
                    total_count = len(agent_ids)
                    
                    if total_count > 0:
                        utilization = active_count / total_count
                        
                        # Spawn more agents if utilization is high
                        if (utilization > self.config.spawn_threshold and 
                            total_count < self.config.max_pool_size):
                            await self.spawn_agent({capability}, 
                                                 lifecycle_policy=LifecyclePolicy.IDLE_TIMEOUT)
                        
                        # Terminate idle agents if utilization is low
                        elif (utilization < self.config.terminate_threshold and 
                              total_count > self.config.min_pool_size):
                            idle_agents = [aid for aid in agent_ids 
                                         if self.agent_states.get(aid) == AgentState.IDLE]
                            if idle_agents:
                                await self.terminate_agent(idle_agents[0], "low_utilization")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                log.error(f"Error in pool manager: {e}")
                await asyncio.sleep(60)
    
    async def _resource_monitor(self):
        """Monitor system resources and adjust agent limits"""
        while self.running:
            try:
                # This would integrate with actual resource monitoring
                # For now, just log current state
                total_agents = len(self.agents)
                active_agents = sum(1 for state in self.agent_states.values() 
                                  if state in [AgentState.ACTIVE, AgentState.BUSY])
                
                log.debug(f"Resource monitor: {active_agents}/{total_agents} agents active")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                log.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(120)
    
    async def _terminate_all_agents(self):
        """Terminate all agents gracefully"""
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            await self.terminate_agent(agent_id, "shutdown")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle manager statistics"""
        return {
            "total_agents": len(self.agents),
            "agents_by_state": {
                state.value: sum(1 for s in self.agent_states.values() if s == state)
                for state in AgentState
            },
            "spawn_count": self.spawn_count,
            "terminate_count": self.terminate_count,
            "active_tasks": len(self.active_tasks),
            "pool_sizes": {cap: len(agents) for cap, agents in self.agent_pools.items()}
        }
