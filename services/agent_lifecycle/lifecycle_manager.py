"""
Agent Lifecycle Manager
Manages the complete lifecycle of agents including creation, monitoring, and termination
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

log = logging.getLogger("agent-lifecycle")

class LifecycleState(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking"""
    agent_id: str
    created_at: datetime
    last_activity: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_runtime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 0.0
        return self.tasks_completed / total_tasks

class AgentLifecycleManager:
    """
    Manages the complete lifecycle of agents
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, AgentMetrics] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the lifecycle manager"""
        if self._initialized:
            return
        
        self._initialized = True
        log.info("✅ Agent Lifecycle Manager initialized")
    
    async def create_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> bool:
        """Create a new agent"""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.agents[agent_id] = {
                "id": agent_id,
                "state": LifecycleState.CREATED,
                "config": agent_config,
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
            
            self.metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            log.info(f"Agent {agent_id} created successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to create agent {agent_id}: {e}")
            return False
    
    async def update_agent_state(self, agent_id: str, state: LifecycleState) -> bool:
        """Update agent state"""
        if agent_id not in self.agents:
            log.error(f"Agent {agent_id} not found")
            return False
        
        try:
            self.agents[agent_id]["state"] = state
            self.agents[agent_id]["last_activity"] = datetime.now()
            
            if agent_id in self.metrics:
                self.metrics[agent_id].last_activity = datetime.now()
            
            log.debug(f"Agent {agent_id} state updated to {state.value}")
            return True
            
        except Exception as e:
            log.error(f"Failed to update agent {agent_id} state: {e}")
            return False
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent"""
        if agent_id not in self.agents:
            log.error(f"Agent {agent_id} not found")
            return False
        
        try:
            await self.update_agent_state(agent_id, LifecycleState.TERMINATING)
            
            # Perform cleanup
            # ... cleanup logic here ...
            
            await self.update_agent_state(agent_id, LifecycleState.TERMINATED)
            log.info(f"Agent {agent_id} terminated successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to terminate agent {agent_id}: {e}")
            return False
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent"""
        return self.metrics.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all agents"""
        return self.agents.copy()
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs"""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if agent["state"] in [LifecycleState.ACTIVE, LifecycleState.BUSY, LifecycleState.IDLE]
        ]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        total_agents = len(self.agents)
        active_agents = len(self.get_active_agents())
        
        states_count = {}
        for agent in self.agents.values():
            state = agent["state"].value
            states_count[state] = states_count.get(state, 0) + 1
        
        total_tasks = sum(m.tasks_completed + m.tasks_failed for m in self.metrics.values())
        total_success = sum(m.tasks_completed for m in self.metrics.values())
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "states_distribution": states_count,
            "total_tasks_processed": total_tasks,
            "overall_success_rate": total_success / total_tasks if total_tasks > 0 else 0.0,
            "initialized": self._initialized
        }

