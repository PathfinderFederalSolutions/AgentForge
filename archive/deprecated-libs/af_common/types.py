"""
AF Common Types - Bridge to existing types
"""

# Import from existing swarm types
try:
    from services.swarm.forge_types import Task, AgentContract, TaskResult
except ImportError:
    from pydantic import BaseModel
    from typing import Dict, Any, Optional
    from datetime import datetime
    
    class Task(BaseModel):
        id: str
        description: str
        metadata: Dict[str, Any] = {}
    
    class AgentContract(BaseModel):
        name: str
        capabilities: list = []
    
    class TaskResult(BaseModel):
        task_id: str
        success: bool
        result: Optional[Dict] = None

# Additional types
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class SystemMetric:
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = None

