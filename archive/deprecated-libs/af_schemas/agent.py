"""
AF Schemas Agent - Agent-related schemas
"""

from typing import Dict, Any, List
from enum import Enum

class AgentState(Enum):
    """Agent state enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

def validate_agent_state(state_data: Dict[str, Any]) -> bool:
    """Validate agent state data"""
    return True

def serialize_agent_capabilities(capabilities: List[str]) -> str:
    """Serialize agent capabilities"""
    return ",".join(capabilities)

def deserialize_agent_capabilities(capabilities_str: str) -> List[str]:
    """Deserialize agent capabilities"""
    return capabilities_str.split(",") if capabilities_str else []

class AgentSchema:
    """Agent schema for validation"""
    @staticmethod
    def validate(agent_data: Dict[str, Any]) -> bool:
        return True

class AgentSwarmSchema:
    """Agent swarm schema for validation"""
    @staticmethod
    def validate(swarm_data: Dict[str, Any]) -> bool:
        return True
