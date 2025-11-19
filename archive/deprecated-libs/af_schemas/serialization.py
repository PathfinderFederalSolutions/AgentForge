"""
AF Schemas Serialization
"""

import json
from typing import Dict, Any

def serialize_task(task: Any) -> str:
    """Serialize task to JSON"""
    if hasattr(task, 'dict'):
        return json.dumps(task.dict())
    return json.dumps(task.__dict__ if hasattr(task, '__dict__') else str(task))

def deserialize_task(task_json: str) -> Dict[str, Any]:
    """Deserialize task from JSON"""
    return json.loads(task_json)

def serialize_agent_state(agent_state: Any) -> str:
    """Serialize agent state to JSON"""
    if hasattr(agent_state, 'dict'):
        return json.dumps(agent_state.dict())
    return json.dumps(agent_state.__dict__ if hasattr(agent_state, '__dict__') else str(agent_state))



