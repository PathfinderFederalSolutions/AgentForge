"""
AF Messaging Protocols
"""

import json
import time
from typing import Dict, Any
from enum import Enum

class MessageProtocol(Enum):
    """Message protocol types"""
    JSON = "json"
    BINARY = "binary"
    TEXT = "text"

def create_message(data: Dict[str, Any], protocol: MessageProtocol = MessageProtocol.JSON) -> str:
    """Create a message in the specified protocol"""
    message = {
        "timestamp": time.time(),
        "protocol": protocol.value,
        "data": data
    }
    
    if protocol == MessageProtocol.JSON:
        return json.dumps(message)
    elif protocol == MessageProtocol.TEXT:
        return str(message)
    else:
        return json.dumps(message)  # Default to JSON

def parse_message(message_str: str) -> Dict[str, Any]:
    """Parse a message from string format"""
    try:
        return json.loads(message_str)
    except json.JSONDecodeError:
        # Fallback for non-JSON messages
        return {"data": message_str, "protocol": "text"}



