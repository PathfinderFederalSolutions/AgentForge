"""
Conflict-free Replicated Data Types (CRDT) Implementation
Basic implementation for distributed memory synchronization
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import uuid


@dataclass
class Op:
    """Operation for CRDT"""
    timestamp: float
    actor_id: str
    operation: str
    key: str
    value: Any
    
    def __init__(self, operation: str, key: str, value: Any, actor_id: str = None):
        self.timestamp = time.time()
        self.actor_id = actor_id or str(uuid.uuid4())
        self.operation = operation
        self.key = key
        self.value = value


class LWWMap:
    """
    Last-Writer-Wins Map CRDT
    Simple CRDT implementation where the most recent write wins
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._actor_ids: Dict[str, str] = {}
    
    def set(self, key: str, value: Any, timestamp: float = None, actor_id: str = None) -> Op:
        """Set a value in the map"""
        timestamp = timestamp or time.time()
        actor_id = actor_id or str(uuid.uuid4())
        
        # Only update if this is a newer timestamp
        if key not in self._timestamps or timestamp > self._timestamps[key]:
            self._data[key] = value
            self._timestamps[key] = timestamp
            self._actor_ids[key] = actor_id
        
        return Op("set", key, value, actor_id)
    
    def get(self, key: str) -> Any:
        """Get a value from the map"""
        return self._data.get(key)
    
    def delete(self, key: str, timestamp: float = None, actor_id: str = None) -> Op:
        """Delete a key from the map"""
        timestamp = timestamp or time.time()
        actor_id = actor_id or str(uuid.uuid4())
        
        # Only delete if this is a newer timestamp
        if key not in self._timestamps or timestamp > self._timestamps[key]:
            self._data.pop(key, None)
            self._timestamps.pop(key, None)
            self._actor_ids.pop(key, None)
        
        return Op("delete", key, None, actor_id)
    
    def apply_op(self, op: Op) -> bool:
        """Apply an operation to the CRDT"""
        if op.operation == "set":
            if op.key not in self._timestamps or op.timestamp > self._timestamps[op.key]:
                self._data[op.key] = op.value
                self._timestamps[op.key] = op.timestamp
                self._actor_ids[op.key] = op.actor_id
                return True
        elif op.operation == "delete":
            if op.key not in self._timestamps or op.timestamp > self._timestamps[op.key]:
                self._data.pop(op.key, None)
                self._timestamps.pop(op.key, None)
                self._actor_ids.pop(op.key, None)
                return True
        return False
    
    def merge(self, other: 'LWWMap') -> None:
        """Merge another LWWMap into this one"""
        for key in other._data:
            if key not in self._timestamps or other._timestamps[key] > self._timestamps[key]:
                self._data[key] = other._data[key]
                self._timestamps[key] = other._timestamps[key]
                self._actor_ids[key] = other._actor_ids[key]
    
    def keys(self):
        """Get all keys"""
        return self._data.keys()
    
    def values(self):
        """Get all values"""
        return self._data.values()
    
    def items(self):
        """Get all key-value pairs"""
        return self._data.items()
    
    def __len__(self):
        return len(self._data)
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __delitem__(self, key):
        self.delete(key)

