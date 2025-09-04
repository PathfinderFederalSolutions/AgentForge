# forge_types.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime

MemoryScope = Literal["task", "summary", "tool", "global"]

class Task(BaseModel):
    id: str
    description: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    memory_scopes: List[MemoryScope] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    budget: int = 0
    priority: int = 0
    deadline: Optional[datetime] = None  # Add default

class AgentContract(BaseModel):
    name: str
    capabilities: List[str]
    memory_scopes: List[MemoryScope]
    tools: List[str]
    budget: int
    deadline: Optional[datetime] = None  # Add default