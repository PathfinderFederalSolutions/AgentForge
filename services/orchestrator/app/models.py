from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict, model_validator
import uuid
import math

TaskType = Literal["gather","analyze","synthesize","review","map","reduce","custom"]
class TaskSpec(BaseModel):
    id: str
    type: TaskType
    args: Dict[str, Any] = Field(default_factory=dict)
    dependsOn: List[str] = Field(default_factory=list)
    review: bool = False
class Constraints(BaseModel):
    deadline_ms: int = 60000
    budget_tokens: int = 100000
class SwarmJob(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Constraints()
    tools_allowed: List[str] = Field(default_factory=list)
    plan: List[TaskSpec]
    reply_subject: Optional[str] = None
class ToolInvocation(BaseModel):
    request_id: str
    task_id: str
    tool: str
    payload: Dict[str, Any]
    attempt: int = 1
    max_attempts: int = 3
    idempotency_key: str
class ToolResult(BaseModel):
    request_id: str
    task_id: str
    status: Literal["ok","error"]
    output: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
class AggregatedResult(BaseModel):
    request_id: str
    status: str
    artifacts: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    validation: Dict[str, Any]
    errors: List[str]

# --- Tactical UI Schemas -----------------------------------------------------

def _clip_lat(lat: float) -> float:
    return max(-90.0, min(90.0, float(lat)))

def _wrap_lon(lon: float) -> float:
    # Normalize to [-180,180]
    x = float(lon)
    while x <= -180.0:
        x += 360.0
    while x > 180.0:
        x -= 360.0
    return x

def _sanitize(s: Optional[str], max_len: int = 256) -> Optional[str]:
    if s is None:
        return None
    # Strip control chars and limit length
    cleaned = ''.join(ch for ch in s if ch.isprintable())
    return cleaned[:max_len]

class ThreatMarker(BaseModel):
    """GeoJSON Point Feature properties for a threat marker."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    # Coordinates as [lon, lat]
    coordinates: Tuple[float, float]
    severity: Optional[Literal["low","medium","high","critical"]] = None
    evidence_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_bounds(self):
        lon, lat = self.coordinates
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            # clip/wrap but keep within bounds
            self.coordinates = (_wrap_lon(lon), _clip_lat(lat))
        # sanitize strings
        self.title = _sanitize(self.title)
        self.description = _sanitize(self.description, 1024)
        if self.evidence_id:
            self.evidence_id = _sanitize(self.evidence_id, 128)
        return self

class ThreatZone(BaseModel):
    """GeoJSON Polygon Feature properties for a threat zone."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    description: Optional[str] = None
    # Polygon as list of linear rings; outer ring required, each ring is list[[lon,lat],...]
    polygon: List[List[Tuple[float, float]]]
    severity: Optional[Literal["low","medium","high","critical"]] = None

    @model_validator(mode="after")
    def _validate_polygon(self):
        # Ensure at least one ring with >= 4 points (closed)
        if not self.polygon or not self.polygon[0] or len(self.polygon[0]) < 4:
            raise ValueError("polygon must include an outer ring with at least 4 coordinates")
        new_poly: List[List[Tuple[float, float]]] = []
        for ring in self.polygon:
            cleaned: List[Tuple[float, float]] = []
            for lon, lat in ring:
                cleaned.append((_wrap_lon(lon), _clip_lat(lat)))
            # Ensure closed ring (first == last)
            if cleaned[0] != cleaned[-1]:
                cleaned.append(cleaned[0])
            new_poly.append(cleaned)
        self.polygon = new_poly
        self.title = _sanitize(self.title)
        self.description = _sanitize(self.description, 1024)
        return self