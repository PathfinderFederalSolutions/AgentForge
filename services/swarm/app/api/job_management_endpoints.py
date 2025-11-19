"""
Job Management Endpoints - Phase 2 Implementation
Complete job lifecycle and swarm activity management
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from enum import Enum

log = logging.getLogger("job-management-api")

# Enums
class JobType(str, Enum):
    TASK = "task"
    CONTINUOUS = "continuous"

class JobStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"

# Request/Response Models
class JobRequest(BaseModel):
    title: str
    description: str
    type: JobType = JobType.TASK
    user_message: str
    data_sources: List[str] = []
    requirements: Dict[str, Any] = {}

class JobResponse(BaseModel):
    id: str
    title: str
    description: str
    type: JobType
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[str] = None
    agents_assigned: int
    agents_active: int
    progress: Optional[int] = None  # For task jobs
    tasks_completed: Optional[int] = None
    total_tasks: Optional[int] = None
    events_processed: Optional[int] = None  # For continuous jobs
    last_event_time: Optional[datetime] = None
    alerts_generated: int
    data_streams: List[str]
    confidence: float
    metadata: Dict[str, Any] = {}

class SwarmActivityResponse(BaseModel):
    id: str
    job_id: str
    agent_id: str
    agent_type: str
    task: str
    status: AgentStatus
    progress: int
    start_time: datetime
    end_time: Optional[datetime] = None
    memory_tier: Optional[str] = None
    capabilities: List[str] = []
    metrics: Dict[str, Any] = {}

class ArchivedJobResponse(BaseModel):
    id: str
    title: str
    description: str
    completed_at: datetime
    duration: str
    agents_used: int
    confidence: float
    output_size: str
    results_summary: str
    metadata: Dict[str, Any] = {}

# Router
router = APIRouter(prefix="/v1/jobs", tags=["job-management"])

# In-memory storage for demonstration (in production, use database)
active_jobs: Dict[str, JobResponse] = {}
archived_jobs: Dict[str, ArchivedJobResponse] = {}
swarm_activities: Dict[str, List[SwarmActivityResponse]] = {}

# Initialize with demo data
def initialize_demo_data():
    """Initialize with demonstration jobs and activities"""
    global active_jobs, archived_jobs, swarm_activities
    
    # Demo active jobs
    job1_id = "demo-job-001"
    job2_id = "demo-job-002"
    job3_id = "demo-job-003"
    
    active_jobs[job1_id] = JobResponse(
        id=job1_id,
        title="Real-time Sensor Fusion",
        description="Analyzing 15 IoT sensor streams for anomaly detection",
        type=JobType.CONTINUOUS,
        status=JobStatus.RUNNING,
        start_time=datetime.now() - timedelta(hours=2),
        agents_assigned=8,
        agents_active=6,
        events_processed=15420,
        last_event_time=datetime.now() - timedelta(seconds=30),
        alerts_generated=23,
        data_streams=["temp-sensors", "pressure-gauges", "flow-meters"],
        confidence=0.94
    )
    
    active_jobs[job2_id] = JobResponse(
        id=job2_id,
        title="Financial Data Analysis",
        description="Processing live market data for trading insights",
        type=JobType.CONTINUOUS,
        status=JobStatus.RUNNING,
        start_time=datetime.now() - timedelta(hours=4),
        agents_assigned=12,
        agents_active=10,
        events_processed=8934,
        last_event_time=datetime.now() - timedelta(seconds=5),
        alerts_generated=7,
        data_streams=["market-feed", "news-api"],
        confidence=0.87
    )
    
    active_jobs[job3_id] = JobResponse(
        id=job3_id,
        title="Website Redesign Project",
        description="Creating responsive web application with modern UI/UX",
        type=JobType.TASK,
        status=JobStatus.RUNNING,
        start_time=datetime.now() - timedelta(hours=1),
        agents_assigned=5,
        agents_active=3,
        progress=65,
        tasks_completed=9,
        total_tasks=20,
        alerts_generated=3,
        data_streams=[],
        confidence=0.91
    )
    
    # Demo swarm activities
    swarm_activities[job1_id] = [
        SwarmActivityResponse(
            id="activity-001",
            job_id=job1_id,
            agent_id="sensor-fusion-001",
            agent_type="neural-mesh",
            task="Analyzing temperature sensor patterns in L3 memory",
            status=AgentStatus.WORKING,
            progress=78,
            start_time=datetime.now() - timedelta(minutes=15),
            memory_tier="L3",
            capabilities=["pattern_recognition", "anomaly_detection"],
            metrics={"sensors_processed": 145, "anomalies_found": 3}
        ),
        SwarmActivityResponse(
            id="activity-002",
            job_id=job1_id,
            agent_id="quantum-coordinator-001",
            agent_type="quantum-scheduler",
            task="Coordinating multi-sensor data fusion",
            status=AgentStatus.WORKING,
            progress=92,
            start_time=datetime.now() - timedelta(minutes=20),
            memory_tier="L2",
            capabilities=["quantum_coordination", "data_fusion"],
            metrics={"coordination_efficiency": 0.94}
        )
    ]
    
    # Demo archived jobs
    archived_jobs["arch-001"] = ArchivedJobResponse(
        id="arch-001",
        title="Quarterly Report Generation",
        description="Generated comprehensive Q4 business analysis",
        completed_at=datetime.now() - timedelta(days=1),
        duration="2h 34m",
        agents_used=15,
        confidence=0.94,
        output_size="47 pages",
        results_summary="Successfully generated comprehensive quarterly report with 15 key insights, 23 visualizations, and strategic recommendations."
    )

# Initialize demo data on startup
initialize_demo_data()

@router.post("/create", response_model=JobResponse)
async def create_job(request: JobRequest):
    """Create a new job"""
    try:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        
        # Determine agent count based on complexity
        agent_count = max(2, len(request.data_sources) * 2 + len(request.user_message.split()) // 10)
        agent_count = min(agent_count, 20)  # Cap at 20 for demo
        
        # Create job
        job = JobResponse(
            id=job_id,
            title=request.title,
            description=request.description,
            type=request.type,
            status=JobStatus.RUNNING,
            start_time=datetime.now(),
            agents_assigned=agent_count,
            agents_active=agent_count,
            progress=0 if request.type == JobType.TASK else None,
            tasks_completed=0 if request.type == JobType.TASK else None,
            total_tasks=10 if request.type == JobType.TASK else None,
            events_processed=0 if request.type == JobType.CONTINUOUS else None,
            last_event_time=datetime.now() if request.type == JobType.CONTINUOUS else None,
            alerts_generated=0,
            data_streams=request.data_sources,
            confidence=0.85,
            metadata={
                "user_message": request.user_message,
                "requirements": request.requirements,
                "created_by": "user_001"
            }
        )
        
        active_jobs[job_id] = job
        
        # Initialize swarm activities
        swarm_activities[job_id] = []
        await _generate_initial_swarm_activities(job_id, job)
        
        log.info(f"Created job {job_id}: {request.title}")
        return job
        
    except Exception as e:
        log.error(f"Job creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")

@router.get("/active", response_model=List[JobResponse])
async def get_active_jobs():
    """Get all active jobs"""
    try:
        return list(active_jobs.values())
    except Exception as e:
        log.error(f"Failed to get active jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/archived", response_model=List[ArchivedJobResponse])
async def get_archived_jobs():
    """Get all archived jobs"""
    try:
        return list(archived_jobs.values())
    except Exception as e:
        log.error(f"Failed to get archived jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get specific job details"""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return active_jobs[job_id]
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running job"""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        if job.status != JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Job is not running")
        
        job.status = JobStatus.PAUSED
        job.agents_active = 0
        
        log.info(f"Paused job {job_id}")
        return {"message": f"Job {job_id} paused successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to pause job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job"""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        if job.status != JobStatus.PAUSED:
            raise HTTPException(status_code=400, detail="Job is not paused")
        
        job.status = JobStatus.RUNNING
        job.agents_active = job.agents_assigned
        
        log.info(f"Resumed job {job_id}")
        return {"message": f"Job {job_id} resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to resume job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/archive")
async def archive_job(job_id: str):
    """Archive a completed job"""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        
        # Calculate duration
        duration_seconds = (datetime.now() - job.start_time).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        duration_str = f"{hours}h {minutes}m"
        
        # Create archived job
        archived_job = ArchivedJobResponse(
            id=job.id,
            title=job.title,
            description=job.description,
            completed_at=datetime.now(),
            duration=duration_str,
            agents_used=job.agents_assigned,
            confidence=job.confidence,
            output_size=f"{job.tasks_completed or job.events_processed or 0} items processed",
            results_summary=f"Successfully completed {job.title} with {job.confidence*100:.0f}% confidence",
            metadata=job.metadata
        )
        
        # Move to archived
        archived_jobs[job_id] = archived_job
        del active_jobs[job_id]
        
        # Clean up swarm activities
        if job_id in swarm_activities:
            del swarm_activities[job_id]
        
        log.info(f"Archived job {job_id}")
        return {"message": f"Job {job_id} archived successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to archive job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/activity", response_model=List[SwarmActivityResponse])
async def get_job_activity(job_id: str):
    """Get swarm activity for a specific job"""
    try:
        if job_id not in active_jobs and job_id not in swarm_activities:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return swarm_activities.get(job_id, [])
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get activity for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity/all", response_model=List[SwarmActivityResponse])
async def get_all_swarm_activity():
    """Get all current swarm activity across jobs"""
    try:
        all_activities = []
        for activities in swarm_activities.values():
            all_activities.extend(activities)
        
        # Sort by start time, most recent first
        all_activities.sort(key=lambda x: x.start_time, reverse=True)
        
        return all_activities[:50]  # Return last 50 activities
        
    except Exception as e:
        log.error(f"Failed to get all swarm activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_initial_swarm_activities(job_id: str, job: JobResponse):
    """Generate initial swarm activities for a new job"""
    activities = []
    
    # Generate activities based on job type and complexity
    agent_types = ["neural-mesh", "quantum-scheduler", "data-processor", "analytics", "universal-io"]
    
    for i in range(min(job.agents_assigned, 6)):  # Show up to 6 activities
        agent_type = agent_types[i % len(agent_types)]
        
        activity = SwarmActivityResponse(
            id=f"activity-{uuid.uuid4().hex[:8]}",
            job_id=job_id,
            agent_id=f"{agent_type}-{i+1:03d}",
            agent_type=agent_type,
            task=_generate_task_for_agent_type(agent_type, job),
            status=AgentStatus.INITIALIZING if i < 2 else AgentStatus.WORKING,
            progress=0 if i < 2 else min(25 + i * 10, 85),
            start_time=datetime.now() - timedelta(seconds=i*30),
            memory_tier=f"L{min(i//2 + 1, 4)}",
            capabilities=_get_capabilities_for_agent_type(agent_type),
            metrics={"efficiency": 0.85 + i * 0.02}
        )
        
        activities.append(activity)
    
    swarm_activities[job_id] = activities

def _generate_task_for_agent_type(agent_type: str, job: JobResponse) -> str:
    """Generate appropriate task description for agent type"""
    task_templates = {
        "neural-mesh": [
            f"Analyzing patterns in {job.title.lower()} using L3 memory",
            "Building knowledge graph connections",
            "Cross-referencing organizational context"
        ],
        "quantum-scheduler": [
            "Coordinating multi-agent task distribution",
            "Optimizing resource allocation across swarm",
            "Managing quantum superposition states"
        ],
        "data-processor": [
            "Processing structured data inputs",
            "Validating data integrity and format",
            "Extracting key features from datasets"
        ],
        "analytics": [
            "Computing statistical metrics and trends",
            "Generating predictive insights",
            "Identifying correlation patterns"
        ],
        "universal-io": [
            "Processing multi-modal input streams",
            "Generating output in requested format",
            "Managing input/output transformations"
        ]
    }
    
    templates = task_templates.get(agent_type, ["Processing assigned task"])
    return templates[hash(job.id) % len(templates)]

def _get_capabilities_for_agent_type(agent_type: str) -> List[str]:
    """Get capabilities list for agent type"""
    capabilities_map = {
        "neural-mesh": ["pattern_recognition", "memory_synthesis", "knowledge_graphs"],
        "quantum-scheduler": ["quantum_coordination", "resource_optimization", "task_distribution"],
        "data-processor": ["data_validation", "format_conversion", "feature_extraction"],
        "analytics": ["statistical_analysis", "predictive_modeling", "trend_analysis"],
        "universal-io": ["multi_modal_processing", "format_generation", "content_transformation"]
    }
    
    return capabilities_map.get(agent_type, ["general_processing"])

# Background task to update job progress
async def update_job_progress():
    """Background task to simulate job progress updates"""
    while True:
        try:
            for job_id, job in active_jobs.items():
                if job.status == JobStatus.RUNNING:
                    # Update continuous jobs
                    if job.type == JobType.CONTINUOUS:
                        if job.events_processed is not None:
                            job.events_processed += max(1, int(job.agents_active * 0.5))
                        job.last_event_time = datetime.now()
                        
                        # Occasionally generate alerts
                        if hash(job_id + str(int(time.time()))) % 100 < 5:  # 5% chance
                            job.alerts_generated += 1
                    
                    # Update task jobs
                    elif job.type == JobType.TASK:
                        if job.progress is not None and job.progress < 100:
                            job.progress = min(job.progress + max(1, job.agents_active // 2), 100)
                        if job.tasks_completed is not None and job.total_tasks is not None:
                            if job.tasks_completed < job.total_tasks:
                                if hash(job_id + str(int(time.time()))) % 20 < 3:  # 15% chance
                                    job.tasks_completed += 1
                    
                    # Update swarm activities
                    if job_id in swarm_activities:
                        for activity in swarm_activities[job_id]:
                            if activity.status == AgentStatus.WORKING and activity.progress < 100:
                                activity.progress = min(activity.progress + max(1, 3), 100)
                                if activity.progress >= 100:
                                    activity.status = AgentStatus.COMPLETED
                                    activity.end_time = datetime.now()
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            log.error(f"Error updating job progress: {e}")
            await asyncio.sleep(5)

# Start background task
asyncio.create_task(update_job_progress())

# Health check
@router.get("/health")
async def job_management_health():
    """Health check for job management system"""
    return {
        "status": "healthy",
        "active_jobs": len(active_jobs),
        "archived_jobs": len(archived_jobs),
        "total_activities": sum(len(activities) for activities in swarm_activities.values()),
        "timestamp": time.time()
    }
