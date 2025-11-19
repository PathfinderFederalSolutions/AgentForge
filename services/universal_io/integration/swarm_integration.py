"""
Universal I/O Swarm Integration
Integrates universal-io with the swarm orchestration system for massive scale processing
Enables deployment of 400+ agents for comprehensive codebase analysis and processing
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import universal-io components
from ..stream.stream_ingestion import StreamIngestionEngine, StreamMessage, StreamConfig, StreamType
from ..stream.event_processor import EventProcessingEngine, ProcessingEvent, EventType
from ..outputs.vertical_generators import (
    DefenseIntelligenceGenerator, HealthcareGenerator, FinanceGenerator,
    VerticalDomain, create_vertical_generator
)
from ..security.zero_trust_framework import ZeroTrustSecurityFramework, SecurityLevel, encrypt_data

# Import swarm system components
try:
    from ...swarm.unified_swarm_system import (
        UnifiedSwarmSystem, UnifiedGoal, SwarmAgent, SwarmExecutionResult,
        SwarmScale, SwarmObjective, SwarmMode
    )
    from ...swarm.integration.unified_integration_bridge import (
        UnifiedIntegrationBridge, IntegrationMode, SynchronizationStrategy
    )
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    log.warning("Swarm system not available - running in standalone mode")

log = logging.getLogger("swarm-integration")

class ProcessingScale(Enum):
    """Processing scale levels"""
    SINGLE_AGENT = "single_agent"          # 1 agent
    SMALL_SWARM = "small_swarm"            # 10-50 agents
    MEDIUM_SWARM = "medium_swarm"          # 50-200 agents
    LARGE_SWARM = "large_swarm"            # 200-500 agents
    MASSIVE_SWARM = "massive_swarm"        # 500+ agents
    CODEBASE_ANALYSIS = "codebase_analysis" # 400+ specialized agents

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"      # Sub-second processing required
    HIGH = "high"              # < 5 seconds
    NORMAL = "normal"          # < 30 seconds
    LOW = "low"                # Best effort
    BATCH = "batch"            # Background processing

class ProcessingObjective(Enum):
    """Processing objectives"""
    REAL_TIME_ANALYSIS = "real_time_analysis"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    CODEBASE_UNDERSTANDING = "codebase_understanding"
    CAPABILITY_DISCOVERY = "capability_discovery"
    INTEGRATION_MAPPING = "integration_mapping"
    QUALITY_ASSESSMENT = "quality_assessment"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

@dataclass
class SwarmProcessingTask:
    """Task for swarm processing"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    objective: ProcessingObjective = ProcessingObjective.REAL_TIME_ANALYSIS
    priority: TaskPriority = TaskPriority.NORMAL
    scale: ProcessingScale = ProcessingScale.MEDIUM_SWARM
    
    # Input data
    input_data: Any = None
    input_metadata: Dict[str, Any] = field(default_factory=dict)
    input_type: str = "unknown"
    
    # Output requirements
    output_format: str = "json"
    output_requirements: Dict[str, Any] = field(default_factory=dict)
    vertical_domain: Optional[VerticalDomain] = None
    
    # Processing constraints
    max_processing_time: float = 300.0  # 5 minutes
    quality_threshold: float = 0.8
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    # Swarm configuration
    agent_specializations: List[str] = field(default_factory=list)
    coordination_strategy: str = "hierarchical"
    fault_tolerance: bool = True
    
    # Progress tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    result: Optional[Any] = None
    confidence: float = 0.0
    agents_used: int = 0
    processing_time: float = 0.0
    
    def mark_started(self):
        """Mark task as started"""
        self.started_at = time.time()
    
    def mark_completed(self, result: Any, confidence: float, agents_used: int):
        """Mark task as completed"""
        self.completed_at = time.time()
        self.result = result
        self.confidence = confidence
        self.agents_used = agents_used
        if self.started_at:
            self.processing_time = self.completed_at - self.started_at
    
    def is_expired(self) -> bool:
        """Check if task has exceeded max processing time"""
        if not self.started_at:
            return False
        return time.time() - self.started_at > self.max_processing_time

@dataclass
class CodebaseAnalysisTask:
    """Specialized task for comprehensive codebase analysis"""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    codebase_path: str = ""
    analysis_depth: str = "comprehensive"  # surface, detailed, comprehensive, exhaustive
    
    # Analysis objectives
    discover_capabilities: bool = True
    map_integrations: bool = True
    assess_quality: bool = True
    identify_patterns: bool = True
    security_analysis: bool = True
    performance_analysis: bool = True
    
    # Agent specializations for codebase analysis
    specialized_agents: Dict[str, int] = field(default_factory=lambda: {
        "python_analyzer": 50,
        "javascript_analyzer": 30,
        "api_analyzer": 20,
        "database_analyzer": 15,
        "security_analyzer": 25,
        "architecture_analyzer": 10,
        "integration_mapper": 15,
        "quality_assessor": 20,
        "performance_profiler": 10,
        "documentation_analyzer": 5
    })
    
    # File type priorities
    file_priorities: Dict[str, int] = field(default_factory=lambda: {
        ".py": 10,
        ".js": 9,
        ".ts": 9,
        ".java": 8,
        ".cpp": 7,
        ".h": 7,
        ".sql": 8,
        ".yaml": 6,
        ".json": 6,
        ".md": 4,
        ".txt": 3
    })
    
    # Results
    total_files_analyzed: int = 0
    capabilities_discovered: List[Dict[str, Any]] = field(default_factory=list)
    integrations_mapped: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    architecture_insights: Dict[str, Any] = field(default_factory=dict)

class UniversalIOSwarmCoordinator:
    """Coordinates universal I/O processing with swarm orchestration"""
    
    def __init__(self):
        # Core components
        self.stream_engine = StreamIngestionEngine()
        self.event_processor = EventProcessingEngine()
        self.security_framework = ZeroTrustSecurityFramework()
        
        # Swarm integration
        if SWARM_AVAILABLE:
            self.swarm_system = UnifiedSwarmSystem("universal_io_swarm")
            self.integration_bridge = UnifiedIntegrationBridge()
        else:
            self.swarm_system = None
            self.integration_bridge = None
        
        # Vertical generators
        self.vertical_generators = {
            VerticalDomain.DEFENSE_INTELLIGENCE: DefenseIntelligenceGenerator(),
            VerticalDomain.HEALTHCARE: HealthcareGenerator(),
            VerticalDomain.FINANCE: FinanceGenerator()
        }
        
        # Active tasks and processing state
        self.active_tasks: Dict[str, SwarmProcessingTask] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.completed_tasks: List[SwarmProcessingTask] = []
        
        # Performance metrics
        self.metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "peak_agents_deployed": 0,
            "total_agent_hours": 0.0,
            "throughput_per_hour": 0.0
        }
        
        log.info("Universal I/O Swarm Coordinator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the swarm coordinator"""
        try:
            # Initialize stream processing
            await self.event_processor.start_processing()
            
            # Initialize swarm system if available
            if self.swarm_system:
                await self.swarm_system.initialize()
                log.info("Swarm system initialized")
            
            # Start background processing
            asyncio.create_task(self._process_task_queue())
            asyncio.create_task(self._monitor_active_tasks())
            
            log.info("Universal I/O Swarm Coordinator ready")
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize swarm coordinator: {e}")
            return False
    
    async def submit_processing_task(self, task: SwarmProcessingTask) -> str:
        """Submit a processing task to the swarm"""
        try:
            # Validate task
            if not self._validate_task(task):
                raise ValueError("Invalid task configuration")
            
            # Encrypt sensitive data if required
            if task.security_level != SecurityLevel.PUBLIC:
                task.input_data = await encrypt_data(task.input_data, task.security_level)
            
            # Add to processing queue
            await self.processing_queue.put(task)
            self.active_tasks[task.task_id] = task
            
            log.info(f"Submitted task {task.task_id} for {task.scale.value} processing")
            return task.task_id
            
        except Exception as e:
            log.error(f"Failed to submit task: {e}")
            raise
    
    async def submit_codebase_analysis(self, analysis_task: CodebaseAnalysisTask) -> str:
        """Submit comprehensive codebase analysis task"""
        try:
            # Create swarm processing task for codebase analysis
            swarm_task = SwarmProcessingTask(
                objective=ProcessingObjective.CODEBASE_UNDERSTANDING,
                priority=TaskPriority.HIGH,
                scale=ProcessingScale.CODEBASE_ANALYSIS,
                input_data=analysis_task,
                input_type="codebase_analysis",
                output_format="comprehensive_analysis_report",
                max_processing_time=3600.0,  # 1 hour for comprehensive analysis
                agent_specializations=list(analysis_task.specialized_agents.keys())
            )
            
            return await self.submit_processing_task(swarm_task)
            
        except Exception as e:
            log.error(f"Failed to submit codebase analysis: {e}")
            raise
    
    async def _process_task_queue(self):
        """Background task to process the task queue"""
        while True:
            try:
                # Get next task from queue
                task = await self.processing_queue.get()
                
                # Process the task
                await self._process_single_task(task)
                
            except Exception as e:
                log.error(f"Task queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_task(self, task: SwarmProcessingTask):
        """Process a single task"""
        try:
            task.mark_started()
            log.info(f"Processing task {task.task_id} with {task.scale.value}")
            
            # Determine processing strategy based on scale and objective
            if task.scale == ProcessingScale.SINGLE_AGENT:
                result = await self._process_with_single_agent(task)
            elif task.scale in [ProcessingScale.SMALL_SWARM, ProcessingScale.MEDIUM_SWARM]:
                result = await self._process_with_medium_swarm(task)
            elif task.scale in [ProcessingScale.LARGE_SWARM, ProcessingScale.MASSIVE_SWARM]:
                result = await self._process_with_large_swarm(task)
            elif task.scale == ProcessingScale.CODEBASE_ANALYSIS:
                result = await self._process_codebase_analysis(task)
            else:
                result = await self._process_with_default_strategy(task)
            
            # Mark task as completed
            task.mark_completed(
                result=result["output"],
                confidence=result["confidence"],
                agents_used=result["agents_used"]
            )
            
            # Update metrics
            self._update_metrics(task)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            log.info(f"Task {task.task_id} completed in {task.processing_time:.2f}s with {task.agents_used} agents")
            
        except Exception as e:
            log.error(f"Task processing failed: {e}")
            task.result = {"error": str(e)}
            task.confidence = 0.0
            
            # Move to completed tasks even if failed
            self.completed_tasks.append(task)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _process_with_single_agent(self, task: SwarmProcessingTask) -> Dict[str, Any]:
        """Process task with single agent"""
        # Use appropriate vertical generator if specified
        if task.vertical_domain and task.vertical_domain in self.vertical_generators:
            generator = self.vertical_generators[task.vertical_domain]
            
            # Create output spec
            from ..output.generators.base import OutputSpec, OutputFormat, GenerationQuality
            spec = OutputSpec(
                format=OutputFormat(task.output_format),
                quality=GenerationQuality.PRODUCTION,
                requirements=task.output_requirements
            )
            
            # Generate output
            output = await generator.generate(task.input_data, spec)
            
            return {
                "output": output.content,
                "confidence": output.confidence,
                "agents_used": 1,
                "processing_method": "single_agent_vertical"
            }
        else:
            # Use generic processing
            return {
                "output": {
                    "processed_data": task.input_data,
                    "metadata": task.input_metadata,
                    "processing_type": "single_agent_generic"
                },
                "confidence": 0.7,
                "agents_used": 1,
                "processing_method": "single_agent_generic"
            }
    
    async def _process_with_medium_swarm(self, task: SwarmProcessingTask) -> Dict[str, Any]:
        """Process task with medium swarm (50-200 agents)"""
        if not self.swarm_system:
            # Fallback to single agent if swarm not available
            return await self._process_with_single_agent(task)
        
        try:
            # Create swarm goal
            goal = UnifiedGoal(
                goal_id=f"medium_swarm_{task.task_id}",
                description=f"Process {task.objective.value} with medium swarm",
                objective=self._map_processing_objective_to_swarm(task.objective),
                expected_scale=SwarmScale.MEDIUM,
                requirements={
                    "input_data": task.input_data,
                    "output_format": task.output_format,
                    "quality_threshold": task.quality_threshold,
                    "specializations": task.agent_specializations
                }
            )
            
            # Execute with swarm
            result = await self.swarm_system.execute_goal(goal)
            
            return {
                "output": result.result,
                "confidence": result.confidence,
                "agents_used": result.total_agents_used,
                "processing_method": "medium_swarm",
                "swarm_metrics": result.execution_metrics
            }
            
        except Exception as e:
            log.error(f"Medium swarm processing failed: {e}")
            # Fallback to single agent
            return await self._process_with_single_agent(task)
    
    async def _process_with_large_swarm(self, task: SwarmProcessingTask) -> Dict[str, Any]:
        """Process task with large swarm (200+ agents)"""
        if not self.swarm_system:
            return await self._process_with_single_agent(task)
        
        try:
            # Create goal for large-scale processing
            goal = UnifiedGoal(
                goal_id=f"large_swarm_{task.task_id}",
                description=f"Large-scale processing: {task.objective.value}",
                objective=SwarmObjective.OPTIMIZE_QUALITY,
                expected_scale=SwarmScale.LARGE if task.scale == ProcessingScale.LARGE_SWARM else SwarmScale.MASSIVE,
                requirements={
                    "input_data": task.input_data,
                    "output_format": task.output_format,
                    "processing_objective": task.objective.value,
                    "agent_specializations": task.agent_specializations,
                    "coordination_strategy": task.coordination_strategy,
                    "fault_tolerance": task.fault_tolerance
                }
            )
            
            # Execute with large swarm
            result = await self.swarm_system.execute_goal(goal)
            
            # Update peak agents metric
            if result.total_agents_used > self.metrics["peak_agents_deployed"]:
                self.metrics["peak_agents_deployed"] = result.total_agents_used
            
            return {
                "output": result.result,
                "confidence": result.confidence,
                "agents_used": result.total_agents_used,
                "processing_method": "large_swarm",
                "swarm_metrics": result.execution_metrics,
                "coordination_efficiency": result.coordination_efficiency
            }
            
        except Exception as e:
            log.error(f"Large swarm processing failed: {e}")
            return await self._process_with_medium_swarm(task)
    
    async def _process_codebase_analysis(self, task: SwarmProcessingTask) -> Dict[str, Any]:
        """Process comprehensive codebase analysis with 400+ specialized agents"""
        if not self.swarm_system:
            return await self._process_with_single_agent(task)
        
        try:
            analysis_task: CodebaseAnalysisTask = task.input_data
            
            # Create specialized goals for different aspects of codebase analysis
            analysis_goals = []
            
            # Capability discovery goal
            if analysis_task.discover_capabilities:
                capability_goal = UnifiedGoal(
                    goal_id=f"capabilities_{analysis_task.analysis_id}",
                    description="Discover all capabilities in codebase",
                    objective=SwarmObjective.COMPREHENSIVE_ANALYSIS,
                    expected_scale=SwarmScale.LARGE,
                    requirements={
                        "codebase_path": analysis_task.codebase_path,
                        "analysis_type": "capability_discovery",
                        "agent_count": analysis_task.specialized_agents.get("python_analyzer", 50),
                        "file_priorities": analysis_task.file_priorities
                    }
                )
                analysis_goals.append(capability_goal)
            
            # Integration mapping goal
            if analysis_task.map_integrations:
                integration_goal = UnifiedGoal(
                    goal_id=f"integrations_{analysis_task.analysis_id}",
                    description="Map all system integrations",
                    objective=SwarmObjective.COMPREHENSIVE_ANALYSIS,
                    expected_scale=SwarmScale.MEDIUM,
                    requirements={
                        "codebase_path": analysis_task.codebase_path,
                        "analysis_type": "integration_mapping",
                        "agent_count": analysis_task.specialized_agents.get("integration_mapper", 15)
                    }
                )
                analysis_goals.append(integration_goal)
            
            # Quality assessment goal
            if analysis_task.assess_quality:
                quality_goal = UnifiedGoal(
                    goal_id=f"quality_{analysis_task.analysis_id}",
                    description="Assess code quality and maintainability",
                    objective=SwarmObjective.QUALITY_ASSESSMENT,
                    expected_scale=SwarmScale.MEDIUM,
                    requirements={
                        "codebase_path": analysis_task.codebase_path,
                        "analysis_type": "quality_assessment",
                        "agent_count": analysis_task.specialized_agents.get("quality_assessor", 20)
                    }
                )
                analysis_goals.append(quality_goal)
            
            # Security analysis goal
            if analysis_task.security_analysis:
                security_goal = UnifiedGoal(
                    goal_id=f"security_{analysis_task.analysis_id}",
                    description="Comprehensive security analysis",
                    objective=SwarmObjective.SECURITY_ANALYSIS,
                    expected_scale=SwarmScale.MEDIUM,
                    requirements={
                        "codebase_path": analysis_task.codebase_path,
                        "analysis_type": "security_analysis",
                        "agent_count": analysis_task.specialized_agents.get("security_analyzer", 25)
                    }
                )
                analysis_goals.append(security_goal)
            
            # Execute all goals in parallel
            results = await asyncio.gather(*[
                self.swarm_system.execute_goal(goal) for goal in analysis_goals
            ], return_exceptions=True)
            
            # Aggregate results
            total_agents_used = 0
            combined_confidence = 0.0
            analysis_results = {
                "analysis_id": analysis_task.analysis_id,
                "codebase_path": analysis_task.codebase_path,
                "analysis_depth": analysis_task.analysis_depth,
                "capabilities_discovered": [],
                "integrations_mapped": [],
                "quality_metrics": {},
                "security_findings": [],
                "architecture_insights": {},
                "processing_summary": {
                    "total_goals_executed": len(analysis_goals),
                    "successful_goals": 0,
                    "failed_goals": 0
                }
            }
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log.error(f"Analysis goal {i} failed: {result}")
                    analysis_results["processing_summary"]["failed_goals"] += 1
                    continue
                
                analysis_results["processing_summary"]["successful_goals"] += 1
                total_agents_used += result.total_agents_used
                combined_confidence += result.confidence
                
                # Extract specific analysis results based on goal type
                goal = analysis_goals[i]
                analysis_type = goal.requirements.get("analysis_type", "unknown")
                
                if analysis_type == "capability_discovery":
                    analysis_results["capabilities_discovered"] = result.result.get("capabilities", [])
                elif analysis_type == "integration_mapping":
                    analysis_results["integrations_mapped"] = result.result.get("integrations", [])
                elif analysis_type == "quality_assessment":
                    analysis_results["quality_metrics"] = result.result.get("quality_metrics", {})
                elif analysis_type == "security_analysis":
                    analysis_results["security_findings"] = result.result.get("security_findings", [])
            
            # Calculate average confidence
            successful_goals = analysis_results["processing_summary"]["successful_goals"]
            if successful_goals > 0:
                combined_confidence /= successful_goals
            
            return {
                "output": analysis_results,
                "confidence": combined_confidence,
                "agents_used": total_agents_used,
                "processing_method": "codebase_analysis_swarm",
                "analysis_depth": analysis_task.analysis_depth,
                "specialized_agents_deployed": analysis_task.specialized_agents
            }
            
        except Exception as e:
            log.error(f"Codebase analysis failed: {e}")
            return {
                "output": {"error": str(e), "analysis_id": task.input_data.analysis_id},
                "confidence": 0.0,
                "agents_used": 0,
                "processing_method": "codebase_analysis_failed"
            }
    
    async def _process_with_default_strategy(self, task: SwarmProcessingTask) -> Dict[str, Any]:
        """Default processing strategy"""
        return await self._process_with_medium_swarm(task)
    
    def _map_processing_objective_to_swarm(self, objective: ProcessingObjective) -> SwarmObjective:
        """Map processing objective to swarm objective"""
        mapping = {
            ProcessingObjective.REAL_TIME_ANALYSIS: SwarmObjective.OPTIMIZE_SPEED,
            ProcessingObjective.COMPREHENSIVE_ANALYSIS: SwarmObjective.COMPREHENSIVE_ANALYSIS,
            ProcessingObjective.CODEBASE_UNDERSTANDING: SwarmObjective.COMPREHENSIVE_ANALYSIS,
            ProcessingObjective.CAPABILITY_DISCOVERY: SwarmObjective.COMPREHENSIVE_ANALYSIS,
            ProcessingObjective.INTEGRATION_MAPPING: SwarmObjective.COMPREHENSIVE_ANALYSIS,
            ProcessingObjective.QUALITY_ASSESSMENT: SwarmObjective.OPTIMIZE_QUALITY,
            ProcessingObjective.SECURITY_ANALYSIS: SwarmObjective.COMPREHENSIVE_ANALYSIS,
            ProcessingObjective.PERFORMANCE_OPTIMIZATION: SwarmObjective.OPTIMIZE_SPEED
        }
        return mapping.get(objective, SwarmObjective.OPTIMIZE_QUALITY)
    
    def _validate_task(self, task: SwarmProcessingTask) -> bool:
        """Validate task configuration"""
        if not task.input_data:
            log.error("Task input data is required")
            return False
        
        if task.scale == ProcessingScale.CODEBASE_ANALYSIS:
            if not isinstance(task.input_data, CodebaseAnalysisTask):
                log.error("Codebase analysis requires CodebaseAnalysisTask input")
                return False
        
        return True
    
    def _update_metrics(self, task: SwarmProcessingTask):
        """Update processing metrics"""
        self.metrics["total_tasks_processed"] += 1
        
        if task.result and "error" not in task.result:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        
        # Update average processing time
        total_tasks = self.metrics["total_tasks_processed"]
        current_avg = self.metrics["avg_processing_time"]
        self.metrics["avg_processing_time"] = (
            (current_avg * (total_tasks - 1) + task.processing_time) / total_tasks
        )
        
        # Update agent hours
        self.metrics["total_agent_hours"] += (task.agents_used * task.processing_time / 3600)
        
        # Update throughput
        if total_tasks > 0:
            total_time_hours = (time.time() - self.metrics.get("start_time", time.time())) / 3600
            if total_time_hours > 0:
                self.metrics["throughput_per_hour"] = total_tasks / total_time_hours
    
    async def _monitor_active_tasks(self):
        """Monitor active tasks for timeouts and issues"""
        while True:
            try:
                current_time = time.time()
                expired_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    if task.is_expired():
                        expired_tasks.append(task_id)
                        log.warning(f"Task {task_id} expired after {task.max_processing_time}s")
                
                # Handle expired tasks
                for task_id in expired_tasks:
                    task = self.active_tasks[task_id]
                    task.result = {"error": "Task timeout", "expired_at": current_time}
                    task.confidence = 0.0
                    
                    self.completed_tasks.append(task)
                    del self.active_tasks[task_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Task monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": {
                    "started_at": task.started_at,
                    "elapsed_time": time.time() - task.started_at if task.started_at else 0,
                    "max_processing_time": task.max_processing_time
                },
                "objective": task.objective.value,
                "scale": task.scale.value,
                "priority": task.priority.value
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task.result,
                    "confidence": task.confidence,
                    "agents_used": task.agents_used,
                    "processing_time": task.processing_time,
                    "completed_at": task.completed_at
                }
        
        return None
    
    def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator metrics"""
        return {
            "processing_metrics": self.metrics,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.processing_queue.qsize(),
            "swarm_available": SWARM_AVAILABLE,
            "vertical_generators": list(self.vertical_generators.keys()),
            "stream_stats": self.stream_engine.get_stream_stats() if hasattr(self.stream_engine, 'get_stream_stats') else {},
            "security_metrics": self.security_framework.get_security_metrics()
        }
    
    async def shutdown(self):
        """Shutdown the coordinator"""
        log.info("Shutting down Universal I/O Swarm Coordinator")
        
        # Stop stream processing
        await self.stream_engine.shutdown()
        
        # Stop event processing
        await self.event_processor.shutdown()
        
        # Cancel remaining tasks
        for task in self.active_tasks.values():
            task.result = {"error": "System shutdown"}
            task.confidence = 0.0
        
        log.info("Universal I/O Swarm Coordinator shutdown complete")

# Convenience functions for common operations
async def process_with_swarm(input_data: Any, objective: ProcessingObjective,
                           scale: ProcessingScale = ProcessingScale.MEDIUM_SWARM,
                           output_format: str = "json",
                           vertical_domain: VerticalDomain = None) -> str:
    """Convenience function to process data with swarm"""
    coordinator = UniversalIOSwarmCoordinator()
    await coordinator.initialize()
    
    task = SwarmProcessingTask(
        input_data=input_data,
        objective=objective,
        scale=scale,
        output_format=output_format,
        vertical_domain=vertical_domain
    )
    
    return await coordinator.submit_processing_task(task)

async def analyze_codebase(codebase_path: str, analysis_depth: str = "comprehensive") -> str:
    """Convenience function to analyze entire codebase"""
    coordinator = UniversalIOSwarmCoordinator()
    await coordinator.initialize()
    
    analysis_task = CodebaseAnalysisTask(
        codebase_path=codebase_path,
        analysis_depth=analysis_depth
    )
    
    return await coordinator.submit_codebase_analysis(analysis_task)

# Global coordinator instance
_global_coordinator = None

async def get_global_coordinator() -> UniversalIOSwarmCoordinator:
    """Get global coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = UniversalIOSwarmCoordinator()
        await _global_coordinator.initialize()
    return _global_coordinator
