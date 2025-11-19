"""
Autonomous Goal Decomposition and Task Planning Framework
Breaks down complex objectives into executable tasks autonomously
Plans optimal execution strategy with resource allocation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger("autonomous-goal-decomposition")

class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"          # <1 minute
    SIMPLE = "simple"            # 1-5 minutes
    MODERATE = "moderate"        # 5-30 minutes
    COMPLEX = "complex"          # 30-120 minutes
    VERY_COMPLEX = "very_complex"  # >120 minutes

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 10    # Must be done immediately
    HIGH = 8         # Should be done soon
    MEDIUM = 5       # Normal priority
    LOW = 3          # Can be delayed
    DEFERRED = 1     # Do when resources available

class TaskStatus(Enum):
    """Task execution status"""
    PLANNED = "planned"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """A single executable task"""
    task_id: str
    description: str
    complexity: TaskComplexity
    priority: TaskPriority
    estimated_duration: float  # seconds
    required_capabilities: List[str]
    required_resources: Dict[str, Any]
    dependencies: List[str]  # task_ids that must complete first
    success_criteria: List[str]
    outputs: List[str]
    status: TaskStatus = TaskStatus.PLANNED
    assigned_agents: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class Goal:
    """High-level goal to be decomposed"""
    goal_id: str
    description: str
    objective: str
    success_metrics: List[str]
    constraints: Dict[str, Any]
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.MEDIUM

@dataclass
class ExecutionPlan:
    """Complete execution plan for a goal"""
    plan_id: str
    goal: Goal
    tasks: List[Task]
    task_graph: Dict[str, List[str]]  # task_id -> dependent task_ids
    critical_path: List[str]  # task_ids in order
    estimated_total_time: float
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    created_at: float
    confidence: float

class AutonomousGoalDecomposer:
    """
    Autonomously decomposes high-level goals into executable tasks.
    Creates optimal execution plans with dependency resolution.
    """
    
    def __init__(self):
        self.decomposition_patterns: Dict[str, List[str]] = {}
        self.task_templates: Dict[str, Task] = {}
        self.execution_history: List[ExecutionPlan] = []
        
        # Learning system
        self.successful_patterns: Dict[str, float] = {}
        self.task_duration_estimates: Dict[str, float] = {}
        
        self._initialize_decomposition_patterns()
        
        log.info("Autonomous Goal Decomposer initialized")
    
    def _initialize_decomposition_patterns(self):
        """Initialize known goal decomposition patterns"""
        
        # Intelligence analysis pattern
        self.decomposition_patterns["intelligence_analysis"] = [
            "data_collection",
            "data_validation",
            "multi_source_fusion",
            "pattern_recognition",
            "threat_assessment",
            "impact_analysis",
            "recommendation_generation"
        ]
        
        # Threat response pattern
        self.decomposition_patterns["threat_response"] = [
            "threat_validation",
            "asset_identification",
            "course_of_action_development",
            "risk_assessment",
            "resource_allocation",
            "execution_planning",
            "monitoring_setup"
        ]
        
        # Infrastructure protection pattern
        self.decomposition_patterns["infrastructure_protection"] = [
            "vulnerability_assessment",
            "critical_asset_identification",
            "threat_modeling",
            "protection_measure_design",
            "redundancy_planning",
            "response_capability_development",
            "monitoring_implementation"
        ]
        
        # Cyber defense pattern
        self.decomposition_patterns["cyber_defense"] = [
            "threat_detection",
            "incident_analysis",
            "containment_planning",
            "eradication_strategy",
            "recovery_planning",
            "lessons_learned"
        ]
        
        # Battle planning pattern
        self.decomposition_patterns["battle_planning"] = [
            "mission_analysis",
            "intelligence_preparation",
            "course_of_action_development",
            "wargaming",
            "decision_briefing_preparation",
            "execution_order_development"
        ]
    
    async def decompose_goal(
        self,
        goal: Goal,
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        Autonomously decompose a goal into executable tasks.
        Returns complete execution plan with dependencies.
        """
        
        start_time = time.time()
        
        log.info(f"Decomposing goal: {goal.description}")
        
        # Step 1: Identify goal pattern
        pattern_type = self._identify_goal_pattern(goal, context)
        log.info(f"Identified pattern: {pattern_type}")
        
        # Step 2: Generate tasks from pattern
        tasks = await self._generate_tasks_from_pattern(goal, pattern_type, context)
        log.info(f"Generated {len(tasks)} tasks")
        
        # Step 3: Analyze dependencies
        task_graph = self._analyze_dependencies(tasks)
        
        # Step 4: Calculate critical path
        critical_path = self._calculate_critical_path(tasks, task_graph)
        
        # Step 5: Estimate total time
        total_time = self._estimate_total_time(tasks, critical_path)
        
        # Step 6: Assess resources
        resources = self._assess_resource_requirements(tasks)
        
        # Step 7: Risk assessment
        risks = await self._assess_risks(goal, tasks, context)
        
        # Step 8: Calculate confidence
        confidence = self._calculate_plan_confidence(goal, tasks, context)
        
        plan = ExecutionPlan(
            plan_id=f"plan_{goal.goal_id}_{int(time.time())}",
            goal=goal,
            tasks=tasks,
            task_graph=task_graph,
            critical_path=critical_path,
            estimated_total_time=total_time,
            resource_requirements=resources,
            risk_assessment=risks,
            created_at=time.time(),
            confidence=confidence
        )
        
        self.execution_history.append(plan)
        
        planning_time = time.time() - start_time
        log.info(f"Goal decomposition complete: {len(tasks)} tasks, "
                f"estimated time {total_time/60:.1f} minutes, "
                f"confidence {confidence:.2%}, "
                f"planning took {planning_time:.2f}s")
        
        return plan
    
    def _identify_goal_pattern(
        self,
        goal: Goal,
        context: Dict[str, Any]
    ) -> str:
        """Identify which decomposition pattern to use"""
        
        goal_lower = goal.description.lower()
        objective_lower = goal.objective.lower()
        
        # Pattern matching
        if any(word in goal_lower or word in objective_lower 
              for word in ["analyze", "assess", "intelligence", "threat"]):
            return "intelligence_analysis"
        
        elif any(word in goal_lower or word in objective_lower 
                for word in ["respond", "counter", "defend", "protect"]):
            return "threat_response"
        
        elif any(word in goal_lower or word in objective_lower 
                for word in ["infrastructure", "critical", "resilience"]):
            return "infrastructure_protection"
        
        elif any(word in goal_lower or word in objective_lower 
                for word in ["cyber", "network", "intrusion"]):
            return "cyber_defense"
        
        elif any(word in goal_lower or word in objective_lower 
                for word in ["battle", "operation", "mission", "campaign"]):
            return "battle_planning"
        
        else:
            return "intelligence_analysis"  # Default
    
    async def _generate_tasks_from_pattern(
        self,
        goal: Goal,
        pattern_type: str,
        context: Dict[str, Any]
    ) -> List[Task]:
        """Generate concrete tasks from pattern"""
        
        pattern = self.decomposition_patterns.get(pattern_type, 
                                                  self.decomposition_patterns["intelligence_analysis"])
        
        tasks = []
        
        for idx, task_type in enumerate(pattern):
            task = self._create_task(
                goal=goal,
                task_type=task_type,
                sequence=idx,
                total_tasks=len(pattern),
                context=context
            )
            tasks.append(task)
        
        # Add goal-specific tasks
        if context and context.get("additional_requirements"):
            for req in context["additional_requirements"]:
                additional_task = self._create_custom_task(goal, req, len(tasks))
                tasks.append(additional_task)
        
        return tasks
    
    def _create_task(
        self,
        goal: Goal,
        task_type: str,
        sequence: int,
        total_tasks: int,
        context: Dict[str, Any]
    ) -> Task:
        """Create a task from template"""
        
        # Determine complexity
        if task_type in ["data_collection", "data_validation"]:
            complexity = TaskComplexity.SIMPLE
            duration = 300  # 5 minutes
        elif task_type in ["pattern_recognition", "threat_assessment"]:
            complexity = TaskComplexity.MODERATE
            duration = 900  # 15 minutes
        elif task_type in ["wargaming", "course_of_action_development"]:
            complexity = TaskComplexity.COMPLEX
            duration = 3600  # 1 hour
        else:
            complexity = TaskComplexity.MODERATE
            duration = 600  # 10 minutes
        
        # Inherit priority from goal
        priority = goal.priority
        
        # First and last tasks are often critical
        if sequence == 0 or sequence == total_tasks - 1:
            if priority.value < TaskPriority.HIGH.value:
                priority = TaskPriority.HIGH
        
        # Determine dependencies (tasks depend on previous tasks in sequence)
        dependencies = []
        if sequence > 0:
            dependencies.append(f"{goal.goal_id}_task_{sequence-1}")
        
        # Determine required capabilities
        capabilities = self._get_capabilities_for_task_type(task_type)
        
        # Success criteria
        success_criteria = [
            f"{task_type.replace('_', ' ').title()} completed successfully",
            "Quality validation passed",
            "Confidence threshold met"
        ]
        
        return Task(
            task_id=f"{goal.goal_id}_task_{sequence}",
            description=f"{task_type.replace('_', ' ').title()} for {goal.objective}",
            complexity=complexity,
            priority=priority,
            estimated_duration=duration,
            required_capabilities=capabilities,
            required_resources={"agents": len(capabilities) * 2, "time": duration},
            dependencies=dependencies,
            success_criteria=success_criteria,
            outputs=[f"{task_type}_result"]
        )
    
    def _create_custom_task(
        self,
        goal: Goal,
        requirement: str,
        sequence: int
    ) -> Task:
        """Create custom task from requirement"""
        
        return Task(
            task_id=f"{goal.goal_id}_custom_{sequence}",
            description=requirement,
            complexity=TaskComplexity.MODERATE,
            priority=goal.priority,
            estimated_duration=600,
            required_capabilities=["general_analysis"],
            required_resources={"agents": 5, "time": 600},
            dependencies=[],
            success_criteria=["Requirement fulfilled"],
            outputs=["custom_result"]
        )
    
    def _get_capabilities_for_task_type(self, task_type: str) -> List[str]:
        """Get required capabilities for task type"""
        
        capability_map = {
            "data_collection": ["data_ingestion", "source_validation"],
            "data_validation": ["data_quality", "validation"],
            "multi_source_fusion": ["fusion", "correlation", "confidence_weighting"],
            "pattern_recognition": ["pattern_matching", "anomaly_detection"],
            "threat_assessment": ["threat_analysis", "risk_calculation"],
            "impact_analysis": ["cascade_modeling", "impact_quantification"],
            "recommendation_generation": ["synthesis", "decision_support"],
            "course_of_action_development": ["coa_generation", "option_analysis"],
            "wargaming": ["simulation", "outcome_prediction"],
            "mission_analysis": ["requirements_analysis", "constraint_identification"]
        }
        
        return capability_map.get(task_type, ["general_analysis"])
    
    def _analyze_dependencies(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build dependency graph"""
        
        graph = {}
        
        for task in tasks:
            graph[task.task_id] = task.dependencies.copy()
        
        return graph
    
    def _calculate_critical_path(
        self,
        tasks: List[Task],
        task_graph: Dict[str, List[str]]
    ) -> List[str]:
        """Calculate critical path through task graph"""
        
        # Topological sort with longest path
        task_dict = {t.task_id: t for t in tasks}
        
        # Calculate earliest start times
        earliest_start = {}
        
        def calculate_earliest(task_id: str) -> float:
            if task_id in earliest_start:
                return earliest_start[task_id]
            
            deps = task_graph.get(task_id, [])
            if not deps:
                earliest_start[task_id] = 0
                return 0
            
            max_dep_finish = max(
                calculate_earliest(dep) + task_dict[dep].estimated_duration
                for dep in deps
            )
            
            earliest_start[task_id] = max_dep_finish
            return max_dep_finish
        
        for task in tasks:
            calculate_earliest(task.task_id)
        
        # Find tasks on critical path (longest path to end)
        critical_path = []
        
        # Simple approach: tasks with zero slack
        for task in sorted(tasks, key=lambda t: earliest_start.get(t.task_id, 0)):
            critical_path.append(task.task_id)
        
        return critical_path
    
    def _estimate_total_time(
        self,
        tasks: List[Task],
        critical_path: List[str]
    ) -> float:
        """Estimate total execution time"""
        
        task_dict = {t.task_id: t for t in tasks}
        
        # Sum critical path durations
        total = sum(task_dict[task_id].estimated_duration for task_id in critical_path)
        
        return total
    
    def _assess_resource_requirements(self, tasks: List[Task]) -> Dict[str, Any]:
        """Assess resource requirements for plan"""
        
        total_agents = sum(t.required_resources.get("agents", 0) for t in tasks)
        peak_agents = max(t.required_resources.get("agents", 0) for t in tasks)
        
        all_capabilities = set()
        for task in tasks:
            all_capabilities.update(task.required_capabilities)
        
        return {
            "total_agent_hours": total_agents,
            "peak_concurrent_agents": peak_agents,
            "unique_capabilities": list(all_capabilities),
            "capability_count": len(all_capabilities),
            "estimated_cost": total_agents * 10  # $10 per agent-hour
        }
    
    async def _assess_risks(
        self,
        goal: Goal,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess execution risks"""
        
        risks = {
            "complexity_risk": "LOW",
            "dependency_risk": "LOW",
            "resource_risk": "LOW",
            "timeline_risk": "LOW",
            "identified_risks": []
        }
        
        # Complexity risk
        complex_tasks = [t for t in tasks if t.complexity in [
            TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX
        ]]
        
        if len(complex_tasks) / len(tasks) > 0.3:
            risks["complexity_risk"] = "HIGH"
            risks["identified_risks"].append(
                f"{len(complex_tasks)} complex tasks may cause delays"
            )
        
        # Dependency risk
        max_deps = max(len(t.dependencies) for t in tasks)
        if max_deps > 3:
            risks["dependency_risk"] = "MEDIUM"
            risks["identified_risks"].append(
                f"Task with {max_deps} dependencies may bottleneck execution"
            )
        
        # Timeline risk
        if goal.deadline:
            estimated_time = sum(t.estimated_duration for t in tasks)
            time_available = goal.deadline - time.time()
            
            if estimated_time > time_available:
                risks["timeline_risk"] = "HIGH"
                risks["identified_risks"].append(
                    f"Estimated time {estimated_time/60:.0f}min exceeds deadline by {(estimated_time - time_available)/60:.0f}min"
                )
        
        return risks
    
    def _calculate_plan_confidence(
        self,
        goal: Goal,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in execution plan"""
        
        confidence = 0.9  # Base confidence
        
        # Reduce for complexity
        complex_tasks = [t for t in tasks if t.complexity in [
            TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX
        ]]
        
        if complex_tasks:
            confidence -= len(complex_tasks) * 0.02
        
        # Reduce for unknown capabilities
        known_capabilities = set(self.decomposition_patterns.keys())
        all_capabilities = set()
        for task in tasks:
            all_capabilities.update(task.required_capabilities)
        
        unknown = all_capabilities - known_capabilities
        if unknown:
            confidence -= len(unknown) * 0.05
        
        # Increase for historical success
        pattern_type = self._identify_goal_pattern(goal, context)
        if pattern_type in self.successful_patterns:
            historical_success = self.successful_patterns[pattern_type]
            confidence = confidence * 0.7 + historical_success * 0.3
        
        return max(min(confidence, 0.99), 0.5)
    
    async def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize execution plan"""
        
        log.info(f"Optimizing plan {plan.plan_id}")
        
        # Identify parallelization opportunities
        parallelizable = self._find_parallelizable_tasks(plan.tasks, plan.task_graph)
        
        if parallelizable:
            log.info(f"Found {len(parallelizable)} parallelization opportunities")
            
            # Adjust estimated time for parallel execution
            time_savings = sum(
                plan.tasks[i].estimated_duration * 0.3 
                for i in range(len(plan.tasks))
                if plan.tasks[i].task_id in parallelizable
            )
            
            plan.estimated_total_time -= time_savings
            
            log.info(f"Parallel optimization saves {time_savings/60:.1f} minutes")
        
        return plan
    
    def _find_parallelizable_tasks(
        self,
        tasks: List[Task],
        task_graph: Dict[str, List[str]]
    ) -> Set[str]:
        """Find tasks that can be executed in parallel"""
        
        parallelizable = set()
        
        # Tasks with same dependencies can run in parallel
        dependency_groups = {}
        for task in tasks:
            dep_key = tuple(sorted(task.dependencies))
            if dep_key not in dependency_groups:
                dependency_groups[dep_key] = []
            dependency_groups[dep_key].append(task.task_id)
        
        for dep_key, task_ids in dependency_groups.items():
            if len(task_ids) > 1:
                parallelizable.update(task_ids)
        
        return parallelizable


# Global instance
autonomous_goal_decomposer = AutonomousGoalDecomposer()


async def decompose_and_plan(
    goal_description: str,
    objective: str,
    success_metrics: List[str] = None,
    constraints: Dict[str, Any] = None,
    deadline: Optional[float] = None,
    context: Dict[str, Any] = None
) -> ExecutionPlan:
    """
    Main entry point: Decompose goal and create execution plan.
    """
    
    goal = Goal(
        goal_id=f"goal_{int(time.time())}",
        description=goal_description,
        objective=objective,
        success_metrics=success_metrics or ["Objective achieved"],
        constraints=constraints or {},
        deadline=deadline
    )
    
    plan = await autonomous_goal_decomposer.decompose_goal(goal, context)
    optimized_plan = await autonomous_goal_decomposer.optimize_plan(plan)
    
    return optimized_plan

