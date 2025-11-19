"""
AgentForge Intelligence Module
Autonomous intelligence analysis with human-level intuition and machine speed

This module provides:
- Intelligent agent specialization and autonomous spawning
- Multi-domain intelligence fusion
- TTP (Tactics, Techniques, Procedures) pattern recognition
- Cascading effect analysis
- Autonomous capability gap detection
- Master intelligence orchestration

Usage:
    from services.swarm.intelligence import process_intelligence
    
    response = await process_intelligence(
        task_description="Analyze multi-source intelligence for threat indicators",
        available_data=[...],
        context={...}
    )
"""

from .agent_specialization_engine import (
    intelligent_specialization_engine,
    AgentDomain,
    AgentCapabilityLevel,
    AgentSpecialization,
    TaskAnalysis,
    analyze_task_and_determine_agents
)

from .capability_gap_analyzer import (
    capability_gap_analyzer,
    CapabilityGap,
    CapabilityGapSeverity,
    AgentSpawnRequest,
    analyze_and_identify_gaps
)

from .multi_domain_fusion import (
    multi_domain_fusion_system,
    IntelligenceInject,
    IntelligenceDomain,
    SourceCredibility,
    InjectCorrelation,
    FusedIntelligence,
    process_intelligence_inject
)

from .ttp_pattern_recognition import (
    ttp_recognition_engine,
    TTPCategory,
    OperationType,
    TTPPattern,
    TTPDetection,
    CampaignAssessment,
    recognize_ttp_patterns
)

from .cascading_effect_analyzer import (
    cascade_analyzer,
    EffectCategory,
    ImpactSeverity,
    SystemType,
    CascadeEffect,
    CascadeAnalysis,
    analyze_cascade_effects
)

from .master_intelligence_orchestrator import (
    master_orchestrator,
    ProcessingPhase,
    IntelligenceRequest,
    IntelligenceResponse,
    process_intelligence
)

from .comprehensive_threat_library import (
    comprehensive_threat_library,
    ComprehensiveThreat,
    ThreatDomain,
    ThreatActor,
    CombatantCommand,
    get_threat_library
)

from .self_healing_orchestrator import (
    self_healing_orchestrator,
    ValidationStatus,
    CorrectionAction,
    ValidationResult,
    CorrectionResult,
    validate_and_heal_result
)

from .swarm_integration_bridge import (
    intelligence_swarm_bridge,
    IntegrationMode,
    IntegratedTask,
    process_with_full_integration,
    connect_swarm_systems
)

from .realtime_intelligence_stream import (
    realtime_intelligence_stream,
    IntelligenceEventType,
    StreamPriority,
    IntelligenceEvent,
    StreamSubscription,
    stream_ttp_detection,
    stream_threat_identified,
    stream_campaign_detected,
    stream_fusion_complete,
    stream_cascade_prediction,
    stream_agent_spawned,
    stream_gap_identified
)

from .autonomous_goal_decomposition import (
    autonomous_goal_decomposer,
    Task,
    TaskComplexity,
    TaskPriority,
    TaskStatus,
    Goal,
    ExecutionPlan,
    decompose_and_plan
)

from .coa_generation import (
    coa_generator,
    COAType,
    EffectType,
    COAElement,
    COAPhase,
    CourseOfAction,
    COAComparison,
    generate_courses_of_action
)

from .wargaming_simulation import (
    wargaming_simulator,
    ForceType,
    OutcomeType,
    CriticalEvent,
    WargameResult,
    WargameComparison,
    simulate_and_compare_coas
)

from .easy_interface import (
    easy,
    EasyIntelligence,
    SimpleRequest,
    SimpleResponse,
    analyze_threat,
    make_plan,
    get_decision,
    analyze
)

__all__ = [
    # Agent Specialization
    "intelligent_specialization_engine",
    "AgentDomain",
    "AgentCapabilityLevel",
    "AgentSpecialization",
    "TaskAnalysis",
    "analyze_task_and_determine_agents",
    
    # Capability Gap Analysis
    "capability_gap_analyzer",
    "CapabilityGap",
    "CapabilityGapSeverity",
    "AgentSpawnRequest",
    "analyze_and_identify_gaps",
    
    # Multi-Domain Fusion
    "multi_domain_fusion_system",
    "IntelligenceInject",
    "IntelligenceDomain",
    "SourceCredibility",
    "InjectCorrelation",
    "FusedIntelligence",
    "process_intelligence_inject",
    
    # TTP Recognition
    "ttp_recognition_engine",
    "TTPCategory",
    "OperationType",
    "TTPPattern",
    "TTPDetection",
    "CampaignAssessment",
    "recognize_ttp_patterns",
    
    # Cascade Analysis
    "cascade_analyzer",
    "EffectCategory",
    "ImpactSeverity",
    "SystemType",
    "CascadeEffect",
    "CascadeAnalysis",
    "analyze_cascade_effects",
    
    # Master Orchestrator
    "master_orchestrator",
    "ProcessingPhase",
    "IntelligenceRequest",
    "IntelligenceResponse",
    "process_intelligence",
    
    # Comprehensive Threat Library
    "comprehensive_threat_library",
    "ComprehensiveThreat",
    "ThreatDomain",
    "ThreatActor",
    "CombatantCommand",
    "get_threat_library",
    
    # Self-Healing
    "self_healing_orchestrator",
    "ValidationStatus",
    "CorrectionAction",
    "ValidationResult",
    "CorrectionResult",
    "validate_and_heal_result",
    
    # Swarm Integration
    "intelligence_swarm_bridge",
    "IntegrationMode",
    "IntegratedTask",
    "process_with_full_integration",
    "connect_swarm_systems",
    
    # Real-Time Streaming
    "realtime_intelligence_stream",
    "IntelligenceEventType",
    "StreamPriority",
    "IntelligenceEvent",
    "StreamSubscription",
    "stream_ttp_detection",
    "stream_threat_identified",
    "stream_campaign_detected",
    "stream_fusion_complete",
    "stream_cascade_prediction",
    "stream_agent_spawned",
    "stream_gap_identified",
    
    # Goal Decomposition & Planning
    "autonomous_goal_decomposer",
    "Task",
    "TaskComplexity",
    "TaskPriority",
    "TaskStatus",
    "Goal",
    "ExecutionPlan",
    "decompose_and_plan",
    
    # COA Generation
    "coa_generator",
    "COAType",
    "EffectType",
    "COAElement",
    "COAPhase",
    "CourseOfAction",
    "COAComparison",
    "generate_courses_of_action",
    
    # Wargaming
    "wargaming_simulator",
    "ForceType",
    "OutcomeType",
    "CriticalEvent",
    "WargameResult",
    "WargameComparison",
    "simulate_and_compare_coas",
    
    # Easy Interface (User-Friendly)
    "easy",
    "EasyIntelligence",
    "SimpleRequest",
    "SimpleResponse",
    "analyze_threat",
    "make_plan",
    "get_decision",
    "analyze",
]

# Version
__version__ = "3.0.0"

