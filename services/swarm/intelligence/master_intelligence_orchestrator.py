"""
Master Intelligence Orchestrator
Coordinates all intelligence capabilities with human-level intuition and machine speed
Autonomous agent spawning, multi-domain fusion, TTP recognition, cascade analysis
Now with real-time streaming support for live battlefield intelligence
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Import streaming capabilities
try:
    from .realtime_intelligence_stream import (
        stream_ttp_detection, stream_threat_identified, stream_campaign_detected,
        stream_fusion_complete, stream_cascade_prediction, stream_agent_spawned,
        stream_gap_identified, StreamPriority
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Import all intelligence systems
from .agent_specialization_engine import (
    intelligent_specialization_engine,
    TaskAnalysis,
    analyze_task_and_determine_agents
)
from .capability_gap_analyzer import (
    capability_gap_analyzer,
    CapabilityGap,
    AgentSpawnRequest,
    analyze_and_identify_gaps
)
from .multi_domain_fusion import (
    multi_domain_fusion_system,
    IntelligenceInject,
    FusedIntelligence,
    IntelligenceDomain,
    SourceCredibility
)
from .ttp_pattern_recognition import (
    ttp_recognition_engine,
    TTPDetection,
    CampaignAssessment,
    recognize_ttp_patterns
)
from .cascading_effect_analyzer import (
    cascade_analyzer,
    CascadeAnalysis,
    analyze_cascade_effects
)
from .autonomous_goal_decomposition import (
    autonomous_goal_decomposer,
    Goal,
    ExecutionPlan,
    decompose_and_plan
)
from .coa_generation import (
    coa_generator,
    CourseOfAction,
    COAComparison,
    generate_courses_of_action
)
from .wargaming_simulation import (
    wargaming_simulator,
    WargameResult,
    WargameComparison,
    simulate_and_compare_coas
)

log = logging.getLogger("master-intelligence-orchestrator")

class ProcessingPhase(Enum):
    """Phases of intelligence processing"""
    INITIALIZATION = "initialization"
    AGENT_PLANNING = "agent_planning"
    DATA_INGESTION = "data_ingestion"
    MULTI_DOMAIN_FUSION = "multi_domain_fusion"
    TTP_RECOGNITION = "ttp_recognition"
    GAP_ANALYSIS = "gap_analysis"
    AGENT_SPAWNING = "agent_spawning"
    CASCADE_ANALYSIS = "cascade_analysis"
    GOAL_DECOMPOSITION = "goal_decomposition"
    COA_GENERATION = "coa_generation"
    WARGAMING = "wargaming"
    SYNTHESIS = "synthesis"
    FINALIZATION = "finalization"

@dataclass
class IntelligenceRequest:
    """Request for intelligence analysis"""
    request_id: str
    task_description: str
    available_data: List[Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    deadline: Optional[float] = None

@dataclass
class IntelligenceResponse:
    """Comprehensive intelligence response"""
    request_id: str
    
    # Core analysis
    task_analysis: TaskAnalysis
    fused_intelligence: List[FusedIntelligence]
    ttp_detections: List[TTPDetection]
    campaign_assessment: Optional[CampaignAssessment]
    cascade_analysis: Optional[CascadeAnalysis]
    
    # Planning and operations
    execution_plan: Optional[ExecutionPlan] = None
    coa_comparison: Optional[COAComparison] = None
    wargame_results: Optional[WargameComparison] = None
    
    # Agent information
    deployed_agents: List[str]
    spawned_agents: List[AgentSpawnRequest]
    agent_count: int
    
    # Quality metrics
    overall_confidence: float
    identified_gaps: List[CapabilityGap]
    
    # Synthesis
    executive_summary: str
    key_findings: List[str]
    threat_assessment: str
    recommended_actions: List[str]
    alternative_hypotheses: List[Dict[str, Any]]
    
    # Metadata
    processing_time: float
    processing_phases: List[Dict[str, Any]]
    timestamp: float

class MasterIntelligenceOrchestrator:
    """
    Master orchestrator for all intelligence capabilities.
    Coordinates autonomous agent spawning, multi-domain fusion,
    TTP recognition, and cascade analysis.
    """
    
    def __init__(self):
        self.active_requests: Dict[str, IntelligenceRequest] = {}
        self.processing_history: List[IntelligenceResponse] = []
        
        # Performance tracking
        self.total_requests_processed = 0
        self.avg_processing_time = 0.0
        self.avg_confidence = 0.0
        self.avg_agent_count = 0
        
        log.info("Master Intelligence Orchestrator initialized")
    
    async def process_intelligence_request(
        self,
        request: IntelligenceRequest
    ) -> IntelligenceResponse:
        """
        Process comprehensive intelligence request.
        Coordinates all intelligence systems to produce high-quality analysis.
        """
        
        start_time = time.time()
        processing_phases = []
        
        log.info(f"Processing intelligence request {request.request_id}")
        log.info(f"Task: {request.task_description[:100]}...")
        log.info(f"Data sources: {len(request.available_data)}")
        
        self.active_requests[request.request_id] = request
        
        # Phase 1: Initialization
        phase_start = time.time()
        await self._log_phase(ProcessingPhase.INITIALIZATION, processing_phases, phase_start)
        
        # Phase 2: Agent Planning
        phase_start = time.time()
        log.info("Phase 2: Agent Planning - Determining required agents...")
        task_analysis = await analyze_task_and_determine_agents(
            request.task_description,
            request.available_data,
            request.context
        )
        await self._log_phase(ProcessingPhase.AGENT_PLANNING, processing_phases, phase_start, {
            "agent_count": task_analysis.recommended_agent_count,
            "specializations": len(task_analysis.required_specializations)
        })
        
        deployed_agents = [spec.agent_type for spec in task_analysis.required_specializations]
        log.info(f"Deploying {task_analysis.recommended_agent_count} agents with "
                f"{len(task_analysis.required_specializations)} specializations")
        
        # Phase 3: Data Ingestion & Inject Processing
        phase_start = time.time()
        log.info("Phase 3: Data Ingestion - Converting data to intelligence injects...")
        injects = await self._convert_to_injects(request.available_data, request.context)
        await self._log_phase(ProcessingPhase.DATA_INGESTION, processing_phases, phase_start, {
            "injects_created": len(injects)
        })
        
        # Phase 4: Multi-Domain Fusion
        phase_start = time.time()
        log.info("Phase 4: Multi-Domain Fusion - Correlating intelligence sources...")
        fused_intelligence_list = []
        for inject in injects:
            fused_intel, correlations = await multi_domain_fusion_system.process_inject(inject)
            fused_intelligence_list.append(fused_intel)
        
        await self._log_phase(ProcessingPhase.MULTI_DOMAIN_FUSION, processing_phases, phase_start, {
            "fusion_results": len(fused_intelligence_list)
        })
        
        # Phase 5: TTP Recognition
        phase_start = time.time()
        log.info("Phase 5: TTP Recognition - Identifying adversary patterns...")
        ttp_detections, campaign_assessment = await recognize_ttp_patterns(
            request.available_data,
            request.context
        )
        await self._log_phase(ProcessingPhase.TTP_RECOGNITION, processing_phases, phase_start, {
            "ttps_detected": len(ttp_detections),
            "campaign_identified": campaign_assessment is not None
        })
        
        # Stream TTP detections
        if STREAMING_AVAILABLE and ttp_detections:
            for ttp in ttp_detections:
                await stream_ttp_detection(
                    ttp_name=ttp.pattern.name,
                    confidence=ttp.confidence,
                    details={
                        "indicators": ttp.observed_indicators,
                        "sequence_match": ttp.sequence_match,
                        "reasoning": ttp.reasoning
                    },
                    priority=StreamPriority.HIGH if ttp.confidence > 0.8 else StreamPriority.MEDIUM
                )
        
        # Stream campaign detection
        if STREAMING_AVAILABLE and campaign_assessment:
            await stream_campaign_detected(
                campaign_type=campaign_assessment.operation_type.value,
                campaign_stage=campaign_assessment.campaign_stage,
                intent=campaign_assessment.intent_assessment,
                details={
                    "threat_level": campaign_assessment.threat_level,
                    "confidence": campaign_assessment.confidence,
                    "predicted_next_steps": campaign_assessment.predicted_next_steps
                },
                priority=StreamPriority.CRITICAL if campaign_assessment.threat_level == "CRITICAL" else StreamPriority.HIGH
            )
        
        # Phase 6: Gap Analysis
        phase_start = time.time()
        log.info("Phase 6: Gap Analysis - Identifying capability gaps...")
        
        current_analysis = self._build_current_analysis(
            fused_intelligence_list, ttp_detections, campaign_assessment
        )
        
        # Calculate current confidence
        if fused_intelligence_list:
            current_confidence = sum(f.confidence for f in fused_intelligence_list) / len(fused_intelligence_list)
        else:
            current_confidence = 0.5
        
        identified_gaps, spawn_requests = await analyze_and_identify_gaps(
            current_analysis,
            request.available_data,
            deployed_agents,
            current_confidence
        )
        
        await self._log_phase(ProcessingPhase.GAP_ANALYSIS, processing_phases, phase_start, {
            "gaps_identified": len(identified_gaps),
            "spawn_requests": len(spawn_requests)
        })
        
        # Stream gap identifications
        if STREAMING_AVAILABLE and identified_gaps:
            for gap in identified_gaps[:5]:  # Top 5 gaps
                await stream_gap_identified(
                    gap_type=gap.gap_type,
                    severity=gap.severity.value,
                    description=gap.description,
                    recommended_actions=gap.recommended_agents,
                    priority=StreamPriority.HIGH if gap.severity.value == "critical" else StreamPriority.MEDIUM
                )
        
        # Phase 7: Agent Spawning (if needed)
        if spawn_requests:
            phase_start = time.time()
            log.info(f"Phase 7: Agent Spawning - Spawning {len(spawn_requests)} additional agents...")
            await self._simulate_agent_spawning(spawn_requests)
            deployed_agents.extend([req.agent_type for req in spawn_requests])
            await self._log_phase(ProcessingPhase.AGENT_SPAWNING, processing_phases, phase_start, {
                "agents_spawned": len(spawn_requests)
            })
            
            # Stream agent spawning events
            if STREAMING_AVAILABLE:
                for spawn_req in spawn_requests:
                    await stream_agent_spawned(
                        agent_type=spawn_req.agent_type,
                        reason=spawn_req.justification,
                        expected_improvement=spawn_req.expected_confidence_gain,
                        priority=StreamPriority.LOW
                    )
        
        # Phase 8: Cascade Analysis (if threat identified)
        cascade_analysis = None
        if ttp_detections or campaign_assessment:
            phase_start = time.time()
            log.info("Phase 8: Cascade Analysis - Predicting cascading effects...")
            
            # Create triggering event from detections
            triggering_event = self._create_triggering_event(ttp_detections, campaign_assessment)
            cascade_analysis = await analyze_cascade_effects(triggering_event, request.context)
            
            await self._log_phase(ProcessingPhase.CASCADE_ANALYSIS, processing_phases, phase_start, {
                "total_effects": cascade_analysis.total_effects,
                "cascade_depth": cascade_analysis.cascade_depth
            })
            
            # Stream cascade prediction
            if STREAMING_AVAILABLE and cascade_analysis:
                await stream_cascade_prediction(
                    triggering_event=cascade_analysis.triggering_event,
                    total_effects=cascade_analysis.total_effects,
                    critical_effects=len(cascade_analysis.critical_effects),
                    economic_impact=cascade_analysis.total_economic_impact,
                    details={
                        "cascade_depth": cascade_analysis.cascade_depth,
                        "affected_population": cascade_analysis.total_affected_population,
                        "confidence": cascade_analysis.confidence
                    },
                    priority=StreamPriority.CRITICAL if cascade_analysis.critical_effects else StreamPriority.HIGH
                )
        
        # Phase 9: Goal Decomposition (if planning requested)
        execution_plan = None
        if request.context.get("include_planning"):
            phase_start = time.time()
            log.info("Phase 9: Goal Decomposition - Creating execution plan...")
            
            execution_plan = await decompose_and_plan(
                goal_description=request.task_description,
                objective=request.context.get("objective", request.task_description),
                success_metrics=request.context.get("success_metrics"),
                constraints=request.context.get("constraints"),
                deadline=request.deadline,
                context=request.context
            )
            
            await self._log_phase(ProcessingPhase.GOAL_DECOMPOSITION, processing_phases, phase_start, {
                "tasks_generated": len(execution_plan.tasks),
                "estimated_duration": execution_plan.estimated_total_time / 60
            })
            
            log.info(f"Execution plan created: {len(execution_plan.tasks)} tasks, "
                    f"{execution_plan.estimated_total_time/60:.1f} minutes")
        
        # Phase 10: COA Generation (if threat response needed)
        coa_comparison = None
        if (ttp_detections or campaign_assessment) and request.context.get("generate_coas"):
            phase_start = time.time()
            log.info("Phase 10: COA Generation - Developing courses of action...")
            
            situation = {
                "ttp_detections": [d.pattern.name for d in ttp_detections],
                "campaign": campaign_assessment.operation_type.value if campaign_assessment else None,
                "threat_level": campaign_assessment.threat_level if campaign_assessment else "MEDIUM",
                "intelligence_confidence": current_confidence
            }
            
            coa_comparison = await generate_courses_of_action(
                situation=situation,
                objective=request.context.get("objective", "Neutralize threat"),
                constraints=request.context.get("constraints"),
                num_coas=request.context.get("num_coas", 4)
            )
            
            await self._log_phase(ProcessingPhase.COA_GENERATION, processing_phases, phase_start, {
                "coas_generated": len(coa_comparison.coas),
                "recommended_coa": coa_comparison.recommended_coa
            })
            
            log.info(f"Generated {len(coa_comparison.coas)} COAs, "
                    f"recommended: {coa_comparison.coas[0].coa_name}")
        
        # Phase 11: Wargaming (if COAs generated)
        wargame_results = None
        if coa_comparison and request.context.get("run_wargaming"):
            phase_start = time.time()
            log.info("Phase 11: Wargaming - Simulating COA execution...")
            
            wargame_results = await simulate_and_compare_coas(
                coas=coa_comparison.coas,
                situation=situation if 'situation' in locals() else {},
                red_force_strategy=request.context.get("red_force_strategy", "defensive")
            )
            
            await self._log_phase(ProcessingPhase.WARGAMING, processing_phases, phase_start, {
                "simulations_run": len(wargame_results.coa_results),
                "best_coa": wargame_results.best_coa_id,
                "worst_coa": wargame_results.worst_coa_id
            })
            
            log.info(f"Wargaming complete: Best COA has {wargame_results.coa_results[0].outcome_probability:.0%} success probability")
        
        # Phase 12: Synthesis
        phase_start = time.time()
        log.info("Phase 12: Synthesis - Generating comprehensive assessment...")
        
        synthesis = await self._synthesize_intelligence(
            task_analysis,
            fused_intelligence_list,
            ttp_detections,
            campaign_assessment,
            cascade_analysis,
            identified_gaps,
            execution_plan,
            coa_comparison,
            wargame_results
        )
        
        await self._log_phase(ProcessingPhase.SYNTHESIS, processing_phases, phase_start)
        
        # Phase 13: Finalization
        processing_time = time.time() - start_time
        
        response = IntelligenceResponse(
            request_id=request.request_id,
            task_analysis=task_analysis,
            fused_intelligence=fused_intelligence_list,
            ttp_detections=ttp_detections,
            campaign_assessment=campaign_assessment,
            cascade_analysis=cascade_analysis,
            execution_plan=execution_plan,
            coa_comparison=coa_comparison,
            wargame_results=wargame_results,
            deployed_agents=deployed_agents,
            spawned_agents=spawn_requests,
            agent_count=len(deployed_agents),
            overall_confidence=synthesis["overall_confidence"],
            identified_gaps=identified_gaps,
            executive_summary=synthesis["executive_summary"],
            key_findings=synthesis["key_findings"],
            threat_assessment=synthesis["threat_assessment"],
            recommended_actions=synthesis["recommended_actions"],
            alternative_hypotheses=synthesis["alternative_hypotheses"],
            processing_time=processing_time,
            processing_phases=processing_phases,
            timestamp=time.time()
        )
        
        self.processing_history.append(response)
        self._update_metrics(response)
        
        log.info(f"Intelligence processing complete in {processing_time:.2f}s")
        log.info(f"Confidence: {response.overall_confidence:.2f}, "
                f"Agents: {response.agent_count}, "
                f"Findings: {len(response.key_findings)}")
        
        return response
    
    async def _convert_to_injects(
        self,
        data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[IntelligenceInject]:
        """Convert raw data to intelligence injects"""
        
        injects = []
        
        for idx, data_item in enumerate(data):
            # Determine domain
            data_type = data_item.get("type", "").lower()
            
            if "sigint" in data_type or "signal" in data_type:
                domain = IntelligenceDomain.SIGINT
            elif "cyber" in data_type or "network" in data_type:
                domain = IntelligenceDomain.CYBINT
            elif "geoint" in data_type or "imagery" in data_type:
                domain = IntelligenceDomain.GEOINT
            elif "osint" in data_type or "open" in data_type:
                domain = IntelligenceDomain.OSINT
            else:
                domain = IntelligenceDomain.OSINT  # Default
            
            # Determine credibility
            credibility_score = data_item.get("credibility", 0.7)
            if credibility_score >= 0.9:
                credibility = SourceCredibility.CONFIRMED
            elif credibility_score >= 0.7:
                credibility = SourceCredibility.PROBABLY_TRUE
            elif credibility_score >= 0.5:
                credibility = SourceCredibility.POSSIBLY_TRUE
            else:
                credibility = SourceCredibility.DOUBTFUL
            
            inject = IntelligenceInject(
                inject_id=f"inject_{idx}_{int(time.time() * 1000)}",
                source_id=data_item.get("source_id", f"source_{idx}"),
                source_name=data_item.get("source", f"Source {idx+1}"),
                timestamp=data_item.get("timestamp", time.time()),
                domain=domain,
                data_type=data_type,
                content=data_item.get("content", data_item),
                credibility=credibility,
                confidence=data_item.get("confidence", 0.7),
                classification=data_item.get("classification", "UNCLASSIFIED"),
                tags=set(data_item.get("tags", [])),
                metadata=data_item.get("metadata", {})
            )
            
            injects.append(inject)
        
        return injects
    
    def _build_current_analysis(
        self,
        fused_intelligence: List[FusedIntelligence],
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment]
    ) -> Dict[str, Any]:
        """Build current analysis state for gap analysis"""
        
        analysis = {
            "fusion_count": len(fused_intelligence),
            "ttp_count": len(ttp_detections),
            "sources": [],
            "analyzed_data_types": set(),
            "temporal_analysis": {},
            "correlation_analysis": {},
            "hypotheses": []
        }
        
        # Extract information from fused intelligence
        for fusion in fused_intelligence:
            analysis["sources"].extend(fusion.source_injects)
            
            # Extract analyzed data types
            for domain in fusion.domains:
                analysis["analyzed_data_types"].add(domain.value)
            
            # Temporal analysis exists if timeline present
            if "timeline" in fusion.fused_assessment:
                analysis["temporal_analysis"] = fusion.fused_assessment["timeline"]
            
            # Correlation analysis exists
            if fusion.correlations:
                analysis["correlation_analysis"]["count"] = len(fusion.correlations)
            
            # Alternative hypotheses
            analysis["hypotheses"].extend(fusion.alternative_hypotheses)
        
        # Add campaign assessment info
        if campaign_assessment:
            analysis["intent"] = campaign_assessment.intent_assessment
            analysis["prediction"] = campaign_assessment.predicted_next_steps
        
        return analysis
    
    def _create_triggering_event(
        self,
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment]
    ) -> Dict[str, Any]:
        """Create triggering event for cascade analysis"""
        
        if campaign_assessment:
            return {
                "id": campaign_assessment.campaign_id,
                "description": campaign_assessment.intent_assessment,
                "type": campaign_assessment.operation_type.value,
                "target": "critical_infrastructure" if any(
                    "infrastructure" in d.pattern.name.lower() 
                    for d in ttp_detections
                ) else "network_assets"
            }
        elif ttp_detections:
            primary_ttp = max(ttp_detections, key=lambda d: d.confidence)
            return {
                "id": primary_ttp.detection_id,
                "description": primary_ttp.pattern.description,
                "type": primary_ttp.pattern.category.value,
                "target": "infrastructure"
            }
        else:
            return {
                "id": "unknown_event",
                "description": "Unspecified threat event",
                "type": "cyber",
                "target": "network"
            }
    
    async def _simulate_agent_spawning(self, spawn_requests: List[AgentSpawnRequest]):
        """Simulate spawning of additional agents"""
        # In real implementation, this would actually spawn agents
        await asyncio.sleep(0.1)  # Simulate spawning time
        log.info(f"Spawned {len(spawn_requests)} additional specialized agents")
    
    async def _synthesize_intelligence(
        self,
        task_analysis: TaskAnalysis,
        fused_intelligence: List[FusedIntelligence],
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment],
        cascade_analysis: Optional[CascadeAnalysis],
        gaps: List[CapabilityGap],
        execution_plan: Optional[ExecutionPlan] = None,
        coa_comparison: Optional[COAComparison] = None,
        wargame_results: Optional[WargameComparison] = None
    ) -> Dict[str, Any]:
        """Synthesize all intelligence into comprehensive assessment"""
        
        # Calculate overall confidence
        confidences = []
        if fused_intelligence:
            confidences.extend([f.confidence for f in fused_intelligence])
        if ttp_detections:
            confidences.extend([d.confidence for d in ttp_detections])
        if campaign_assessment:
            confidences.append(campaign_assessment.confidence)
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Account for gaps
        if gaps:
            critical_gaps = [g for g in gaps if g.severity.value == "critical"]
            confidence_penalty = len(critical_gaps) * 0.05
            overall_confidence = max(overall_confidence - confidence_penalty, 0.3)
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            fused_intelligence, ttp_detections, campaign_assessment, cascade_analysis,
            execution_plan, coa_comparison, wargame_results
        )
        
        # Extract key findings
        key_findings = self._extract_key_findings(
            fused_intelligence, ttp_detections, campaign_assessment,
            execution_plan, coa_comparison
        )
        
        # Generate threat assessment
        threat_assessment = self._generate_threat_assessment(
            ttp_detections, campaign_assessment, cascade_analysis, wargame_results
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            ttp_detections, campaign_assessment, cascade_analysis,
            coa_comparison, wargame_results
        )
        
        # Collect alternative hypotheses
        alternative_hypotheses = []
        for fusion in fused_intelligence:
            alternative_hypotheses.extend(fusion.alternative_hypotheses)
        
        return {
            "overall_confidence": overall_confidence,
            "executive_summary": exec_summary,
            "key_findings": key_findings,
            "threat_assessment": threat_assessment,
            "recommended_actions": recommended_actions,
            "alternative_hypotheses": alternative_hypotheses
        }
    
    def _generate_executive_summary(
        self,
        fused_intelligence: List[FusedIntelligence],
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment],
        cascade_analysis: Optional[CascadeAnalysis],
        execution_plan: Optional[ExecutionPlan],
        coa_comparison: Optional[COAComparison],
        wargame_results: Optional[WargameComparison]
    ) -> str:
        """Generate executive summary"""
        
        summary_parts = []
        
        # Multi-source intelligence
        if fused_intelligence:
            total_sources = sum(len(f.source_injects) for f in fused_intelligence)
            summary_parts.append(
                f"Analysis of {total_sources} intelligence sources across "
                f"{len(fused_intelligence)} fusion events."
            )
        
        # TTP detection
        if ttp_detections:
            summary_parts.append(
                f"Identified {len(ttp_detections)} adversary TTP patterns "
                f"with average confidence {sum(d.confidence for d in ttp_detections) / len(ttp_detections):.2f}."
            )
        
        # Campaign
        if campaign_assessment:
            summary_parts.append(
                f"Multi-stage {campaign_assessment.operation_type.value} campaign identified "
                f"at {campaign_assessment.campaign_stage} stage. "
                f"Threat level: {campaign_assessment.threat_level}."
            )
        
        # Cascade
        if cascade_analysis:
            summary_parts.append(
                f"Cascading effect analysis predicts {cascade_analysis.total_effects} effects "
                f"with {len(cascade_analysis.critical_effects)} critical impacts."
            )
        
        # Planning
        if execution_plan:
            summary_parts.append(
                f"Execution plan developed: {len(execution_plan.tasks)} tasks, "
                f"estimated duration {execution_plan.estimated_total_time/3600:.1f} hours."
            )
        
        # COA
        if coa_comparison:
            summary_parts.append(
                f"Generated and compared {len(coa_comparison.coas)} courses of action. "
                f"Recommended: {coa_comparison.coas[0].coa_name}."
            )
        
        # Wargaming
        if wargame_results:
            best_outcome = wargame_results.coa_results[0].outcome.value
            success_prob = wargame_results.coa_results[0].outcome_probability
            summary_parts.append(
                f"Wargaming simulation predicts {best_outcome.replace('_', ' ')} "
                f"with {success_prob:.0%} probability."
            )
        
        return " ".join(summary_parts)
    
    def _extract_key_findings(
        self,
        fused_intelligence: List[FusedIntelligence],
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment],
        execution_plan: Optional[ExecutionPlan],
        coa_comparison: Optional[COAComparison]
    ) -> List[str]:
        """Extract key findings"""
        
        findings = []
        
        # From fused intelligence
        for fusion in fused_intelligence[:3]:  # Top 3
            if "summary" in fusion.fused_assessment:
                findings.append(fusion.fused_assessment["summary"])
        
        # From TTPs
        for ttp in ttp_detections[:3]:  # Top 3
            findings.append(
                f"{ttp.pattern.name} detected with {ttp.confidence:.0%} confidence"
            )
        
        # From campaign
        if campaign_assessment:
            findings.append(campaign_assessment.intent_assessment)
        
        # From planning
        if execution_plan:
            findings.append(
                f"Execution requires {len(execution_plan.tasks)} tasks over "
                f"{execution_plan.estimated_total_time/3600:.1f} hours"
            )
        
        # From COA/wargaming
        if coa_comparison:
            findings.append(
                f"Best COA: {coa_comparison.coas[0].coa_name} "
                f"(feasibility: {coa_comparison.coas[0].feasibility_score:.0%})"
            )
        
        return findings[:7]  # Top 7 findings
    
    def _generate_threat_assessment(
        self,
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment],
        cascade_analysis: Optional[CascadeAnalysis],
        wargame_results: Optional[WargameComparison]
    ) -> str:
        """Generate threat assessment"""
        
        assessment_parts = []
        
        if campaign_assessment:
            assessment_parts.append(
                f"{campaign_assessment.threat_level} threat level. "
                f"{campaign_assessment.intent_assessment} "
                f"Current stage: {campaign_assessment.campaign_stage}."
            )
        elif ttp_detections:
            avg_conf = sum(d.confidence for d in ttp_detections) / len(ttp_detections)
            if avg_conf > 0.8:
                level = "HIGH"
            elif avg_conf > 0.6:
                level = "ELEVATED"
            else:
                level = "MODERATE"
            
            assessment_parts.append(
                f"{level} threat based on {len(ttp_detections)} detected TTP patterns."
            )
        else:
            assessment_parts.append("Insufficient evidence for threat level determination.")
        
        # Add wargaming assessment if available
        if wargame_results:
            best_result = wargame_results.coa_results[0]
            assessment_parts.append(
                f"Wargaming indicates {best_result.outcome.value.replace('_', ' ')} likely "
                f"({best_result.outcome_probability:.0%} probability) if recommended COA executed."
            )
        
        return " ".join(assessment_parts)
    
    def _generate_recommendations(
        self,
        ttp_detections: List[TTPDetection],
        campaign_assessment: Optional[CampaignAssessment],
        cascade_analysis: Optional[CascadeAnalysis],
        coa_comparison: Optional[COAComparison],
        wargame_results: Optional[WargameComparison]
    ) -> List[str]:
        """Generate recommended actions"""
        
        recommendations = []
        
        # From COA/wargaming (highest priority)
        if wargame_results:
            # Use wargaming recommendation as top recommendation
            best_coa = wargame_results.coa_results[0]
            recommendations.append(
                f"Execute {best_coa.coa.coa_name} ({best_coa.outcome_probability:.0%} success probability)"
            )
            recommendations.extend(best_coa.recommendations[:2])
        
        elif coa_comparison:
            # Use COA recommendation
            recommended_coa = coa_comparison.coas[0]
            recommendations.append(
                f"Recommend {recommended_coa.coa_name} "
                f"(overall score: {recommended_coa.overall_score:.2f})"
            )
            recommendations.extend(recommended_coa.advantages[:2])
        
        # From TTPs
        for ttp in ttp_detections:
            if ttp.pattern.mitigation:
                recommendations.extend(ttp.pattern.mitigation[:2])
        
        # From cascade analysis
        if cascade_analysis and cascade_analysis.critical_effects:
            for effect in cascade_analysis.critical_effects[:3]:
                if effect.mitigation_options:
                    recommendations.extend(effect.mitigation_options[:2])
        
        # Deduplicate and prioritize
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:7]
    
    async def _log_phase(
        self,
        phase: ProcessingPhase,
        phases_list: List[Dict[str, Any]],
        start_time: float,
        metrics: Dict[str, Any] = None
    ):
        """Log processing phase"""
        
        phase_time = time.time() - start_time
        phase_log = {
            "phase": phase.value,
            "duration": phase_time,
            "timestamp": time.time()
        }
        
        if metrics:
            phase_log["metrics"] = metrics
        
        phases_list.append(phase_log)
        log.debug(f"Phase {phase.value} completed in {phase_time:.3f}s")
    
    def _update_metrics(self, response: IntelligenceResponse):
        """Update performance metrics"""
        
        self.total_requests_processed += 1
        
        # Update averages
        n = self.total_requests_processed
        self.avg_processing_time = (
            (self.avg_processing_time * (n-1) + response.processing_time) / n
        )
        self.avg_confidence = (
            (self.avg_confidence * (n-1) + response.overall_confidence) / n
        )
        self.avg_agent_count = (
            (self.avg_agent_count * (n-1) + response.agent_count) / n
        )


# Global instance
master_orchestrator = MasterIntelligenceOrchestrator()


async def process_intelligence(
    task_description: str,
    available_data: List[Dict[str, Any]],
    context: Dict[str, Any] = None,
    priority: int = 5
) -> IntelligenceResponse:
    """
    Main entry point: Process comprehensive intelligence request.
    Coordinates all intelligence systems for high-quality analysis.
    """
    
    request = IntelligenceRequest(
        request_id=f"intel_{int(time.time() * 1000)}",
        task_description=task_description,
        available_data=available_data,
        context=context or {},
        priority=priority
    )
    
    return await master_orchestrator.process_intelligence_request(request)

