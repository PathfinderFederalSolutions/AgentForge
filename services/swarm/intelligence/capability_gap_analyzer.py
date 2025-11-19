"""
Autonomous Capability Gap Analysis Engine
Real-time identification of missing capabilities and dynamic agent spawning
Continuously learns and adapts to improve intelligence quality
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger("capability-gap-analyzer")

class CapabilityGapSeverity(Enum):
    """Severity of capability gaps"""
    CRITICAL = "critical"      # Major blind spot, immediate spawning required
    HIGH = "high"             # Significant gap, high priority spawning
    MEDIUM = "medium"         # Moderate gap, spawn when resources available
    LOW = "low"              # Minor gap, opportunistic spawning

@dataclass
class CapabilityGap:
    """Identified capability gap"""
    gap_id: str
    gap_type: str
    severity: CapabilityGapSeverity
    description: str
    missing_capabilities: List[str]
    recommended_agents: List[str]
    confidence_impact: float  # How much this gap reduces confidence
    detected_at: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentSpawnRequest:
    """Request to spawn new specialized agent"""
    request_id: str
    agent_type: str
    specialization: str
    priority: int
    justification: str
    expected_confidence_gain: float
    required_inputs: List[str]
    output_expectations: List[str]

class AutonomousCapabilityGapAnalyzer:
    """
    Continuously monitors analysis quality and autonomously identifies
    capability gaps, spawning specialized agents as needed
    """
    
    def __init__(self):
        self.identified_gaps: Dict[str, CapabilityGap] = {}
        self.spawn_history: List[AgentSpawnRequest] = []
        self.capability_registry: Dict[str, Set[str]] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Learning system
        self.gap_patterns: Dict[str, List[str]] = {}
        self.successful_spawns: Dict[str, float] = {}
        self.failed_spawns: Dict[str, str] = {}
        
        # Thresholds
        self.confidence_threshold = 0.85
        self.gap_detection_sensitivity = 0.7
        
        self._initialize_capability_registry()
        
        log.info("Autonomous Capability Gap Analyzer initialized")
    
    def _initialize_capability_registry(self):
        """Initialize known capability categories"""
        
        self.capability_registry = {
            "signal_processing": {
                "rf_analysis", "frequency_analysis", "signal_demodulation",
                "communication_protocol_analysis", "spread_spectrum_analysis"
            },
            "acoustic_analysis": {
                "sonar_processing", "acoustic_signature_matching",
                "underwater_sound_analysis", "noise_filtering", "bearing_estimation"
            },
            "cyber_analysis": {
                "packet_analysis", "malware_reverse_engineering",
                "network_topology_mapping", "intrusion_detection", "log_analysis"
            },
            "pattern_recognition": {
                "anomaly_detection", "clustering", "sequence_detection",
                "behavioral_profiling", "trend_analysis"
            },
            "temporal_reasoning": {
                "timeline_construction", "causality_inference",
                "sequence_prediction", "temporal_correlation", "event_sequencing"
            },
            "geospatial_reasoning": {
                "route_analysis", "terrain_analysis", "proximity_analysis",
                "movement_prediction", "spatial_correlation"
            },
            "multi_source_fusion": {
                "data_correlation", "confidence_weighting", "source_credibility",
                "contradiction_resolution", "evidence_synthesis"
            },
            "threat_modeling": {
                "intent_assessment", "capability_analysis", "tactic_identification",
                "threat_prioritization", "adversary_profiling"
            },
            "predictive_modeling": {
                "forecasting", "scenario_generation", "probability_estimation",
                "outcome_prediction", "what_if_analysis"
            },
            "impact_analysis": {
                "cascading_effects", "dependency_analysis", "resilience_assessment",
                "recovery_estimation", "risk_quantification"
            },
            "deception_detection": {
                "consistency_checking", "source_verification",
                "credibility_assessment", "manipulation_detection", "truth_inference"
            },
            "network_analysis": {
                "graph_analysis", "centrality_analysis", "community_detection",
                "influence_mapping", "relationship_extraction"
            }
        }
    
    async def analyze_analysis_quality(
        self,
        current_analysis: Dict[str, Any],
        available_data: List[Dict[str, Any]],
        deployed_agents: List[str],
        confidence_score: float
    ) -> Tuple[List[CapabilityGap], List[AgentSpawnRequest]]:
        """
        Analyze the quality of current analysis and identify capability gaps.
        Returns identified gaps and recommended agent spawns.
        """
        
        log.info(f"Analyzing analysis quality (confidence: {confidence_score:.2f})")
        
        identified_gaps = []
        spawn_requests = []
        
        # 1. Confidence gap analysis
        if confidence_score < self.confidence_threshold:
            confidence_gaps = self._identify_confidence_gaps(
                current_analysis, deployed_agents, confidence_score
            )
            identified_gaps.extend(confidence_gaps)
        
        # 2. Coverage gap analysis
        coverage_gaps = self._identify_coverage_gaps(
            available_data, deployed_agents, current_analysis
        )
        identified_gaps.extend(coverage_gaps)
        
        # 3. Temporal gap analysis
        temporal_gaps = self._identify_temporal_gaps(
            available_data, current_analysis
        )
        identified_gaps.extend(temporal_gaps)
        
        # 4. Multi-source correlation gaps
        correlation_gaps = self._identify_correlation_gaps(
            available_data, current_analysis, deployed_agents
        )
        identified_gaps.extend(correlation_gaps)
        
        # 5. Reasoning depth gaps
        reasoning_gaps = self._identify_reasoning_gaps(
            current_analysis, deployed_agents
        )
        identified_gaps.extend(reasoning_gaps)
        
        # Generate spawn requests for critical and high severity gaps
        for gap in identified_gaps:
            if gap.severity in [CapabilityGapSeverity.CRITICAL, CapabilityGapSeverity.HIGH]:
                spawn_req = self._generate_spawn_request(gap)
                spawn_requests.append(spawn_req)
        
        # Store identified gaps
        for gap in identified_gaps:
            self.identified_gaps[gap.gap_id] = gap
        
        log.info(f"Identified {len(identified_gaps)} capability gaps, "
                f"recommending {len(spawn_requests)} agent spawns")
        
        return identified_gaps, spawn_requests
    
    def _identify_confidence_gaps(
        self,
        current_analysis: Dict[str, Any],
        deployed_agents: List[str],
        confidence_score: float
    ) -> List[CapabilityGap]:
        """Identify what's causing low confidence"""
        
        gaps = []
        confidence_deficit = self.confidence_threshold - confidence_score
        
        if confidence_deficit > 0.3:
            # Critical confidence gap
            gaps.append(CapabilityGap(
                gap_id=f"conf_critical_{int(time.time())}",
                gap_type="confidence_critical",
                severity=CapabilityGapSeverity.CRITICAL,
                description=f"Confidence score {confidence_score:.2f} significantly below threshold {self.confidence_threshold}",
                missing_capabilities=["source_verification", "evidence_synthesis", "confidence_calibration"],
                recommended_agents=["confidence_calibrator", "evidence_verifier", "source_credibility_assessor"],
                confidence_impact=confidence_deficit,
                detected_at=time.time()
            ))
        
        # Check for single-source reliance
        sources = current_analysis.get("sources", [])
        if len(sources) < 3:
            gaps.append(CapabilityGap(
                gap_id=f"conf_sources_{int(time.time())}",
                gap_type="insufficient_sources",
                severity=CapabilityGapSeverity.HIGH,
                description=f"Analysis relies on only {len(sources)} source(s), need multi-source confirmation",
                missing_capabilities=["multi_source_fusion", "cross_validation"],
                recommended_agents=["source_diversifier", "cross_validator"],
                confidence_impact=0.15,
                detected_at=time.time()
            ))
        
        return gaps
    
    def _identify_coverage_gaps(
        self,
        available_data: List[Dict[str, Any]],
        deployed_agents: List[str],
        current_analysis: Dict[str, Any]
    ) -> List[CapabilityGap]:
        """Identify data that isn't being analyzed"""
        
        gaps = []
        
        # Identify data types present
        data_types = set()
        for data_item in available_data:
            dtype = data_item.get("type", "unknown")
            data_types.add(dtype)
        
        # Check for unanalyzed data types
        analyzed_types = current_analysis.get("analyzed_data_types", set())
        unanalyzed = data_types - analyzed_types
        
        if unanalyzed:
            for dtype in unanalyzed:
                gaps.append(CapabilityGap(
                    gap_id=f"coverage_{dtype}_{int(time.time())}",
                    gap_type="unanalyzed_data",
                    severity=CapabilityGapSeverity.HIGH,
                    description=f"Data type '{dtype}' present but not analyzed",
                    missing_capabilities=[f"{dtype}_processing", f"{dtype}_interpretation"],
                    recommended_agents=[f"{dtype}_specialist", f"{dtype}_analyzer"],
                    confidence_impact=0.1,
                    detected_at=time.time(),
                    context={"data_type": dtype}
                ))
        
        # Check for domain-specific analysis gaps
        if any("acoustic" in str(d).lower() for d in available_data):
            if not any("acoustic" in agent.lower() for agent in deployed_agents):
                gaps.append(CapabilityGap(
                    gap_id=f"domain_acoustic_{int(time.time())}",
                    gap_type="missing_domain_expertise",
                    severity=CapabilityGapSeverity.HIGH,
                    description="Acoustic data present but no acoustic analysis specialist deployed",
                    missing_capabilities=["acoustic_signature_analysis", "sonar_processing"],
                    recommended_agents=["acoustic_specialist", "sonar_analyst"],
                    confidence_impact=0.2,
                    detected_at=time.time()
                ))
        
        return gaps
    
    def _identify_temporal_gaps(
        self,
        available_data: List[Dict[str, Any]],
        current_analysis: Dict[str, Any]
    ) -> List[CapabilityGap]:
        """Identify temporal correlation gaps"""
        
        gaps = []
        
        # Check if data has timestamps
        timestamped_data = [d for d in available_data if "timestamp" in d or "time" in d]
        
        if len(timestamped_data) >= 2:
            # Multiple timestamped data points, check for temporal analysis
            temporal_analysis = current_analysis.get("temporal_analysis", {})
            
            if not temporal_analysis:
                gaps.append(CapabilityGap(
                    gap_id=f"temporal_missing_{int(time.time())}",
                    gap_type="missing_temporal_analysis",
                    severity=CapabilityGapSeverity.HIGH,
                    description=f"{len(timestamped_data)} timestamped data points but no temporal correlation analysis",
                    missing_capabilities=["temporal_correlation", "sequence_analysis", "causality_inference"],
                    recommended_agents=["temporal_analyst", "sequence_correlator", "causality_reasoner"],
                    confidence_impact=0.15,
                    detected_at=time.time()
                ))
            
            # Check for timeline gaps (data points far apart in time)
            timestamps = []
            for d in timestamped_data:
                ts = d.get("timestamp", d.get("time"))
                if ts:
                    timestamps.append(ts)
            
            if len(timestamps) >= 2:
                timestamps.sort()
                max_gap = max(timestamps[i+1] - timestamps[i] 
                             for i in range(len(timestamps)-1))
                
                if max_gap > 3600:  # 1 hour gap
                    gaps.append(CapabilityGap(
                        gap_id=f"temporal_gap_{int(time.time())}",
                        gap_type="temporal_coverage_gap",
                        severity=CapabilityGapSeverity.MEDIUM,
                        description=f"Large temporal gap ({max_gap/3600:.1f} hours) between data points",
                        missing_capabilities=["gap_interpolation", "missing_data_inference"],
                        recommended_agents=["gap_interpolator", "temporal_estimator"],
                        confidence_impact=0.08,
                        detected_at=time.time()
                    ))
        
        return gaps
    
    def _identify_correlation_gaps(
        self,
        available_data: List[Dict[str, Any]],
        current_analysis: Dict[str, Any],
        deployed_agents: List[str]
    ) -> List[CapabilityGap]:
        """Identify multi-source correlation gaps"""
        
        gaps = []
        
        # Check if multiple data sources exist
        if len(available_data) >= 3:
            correlation_analysis = current_analysis.get("correlation_analysis", {})
            
            # Check for cross-source correlation
            if not correlation_analysis or len(correlation_analysis) == 0:
                gaps.append(CapabilityGap(
                    gap_id=f"correlation_missing_{int(time.time())}",
                    gap_type="missing_correlation_analysis",
                    severity=CapabilityGapSeverity.CRITICAL,
                    description=f"{len(available_data)} data sources but no cross-source correlation performed",
                    missing_capabilities=["cross_source_correlation", "data_fusion", "evidence_synthesis"],
                    recommended_agents=["correlation_specialist", "fusion_analyst", "evidence_synthesizer"],
                    confidence_impact=0.25,
                    detected_at=time.time()
                ))
            
            # Check for geospatial correlation if location data exists
            has_location = any("location" in str(d).lower() or "position" in str(d).lower() 
                              for d in available_data)
            if has_location:
                if not any("geospatial" in agent.lower() or "spatial" in agent.lower() 
                          for agent in deployed_agents):
                    gaps.append(CapabilityGap(
                        gap_id=f"geospatial_corr_{int(time.time())}",
                        gap_type="missing_geospatial_correlation",
                        severity=CapabilityGapSeverity.HIGH,
                        description="Location data present but no geospatial correlation analysis",
                        missing_capabilities=["geospatial_correlation", "proximity_analysis"],
                        recommended_agents=["geospatial_correlator", "spatial_analyst"],
                        confidence_impact=0.12,
                        detected_at=time.time()
                    ))
        
        return gaps
    
    def _identify_reasoning_gaps(
        self,
        current_analysis: Dict[str, Any],
        deployed_agents: List[str]
    ) -> List[CapabilityGap]:
        """Identify gaps in reasoning depth"""
        
        gaps = []
        
        # Check for intent analysis
        if "intent" not in current_analysis and "intent_analysis" not in current_analysis:
            gaps.append(CapabilityGap(
                gap_id=f"reasoning_intent_{int(time.time())}",
                gap_type="missing_intent_analysis",
                severity=CapabilityGapSeverity.HIGH,
                description="No adversary intent assessment in analysis",
                missing_capabilities=["intent_inference", "motive_analysis", "goal_prediction"],
                recommended_agents=["intent_analyst", "behavioral_profiler", "adversary_modeler"],
                confidence_impact=0.15,
                detected_at=time.time()
            ))
        
        # Check for predictive analysis
        if "prediction" not in current_analysis and "forecast" not in current_analysis:
            gaps.append(CapabilityGap(
                gap_id=f"reasoning_predict_{int(time.time())}",
                gap_type="missing_predictive_analysis",
                severity=CapabilityGapSeverity.MEDIUM,
                description="No predictive/forecasting analysis present",
                missing_capabilities=["forecasting", "scenario_projection", "outcome_prediction"],
                recommended_agents=["predictive_modeler", "scenario_analyst", "forecaster"],
                confidence_impact=0.1,
                detected_at=time.time()
            ))
        
        # Check for impact analysis
        if "impact" not in current_analysis and "consequences" not in current_analysis:
            gaps.append(CapabilityGap(
                gap_id=f"reasoning_impact_{int(time.time())}",
                gap_type="missing_impact_analysis",
                severity=CapabilityGapSeverity.MEDIUM,
                description="No impact/consequence analysis present",
                missing_capabilities=["impact_assessment", "consequence_analysis", "cascading_effects"],
                recommended_agents=["impact_assessor", "consequence_analyst", "cascade_modeler"],
                confidence_impact=0.1,
                detected_at=time.time()
            ))
        
        # Check for alternative hypothesis generation
        hypotheses = current_analysis.get("hypotheses", [])
        if len(hypotheses) < 2:
            gaps.append(CapabilityGap(
                gap_id=f"reasoning_hypothesis_{int(time.time())}",
                gap_type="insufficient_hypotheses",
                severity=CapabilityGapSeverity.MEDIUM,
                description="Fewer than 2 alternative hypotheses generated",
                missing_capabilities=["hypothesis_generation", "alternative_reasoning", "abductive_reasoning"],
                recommended_agents=["hypothesis_generator", "alternative_reasoner", "devil_advocate"],
                confidence_impact=0.08,
                detected_at=time.time()
            ))
        
        return gaps
    
    def _generate_spawn_request(self, gap: CapabilityGap) -> AgentSpawnRequest:
        """Generate agent spawn request from capability gap"""
        
        # Prioritize by severity
        priority_map = {
            CapabilityGapSeverity.CRITICAL: 10,
            CapabilityGapSeverity.HIGH: 8,
            CapabilityGapSeverity.MEDIUM: 5,
            CapabilityGapSeverity.LOW: 3
        }
        
        # Select best agent type
        agent_type = gap.recommended_agents[0] if gap.recommended_agents else "general_specialist"
        
        return AgentSpawnRequest(
            request_id=f"spawn_{gap.gap_id}",
            agent_type=agent_type,
            specialization=gap.gap_type,
            priority=priority_map[gap.severity],
            justification=gap.description,
            expected_confidence_gain=gap.confidence_impact,
            required_inputs=gap.missing_capabilities,
            output_expectations=[f"{cap}_results" for cap in gap.missing_capabilities]
        )
    
    async def learn_from_spawn(
        self,
        spawn_request: AgentSpawnRequest,
        actual_confidence_gain: float,
        success: bool
    ):
        """Learn from agent spawn results"""
        
        if success:
            self.successful_spawns[spawn_request.agent_type] = actual_confidence_gain
            log.info(f"Spawn successful: {spawn_request.agent_type} gained {actual_confidence_gain:.3f} confidence")
        else:
            self.failed_spawns[spawn_request.agent_type] = spawn_request.justification
            log.warning(f"Spawn failed: {spawn_request.agent_type}")
        
        # Update gap patterns
        if spawn_request.specialization not in self.gap_patterns:
            self.gap_patterns[spawn_request.specialization] = []
        self.gap_patterns[spawn_request.specialization].append(spawn_request.agent_type)


# Global instance
capability_gap_analyzer = AutonomousCapabilityGapAnalyzer()


async def analyze_and_identify_gaps(
    current_analysis: Dict[str, Any],
    available_data: List[Dict[str, Any]],
    deployed_agents: List[str],
    confidence_score: float
) -> Tuple[List[CapabilityGap], List[AgentSpawnRequest]]:
    """
    Main entry point: Analyze analysis quality and identify capability gaps.
    Returns gaps and recommended agent spawns.
    """
    return await capability_gap_analyzer.analyze_analysis_quality(
        current_analysis, available_data, deployed_agents, confidence_score
    )

