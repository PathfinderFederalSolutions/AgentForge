"""
TTP (Tactics, Techniques, Procedures) Pattern Recognition Engine
Identifies adversary behavioral patterns and predicts intent
Works like experienced intelligence analyst recognizing threat patterns
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger("ttp-pattern-recognition")

class TTPCategory(Enum):
    """Categories of TTPs"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

class OperationType(Enum):
    """Types of operations"""
    CYBER_ATTACK = "cyber_attack"
    PHYSICAL_ATTACK = "physical_attack"
    HYBRID_OPERATION = "hybrid_operation"
    INTELLIGENCE_COLLECTION = "intelligence_collection"
    SABOTAGE = "sabotage"
    DECEPTION = "deception"
    DENIAL_OF_SERVICE = "denial_of_service"

@dataclass
class TTPPattern:
    """A defined TTP pattern"""
    pattern_id: str
    name: str
    category: TTPCategory
    operation_types: List[OperationType]
    indicators: List[str]
    typical_sequence: List[str]
    associated_actors: List[str]
    confidence_threshold: float
    description: str
    mitigation: List[str] = field(default_factory=list)

@dataclass
class TTPDetection:
    """Detected TTP in observed data"""
    detection_id: str
    pattern: TTPPattern
    confidence: float
    observed_indicators: List[str]
    sequence_match: float
    timestamp: float
    evidence: List[Dict[str, Any]]
    reasoning: List[str]

@dataclass
class CampaignAssessment:
    """Assessment of multi-stage campaign"""
    campaign_id: str
    detected_ttps: List[TTPDetection]
    operation_type: OperationType
    campaign_stage: str
    intent_assessment: str
    predicted_next_steps: List[str]
    threat_level: str
    confidence: float
    timeline: List[Dict[str, Any]]

class TTPPatternRecognitionEngine:
    """
    Recognizes adversary TTPs from observed behaviors.
    Identifies multi-stage campaigns and predicts adversary intent.
    """
    
    def __init__(self):
        self.ttp_library: Dict[str, TTPPattern] = {}
        self.detection_history: List[TTPDetection] = []
        self.campaign_tracking: Dict[str, CampaignAssessment] = {}
        
        # Pattern matching configuration
        self.sequence_weight = 0.4
        self.indicator_weight = 0.6
        
        self._initialize_ttp_library()
        
        log.info("TTP Pattern Recognition Engine initialized")
    
    def _initialize_ttp_library(self):
        """Initialize library of known TTP patterns"""
        
        # Submarine infiltration patterns
        self.ttp_library["submarine_infiltration"] = TTPPattern(
            pattern_id="submarine_infiltration",
            name="Submarine Infiltration Operation",
            category=TTPCategory.RECONNAISSANCE,
            operation_types=[OperationType.INTELLIGENCE_COLLECTION, OperationType.SABOTAGE],
            indicators=[
                "acoustic_anomaly",
                "sonar_contact",
                "ais_spoofing",
                "satcom_burst",
                "gnss_interference",
                "submarine_signature"
            ],
            typical_sequence=[
                "acoustic_detection",
                "communications_establish",
                "positioning_maneuver",
                "target_approach"
            ],
            associated_actors=["nation_state_maritime", "submarine_force"],
            confidence_threshold=0.7,
            description="Covert submarine operation for reconnaissance or sabotage preparation",
            mitigation=[
                "Deploy ASW assets",
                "Increase maritime patrol frequency",
                "Enhance acoustic monitoring"
            ]
        )
        
        # Infrastructure sabotage patterns
        self.ttp_library["infrastructure_sabotage_prep"] = TTPPattern(
            pattern_id="infrastructure_sabotage_prep",
            name="Infrastructure Sabotage Preparation",
            category=TTPCategory.RESOURCE_DEVELOPMENT,
            operation_types=[OperationType.SABOTAGE, OperationType.HYBRID_OPERATION],
            indicators=[
                "critical_infrastructure_reconnaissance",
                "usv_deployment",
                "cable_route_surveillance",
                "sabotage_equipment_staging",
                "proximity_to_infrastructure"
            ],
            typical_sequence=[
                "reconnaissance",
                "equipment_positioning",
                "approach_planning",
                "timing_coordination"
            ],
            associated_actors=["nation_state", "hybrid_threat"],
            confidence_threshold=0.75,
            description="Preparation for sabotage of critical infrastructure (cables, pipelines, power)",
            mitigation=[
                "Pre-position repair capabilities",
                "Increase infrastructure monitoring",
                "Establish backup systems",
                "Deploy protection assets"
            ]
        )
        
        # Cyber + maritime coordination
        self.ttp_library["cyber_maritime_coordination"] = TTPPattern(
            pattern_id="cyber_maritime_coordination",
            name="Cyber-Maritime Coordinated Operation",
            category=TTPCategory.COMMAND_AND_CONTROL,
            operation_types=[OperationType.HYBRID_OPERATION],
            indicators=[
                "cyber_intrusion",
                "maritime_activity",
                "synchronized_timing",
                "communication_coordination",
                "network_scanning"
            ],
            typical_sequence=[
                "cyber_reconnaissance",
                "network_compromise",
                "maritime_positioning",
                "synchronized_action"
            ],
            associated_actors=["advanced_persistent_threat", "nation_state"],
            confidence_threshold=0.8,
            description="Coordinated cyber and maritime operation for maximum impact",
            mitigation=[
                "Network segmentation",
                "Maritime-cyber fusion cell activation",
                "Incident response coordination"
            ]
        )
        
        # Electronic warfare patterns
        self.ttp_library["electronic_warfare_prep"] = TTPPattern(
            pattern_id="electronic_warfare_prep",
            name="Electronic Warfare Preparation",
            category=TTPCategory.DEFENSE_EVASION,
            operation_types=[OperationType.DECEPTION, OperationType.DENIAL_OF_SERVICE],
            indicators=[
                "gnss_spoofing",
                "rf_jamming",
                "communications_disruption",
                "radar_interference",
                "electronic_signature"
            ],
            typical_sequence=[
                "spectrum_reconnaissance",
                "jamming_test",
                "spoofing_initiation",
                "full_disruption"
            ],
            associated_actors=["nation_state", "electronic_warfare_unit"],
            confidence_threshold=0.75,
            description="Preparation for electronic warfare operations to degrade C2",
            mitigation=[
                "Switch to alternate frequencies",
                "Activate backup navigation",
                "Deploy EW countermeasures"
            ]
        )
        
        # Multi-domain deception
        self.ttp_library["multi_domain_deception"] = TTPPattern(
            pattern_id="multi_domain_deception",
            name="Multi-Domain Deception Operation",
            category=TTPCategory.DEFENSE_EVASION,
            operation_types=[OperationType.DECEPTION],
            indicators=[
                "false_flag_activity",
                "decoy_deployment",
                "information_manipulation",
                "contradictory_signals",
                "maskirovka"
            ],
            typical_sequence=[
                "deception_planning",
                "false_indicators",
                "attention_misdirection",
                "actual_operation_elsewhere"
            ],
            associated_actors=["nation_state", "intelligence_service"],
            confidence_threshold=0.65,
            description="Sophisticated deception to mislead adversary analysis",
            mitigation=[
                "Multi-source verification",
                "Red team analysis",
                "Alternative hypothesis generation"
            ]
        )
        
        # APT cyber intrusion
        self.ttp_library["apt_cyber_intrusion"] = TTPPattern(
            pattern_id="apt_cyber_intrusion",
            name="APT Cyber Intrusion Campaign",
            category=TTPCategory.INITIAL_ACCESS,
            operation_types=[OperationType.CYBER_ATTACK, OperationType.INTELLIGENCE_COLLECTION],
            indicators=[
                "spear_phishing",
                "zero_day_exploit",
                "credential_theft",
                "lateral_movement",
                "data_exfiltration",
                "persistence_mechanism"
            ],
            typical_sequence=[
                "reconnaissance",
                "initial_compromise",
                "establish_foothold",
                "lateral_movement",
                "objective_completion"
            ],
            associated_actors=["apt_group", "nation_state_cyber"],
            confidence_threshold=0.8,
            description="Advanced persistent threat campaign for long-term access",
            mitigation=[
                "Network isolation",
                "Credential rotation",
                "Hunt for persistence",
                "Forensic analysis"
            ]
        )
        
        # Supply chain compromise
        self.ttp_library["supply_chain_compromise"] = TTPPattern(
            pattern_id="supply_chain_compromise",
            name="Supply Chain Compromise",
            category=TTPCategory.INITIAL_ACCESS,
            operation_types=[OperationType.CYBER_ATTACK],
            indicators=[
                "vendor_compromise",
                "software_update_manipulation",
                "trusted_relationship_abuse",
                "third_party_access",
                "backdoor_insertion"
            ],
            typical_sequence=[
                "supplier_reconnaissance",
                "supplier_compromise",
                "malicious_update",
                "widespread_deployment"
            ],
            associated_actors=["nation_state_cyber", "apt_group"],
            confidence_threshold=0.85,
            description="Compromise through trusted supply chain relationships",
            mitigation=[
                "Vendor security assessment",
                "Code signing verification",
                "Supply chain monitoring"
            ]
        )
        
        log.info(f"Initialized {len(self.ttp_library)} TTP patterns")
    
    async def analyze_for_ttps(
        self,
        observed_data: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> Tuple[List[TTPDetection], Optional[CampaignAssessment]]:
        """
        Analyze observed data for TTP patterns.
        Returns detected TTPs and campaign assessment if multi-stage operation detected.
        """
        
        log.info(f"Analyzing {len(observed_data)} observations for TTP patterns")
        
        # Extract indicators from observed data
        observed_indicators = self._extract_indicators(observed_data)
        
        # Detect individual TTPs
        detections = []
        for pattern in self.ttp_library.values():
            detection = await self._detect_ttp_pattern(
                pattern, observed_indicators, observed_data
            )
            if detection:
                detections.append(detection)
                self.detection_history.append(detection)
        
        # Assess if this is part of a campaign
        campaign_assessment = None
        if len(detections) >= 2:
            campaign_assessment = await self._assess_campaign(detections, observed_data)
        
        log.info(f"Detected {len(detections)} TTP patterns")
        if campaign_assessment:
            log.info(f"Campaign identified: {campaign_assessment.operation_type.value}, "
                    f"stage: {campaign_assessment.campaign_stage}")
        
        return detections, campaign_assessment
    
    def _extract_indicators(self, observed_data: List[Dict[str, Any]]) -> Set[str]:
        """Extract indicators from observed data"""
        
        indicators = set()
        
        for observation in observed_data:
            # Direct indicators
            if "indicators" in observation:
                indicators.update(observation["indicators"])
            
            # Infer indicators from data type and content
            data_type = observation.get("type", "").lower()
            content = observation.get("content", {})
            
            if "acoustic" in data_type or "sonar" in data_type:
                indicators.add("acoustic_anomaly")
                if content.get("signature_match"):
                    indicators.add("submarine_signature")
            
            if "ais" in data_type:
                if content.get("spoofing") or content.get("anomaly"):
                    indicators.add("ais_spoofing")
            
            if "sigint" in data_type or "signal" in data_type:
                indicators.add("signal_intercept")
                if "satcom" in str(content).lower():
                    indicators.add("satcom_burst")
                if "gnss" in str(content).lower() or "gps" in str(content).lower():
                    indicators.add("gnss_interference")
            
            if "cyber" in data_type or "network" in data_type:
                indicators.add("cyber_activity")
                if "intrusion" in str(content).lower():
                    indicators.add("cyber_intrusion")
                if "scan" in str(content).lower():
                    indicators.add("network_scanning")
            
            if "infrastructure" in str(observation).lower():
                indicators.add("critical_infrastructure_reconnaissance")
            
            # Extract tags
            if "tags" in observation:
                indicators.update(observation["tags"])
        
        return indicators
    
    async def _detect_ttp_pattern(
        self,
        pattern: TTPPattern,
        observed_indicators: Set[str],
        observed_data: List[Dict[str, Any]]
    ) -> Optional[TTPDetection]:
        """Detect if a specific TTP pattern is present"""
        
        # Calculate indicator match score
        pattern_indicators = set(pattern.indicators)
        matched_indicators = observed_indicators & pattern_indicators
        
        if not matched_indicators:
            return None
        
        indicator_score = len(matched_indicators) / len(pattern_indicators)
        
        # Calculate sequence match score
        sequence_score = self._calculate_sequence_match(
            pattern.typical_sequence, observed_data
        )
        
        # Calculate overall confidence
        confidence = (
            indicator_score * self.indicator_weight +
            sequence_score * self.sequence_weight
        )
        
        # Check if meets threshold
        if confidence < pattern.confidence_threshold:
            return None
        
        # Generate reasoning
        reasoning = [
            f"Matched {len(matched_indicators)}/{len(pattern_indicators)} pattern indicators",
            f"Indicator match score: {indicator_score:.2f}",
            f"Sequence match score: {sequence_score:.2f}",
            f"Overall confidence: {confidence:.2f}"
        ]
        
        if sequence_score > 0.7:
            reasoning.append("Observed behavior follows expected TTP sequence")
        
        # Collect evidence
        evidence = []
        for obs in observed_data:
            obs_indicators = set(obs.get("indicators", []))
            if obs_indicators & matched_indicators:
                evidence.append({
                    "observation_id": obs.get("id", "unknown"),
                    "type": obs.get("type"),
                    "timestamp": obs.get("timestamp"),
                    "matched_indicators": list(obs_indicators & matched_indicators)
                })
        
        return TTPDetection(
            detection_id=f"ttp_{pattern.pattern_id}_{int(time.time())}",
            pattern=pattern,
            confidence=confidence,
            observed_indicators=list(matched_indicators),
            sequence_match=sequence_score,
            timestamp=time.time(),
            evidence=evidence,
            reasoning=reasoning
        )
    
    def _calculate_sequence_match(
        self,
        expected_sequence: List[str],
        observed_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well observed data matches expected sequence"""
        
        if not expected_sequence:
            return 0.5  # No sequence to match
        
        # Sort observed data by timestamp
        sorted_obs = sorted(observed_data, key=lambda x: x.get("timestamp", 0))
        
        # Extract sequence of observed activities
        observed_sequence = []
        for obs in sorted_obs:
            activity = obs.get("activity") or obs.get("type") or obs.get("event_type")
            if activity:
                observed_sequence.append(activity.lower())
        
        if not observed_sequence:
            return 0.0
        
        # Calculate longest common subsequence
        matches = 0
        expected_idx = 0
        
        for obs_activity in observed_sequence:
            for i in range(expected_idx, len(expected_sequence)):
                if expected_sequence[i].lower() in obs_activity or obs_activity in expected_sequence[i].lower():
                    matches += 1
                    expected_idx = i + 1
                    break
        
        sequence_score = matches / len(expected_sequence)
        return sequence_score
    
    async def _assess_campaign(
        self,
        detections: List[TTPDetection],
        observed_data: List[Dict[str, Any]]
    ) -> CampaignAssessment:
        """Assess if detections indicate a multi-stage campaign"""
        
        # Determine operation type from detected TTPs
        operation_types = []
        for detection in detections:
            operation_types.extend(detection.pattern.operation_types)
        
        # Most common operation type
        from collections import Counter
        op_type_counts = Counter(operation_types)
        primary_operation_type = op_type_counts.most_common(1)[0][0]
        
        # Determine campaign stage
        categories = [d.pattern.category for d in detections]
        if TTPCategory.RECONNAISSANCE in categories:
            stage = "reconnaissance_phase"
        elif TTPCategory.RESOURCE_DEVELOPMENT in categories:
            stage = "preparation_phase"
        elif TTPCategory.INITIAL_ACCESS in categories:
            stage = "initial_access_phase"
        elif TTPCategory.EXECUTION in categories or TTPCategory.IMPACT in categories:
            stage = "execution_phase"
        else:
            stage = "intermediate_phase"
        
        # Assess intent
        intent = self._assess_intent(detections, primary_operation_type)
        
        # Predict next steps
        next_steps = self._predict_next_steps(detections, stage)
        
        # Determine threat level
        threat_level = self._determine_threat_level(detections, stage)
        
        # Calculate campaign confidence
        avg_confidence = sum(d.confidence for d in detections) / len(detections)
        campaign_confidence = min(avg_confidence * 1.1, 1.0)  # Boost for multiple TTPs
        
        # Construct timeline
        timeline = []
        for detection in sorted(detections, key=lambda d: d.timestamp):
            timeline.append({
                "timestamp": detection.timestamp,
                "ttp": detection.pattern.name,
                "category": detection.pattern.category.value,
                "confidence": detection.confidence
            })
        
        campaign_id = f"campaign_{int(time.time())}"
        
        return CampaignAssessment(
            campaign_id=campaign_id,
            detected_ttps=detections,
            operation_type=primary_operation_type,
            campaign_stage=stage,
            intent_assessment=intent,
            predicted_next_steps=next_steps,
            threat_level=threat_level,
            confidence=campaign_confidence,
            timeline=timeline
        )
    
    def _assess_intent(self, detections: List[TTPDetection], operation_type: OperationType) -> str:
        """Assess adversary intent from TTPs"""
        
        if operation_type == OperationType.SABOTAGE:
            return "Adversary intends to disrupt or destroy critical infrastructure"
        elif operation_type == OperationType.INTELLIGENCE_COLLECTION:
            return "Adversary conducting reconnaissance for future operations or intelligence gathering"
        elif operation_type == OperationType.CYBER_ATTACK:
            return "Adversary preparing or executing cyber attack for disruption or data theft"
        elif operation_type == OperationType.HYBRID_OPERATION:
            return "Coordinated multi-domain operation combining cyber, physical, and information warfare"
        elif operation_type == OperationType.DECEPTION:
            return "Adversary attempting to mislead analysis and mask true intent"
        else:
            return "Intent unclear, requires additional intelligence"
    
    def _predict_next_steps(self, detections: List[TTPDetection], current_stage: str) -> List[str]:
        """Predict adversary's next likely actions"""
        
        predictions = []
        
        if current_stage == "reconnaissance_phase":
            predictions = [
                "Equipment and personnel positioning",
                "Final target selection",
                "Timing coordination",
                "Backup plan preparation"
            ]
        elif current_stage == "preparation_phase":
            predictions = [
                "Final approach to target",
                "Execution timing determination",
                "Communications establishment",
                "Execute operation"
            ]
        elif current_stage == "initial_access_phase":
            predictions = [
                "Establish persistence",
                "Lateral movement",
                "Privilege escalation",
                "Objective execution"
            ]
        else:
            predictions = [
                "Continue current operations",
                "Assess operation success",
                "Prepare exfiltration or withdrawal",
                "Execute final objectives"
            ]
        
        return predictions
    
    def _determine_threat_level(self, detections: List[TTPDetection], stage: str) -> str:
        """Determine overall threat level"""
        
        avg_confidence = sum(d.confidence for d in detections) / len(detections)
        detection_count = len(detections)
        
        # Higher threat in later stages
        stage_multiplier = 1.0
        if "execution" in stage or "impact" in stage:
            stage_multiplier = 1.5
        elif "preparation" in stage or "access" in stage:
            stage_multiplier = 1.2
        
        threat_score = (avg_confidence * detection_count * stage_multiplier) / 2
        
        if threat_score > 2.0:
            return "CRITICAL"
        elif threat_score > 1.5:
            return "HIGH"
        elif threat_score > 1.0:
            return "ELEVATED"
        else:
            return "MODERATE"


# Global instance
ttp_recognition_engine = TTPPatternRecognitionEngine()


async def recognize_ttp_patterns(
    observed_data: List[Dict[str, Any]],
    context: Dict[str, Any] = None
) -> Tuple[List[TTPDetection], Optional[CampaignAssessment]]:
    """
    Main entry point: Recognize TTP patterns in observed data.
    Returns detected TTPs and campaign assessment if applicable.
    """
    return await ttp_recognition_engine.analyze_for_ttps(observed_data, context)

