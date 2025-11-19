"""
Intelligent Agent Specialization Engine
Autonomously determines what specialized agents are needed for any task
Provides quantum-speed analysis with human-level intuition
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

log = logging.getLogger("agent-specialization-engine")

class AgentDomain(Enum):
    """Domains of agent expertise"""
    # Intelligence domains
    SIGINT = "signals_intelligence"
    HUMINT = "human_intelligence"
    OSINT = "open_source_intelligence"
    GEOINT = "geospatial_intelligence"
    MASINT = "measurement_signature_intelligence"
    CYBINT = "cyber_intelligence"
    FININT = "financial_intelligence"
    
    # Analysis domains
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MODELING = "predictive_modeling"
    CAUSAL_ANALYSIS = "causal_analysis"
    TEMPORAL_CORRELATION = "temporal_correlation"
    GEOSPATIAL_ANALYSIS = "geospatial_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    
    # Domain expertise
    MARITIME_OPERATIONS = "maritime_operations"
    CYBER_OPERATIONS = "cyber_operations"
    INFRASTRUCTURE_ANALYSIS = "infrastructure_analysis"
    THREAT_ASSESSMENT = "threat_assessment"
    DECEPTION_DETECTION = "deception_detection"
    COUNTERINTELLIGENCE = "counterintelligence"
    
    # Technical domains
    ACOUSTIC_ANALYSIS = "acoustic_analysis"
    ELECTROMAGNETIC_ANALYSIS = "electromagnetic_analysis"
    COMMUNICATIONS_ANALYSIS = "communications_analysis"
    SENSOR_FUSION = "sensor_fusion"
    IMAGE_ANALYSIS = "image_analysis"
    VIDEO_ANALYSIS = "video_analysis"
    
    # Operational domains
    FORCE_TRACKING = "force_tracking"
    LOGISTICS_ANALYSIS = "logistics_analysis"
    COMMAND_CONTROL = "command_control"
    BATTLE_DAMAGE_ASSESSMENT = "battle_damage_assessment"
    
    # Synthesis domains
    INTELLIGENCE_SYNTHESIS = "intelligence_synthesis"
    COURSE_OF_ACTION = "course_of_action"
    IMPACT_ASSESSMENT = "impact_assessment"
    DECISION_SUPPORT = "decision_support"

class AgentCapabilityLevel(Enum):
    """Agent capability sophistication levels"""
    BASIC = "basic"          # Simple data processing
    INTERMEDIATE = "intermediate"  # Pattern recognition
    ADVANCED = "advanced"    # Complex analysis
    EXPERT = "expert"       # Domain expertise with intuition
    MASTER = "master"       # Multi-domain synthesis with prediction

@dataclass
class AgentSpecialization:
    """Definition of a specialized agent"""
    agent_type: str
    domain: AgentDomain
    capability_level: AgentCapabilityLevel
    required_inputs: List[str]
    output_types: List[str]
    confidence_threshold: float
    processing_priority: int
    dependencies: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    reasoning_capabilities: List[str] = field(default_factory=list)

@dataclass
class TaskAnalysis:
    """Analysis of what a task requires"""
    task_id: str
    task_description: str
    identified_domains: List[AgentDomain]
    required_specializations: List[AgentSpecialization]
    estimated_complexity: float
    recommended_agent_count: int
    processing_strategy: str
    confidence: float
    reasoning: List[str]

class IntelligentSpecializationEngine:
    """
    Autonomously determines what specialized agents are needed for any task.
    Operates like an intelligence analyst with quantum-speed processing.
    """
    
    def __init__(self):
        self.specialization_library: Dict[str, AgentSpecialization] = {}
        self.domain_keywords: Dict[AgentDomain, Set[str]] = {}
        self.capability_patterns: Dict[str, List[AgentDomain]] = {}
        self.learned_associations: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.analysis_history: List[TaskAnalysis] = []
        self.specialization_effectiveness: Dict[str, float] = {}
        
        self._initialize_specialization_library()
        self._initialize_domain_keywords()
        self._initialize_capability_patterns()
        
        log.info("Intelligent Specialization Engine initialized")
    
    def _initialize_specialization_library(self):
        """Initialize library of available agent specializations"""
        
        # SIGINT specialists
        self.specialization_library["sigint_collector"] = AgentSpecialization(
            agent_type="sigint_collector",
            domain=AgentDomain.SIGINT,
            capability_level=AgentCapabilityLevel.INTERMEDIATE,
            required_inputs=["communications_data", "signal_intercepts"],
            output_types=["signal_patterns", "communication_metadata"],
            confidence_threshold=0.7,
            processing_priority=9,
            expertise_areas=["rf_analysis", "signal_processing", "frequency_analysis"]
        )
        
        self.specialization_library["sigint_analyst"] = AgentSpecialization(
            agent_type="sigint_analyst",
            domain=AgentDomain.SIGINT,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["signal_patterns", "communication_metadata", "historical_sigint"],
            output_types=["intelligence_assessment", "threat_indicators"],
            confidence_threshold=0.8,
            processing_priority=8,
            dependencies=["sigint_collector"],
            expertise_areas=["pattern_recognition", "adversary_ttps", "communications_protocols"],
            reasoning_capabilities=["temporal_correlation", "behavioral_analysis"]
        )
        
        # Maritime specialists
        self.specialization_library["maritime_tracker"] = AgentSpecialization(
            agent_type="maritime_tracker",
            domain=AgentDomain.MARITIME_OPERATIONS,
            capability_level=AgentCapabilityLevel.INTERMEDIATE,
            required_inputs=["ais_data", "radar_tracks", "satellite_imagery"],
            output_types=["vessel_tracks", "maritime_patterns"],
            confidence_threshold=0.75,
            processing_priority=8,
            expertise_areas=["vessel_identification", "route_analysis", "ais_correlation"]
        )
        
        self.specialization_library["submarine_specialist"] = AgentSpecialization(
            agent_type="submarine_specialist",
            domain=AgentDomain.MARITIME_OPERATIONS,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["acoustic_data", "sonar_contacts", "oceanographic_data"],
            output_types=["submarine_tracks", "threat_assessment", "intent_analysis"],
            confidence_threshold=0.8,
            processing_priority=10,
            expertise_areas=["acoustic_signatures", "submarine_tactics", "underwater_warfare"],
            reasoning_capabilities=["intent_inference", "tactical_prediction"]
        )
        
        # Cyber specialists
        self.specialization_library["cyber_monitor"] = AgentSpecialization(
            agent_type="cyber_monitor",
            domain=AgentDomain.CYBER_OPERATIONS,
            capability_level=AgentCapabilityLevel.INTERMEDIATE,
            required_inputs=["network_traffic", "intrusion_alerts", "log_data"],
            output_types=["cyber_events", "anomaly_indicators"],
            confidence_threshold=0.7,
            processing_priority=9,
            expertise_areas=["network_analysis", "malware_detection", "intrusion_detection"]
        )
        
        self.specialization_library["cyber_threat_analyst"] = AgentSpecialization(
            agent_type="cyber_threat_analyst",
            domain=AgentDomain.CYBER_OPERATIONS,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["cyber_events", "threat_intelligence", "vulnerability_data"],
            output_types=["threat_assessment", "attribution_analysis", "impact_prediction"],
            confidence_threshold=0.85,
            processing_priority=8,
            dependencies=["cyber_monitor"],
            expertise_areas=["apt_tactics", "malware_analysis", "threat_attribution"],
            reasoning_capabilities=["campaign_correlation", "adversary_profiling"]
        )
        
        # Pattern recognition specialists
        self.specialization_library["anomaly_detector"] = AgentSpecialization(
            agent_type="anomaly_detector",
            domain=AgentDomain.ANOMALY_DETECTION,
            capability_level=AgentCapabilityLevel.ADVANCED,
            required_inputs=["time_series_data", "baseline_patterns"],
            output_types=["anomalies", "deviation_scores"],
            confidence_threshold=0.75,
            processing_priority=7,
            expertise_areas=["statistical_analysis", "outlier_detection", "baseline_modeling"]
        )
        
        self.specialization_library["pattern_correlator"] = AgentSpecialization(
            agent_type="pattern_correlator",
            domain=AgentDomain.PATTERN_RECOGNITION,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["multi_source_data", "temporal_sequences"],
            output_types=["correlated_patterns", "causal_relationships"],
            confidence_threshold=0.8,
            processing_priority=8,
            expertise_areas=["cross_domain_correlation", "causal_inference", "temporal_reasoning"],
            reasoning_capabilities=["multi_source_fusion", "hypothesis_generation"]
        )
        
        # Temporal correlation specialists
        self.specialization_library["temporal_analyst"] = AgentSpecialization(
            agent_type="temporal_analyst",
            domain=AgentDomain.TEMPORAL_CORRELATION,
            capability_level=AgentCapabilityLevel.ADVANCED,
            required_inputs=["timestamped_events", "temporal_sequences"],
            output_types=["temporal_relationships", "sequence_patterns"],
            confidence_threshold=0.75,
            processing_priority=7,
            expertise_areas=["time_series_analysis", "event_sequencing", "causality_detection"]
        )
        
        # Geospatial specialists
        self.specialization_library["geospatial_analyst"] = AgentSpecialization(
            agent_type="geospatial_analyst",
            domain=AgentDomain.GEOSPATIAL_ANALYSIS,
            capability_level=AgentCapabilityLevel.ADVANCED,
            required_inputs=["location_data", "imagery", "terrain_data"],
            output_types=["spatial_analysis", "movement_patterns", "terrain_assessment"],
            confidence_threshold=0.8,
            processing_priority=7,
            expertise_areas=["gis_analysis", "imagery_interpretation", "route_analysis"]
        )
        
        # Infrastructure specialists
        self.specialization_library["infrastructure_analyst"] = AgentSpecialization(
            agent_type="infrastructure_analyst",
            domain=AgentDomain.INFRASTRUCTURE_ANALYSIS,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["infrastructure_data", "vulnerability_data", "dependencies"],
            output_types=["critical_nodes", "vulnerability_assessment", "cascade_analysis"],
            confidence_threshold=0.85,
            processing_priority=8,
            expertise_areas=["critical_infrastructure", "network_topology", "cascading_failures"],
            reasoning_capabilities=["impact_prediction", "resilience_analysis"]
        )
        
        # Threat assessment specialists
        self.specialization_library["threat_assessor"] = AgentSpecialization(
            agent_type="threat_assessor",
            domain=AgentDomain.THREAT_ASSESSMENT,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["threat_indicators", "intelligence_reports", "historical_threats"],
            output_types=["threat_level", "probability_assessment", "risk_analysis"],
            confidence_threshold=0.85,
            processing_priority=9,
            expertise_areas=["threat_modeling", "risk_analysis", "adversary_capabilities"],
            reasoning_capabilities=["intent_assessment", "capability_analysis", "probability_estimation"]
        )
        
        # Deception detection specialists
        self.specialization_library["deception_detector"] = AgentSpecialization(
            agent_type="deception_detector",
            domain=AgentDomain.DECEPTION_DETECTION,
            capability_level=AgentCapabilityLevel.EXPERT,
            required_inputs=["information_sources", "behavioral_patterns", "context_data"],
            output_types=["deception_indicators", "credibility_scores", "manipulation_analysis"],
            confidence_threshold=0.8,
            processing_priority=9,
            expertise_areas=["behavioral_analysis", "information_manipulation", "credibility_assessment"],
            reasoning_capabilities=["consistency_checking", "source_verification", "intent_inference"]
        )
        
        # Predictive modeling specialists
        self.specialization_library["predictive_modeler"] = AgentSpecialization(
            agent_type="predictive_modeler",
            domain=AgentDomain.PREDICTIVE_MODELING,
            capability_level=AgentCapabilityLevel.ADVANCED,
            required_inputs=["historical_data", "current_state", "trend_data"],
            output_types=["predictions", "confidence_intervals", "scenario_forecasts"],
            confidence_threshold=0.75,
            processing_priority=7,
            expertise_areas=["statistical_modeling", "machine_learning", "forecasting"]
        )
        
        # Intelligence synthesis specialists (Master level)
        self.specialization_library["intelligence_synthesizer"] = AgentSpecialization(
            agent_type="intelligence_synthesizer",
            domain=AgentDomain.INTELLIGENCE_SYNTHESIS,
            capability_level=AgentCapabilityLevel.MASTER,
            required_inputs=["all_source_intelligence", "analysis_results", "context"],
            output_types=["synthesized_intelligence", "comprehensive_assessment", "key_judgments"],
            confidence_threshold=0.9,
            processing_priority=10,
            dependencies=["sigint_analyst", "cyber_threat_analyst", "threat_assessor"],
            expertise_areas=["all_source_fusion", "strategic_analysis", "intelligence_production"],
            reasoning_capabilities=["holistic_synthesis", "strategic_reasoning", "judgment_formation"]
        )
        
        # Course of action specialists
        self.specialization_library["coa_developer"] = AgentSpecialization(
            agent_type="coa_developer",
            domain=AgentDomain.COURSE_OF_ACTION,
            capability_level=AgentCapabilityLevel.MASTER,
            required_inputs=["intelligence_assessment", "operational_context", "capabilities"],
            output_types=["courses_of_action", "risk_benefit_analysis", "recommendations"],
            confidence_threshold=0.85,
            processing_priority=10,
            dependencies=["intelligence_synthesizer"],
            expertise_areas=["operational_planning", "decision_analysis", "risk_assessment"],
            reasoning_capabilities=["strategic_planning", "option_generation", "consequence_analysis"]
        )
        
        # Impact assessment specialists
        self.specialization_library["impact_assessor"] = AgentSpecialization(
            agent_type="impact_assessor",
            domain=AgentDomain.IMPACT_ASSESSMENT,
            capability_level=AgentCapabilityLevel.ADVANCED,
            required_inputs=["event_data", "system_state", "dependency_maps"],
            output_types=["impact_analysis", "cascading_effects", "recovery_estimates"],
            confidence_threshold=0.8,
            processing_priority=8,
            expertise_areas=["systems_analysis", "cascading_failures", "resilience_modeling"],
            reasoning_capabilities=["cascade_prediction", "second_order_effects", "timeline_estimation"]
        )
        
        log.info(f"Initialized {len(self.specialization_library)} agent specializations")
    
    def _initialize_domain_keywords(self):
        """Initialize keyword mappings for domain identification"""
        
        self.domain_keywords = {
            AgentDomain.SIGINT: {
                "signal", "communications", "intercept", "rf", "radio", "frequency",
                "satcom", "gnss", "spoofing", "electromagnetic", "transmission"
            },
            AgentDomain.MARITIME_OPERATIONS: {
                "maritime", "naval", "ship", "vessel", "submarine", "sonar", "acoustic",
                "ais", "underwater", "ocean", "sea", "port", "harbor", "kilo"
            },
            AgentDomain.CYBER_OPERATIONS: {
                "cyber", "network", "hacking", "malware", "intrusion", "server",
                "digital", "computer", "internet", "firewall", "breach", "exploit"
            },
            AgentDomain.INFRASTRUCTURE_ANALYSIS: {
                "infrastructure", "cable", "pipeline", "power", "utility", "critical",
                "facility", "network", "grid", "supply", "logistics"
            },
            AgentDomain.GEOSPATIAL_ANALYSIS: {
                "location", "position", "coordinates", "gps", "map", "terrain",
                "geography", "spatial", "route", "area", "region"
            },
            AgentDomain.PATTERN_RECOGNITION: {
                "pattern", "trend", "correlation", "relationship", "connection",
                "association", "sequence", "series", "recurring"
            },
            AgentDomain.ANOMALY_DETECTION: {
                "anomaly", "unusual", "abnormal", "outlier", "deviation", "suspicious",
                "irregular", "unexpected", "strange"
            },
            AgentDomain.TEMPORAL_CORRELATION: {
                "time", "temporal", "sequence", "timeline", "chronology", "synchronous",
                "simultaneous", "concurrent", "timing", "before", "after"
            },
            AgentDomain.THREAT_ASSESSMENT: {
                "threat", "risk", "danger", "hostile", "adversary", "enemy",
                "malicious", "attack", "target", "capability", "intent"
            },
            AgentDomain.DECEPTION_DETECTION: {
                "deception", "false", "fake", "misleading", "disinformation",
                "manipulation", "fraud", "spoofing", "masking", "hiding"
            },
            AgentDomain.PREDICTIVE_MODELING: {
                "predict", "forecast", "future", "projection", "estimate",
                "anticipate", "expect", "likely", "probable"
            },
            AgentDomain.IMPACT_ASSESSMENT: {
                "impact", "effect", "consequence", "result", "damage", "loss",
                "cascade", "disruption", "degradation"
            }
        }
    
    def _initialize_capability_patterns(self):
        """Initialize patterns that indicate needed capabilities"""
        
        self.capability_patterns = {
            "multi_source_correlation": [
                AgentDomain.SIGINT, AgentDomain.CYBER_OPERATIONS,
                AgentDomain.PATTERN_RECOGNITION, AgentDomain.TEMPORAL_CORRELATION
            ],
            "submarine_detection": [
                AgentDomain.MARITIME_OPERATIONS, AgentDomain.SIGINT,
                AgentDomain.GEOSPATIAL_ANALYSIS, AgentDomain.PATTERN_RECOGNITION
            ],
            "infrastructure_vulnerability": [
                AgentDomain.INFRASTRUCTURE_ANALYSIS, AgentDomain.CYBER_OPERATIONS,
                AgentDomain.IMPACT_ASSESSMENT, AgentDomain.GEOSPATIAL_ANALYSIS
            ],
            "threat_characterization": [
                AgentDomain.THREAT_ASSESSMENT, AgentDomain.PATTERN_RECOGNITION,
                AgentDomain.PREDICTIVE_MODELING, AgentDomain.INTELLIGENCE_SYNTHESIS
            ],
            "coordinated_attack": [
                AgentDomain.TEMPORAL_CORRELATION, AgentDomain.PATTERN_RECOGNITION,
                AgentDomain.THREAT_ASSESSMENT, AgentDomain.DECEPTION_DETECTION
            ]
        }
    
    async def analyze_task_requirements(
        self,
        task_description: str,
        available_data: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> TaskAnalysis:
        """
        Autonomously analyze what agents are needed for a task.
        Operates like an intelligence analyst with quantum-speed processing.
        """
        
        start_time = time.time()
        task_id = f"task_{int(time.time() * 1000)}"
        
        log.info(f"Analyzing task requirements: {task_description[:100]}...")
        
        # Step 1: Identify domains from task description
        identified_domains = self._identify_domains(task_description, available_data, context)
        
        # Step 2: Determine complexity and scope
        complexity_score = self._assess_complexity(task_description, identified_domains, available_data)
        
        # Step 3: Select required specializations
        required_specializations = self._select_specializations(
            identified_domains, complexity_score, available_data
        )
        
        # Step 4: Determine agent count
        recommended_count = self._calculate_agent_count(
            required_specializations, complexity_score, available_data
        )
        
        # Step 5: Determine processing strategy
        strategy = self._determine_strategy(
            required_specializations, complexity_score, recommended_count
        )
        
        # Step 6: Generate reasoning
        reasoning = self._generate_reasoning(
            task_description, identified_domains, required_specializations, strategy
        )
        
        analysis = TaskAnalysis(
            task_id=task_id,
            task_description=task_description,
            identified_domains=identified_domains,
            required_specializations=required_specializations,
            estimated_complexity=complexity_score,
            recommended_agent_count=recommended_count,
            processing_strategy=strategy,
            confidence=0.85,
            reasoning=reasoning
        )
        
        self.analysis_history.append(analysis)
        
        processing_time = time.time() - start_time
        log.info(f"Task analysis completed in {processing_time:.3f}s: "
                f"{len(required_specializations)} specializations, "
                f"{recommended_count} agents, strategy={strategy}")
        
        return analysis
    
    def _identify_domains(
        self,
        task_description: str,
        available_data: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> List[AgentDomain]:
        """Identify which domains are relevant to the task"""
        
        task_lower = task_description.lower()
        identified = set()
        
        # Keyword matching
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                identified.add(domain)
        
        # Data type analysis
        if available_data:
            for data_item in available_data:
                data_type = data_item.get("type", "").lower()
                data_name = data_item.get("name", "").lower()
                
                # Match data types to domains
                if any(sig in data_type or sig in data_name 
                      for sig in ["signal", "comm", "rf", "satcom"]):
                    identified.add(AgentDomain.SIGINT)
                
                if any(mar in data_type or mar in data_name 
                      for mar in ["ais", "maritime", "vessel", "acoustic"]):
                    identified.add(AgentDomain.MARITIME_OPERATIONS)
                
                if any(cyber in data_type or cyber in data_name 
                      for cyber in ["network", "cyber", "log", "traffic"]):
                    identified.add(AgentDomain.CYBER_OPERATIONS)
        
        # Always include foundational domains
        identified.add(AgentDomain.PATTERN_RECOGNITION)
        identified.add(AgentDomain.INTELLIGENCE_SYNTHESIS)
        
        # If temporal indicators present, add temporal correlation
        if any(word in task_lower for word in ["time", "when", "sequence", "before", "after"]):
            identified.add(AgentDomain.TEMPORAL_CORRELATION)
        
        # If geospatial indicators present
        if any(word in task_lower for word in ["where", "location", "position", "area", "route"]):
            identified.add(AgentDomain.GEOSPATIAL_ANALYSIS)
        
        # If threat indicators present
        if any(word in task_lower for word in ["threat", "risk", "attack", "hostile", "adversary"]):
            identified.add(AgentDomain.THREAT_ASSESSMENT)
        
        return list(identified)
    
    def _assess_complexity(
        self,
        task_description: str,
        identified_domains: List[AgentDomain],
        available_data: List[Dict[str, Any]] = None
    ) -> float:
        """Assess task complexity on 0-1 scale"""
        
        complexity = 0.3  # Base complexity
        
        # Domain count factor
        complexity += len(identified_domains) * 0.05
        
        # Data source count factor
        if available_data:
            complexity += min(len(available_data) * 0.1, 0.3)
        
        # Task description complexity
        word_count = len(task_description.split())
        complexity += min(word_count / 200, 0.2)
        
        # Multi-source correlation indicators
        if any(word in task_description.lower() 
              for word in ["correlate", "combine", "fuse", "integrate", "synthesize"]):
            complexity += 0.1
        
        # Temporal complexity
        if any(word in task_description.lower() 
              for word in ["timeline", "sequence", "chronology", "track"]):
            complexity += 0.1
        
        # Predictive requirements
        if any(word in task_description.lower() 
              for word in ["predict", "forecast", "anticipate", "project"]):
            complexity += 0.15
        
        return min(complexity, 1.0)
    
    def _select_specializations(
        self,
        identified_domains: List[AgentDomain],
        complexity_score: float,
        available_data: List[Dict[str, Any]] = None
    ) -> List[AgentSpecialization]:
        """Select which specialized agents are needed"""
        
        selected = []
        selected_types = set()
        
        # Select agents for each identified domain
        for domain in identified_domains:
            domain_agents = [
                spec for spec in self.specialization_library.values()
                if spec.domain == domain
            ]
            
            # Sort by capability level and priority
            domain_agents.sort(
                key=lambda x: (x.capability_level.value, -x.processing_priority)
            )
            
            # Select appropriate level based on complexity
            for agent in domain_agents:
                if agent.agent_type not in selected_types:
                    # Check if complexity justifies this agent
                    if complexity_score >= 0.7 and agent.capability_level in [
                        AgentCapabilityLevel.EXPERT, AgentCapabilityLevel.MASTER
                    ]:
                        selected.append(agent)
                        selected_types.add(agent.agent_type)
                        break
                    elif complexity_score >= 0.4 and agent.capability_level in [
                        AgentCapabilityLevel.INTERMEDIATE, AgentCapabilityLevel.ADVANCED
                    ]:
                        selected.append(agent)
                        selected_types.add(agent.agent_type)
                        break
                    elif agent.capability_level == AgentCapabilityLevel.BASIC:
                        selected.append(agent)
                        selected_types.add(agent.agent_type)
                        break
        
        # Add dependency agents
        all_dependencies = set()
        for spec in selected:
            all_dependencies.update(spec.dependencies)
        
        for dep_type in all_dependencies:
            if dep_type in self.specialization_library and dep_type not in selected_types:
                selected.append(self.specialization_library[dep_type])
                selected_types.add(dep_type)
        
        # Ensure synthesis agent is included for complex tasks
        if complexity_score >= 0.5 and "intelligence_synthesizer" not in selected_types:
            selected.append(self.specialization_library["intelligence_synthesizer"])
        
        return selected
    
    def _calculate_agent_count(
        self,
        required_specializations: List[AgentSpecialization],
        complexity_score: float,
        available_data: List[Dict[str, Any]] = None
    ) -> int:
        """Calculate how many agents to deploy"""
        
        # Base count from specializations
        base_count = len(required_specializations) * 2
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (complexity_score * 2.0)
        
        # Data volume multiplier
        data_multiplier = 1.0
        if available_data:
            data_count = len(available_data)
            if data_count > 100:
                data_multiplier = min(data_count / 10, 20.0)
            elif data_count > 10:
                data_multiplier = min(data_count / 5, 10.0)
        
        # Calculate total
        total = int(base_count * complexity_multiplier * data_multiplier)
        
        # Ensure minimum and maximum
        total = max(total, len(required_specializations))
        total = min(total, 5000)  # Cap at 5000 agents
        
        return total
    
    def _determine_strategy(
        self,
        required_specializations: List[AgentSpecialization],
        complexity_score: float,
        agent_count: int
    ) -> str:
        """Determine processing strategy"""
        
        if agent_count > 1000:
            return "massive_parallel_swarm"
        elif agent_count > 200:
            return "large_hierarchical_swarm"
        elif agent_count > 50:
            return "medium_coordinated_swarm"
        elif complexity_score > 0.7:
            return "expert_analysis_pipeline"
        else:
            return "standard_multi_agent"
    
    def _generate_reasoning(
        self,
        task_description: str,
        identified_domains: List[AgentDomain],
        required_specializations: List[AgentSpecialization],
        strategy: str
    ) -> List[str]:
        """Generate human-readable reasoning for decisions"""
        
        reasoning = []
        
        reasoning.append(
            f"Identified {len(identified_domains)} relevant intelligence domains: "
            f"{', '.join(d.value for d in identified_domains[:5])}"
        )
        
        reasoning.append(
            f"Selected {len(required_specializations)} specialized agent types "
            f"based on domain requirements and task complexity"
        )
        
        expert_agents = [s for s in required_specializations 
                        if s.capability_level in [AgentCapabilityLevel.EXPERT, AgentCapabilityLevel.MASTER]]
        if expert_agents:
            reasoning.append(
                f"Deploying {len(expert_agents)} expert-level agents for high-confidence analysis: "
                f"{', '.join(a.agent_type for a in expert_agents[:3])}"
            )
        
        reasoning.append(f"Processing strategy: {strategy}")
        
        return reasoning


# Global instance
intelligent_specialization_engine = IntelligentSpecializationEngine()


async def analyze_task_and_determine_agents(
    task_description: str,
    available_data: List[Dict[str, Any]] = None,
    context: Dict[str, Any] = None
) -> TaskAnalysis:
    """
    Main entry point: Analyze a task and determine what agents are needed.
    Returns complete analysis with recommended specializations and agent count.
    """
    return await intelligent_specialization_engine.analyze_task_requirements(
        task_description, available_data, context
    )

