"""
Course of Action (COA) Generation System
Generates military-grade courses of action with risk/benefit analysis
Supports decision-making for all operational scenarios
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger("coa-generation")

class COAType(Enum):
    """Types of courses of action"""
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"
    DELAY = "delay"
    WITHDRAW = "withdraw"
    REINFORCE = "reinforce"
    MANEUVER = "maneuver"
    COMBINED_ARMS = "combined_arms"
    SPECIAL_OPERATIONS = "special_operations"
    CYBER_OPERATIONS = "cyber_operations"
    INFORMATION_OPERATIONS = "information_operations"

class EffectType(Enum):
    """Types of desired effects"""
    DEFEAT = "defeat"
    DESTROY = "destroy"
    DISRUPT = "disrupt"
    DEGRADE = "degrade"
    DENY = "deny"
    DECEIVE = "deceive"
    DETER = "deter"
    DELAY_EFFECT = "delay"
    CONTAIN = "contain"
    NEUTRALIZE = "neutralize"

@dataclass
class COAElement:
    """Element of a course of action"""
    element_id: str
    element_type: str  # unit, asset, capability
    name: str
    capabilities: List[str]
    location: Optional[Dict[str, Any]] = None
    timing: Optional[str] = None
    mission: Optional[str] = None

@dataclass
class COAPhase:
    """Phase within a course of action"""
    phase_id: str
    phase_name: str
    sequence: int
    duration_estimate: float  # seconds
    elements_involved: List[str]  # element_ids
    objectives: List[str]
    success_criteria: List[str]
    risks: List[str]

@dataclass
class CourseOfAction:
    """Complete course of action"""
    coa_id: str
    coa_name: str
    coa_type: COAType
    desired_effects: List[EffectType]
    
    # Structure
    elements: List[COAElement]
    phases: List[COAPhase]
    
    # Analysis
    advantages: List[str]
    disadvantages: List[str]
    risks: List[str]
    resource_requirements: Dict[str, Any]
    estimated_duration: float
    probability_of_success: float
    
    # Evaluation
    feasibility_score: float  # 0-1
    acceptability_score: float  # 0-1
    suitability_score: float  # 0-1
    overall_score: float  # 0-1
    
    # Metadata
    created_at: float
    confidence: float

@dataclass
class COAComparison:
    """Comparison of multiple COAs"""
    comparison_id: str
    coas: List[CourseOfAction]
    recommended_coa: str  # coa_id
    recommendation_rationale: List[str]
    comparison_matrix: Dict[str, Dict[str, Any]]
    decision_brief: str

class COAGenerator:
    """
    Generates military-grade courses of action.
    Provides multiple options with risk/benefit analysis.
    """
    
    def __init__(self):
        self.coa_history: List[CourseOfAction] = []
        self.comparison_history: List[COAComparison] = []
        
        # COA templates
        self.coa_templates: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_coa_templates()
        
        log.info("COA Generator initialized")
    
    def _initialize_coa_templates(self):
        """Initialize COA templates"""
        
        # Defensive COA template
        self.coa_templates["defensive"] = {
            "phases": [
                {"name": "Prepare Defenses", "objectives": ["Establish defensive positions", "Deploy obstacles"]},
                {"name": "Detect & Warn", "objectives": ["Early warning", "Intelligence collection"]},
                {"name": "Engage", "objectives": ["Engage enemy forces", "Attrit enemy"]},
                {"name": "Counterattack", "objectives": ["Exploit enemy weakness", "Restore positions"]}
            ],
            "typical_elements": ["infantry_units", "artillery", "air_defense", "engineers"],
            "effects": [EffectType.DEFEAT, EffectType.DELAY_EFFECT, EffectType.DISRUPT]
        }
        
        # Offensive COA template
        self.coa_templates["offensive"] = {
            "phases": [
                {"name": "Intelligence Preparation", "objectives": ["Identify enemy", "Find vulnerabilities"]},
                {"name": "Shaping Operations", "objectives": ["Isolate objective", "Suppress defenses"]},
                {"name": "Decisive Operation", "objectives": ["Seize objective", "Defeat enemy"]},
                {"name": "Exploitation", "objectives": ["Expand success", "Prevent reorganization"]}
            ],
            "typical_elements": ["maneuver_units", "fires", "aviation", "special_operations"],
            "effects": [EffectType.DESTROY, EffectType.DEFEAT, EffectType.DISRUPT]
        }
        
        # Cyber operations template
        self.coa_templates["cyber"] = {
            "phases": [
                {"name": "Reconnaissance", "objectives": ["Map network", "Identify vulnerabilities"]},
                {"name": "Initial Access", "objectives": ["Establish foothold", "Gain credentials"]},
                {"name": "Execution", "objectives": ["Achieve effects", "Maintain access"]},
                {"name": "Exfiltration", "objectives": ["Remove presence", "Cover tracks"]}
            ],
            "typical_elements": ["cyber_teams", "tools", "infrastructure"],
            "effects": [EffectType.DISRUPT, EffectType.DEGRADE, EffectType.DENY]
        }
    
    async def generate_coas(
        self,
        situation: Dict[str, Any],
        objective: str,
        constraints: Dict[str, Any] = None,
        num_coas: int = 4
    ) -> List[CourseOfAction]:
        """
        Generate multiple courses of action for a situation.
        Returns ranked list of COAs.
        """
        
        log.info(f"Generating {num_coas} COAs for objective: {objective}")
        
        coas = []
        
        # Generate different COA types
        coa_types = [
            COAType.OFFENSIVE,
            COAType.DEFENSIVE,
            COAType.COMBINED_ARMS,
            COAType.SPECIAL_OPERATIONS
        ]
        
        for idx, coa_type in enumerate(coa_types[:num_coas]):
            coa = await self._generate_single_coa(
                situation=situation,
                objective=objective,
                coa_type=coa_type,
                coa_number=idx + 1,
                constraints=constraints
            )
            coas.append(coa)
        
        # Rank COAs
        ranked_coas = sorted(coas, key=lambda c: c.overall_score, reverse=True)
        
        log.info(f"Generated {len(ranked_coas)} COAs, top score: {ranked_coas[0].overall_score:.2f}")
        
        return ranked_coas
    
    async def _generate_single_coa(
        self,
        situation: Dict[str, Any],
        objective: str,
        coa_type: COAType,
        coa_number: int,
        constraints: Dict[str, Any]
    ) -> CourseOfAction:
        """Generate a single COA"""
        
        coa_id = f"coa_{int(time.time())}_{coa_number}"
        
        # Get template if available
        template_key = coa_type.value.replace("_", " ").split()[0]  # "offensive", "defensive", etc.
        template = self.coa_templates.get(template_key, {})
        
        # Generate elements (units, assets, capabilities)
        elements = self._generate_coa_elements(situation, coa_type)
        
        # Generate phases
        phases = self._generate_coa_phases(
            coa_type, objective, elements, template
        )
        
        # Determine desired effects
        desired_effects = template.get("effects", [EffectType.DEFEAT])
        
        # Analyze COA
        advantages, disadvantages = self._analyze_coa(coa_type, elements, phases)
        risks = self._identify_coa_risks(coa_type, elements, phases)
        resources = self._calculate_resource_requirements(elements, phases)
        
        # Estimate duration
        duration = sum(phase.duration_estimate for phase in phases)
        
        # Calculate probability of success
        prob_success = self._calculate_probability_of_success(
            coa_type, elements, phases, situation
        )
        
        # Evaluate against criteria
        feasibility = self._evaluate_feasibility(elements, resources, constraints)
        acceptability = self._evaluate_acceptability(risks, prob_success)
        suitability = self._evaluate_suitability(desired_effects, objective)
        
        overall_score = (feasibility + acceptability + suitability) / 3
        
        coa = CourseOfAction(
            coa_id=coa_id,
            coa_name=f"COA {coa_number}: {coa_type.value.title()} Option",
            coa_type=coa_type,
            desired_effects=desired_effects,
            elements=elements,
            phases=phases,
            advantages=advantages,
            disadvantages=disadvantages,
            risks=risks,
            resource_requirements=resources,
            estimated_duration=duration,
            probability_of_success=prob_success,
            feasibility_score=feasibility,
            acceptability_score=acceptability,
            suitability_score=suitability,
            overall_score=overall_score,
            created_at=time.time(),
            confidence=0.85
        )
        
        self.coa_history.append(coa)
        
        return coa
    
    def _generate_coa_elements(
        self,
        situation: Dict[str, Any],
        coa_type: COAType
    ) -> List[COAElement]:
        """Generate COA elements based on situation and type"""
        
        elements = []
        
        if coa_type == COAType.OFFENSIVE:
            elements.extend([
                COAElement(
                    element_id="elem_maneuver_1",
                    element_type="unit",
                    name="Maneuver Element",
                    capabilities=["mobility", "firepower", "communications"],
                    mission="Seize objective"
                ),
                COAElement(
                    element_id="elem_fires_1",
                    element_type="fires",
                    name="Fire Support Element",
                    capabilities=["artillery", "precision_strike"],
                    mission="Suppress enemy defenses"
                ),
                COAElement(
                    element_id="elem_air_1",
                    element_type="aviation",
                    name="Close Air Support",
                    capabilities=["air_interdiction", "cas"],
                    mission="Provide air support"
                )
            ])
        
        elif coa_type == COAType.DEFENSIVE:
            elements.extend([
                COAElement(
                    element_id="elem_def_1",
                    element_type="unit",
                    name="Defensive Force",
                    capabilities=["defensive_positions", "anti_tank", "air_defense"],
                    mission="Defend key terrain"
                ),
                COAElement(
                    element_id="elem_reserve_1",
                    element_type="reserve",
                    name="Reserve Force",
                    capabilities=["counterattack", "reinforcement"],
                    mission="Counterattack or reinforce"
                )
            ])
        
        elif coa_type == COAType.SPECIAL_OPERATIONS:
            elements.extend([
                COAElement(
                    element_id="elem_sof_1",
                    element_type="special_operations",
                    name="SOF Team",
                    capabilities=["covert_ops", "direct_action", "special_reconnaissance"],
                    mission="Execute special operation"
                ),
                COAElement(
                    element_id="elem_support_1",
                    element_type="support",
                    name="Support Element",
                    capabilities=["isr", "fires", "exfiltration"],
                    mission="Support SOF operation"
                )
            ])
        
        elif coa_type == COAType.CYBER_OPERATIONS:
            elements.extend([
                COAElement(
                    element_id="elem_cyber_1",
                    element_type="cyber_team",
                    name="Offensive Cyber Team",
                    capabilities=["network_exploitation", "malware_deployment"],
                    mission="Achieve cyber effects"
                ),
                COAElement(
                    element_id="elem_intel_1",
                    element_type="intelligence",
                    name="Cyber Intelligence Team",
                    capabilities=["network_mapping", "vulnerability_analysis"],
                    mission="Enable cyber operations"
                )
            ])
        
        return elements
    
    def _generate_coa_phases(
        self,
        coa_type: COAType,
        objective: str,
        elements: List[COAElement],
        template: Dict[str, Any]
    ) -> List[COAPhase]:
        """Generate execution phases for COA"""
        
        phases = []
        template_phases = template.get("phases", [])
        
        for idx, phase_template in enumerate(template_phases):
            phase = COAPhase(
                phase_id=f"phase_{idx}",
                phase_name=phase_template["name"],
                sequence=idx,
                duration_estimate=self._estimate_phase_duration(phase_template["name"]),
                elements_involved=[e.element_id for e in elements],
                objectives=phase_template["objectives"],
                success_criteria=[f"{obj} achieved" for obj in phase_template["objectives"]],
                risks=[f"Risk of failure in {phase_template['name']}"]
            )
            phases.append(phase)
        
        return phases
    
    def _estimate_phase_duration(self, phase_name: str) -> float:
        """Estimate phase duration"""
        
        if "preparation" in phase_name.lower() or "intelligence" in phase_name.lower():
            return 7200  # 2 hours
        elif "shaping" in phase_name.lower():
            return 3600  # 1 hour
        elif "decisive" in phase_name.lower() or "execution" in phase_name.lower():
            return 1800  # 30 minutes
        else:
            return 3600  # 1 hour default
    
    def _analyze_coa(
        self,
        coa_type: COAType,
        elements: List[COAElement],
        phases: List[COAPhase]
    ) -> Tuple[List[str], List[str]]:
        """Analyze COA advantages and disadvantages"""
        
        advantages = []
        disadvantages = []
        
        if coa_type == COAType.OFFENSIVE:
            advantages.extend([
                "Initiative and momentum",
                "Dictates tempo of operations",
                "Can achieve decisive results quickly"
            ])
            disadvantages.extend([
                "Higher risk and casualties",
                "Requires more resources",
                "Vulnerable during movement"
            ])
        
        elif coa_type == COAType.DEFENSIVE:
            advantages.extend([
                "Prepared positions and obstacles",
                "Economy of force",
                "Reduced vulnerability"
            ])
            disadvantages.extend([
                "Surrenders initiative to enemy",
                "Limited offensive options",
                "May result in attrition"
            ])
        
        elif coa_type == COAType.SPECIAL_OPERATIONS:
            advantages.extend([
                "Low signature and risk to main force",
                "Can achieve strategic effects with minimal resources",
                "Flexibility and adaptability"
            ])
            disadvantages.extend([
                "Limited combat power",
                "Dependent on intelligence",
                "High risk to SOF personnel"
            ])
        
        elif coa_type == COAType.CYBER_OPERATIONS:
            advantages.extend([
                "No physical risk to personnel",
                "Rapid effects across distance",
                "Potentially low cost"
            ])
            disadvantages.extend([
                "Attribution challenges",
                "Potential for escalation",
                "May be reversible"
            ])
        
        return advantages, disadvantages
    
    def _identify_coa_risks(
        self,
        coa_type: COAType,
        elements: List[COAElement],
        phases: List[COAPhase]
    ) -> List[str]:
        """Identify risks for COA"""
        
        risks = []
        
        # General risks
        risks.append("Intelligence may be incomplete or inaccurate")
        risks.append("Enemy may adapt or counter")
        
        # Type-specific risks
        if coa_type == COAType.OFFENSIVE:
            risks.extend([
                "Casualties during assault",
                "Culmination of attack before objective secured",
                "Enemy reserves or reinforcements"
            ])
        
        elif coa_type == COAType.CYBER_OPERATIONS:
            risks.extend([
                "Detection and attribution",
                "Cyber counterattack",
                "Unintended cascading effects"
            ])
        
        # Phase-specific risks
        for phase in phases:
            if "decisive" in phase.phase_name.lower():
                risks.append(f"Failure during {phase.phase_name} could jeopardize entire operation")
        
        return risks
    
    def _calculate_resource_requirements(
        self,
        elements: List[COAElement],
        phases: List[COAPhase]
    ) -> Dict[str, Any]:
        """Calculate resource requirements"""
        
        return {
            "personnel": len(elements) * 100,  # Rough estimate
            "duration": sum(p.duration_estimate for p in phases),
            "elements_required": len(elements),
            "phases": len(phases),
            "capabilities_needed": list(set(
                cap for elem in elements for cap in elem.capabilities
            ))
        }
    
    def _calculate_probability_of_success(
        self,
        coa_type: COAType,
        elements: List[COAElement],
        phases: List[COAPhase],
        situation: Dict[str, Any]
    ) -> float:
        """Calculate probability of success"""
        
        base_probability = 0.7
        
        # Adjust for COA type
        if coa_type in [COAType.DEFENSIVE, COAType.DELAY]:
            base_probability += 0.1  # Defensive easier to execute
        elif coa_type in [COAType.OFFENSIVE, COAType.SPECIAL_OPERATIONS]:
            base_probability -= 0.1  # Offensive riskier
        
        # Adjust for complexity
        if len(phases) > 5:
            base_probability -= 0.05
        
        # Adjust for intelligence quality
        intel_confidence = situation.get("intelligence_confidence", 0.75)
        base_probability = base_probability * 0.7 + intel_confidence * 0.3
        
        return max(min(base_probability, 0.95), 0.3)
    
    def _evaluate_feasibility(
        self,
        elements: List[COAElement],
        resources: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> float:
        """Evaluate if COA is feasible"""
        
        score = 0.9
        
        # Check resource constraints
        if constraints:
            if "max_personnel" in constraints:
                if resources["personnel"] > constraints["max_personnel"]:
                    score -= 0.3
            
            if "max_duration" in constraints:
                if resources["duration"] > constraints["max_duration"]:
                    score -= 0.2
        
        return max(score, 0.3)
    
    def _evaluate_acceptability(
        self,
        risks: List[str],
        prob_success: float
    ) -> float:
        """Evaluate if COA is acceptable (risk vs. benefit)"""
        
        # High success probability → more acceptable
        score = prob_success
        
        # Many risks → less acceptable
        risk_penalty = min(len(risks) * 0.02, 0.3)
        score -= risk_penalty
        
        return max(score, 0.3)
    
    def _evaluate_suitability(
        self,
        desired_effects: List[EffectType],
        objective: str
    ) -> float:
        """Evaluate if COA achieves desired effects"""
        
        # Check if effects match objective
        objective_lower = objective.lower()
        
        score = 0.8
        
        # Boost for matching effects
        effect_keywords = {
            EffectType.DEFEAT: ["defeat", "destroy", "eliminate"],
            EffectType.DISRUPT: ["disrupt", "interfere", "impede"],
            EffectType.DENY: ["deny", "prevent", "block"],
            EffectType.DELAY_EFFECT: ["delay", "slow", "buy time"]
        }
        
        for effect in desired_effects:
            keywords = effect_keywords.get(effect, [])
            if any(keyword in objective_lower for keyword in keywords):
                score += 0.05
        
        return min(score, 1.0)
    
    async def compare_coas(
        self,
        coas: List[CourseOfAction],
        decision_criteria: List[str] = None
    ) -> COAComparison:
        """
        Compare multiple COAs and recommend best option.
        Returns comprehensive comparison with rationale.
        """
        
        log.info(f"Comparing {len(coas)} COAs")
        
        # Build comparison matrix
        comparison_matrix = {}
        
        for coa in coas:
            comparison_matrix[coa.coa_id] = {
                "name": coa.coa_name,
                "type": coa.coa_type.value,
                "overall_score": coa.overall_score,
                "feasibility": coa.feasibility_score,
                "acceptability": coa.acceptability_score,
                "suitability": coa.suitability_score,
                "prob_success": coa.probability_of_success,
                "duration": coa.estimated_duration,
                "risk_count": len(coa.risks)
            }
        
        # Recommend best COA
        recommended = max(coas, key=lambda c: c.overall_score)
        
        # Generate rationale
        rationale = [
            f"Highest overall score: {recommended.overall_score:.2f}",
            f"Probability of success: {recommended.probability_of_success:.0%}",
            f"Feasibility: {recommended.feasibility_score:.2f}",
            f"Acceptability: {recommended.acceptability_score:.2f}",
            f"Suitability: {recommended.suitability_score:.2f}"
        ]
        
        # Add comparative rationale
        for coa in coas:
            if coa.coa_id != recommended.coa_id:
                score_diff = recommended.overall_score - coa.overall_score
                rationale.append(
                    f"{recommended.coa_name} scores {score_diff:.2f} higher than {coa.coa_name}"
                )
        
        # Generate decision brief
        decision_brief = self._generate_decision_brief(coas, recommended, rationale)
        
        comparison = COAComparison(
            comparison_id=f"comp_{int(time.time())}",
            coas=coas,
            recommended_coa=recommended.coa_id,
            recommendation_rationale=rationale,
            comparison_matrix=comparison_matrix,
            decision_brief=decision_brief
        )
        
        self.comparison_history.append(comparison)
        
        log.info(f"Recommended COA: {recommended.coa_name} (score: {recommended.overall_score:.2f})")
        
        return comparison
    
    def _generate_decision_brief(
        self,
        coas: List[CourseOfAction],
        recommended: CourseOfAction,
        rationale: List[str]
    ) -> str:
        """Generate decision brief for commander"""
        
        brief = f"""
COURSE OF ACTION DECISION BRIEF

SITUATION: {len(coas)} courses of action developed and analyzed.

RECOMMENDED COA: {recommended.coa_name}

RATIONALE:
{chr(10).join(f"  • {r}" for r in rationale)}

ADVANTAGES:
{chr(10).join(f"  • {a}" for a in recommended.advantages[:3])}

RISKS:
{chr(10).join(f"  • {r}" for r in recommended.risks[:3])}

EXECUTION:
  • Duration: {recommended.estimated_duration / 3600:.1f} hours
  • Phases: {len(recommended.phases)}
  • Probability of Success: {recommended.probability_of_success:.0%}

ALTERNATIVE OPTIONS:
{chr(10).join(f"  • {coa.coa_name}: Overall score {coa.overall_score:.2f}" for coa in coas if coa.coa_id != recommended.coa_id)}

RECOMMENDATION: Execute {recommended.coa_name}
"""
        
        return brief.strip()


# Global instance
coa_generator = COAGenerator()


async def generate_courses_of_action(
    situation: Dict[str, Any],
    objective: str,
    constraints: Dict[str, Any] = None,
    num_coas: int = 4
) -> COAComparison:
    """
    Main entry point: Generate and compare courses of action.
    Returns comparison with recommended COA.
    """
    
    # Generate COAs
    coas = await coa_generator.generate_coas(
        situation=situation,
        objective=objective,
        constraints=constraints,
        num_coas=num_coas
    )
    
    # Compare and recommend
    comparison = await coa_generator.compare_coas(coas)
    
    return comparison

