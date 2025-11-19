"""
Wargaming Simulation Engine
Simulates COA execution with red team/blue team modeling
Predicts outcomes, identifies vulnerabilities, refines plans
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .coa_generation import CourseOfAction, COAPhase, EffectType

log = logging.getLogger("wargaming-simulation")

class ForceType(Enum):
    """Force types in wargaming"""
    BLUE_FORCE = "blue_force"      # Friendly forces
    RED_FORCE = "red_force"        # Adversary forces
    GREEN_FORCE = "green_force"    # Neutral/civilian
    WHITE_CELL = "white_cell"      # Control/simulation

class OutcomeType(Enum):
    """Possible wargame outcomes"""
    DECISIVE_VICTORY = "decisive_victory"
    MARGINAL_VICTORY = "marginal_victory"
    STALEMATE = "stalemate"
    MARGINAL_DEFEAT = "marginal_defeat"
    DECISIVE_DEFEAT = "decisive_defeat"

class CriticalEvent(Enum):
    """Critical events during simulation"""
    ENEMY_REINFORCEMENT = "enemy_reinforcement"
    FRIENDLY_CASUALTY = "friendly_casualty"
    OBJECTIVE_SEIZED = "objective_seized"
    COMMUNICATION_LOSS = "communication_loss"
    SUPPLY_DISRUPTION = "supply_disruption"
    WEATHER_CHANGE = "weather_change"
    CIVILIAN_INCIDENT = "civilian_incident"

@dataclass
class Force:
    """Representation of a force in wargaming"""
    force_id: str
    force_type: ForceType
    name: str
    combat_power: float  # 0-1
    morale: float  # 0-1
    supply_status: float  # 0-1
    position: Dict[str, Any]
    capabilities: List[str]
    losses: float = 0.0

@dataclass
class WargameEvent:
    """Event during wargaming"""
    event_id: str
    timestamp: float
    phase: str
    event_type: CriticalEvent
    description: str
    blue_force_impact: float  # -1 to 1
    red_force_impact: float  # -1 to 1
    mitigation: Optional[str] = None

@dataclass
class WargameResult:
    """Result of wargaming simulation"""
    simulation_id: str
    coa: CourseOfAction
    outcome: OutcomeType
    
    # Final states
    blue_force_state: Force
    red_force_state: Force
    
    # Timeline
    events: List[WargameEvent]
    phase_results: List[Dict[str, Any]]
    
    # Analysis
    objectives_achieved: List[str]
    objectives_failed: List[str]
    critical_factors: List[str]
    vulnerabilities_identified: List[str]
    recommendations: List[str]
    
    # Metrics
    final_blue_combat_power: float
    final_red_combat_power: float
    blue_force_casualties: float
    red_force_casualties: float
    duration_actual: float
    
    # Confidence
    simulation_confidence: float
    outcome_probability: float

@dataclass
class WargameComparison:
    """Comparison of wargaming results for multiple COAs"""
    comparison_id: str
    coa_results: List[WargameResult]
    best_coa_id: str
    worst_coa_id: str
    recommendation: str
    comparison_analysis: Dict[str, Any]

class WargamingSimulator:
    """
    Simulates COA execution through wargaming.
    Red team vs. blue team modeling with outcome prediction.
    """
    
    def __init__(self):
        self.simulation_history: List[WargameResult] = []
        self.event_probability_models: Dict[str, float] = {}
        
        # Red force models (adversary behavior)
        self.red_force_tactics: Dict[str, List[str]] = {}
        
        self._initialize_models()
        
        log.info("Wargaming Simulator initialized")
    
    def _initialize_models(self):
        """Initialize simulation models"""
        
        # Event probabilities
        self.event_probability_models = {
            CriticalEvent.ENEMY_REINFORCEMENT.value: 0.3,
            CriticalEvent.FRIENDLY_CASUALTY.value: 0.4,
            CriticalEvent.COMMUNICATION_LOSS.value: 0.2,
            CriticalEvent.SUPPLY_DISRUPTION.value: 0.25,
            CriticalEvent.WEATHER_CHANGE.value: 0.15,
            CriticalEvent.CIVILIAN_INCIDENT.value: 0.1
        }
        
        # Red force tactics
        self.red_force_tactics["defensive"] = [
            "prepared_positions",
            "obstacles_and_minefields",
            "fire_support_planning",
            "counterattack_reserves"
        ]
        
        self.red_force_tactics["offensive"] = [
            "reconnaissance",
            "artillery_preparation",
            "combined_arms_assault",
            "exploitation"
        ]
    
    async def simulate_coa(
        self,
        coa: CourseOfAction,
        situation: Dict[str, Any],
        red_force_strategy: str = "defensive"
    ) -> WargameResult:
        """
        Simulate execution of a COA through wargaming.
        Returns detailed simulation results.
        """
        
        simulation_id = f"sim_{coa.coa_id}_{int(time.time())}"
        
        log.info(f"Starting wargame simulation: {coa.coa_name}")
        log.info(f"Red force strategy: {red_force_strategy}")
        
        # Initialize forces
        blue_force = Force(
            force_id="blue_main",
            force_type=ForceType.BLUE_FORCE,
            name="Friendly Forces",
            combat_power=1.0,
            morale=0.9,
            supply_status=1.0,
            position={"status": "assembly_area"},
            capabilities=[elem.capabilities for elem in coa.elements][0] if coa.elements else []
        )
        
        red_force = Force(
            force_id="red_main",
            force_type=ForceType.RED_FORCE,
            name="Adversary Forces",
            combat_power=0.8,  # Assume slightly weaker adversary
            morale=0.75,
            supply_status=0.9,
            position={"status": "defensive_positions"},
            capabilities=self.red_force_tactics.get(red_force_strategy, [])
        )
        
        # Simulate each phase
        events = []
        phase_results = []
        objectives_achieved = []
        objectives_failed = []
        
        for phase in coa.phases:
            phase_result = await self._simulate_phase(
                phase, blue_force, red_force, coa, situation
            )
            
            phase_results.append(phase_result)
            events.extend(phase_result["events"])
            objectives_achieved.extend(phase_result["objectives_achieved"])
            objectives_failed.extend(phase_result["objectives_failed"])
            
            # Update force states
            blue_force.combat_power *= phase_result["blue_power_modifier"]
            red_force.combat_power *= phase_result["red_power_modifier"]
            blue_force.morale *= phase_result["blue_morale_modifier"]
            
            # Check if simulation should end early
            if blue_force.combat_power < 0.3 or red_force.combat_power < 0.1:
                log.info("Simulation terminated early due to force degradation")
                break
        
        # Determine outcome
        outcome = self._determine_outcome(
            blue_force, red_force, objectives_achieved, objectives_failed
        )
        
        # Identify critical factors
        critical_factors = self._identify_critical_factors(events, phase_results)
        
        # Identify vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(events, phase_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            outcome, vulnerabilities, phase_results
        )
        
        # Calculate metrics
        blue_casualties = 1.0 - blue_force.combat_power
        red_casualties = 1.0 - red_force.combat_power
        duration_actual = sum(p.duration_estimate for p in coa.phases)
        
        result = WargameResult(
            simulation_id=simulation_id,
            coa=coa,
            outcome=outcome,
            blue_force_state=blue_force,
            red_force_state=red_force,
            events=events,
            phase_results=phase_results,
            objectives_achieved=objectives_achieved,
            objectives_failed=objectives_failed,
            critical_factors=critical_factors,
            vulnerabilities_identified=vulnerabilities,
            recommendations=recommendations,
            final_blue_combat_power=blue_force.combat_power,
            final_red_combat_power=red_force.combat_power,
            blue_force_casualties=blue_casualties,
            red_force_casualties=red_casualties,
            duration_actual=duration_actual,
            simulation_confidence=0.75,
            outcome_probability=self._calculate_outcome_probability(outcome)
        )
        
        self.simulation_history.append(result)
        
        log.info(f"Wargame complete: {outcome.value}, "
                f"Blue power: {blue_force.combat_power:.2f}, "
                f"Red power: {red_force.combat_power:.2f}")
        
        return result
    
    async def _simulate_phase(
        self,
        phase: COAPhase,
        blue_force: Force,
        red_force: Force,
        coa: CourseOfAction,
        situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate a single phase of COA"""
        
        log.debug(f"Simulating phase: {phase.phase_name}")
        
        events = []
        objectives_achieved = []
        objectives_failed = []
        
        # Base modifiers
        blue_power_modifier = 0.95  # Slight attrition
        red_power_modifier = 0.90   # More attrition for defender/loser
        blue_morale_modifier = 1.0
        
        # Simulate phase execution
        await asyncio.sleep(0.1)  # Simulate computation time
        
        # Check for critical events
        for event_type, probability in self.event_probability_models.items():
            if random.random() < probability:
                # Event occurs
                event = self._create_event(
                    phase.phase_name,
                    CriticalEvent(event_type),
                    blue_force,
                    red_force
                )
                events.append(event)
                
                # Apply event impacts
                blue_power_modifier *= (1 + event.blue_force_impact * 0.1)
                red_power_modifier *= (1 + event.red_force_impact * 0.1)
        
        # Evaluate phase objectives
        for objective in phase.objectives:
            # Success probability based on force ratio
            success_prob = blue_force.combat_power / (blue_force.combat_power + red_force.combat_power)
            success_prob *= blue_force.morale
            
            if random.random() < success_prob:
                objectives_achieved.append(objective)
                blue_morale_modifier = 1.05  # Success boosts morale
            else:
                objectives_failed.append(objective)
                blue_morale_modifier = 0.95  # Failure reduces morale
        
        return {
            "phase_name": phase.phase_name,
            "blue_power_modifier": blue_power_modifier,
            "red_power_modifier": red_power_modifier,
            "blue_morale_modifier": blue_morale_modifier,
            "events": events,
            "objectives_achieved": objectives_achieved,
            "objectives_failed": objectives_failed,
            "duration": phase.duration_estimate
        }
    
    def _create_event(
        self,
        phase_name: str,
        event_type: CriticalEvent,
        blue_force: Force,
        red_force: Force
    ) -> WargameEvent:
        """Create a wargame event"""
        
        # Event impacts
        impact_map = {
            CriticalEvent.ENEMY_REINFORCEMENT: (-0.2, 0.3),  # Bad for blue, good for red
            CriticalEvent.FRIENDLY_CASUALTY: (-0.15, 0.05),
            CriticalEvent.COMMUNICATION_LOSS: (-0.1, 0.0),
            CriticalEvent.SUPPLY_DISRUPTION: (-0.2, 0.0),
            CriticalEvent.WEATHER_CHANGE: (-0.05, -0.05),
            CriticalEvent.CIVILIAN_INCIDENT: (-0.1, -0.05)
        }
        
        blue_impact, red_impact = impact_map.get(event_type, (0, 0))
        
        # Event descriptions
        descriptions = {
            CriticalEvent.ENEMY_REINFORCEMENT: "Enemy reinforcements arrive",
            CriticalEvent.FRIENDLY_CASUALTY: "Friendly force takes casualties",
            CriticalEvent.COMMUNICATION_LOSS: "Communications disrupted",
            CriticalEvent.SUPPLY_DISRUPTION: "Supply lines interrupted",
            CriticalEvent.WEATHER_CHANGE: "Weather deteriorates",
            CriticalEvent.CIVILIAN_INCIDENT: "Civilian casualties reported"
        }
        
        return WargameEvent(
            event_id=f"event_{int(time.time() * 1000)}",
            timestamp=time.time(),
            phase=phase_name,
            event_type=event_type,
            description=descriptions[event_type],
            blue_force_impact=blue_impact,
            red_force_impact=red_impact,
            mitigation=f"Adjust plan to account for {event_type.value}"
        )
    
    def _determine_outcome(
        self,
        blue_force: Force,
        red_force: Force,
        objectives_achieved: List[str],
        objectives_failed: List[str]
    ) -> OutcomeType:
        """Determine overall wargame outcome"""
        
        # Calculate success ratio
        total_objectives = len(objectives_achieved) + len(objectives_failed)
        success_ratio = len(objectives_achieved) / max(total_objectives, 1)
        
        # Calculate force ratio
        force_ratio = blue_force.combat_power / max(red_force.combat_power, 0.1)
        
        # Weighted outcome determination
        outcome_score = success_ratio * 0.6 + (force_ratio / 2) * 0.4
        
        if outcome_score > 0.8:
            return OutcomeType.DECISIVE_VICTORY
        elif outcome_score > 0.6:
            return OutcomeType.MARGINAL_VICTORY
        elif outcome_score > 0.4:
            return OutcomeType.STALEMATE
        elif outcome_score > 0.2:
            return OutcomeType.MARGINAL_DEFEAT
        else:
            return OutcomeType.DECISIVE_DEFEAT
    
    def _identify_critical_factors(
        self,
        events: List[WargameEvent],
        phase_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify factors that critically affected outcome"""
        
        factors = []
        
        # Significant events
        significant_events = [e for e in events if abs(e.blue_force_impact) > 0.15]
        if significant_events:
            factors.append(f"{len(significant_events)} critical events significantly impacted outcome")
        
        # Failed objectives
        total_failed = sum(len(p["objectives_failed"]) for p in phase_results)
        if total_failed > 0:
            factors.append(f"{total_failed} objectives failed during execution")
        
        # Specific event types
        reinforcement_events = [e for e in events if e.event_type == CriticalEvent.ENEMY_REINFORCEMENT]
        if reinforcement_events:
            factors.append("Enemy reinforcements arrived at critical moment")
        
        return factors
    
    def _identify_vulnerabilities(
        self,
        events: List[WargameEvent],
        phase_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify vulnerabilities exposed during simulation"""
        
        vulnerabilities = []
        
        # Communication vulnerabilities
        comm_loss = [e for e in events if e.event_type == CriticalEvent.COMMUNICATION_LOSS]
        if comm_loss:
            vulnerabilities.append("Communications vulnerable to disruption")
        
        # Supply vulnerabilities
        supply_disruption = [e for e in events if e.event_type == CriticalEvent.SUPPLY_DISRUPTION]
        if supply_disruption:
            vulnerabilities.append("Supply lines vulnerable to interdiction")
        
        # Phase-specific vulnerabilities
        for phase_result in phase_results:
            if phase_result["objectives_failed"]:
                vulnerabilities.append(
                    f"Vulnerability in {phase_result['phase_name']}: "
                    f"{', '.join(phase_result['objectives_failed'][:2])}"
                )
        
        return vulnerabilities
    
    def _generate_recommendations(
        self,
        outcome: OutcomeType,
        vulnerabilities: List[str],
        phase_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on simulation"""
        
        recommendations = []
        
        # Outcome-based recommendations
        if outcome in [OutcomeType.MARGINAL_DEFEAT, OutcomeType.DECISIVE_DEFEAT]:
            recommendations.append("Consider alternative COA - current plan high risk of failure")
            recommendations.append("Increase combat power or reduce scope")
        
        elif outcome == OutcomeType.STALEMATE:
            recommendations.append("Add decisive element to break stalemate")
            recommendations.append("Consider different approach or timing")
        
        # Vulnerability-based recommendations
        for vulnerability in vulnerabilities[:3]:
            if "communication" in vulnerability.lower():
                recommendations.append("Establish redundant communications")
            elif "supply" in vulnerability.lower():
                recommendations.append("Pre-position supplies, establish alternate routes")
        
        # Phase-based recommendations
        failed_phases = [p for p in phase_results if p["objectives_failed"]]
        if failed_phases:
            recommendations.append(
                f"Reinforce {failed_phases[0]['phase_name']} with additional resources"
            )
        
        return recommendations
    
    def _calculate_outcome_probability(self, outcome: OutcomeType) -> float:
        """Calculate probability of this outcome"""
        
        # Map outcomes to probabilities
        prob_map = {
            OutcomeType.DECISIVE_VICTORY: 0.9,
            OutcomeType.MARGINAL_VICTORY: 0.75,
            OutcomeType.STALEMATE: 0.6,
            OutcomeType.MARGINAL_DEFEAT: 0.4,
            OutcomeType.DECISIVE_DEFEAT: 0.2
        }
        
        return prob_map.get(outcome, 0.5)
    
    async def wargame_all_coas(
        self,
        coas: List[CourseOfAction],
        situation: Dict[str, Any],
        red_force_strategy: str = "defensive"
    ) -> WargameComparison:
        """
        Wargame all COAs and compare results.
        Returns comparison with best/worst COAs identified.
        """
        
        log.info(f"Wargaming {len(coas)} COAs")
        
        # Simulate each COA
        results = []
        for coa in coas:
            result = await self.simulate_coa(coa, situation, red_force_strategy)
            results.append(result)
        
        # Find best and worst
        best_result = max(results, key=lambda r: r.outcome_probability)
        worst_result = min(results, key=lambda r: r.outcome_probability)
        
        # Generate comparison analysis
        comparison_analysis = {
            "best_coa": {
                "name": best_result.coa.coa_name,
                "outcome": best_result.outcome.value,
                "probability": best_result.outcome_probability,
                "casualties": best_result.blue_force_casualties
            },
            "worst_coa": {
                "name": worst_result.coa.coa_name,
                "outcome": worst_result.outcome.value,
                "probability": worst_result.outcome_probability,
                "casualties": worst_result.blue_force_casualties
            },
            "outcome_distribution": {
                outcome.value: len([r for r in results if r.outcome == outcome])
                for outcome in OutcomeType
            }
        }
        
        # Generate recommendation
        recommendation = self._generate_wargame_recommendation(results, best_result)
        
        comparison = WargameComparison(
            comparison_id=f"wargame_comp_{int(time.time())}",
            coa_results=results,
            best_coa_id=best_result.coa.coa_id,
            worst_coa_id=worst_result.coa.coa_id,
            recommendation=recommendation,
            comparison_analysis=comparison_analysis
        )
        
        log.info(f"Wargaming complete. Best: {best_result.coa.coa_name}, "
                f"Worst: {worst_result.coa.coa_name}")
        
        return comparison
    
    def _generate_wargame_recommendation(
        self,
        results: List[WargameResult],
        best_result: WargameResult
    ) -> str:
        """Generate overall wargaming recommendation"""
        
        recommendation = f"""
WARGAMING RECOMMENDATION

Based on simulation of {len(results)} courses of action:

RECOMMENDED: {best_result.coa.coa_name}
  • Outcome: {best_result.outcome.value.replace('_', ' ').title()}
  • Success Probability: {best_result.outcome_probability:.0%}
  • Expected Casualties: {best_result.blue_force_casualties:.0%}
  • Critical Factors: {len(best_result.critical_factors)}
  • Vulnerabilities: {len(best_result.vulnerabilities_identified)}

KEY CONSIDERATIONS:
{chr(10).join(f"  • {rec}" for rec in best_result.recommendations[:3])}

VULNERABILITIES TO MITIGATE:
{chr(10).join(f"  • {vuln}" for vuln in best_result.vulnerabilities_identified[:3])}

DECISION: Recommend execution of {best_result.coa.coa_name} with mitigation of identified vulnerabilities.
"""
        
        return recommendation.strip()


# Global instance
wargaming_simulator = WargamingSimulator()


async def simulate_and_compare_coas(
    coas: List[CourseOfAction],
    situation: Dict[str, Any],
    red_force_strategy: str = "defensive"
) -> WargameComparison:
    """
    Main entry point: Simulate all COAs and compare results.
    Returns wargame comparison with recommendations.
    """
    return await wargaming_simulator.wargame_all_coas(coas, situation, red_force_strategy)

