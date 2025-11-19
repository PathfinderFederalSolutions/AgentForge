"""
Easy Interface for AgentForge Intelligence
User-friendly wrapper for all capabilities
Makes everything accessible with minimal code
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .master_intelligence_orchestrator import process_intelligence, IntelligenceResponse

log = logging.getLogger("easy-interface")

@dataclass
class SimpleRequest:
    """Simplified request interface"""
    query: str                                    # What you want to know/do
    data: List[Dict[str, Any]] = None            # Optional data sources
    include_planning: bool = False                # Add goal decomposition
    include_coas: bool = False                    # Add COA generation
    include_wargaming: bool = False               # Add wargaming simulation
    objective: Optional[str] = None               # What you want to achieve
    constraints: Optional[Dict[str, Any]] = None  # Any limitations

@dataclass
class SimpleResponse:
    """Simplified response interface"""
    # Summary
    summary: str
    confidence: float
    threat_level: str
    
    # Key information
    findings: List[str]
    threats_detected: List[str]
    recommendations: List[str]
    
    # Planning (if requested)
    execution_plan: Optional[str] = None
    recommended_coa: Optional[str] = None
    wargame_outcome: Optional[str] = None
    decision_brief: Optional[str] = None
    
    # Full details (for power users)
    full_response: Optional[IntelligenceResponse] = None

class EasyIntelligence:
    """
    Easy-to-use interface for all AgentForge intelligence capabilities.
    Hides complexity, provides simple interface for users.
    """
    
    async def analyze(
        self,
        query: str,
        data: List[Dict[str, Any]] = None,
        **kwargs
    ) -> SimpleResponse:
        """
        Analyze anything with one simple call.
        
        Examples:
            # Simple intelligence
            result = await easy.analyze("What's this submarine doing?", data=[...])
            
            # With planning
            result = await easy.analyze(
                "How do I respond to this threat?",
                data=[...],
                include_planning=True
            )
            
            # Complete package
            result = await easy.analyze(
                "What should I do?",
                data=[...],
                include_planning=True,
                include_coas=True,
                include_wargaming=True,
                objective="Neutralize threat"
            )
        """
        
        # Build context
        context = {
            "dataSources": data or [],
            "include_planning": kwargs.get("include_planning", False),
            "generate_coas": kwargs.get("include_coas", False),
            "run_wargaming": kwargs.get("include_wargaming", False),
            "objective": kwargs.get("objective", query),
            "constraints": kwargs.get("constraints", {}),
            "num_coas": kwargs.get("num_coas", 4),
            "red_force_strategy": kwargs.get("red_force_strategy", "defensive")
        }
        
        # Process with full intelligence system
        full_response = await process_intelligence(
            task_description=query,
            available_data=data or [],
            context=context
        )
        
        # Build simple response
        simple = self._simplify_response(full_response)
        simple.full_response = full_response  # Include full details
        
        return simple
    
    def _simplify_response(self, response: IntelligenceResponse) -> SimpleResponse:
        """Convert complex response to simple format"""
        
        # Extract threat level from assessment
        threat_level = "UNKNOWN"
        if "CRITICAL" in response.threat_assessment.upper():
            threat_level = "CRITICAL"
        elif "HIGH" in response.threat_assessment.upper():
            threat_level = "HIGH"
        elif "ELEVATED" in response.threat_assessment.upper():
            threat_level = "ELEVATED"
        elif "MODERATE" in response.threat_assessment.upper():
            threat_level = "MODERATE"
        elif "LOW" in response.threat_assessment.upper():
            threat_level = "LOW"
        
        # Extract threats
        threats = [
            ttp.pattern.name for ttp in response.ttp_detections
        ]
        
        # Build execution plan summary
        execution_plan_summary = None
        if response.execution_plan:
            execution_plan_summary = (
                f"{len(response.execution_plan.tasks)} tasks, "
                f"{response.execution_plan.estimated_total_time/3600:.1f} hours, "
                f"{response.execution_plan.confidence:.0%} confidence"
            )
        
        # Build COA summary
        recommended_coa_summary = None
        if response.coa_comparison:
            best_coa = response.coa_comparison.coas[0]
            recommended_coa_summary = (
                f"{best_coa.coa_name}: "
                f"{best_coa.coa_type.value}, "
                f"{best_coa.probability_of_success:.0%} success probability"
            )
        
        # Build wargame summary
        wargame_summary = None
        if response.wargame_results:
            best = response.wargame_results.coa_results[0]
            wargame_summary = (
                f"{best.outcome.value.replace('_', ' ').title()}, "
                f"{best.outcome_probability:.0%} probability, "
                f"{best.blue_force_casualties:.0%} casualties"
            )
        
        # Get decision brief if available
        decision_brief = None
        if response.coa_comparison:
            decision_brief = response.coa_comparison.decision_brief
        elif response.wargame_results:
            decision_brief = response.wargame_results.recommendation
        
        return SimpleResponse(
            summary=response.executive_summary,
            confidence=response.overall_confidence,
            threat_level=threat_level,
            findings=response.key_findings,
            threats_detected=threats,
            recommendations=response.recommended_actions,
            execution_plan=execution_plan_summary,
            recommended_coa=recommended_coa_summary,
            wargame_outcome=wargame_summary,
            decision_brief=decision_brief
        )

    async def quick_threat_check(self, data: List[Dict[str, Any]]) -> str:
        """
        Quick threat check - simplest possible interface.
        
        Example:
            threat = await easy.quick_threat_check([
                {"type": "acoustic", "content": {"submarine": true}}
            ])
            print(threat)  # "HIGH threat: Submarine Infiltration Operation detected"
        """
        
        response = await self.analyze("Check for threats", data=data)
        
        if response.threats_detected:
            return f"{response.threat_level} threat: {', '.join(response.threats_detected)}"
        else:
            return "No threats detected"
    
    async def quick_plan(self, goal: str) -> str:
        """
        Quick planning - simplest planning interface.
        
        Example:
            plan = await easy.quick_plan("Neutralize submarine")
            print(plan)  # "7 tasks, 65 minutes estimated"
        """
        
        response = await self.analyze(
            goal,
            include_planning=True
        )
        
        if response.execution_plan:
            return response.execution_plan
        else:
            return "Planning not generated"
    
    async def quick_decision(
        self,
        situation: str,
        objective: str,
        data: List[Dict[str, Any]] = None
    ) -> str:
        """
        Quick decision support - get recommendation immediately.
        
        Example:
            decision = await easy.quick_decision(
                "Submarine detected threatening cables",
                "Protect infrastructure",
                data=[...]
            )
            print(decision)  # Full decision brief
        """
        
        response = await self.analyze(
            situation,
            data=data,
            include_coas=True,
            include_wargaming=True,
            objective=objective
        )
        
        if response.decision_brief:
            return response.decision_brief
        else:
            return f"Recommendation: {response.recommendations[0] if response.recommendations else 'Analyze further'}"


# Global instance
easy = EasyIntelligence()


# Ultra-simple functions for one-liners

async def analyze_threat(data: List[Dict[str, Any]]) -> str:
    """One-liner threat analysis"""
    return await easy.quick_threat_check(data)


async def make_plan(goal: str) -> str:
    """One-liner planning"""
    return await easy.quick_plan(goal)


async def get_decision(situation: str, objective: str, data: List[Dict[str, Any]] = None) -> str:
    """One-liner decision support"""
    return await easy.quick_decision(situation, objective, data)


async def analyze(query: str, **kwargs) -> SimpleResponse:
    """One-liner full analysis"""
    return await easy.analyze(query, **kwargs)


# Export easy interface
__all__ = [
    "easy",
    "EasyIntelligence",
    "SimpleRequest",
    "SimpleResponse",
    "analyze_threat",
    "make_plan",
    "get_decision",
    "analyze"
]

