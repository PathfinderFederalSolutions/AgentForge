"""
Cascading Effect Analyzer
Predicts second and third-order effects of events
Models infrastructure dependencies and cascade failures
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger("cascading-effect-analyzer")

class EffectCategory(Enum):
    """Categories of effects"""
    IMMEDIATE = "immediate"        # 0-1 hour
    SHORT_TERM = "short_term"      # 1-24 hours
    MEDIUM_TERM = "medium_term"    # 1-7 days
    LONG_TERM = "long_term"        # >7 days

class ImpactSeverity(Enum):
    """Severity of impact"""
    CRITICAL = "critical"      # Mission-critical failure
    HIGH = "high"             # Significant degradation
    MEDIUM = "medium"         # Moderate impact
    LOW = "low"              # Minor impact

class SystemType(Enum):
    """Types of systems that can be affected"""
    COMMUNICATIONS = "communications"
    POWER_GRID = "power_grid"
    TRANSPORTATION = "transportation"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    WATER_SUPPLY = "water_supply"
    INTERNET = "internet"
    SUPPLY_CHAIN = "supply_chain"
    MILITARY_C2 = "military_c2"
    CIVILIAN_SERVICES = "civilian_services"

@dataclass
class SystemDependency:
    """Dependency relationship between systems"""
    dependent_system: SystemType
    required_system: SystemType
    criticality: float  # 0-1, how critical is this dependency
    degradation_threshold: float  # 0-1, at what performance level does dependent fail

@dataclass
class CascadeEffect:
    """A single cascading effect"""
    effect_id: str
    caused_by: str  # event or effect that caused this
    affected_system: SystemType
    effect_category: EffectCategory
    impact_severity: ImpactSeverity
    description: str
    probability: float
    estimated_start_time: float
    estimated_duration: float  # seconds
    affected_population: Optional[int] = None
    economic_impact: Optional[float] = None  # dollars
    mitigation_options: List[str] = field(default_factory=list)

@dataclass
class CascadeAnalysis:
    """Complete cascade analysis"""
    analysis_id: str
    triggering_event: str
    total_effects: int
    cascade_depth: int  # how many levels of effects
    timeline: List[CascadeEffect]
    critical_effects: List[CascadeEffect]
    total_economic_impact: float
    total_affected_population: int
    confidence: float
    analysis_timestamp: float

class CascadingEffectAnalyzer:
    """
    Analyzes cascading effects of events on interconnected systems.
    Predicts second, third, and nth-order consequences.
    """
    
    def __init__(self):
        self.system_dependencies: List[SystemDependency] = []
        self.system_baselines: Dict[SystemType, float] = {}
        self.cascade_history: List[CascadeAnalysis] = []
        
        # Performance impact models
        self.impact_models: Dict[str, callable] = {}
        
        self._initialize_dependencies()
        self._initialize_baselines()
        
        log.info("Cascading Effect Analyzer initialized")
    
    def _initialize_dependencies(self):
        """Initialize known system dependencies"""
        
        # Communications depend on power
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.COMMUNICATIONS,
            required_system=SystemType.POWER_GRID,
            criticality=0.9,
            degradation_threshold=0.3
        ))
        
        # Internet depends on power and communications
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.INTERNET,
            required_system=SystemType.POWER_GRID,
            criticality=0.95,
            degradation_threshold=0.2
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.INTERNET,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.85,
            degradation_threshold=0.4
        ))
        
        # Financial systems depend on power, internet
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.FINANCIAL,
            required_system=SystemType.POWER_GRID,
            criticality=0.95,
            degradation_threshold=0.1
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.FINANCIAL,
            required_system=SystemType.INTERNET,
            criticality=0.9,
            degradation_threshold=0.2
        ))
        
        # Healthcare depends on power, communications, supply chain
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.HEALTHCARE,
            required_system=SystemType.POWER_GRID,
            criticality=1.0,
            degradation_threshold=0.1
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.HEALTHCARE,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.8,
            degradation_threshold=0.3
        ))
        
        # Military C2 depends on communications, power
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.MILITARY_C2,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.95,
            degradation_threshold=0.2
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.MILITARY_C2,
            required_system=SystemType.POWER_GRID,
            criticality=0.9,
            degradation_threshold=0.3
        ))
        
        # Transportation depends on communications, power
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.TRANSPORTATION,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.7,
            degradation_threshold=0.4
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.TRANSPORTATION,
            required_system=SystemType.POWER_GRID,
            criticality=0.75,
            degradation_threshold=0.3
        ))
        
        # Supply chain depends on transportation, communications
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.SUPPLY_CHAIN,
            required_system=SystemType.TRANSPORTATION,
            criticality=0.85,
            degradation_threshold=0.3
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.SUPPLY_CHAIN,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.75,
            degradation_threshold=0.4
        ))
        
        # Civilian services depend on multiple systems
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.CIVILIAN_SERVICES,
            required_system=SystemType.POWER_GRID,
            criticality=0.8,
            degradation_threshold=0.3
        ))
        
        self.system_dependencies.append(SystemDependency(
            dependent_system=SystemType.CIVILIAN_SERVICES,
            required_system=SystemType.COMMUNICATIONS,
            criticality=0.7,
            degradation_threshold=0.4
        ))
        
        log.info(f"Initialized {len(self.system_dependencies)} system dependencies")
    
    def _initialize_baselines(self):
        """Initialize baseline performance levels"""
        
        for system_type in SystemType:
            self.system_baselines[system_type] = 1.0  # 100% normal operation
    
    async def analyze_cascading_effects(
        self,
        triggering_event: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> CascadeAnalysis:
        """
        Analyze cascading effects of a triggering event.
        Returns complete cascade analysis with timeline and impacts.
        """
        
        event_id = triggering_event.get("id", f"event_{int(time.time())}")
        event_desc = triggering_event.get("description", "Unknown event")
        
        log.info(f"Analyzing cascading effects for: {event_desc}")
        
        # Determine immediate effects
        immediate_effects = await self._determine_immediate_effects(triggering_event)
        
        # Calculate cascade effects (up to 5 levels deep)
        all_effects = immediate_effects.copy()
        current_level_effects = immediate_effects
        cascade_depth = 1
        
        for level in range(2, 6):  # Levels 2-5
            next_level_effects = await self._calculate_next_level_effects(
                current_level_effects, all_effects
            )
            
            if not next_level_effects:
                break
            
            all_effects.extend(next_level_effects)
            current_level_effects = next_level_effects
            cascade_depth = level
        
        # Sort effects by time
        timeline = sorted(all_effects, key=lambda e: e.estimated_start_time)
        
        # Identify critical effects
        critical_effects = [
            e for e in all_effects
            if e.impact_severity in [ImpactSeverity.CRITICAL, ImpactSeverity.HIGH]
        ]
        
        # Calculate total impacts
        total_economic = sum(
            e.economic_impact for e in all_effects if e.economic_impact
        )
        
        total_population = max(
            (e.affected_population for e in all_effects if e.affected_population),
            default=0
        )
        
        # Calculate confidence
        avg_probability = sum(e.probability for e in all_effects) / len(all_effects) if all_effects else 0.5
        confidence = avg_probability * 0.85  # Slight discount for cascade uncertainty
        
        analysis = CascadeAnalysis(
            analysis_id=f"cascade_{event_id}_{int(time.time())}",
            triggering_event=event_desc,
            total_effects=len(all_effects),
            cascade_depth=cascade_depth,
            timeline=timeline,
            critical_effects=critical_effects,
            total_economic_impact=total_economic,
            total_affected_population=total_population,
            confidence=confidence,
            analysis_timestamp=time.time()
        )
        
        self.cascade_history.append(analysis)
        
        log.info(f"Cascade analysis complete: {len(all_effects)} effects identified, "
                f"depth={cascade_depth}, {len(critical_effects)} critical")
        
        return analysis
    
    async def _determine_immediate_effects(
        self,
        triggering_event: Dict[str, Any]
    ) -> List[CascadeEffect]:
        """Determine immediate (first-order) effects of event"""
        
        effects = []
        event_type = triggering_event.get("type", "").lower()
        event_target = triggering_event.get("target", "").lower()
        
        # Cable/communications infrastructure damage
        if any(word in event_type or word in event_target 
              for word in ["cable", "undersea", "communications", "fiber"]):
            
            # Immediate internet capacity loss
            effects.append(CascadeEffect(
                effect_id=f"effect_1_{int(time.time() * 1000)}",
                caused_by="triggering_event",
                affected_system=SystemType.INTERNET,
                effect_category=EffectCategory.IMMEDIATE,
                impact_severity=ImpactSeverity.CRITICAL,
                description="40% internet capacity loss in affected region",
                probability=0.95,
                estimated_start_time=time.time(),
                estimated_duration=14 * 24 * 3600,  # 2 weeks
                affected_population=5_000_000,
                economic_impact=12_000_000,  # $12M/hour
                mitigation_options=[
                    "Activate backup satellite links",
                    "Reroute traffic through alternate cables",
                    "Deploy emergency repair ships"
                ]
            ))
            
            # Communications degradation
            effects.append(CascadeEffect(
                effect_id=f"effect_2_{int(time.time() * 1000)}",
                caused_by="triggering_event",
                affected_system=SystemType.COMMUNICATIONS,
                effect_category=EffectCategory.IMMEDIATE,
                impact_severity=ImpactSeverity.HIGH,
                description="25% communications degradation due to network congestion",
                probability=0.9,
                estimated_start_time=time.time() + 300,  # 5 minutes
                estimated_duration=7 * 24 * 3600,  # 1 week
                affected_population=5_000_000,
                economic_impact=5_000_000,
                mitigation_options=[
                    "Priority routing for critical services",
                    "Bandwidth throttling of non-essential services"
                ]
            ))
        
        # Power grid attack
        elif any(word in event_type or word in event_target 
                for word in ["power", "grid", "substation", "transformer"]):
            
            effects.append(CascadeEffect(
                effect_id=f"effect_1_{int(time.time() * 1000)}",
                caused_by="triggering_event",
                affected_system=SystemType.POWER_GRID,
                effect_category=EffectCategory.IMMEDIATE,
                impact_severity=ImpactSeverity.CRITICAL,
                description="Regional power outage affecting 60% of grid capacity",
                probability=0.9,
                estimated_start_time=time.time(),
                estimated_duration=48 * 3600,  # 48 hours
                affected_population=2_000_000,
                economic_impact=50_000_000,
                mitigation_options=[
                    "Load shedding to preserve critical services",
                    "Deploy mobile generators",
                    "Emergency grid reconfiguration"
                ]
            ))
        
        # Cyber attack
        elif "cyber" in event_type or "network" in event_type:
            
            effects.append(CascadeEffect(
                effect_id=f"effect_1_{int(time.time() * 1000)}",
                caused_by="triggering_event",
                affected_system=SystemType.INTERNET,
                effect_category=EffectCategory.IMMEDIATE,
                impact_severity=ImpactSeverity.HIGH,
                description="Network services degraded by 50% due to cyber attack",
                probability=0.85,
                estimated_start_time=time.time(),
                estimated_duration=24 * 3600,  # 24 hours
                affected_population=1_000_000,
                economic_impact=10_000_000,
                mitigation_options=[
                    "Isolate affected networks",
                    "Deploy countermeasures",
                    "Activate backup systems"
                ]
            ))
        
        return effects
    
    async def _calculate_next_level_effects(
        self,
        current_effects: List[CascadeEffect],
        all_existing_effects: List[CascadeEffect]
    ) -> List[CascadeEffect]:
        """Calculate next level of cascading effects"""
        
        next_effects = []
        already_affected_systems = set(e.affected_system for e in all_existing_effects)
        
        for current_effect in current_effects:
            # Find systems that depend on the affected system
            dependent_systems = [
                dep for dep in self.system_dependencies
                if dep.required_system == current_effect.affected_system
                and dep.dependent_system not in already_affected_systems
            ]
            
            for dependency in dependent_systems:
                # Calculate if degradation threshold is exceeded
                # Estimate performance degradation from current effect
                performance_loss = self._estimate_performance_loss(current_effect)
                
                if performance_loss >= dependency.degradation_threshold:
                    # This system will cascade fail
                    cascade_effect = self._create_cascade_effect(
                        current_effect, dependency, performance_loss
                    )
                    next_effects.append(cascade_effect)
        
        return next_effects
    
    def _estimate_performance_loss(self, effect: CascadeEffect) -> float:
        """Estimate performance loss from an effect"""
        
        severity_map = {
            ImpactSeverity.CRITICAL: 0.8,
            ImpactSeverity.HIGH: 0.6,
            ImpactSeverity.MEDIUM: 0.4,
            ImpactSeverity.LOW: 0.2
        }
        
        return severity_map[effect.impact_severity]
    
    def _create_cascade_effect(
        self,
        causing_effect: CascadeEffect,
        dependency: SystemDependency,
        performance_loss: float
    ) -> CascadeEffect:
        """Create a cascading effect"""
        
        # Determine severity based on criticality and performance loss
        if dependency.criticality > 0.9 and performance_loss > 0.7:
            severity = ImpactSeverity.CRITICAL
        elif dependency.criticality > 0.7 or performance_loss > 0.6:
            severity = ImpactSeverity.HIGH
        elif dependency.criticality > 0.5 or performance_loss > 0.4:
            severity = ImpactSeverity.MEDIUM
        else:
            severity = ImpactSeverity.LOW
        
        # Determine timing (cascades take time to propagate)
        if causing_effect.effect_category == EffectCategory.IMMEDIATE:
            category = EffectCategory.SHORT_TERM
            start_delay = 3600  # 1 hour
        elif causing_effect.effect_category == EffectCategory.SHORT_TERM:
            category = EffectCategory.MEDIUM_TERM
            start_delay = 24 * 3600  # 1 day
        else:
            category = EffectCategory.LONG_TERM
            start_delay = 7 * 24 * 3600  # 1 week
        
        # Estimate impacts (scaled by criticality)
        affected_pop = int((causing_effect.affected_population or 0) * dependency.criticality)
        economic = (causing_effect.economic_impact or 0) * dependency.criticality * 0.7
        
        # Generate description
        description = (f"{dependency.dependent_system.value} degraded by {int(performance_loss * 100)}% "
                      f"due to {causing_effect.affected_system.value} failure")
        
        # Mitigation options
        mitigations = [
            f"Activate backup systems for {dependency.dependent_system.value}",
            f"Reduce {dependency.dependent_system.value} load to essential services only",
            f"Deploy redundant {dependency.required_system.value} capacity"
        ]
        
        return CascadeEffect(
            effect_id=f"effect_{dependency.dependent_system.value}_{int(time.time() * 1000)}",
            caused_by=causing_effect.effect_id,
            affected_system=dependency.dependent_system,
            effect_category=category,
            impact_severity=severity,
            description=description,
            probability=causing_effect.probability * dependency.criticality,
            estimated_start_time=causing_effect.estimated_start_time + start_delay,
            estimated_duration=causing_effect.estimated_duration * 0.8,
            affected_population=affected_pop if affected_pop > 0 else None,
            economic_impact=economic if economic > 0 else None,
            mitigation_options=mitigations
        )


# Global instance
cascade_analyzer = CascadingEffectAnalyzer()


async def analyze_cascade_effects(
    triggering_event: Dict[str, Any],
    context: Dict[str, Any] = None
) -> CascadeAnalysis:
    """
    Main entry point: Analyze cascading effects of an event.
    Returns complete cascade analysis with timeline and impacts.
    """
    return await cascade_analyzer.analyze_cascading_effects(triggering_event, context)

