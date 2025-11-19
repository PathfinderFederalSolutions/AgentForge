"""
Multi-Domain Intelligence Fusion System
Cross-source correlation, inject processing, and confidence-weighted fusion
Operates like human intelligence analyst with machine speed
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

log = logging.getLogger("multi-domain-fusion")

class IntelligenceDomain(Enum):
    """Intelligence domains"""
    SIGINT = "signals_intelligence"
    HUMINT = "human_intelligence"
    GEOINT = "geospatial_intelligence"
    MASINT = "measurement_signature_intelligence"
    OSINT = "open_source_intelligence"
    CYBINT = "cyber_intelligence"
    FININT = "financial_intelligence"
    TECHINT = "technical_intelligence"

class SourceCredibility(Enum):
    """Source credibility levels"""
    CONFIRMED = "confirmed"      # 1.0 - Multiple independent confirmations
    PROBABLY_TRUE = "probably_true"  # 0.8 - Single reliable source
    POSSIBLY_TRUE = "possibly_true"  # 0.6 - Plausible but unconfirmed
    DOUBTFUL = "doubtful"       # 0.4 - Contradicted by other sources
    IMPROBABLE = "improbable"   # 0.2 - Highly unlikely
    CANNOT_JUDGE = "cannot_judge"  # 0.5 - Insufficient information

@dataclass
class IntelligenceInject:
    """
    An intelligence inject - a single observation or report from a source.
    Multiple injects get correlated and fused into comprehensive assessments.
    """
    inject_id: str
    source_id: str
    source_name: str
    timestamp: float
    domain: IntelligenceDomain
    data_type: str
    content: Dict[str, Any]
    credibility: SourceCredibility
    confidence: float  # 0-1 scale
    classification: str = "UNCLASSIFIED"
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InjectCorrelation:
    """Correlation between multiple injects"""
    correlation_id: str
    correlated_injects: List[str]  # inject_ids
    correlation_type: str  # temporal, spatial, causal, semantic
    correlation_strength: float  # 0-1
    temporal_proximity: Optional[float] = None  # seconds
    spatial_proximity: Optional[float] = None   # meters
    semantic_similarity: Optional[float] = None
    reasoning: List[str] = field(default_factory=list)

@dataclass
class FusedIntelligence:
    """Result of fusing multiple injects"""
    fusion_id: str
    source_injects: List[str]  # inject_ids
    domains: Set[IntelligenceDomain]
    fused_assessment: Dict[str, Any]
    confidence: float
    credibility: SourceCredibility
    correlations: List[InjectCorrelation]
    created_at: float
    reasoning_chain: List[str]
    alternative_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

class MultiDomainIntelligenceFusion:
    """
    Fuses intelligence from multiple domains and sources.
    Performs temporal correlation, credibility weighting, and confidence scoring.
    """
    
    def __init__(self):
        self.active_injects: Dict[str, IntelligenceInject] = {}
        self.inject_history: List[IntelligenceInject] = []
        self.fusion_results: Dict[str, FusedIntelligence] = {}
        self.correlation_cache: Dict[str, InjectCorrelation] = {}
        
        # Temporal correlation window (seconds)
        self.temporal_window = 7200  # 2 hours
        
        # Spatial correlation threshold (meters)
        self.spatial_threshold = 50000  # 50km
        
        # Credibility weights
        self.credibility_weights = {
            SourceCredibility.CONFIRMED: 1.0,
            SourceCredibility.PROBABLY_TRUE: 0.8,
            SourceCredibility.POSSIBLY_TRUE: 0.6,
            SourceCredibility.DOUBTFUL: 0.4,
            SourceCredibility.IMPROBABLE: 0.2,
            SourceCredibility.CANNOT_JUDGE: 0.5
        }
        
        log.info("Multi-Domain Intelligence Fusion system initialized")
    
    async def process_inject(
        self,
        inject: IntelligenceInject
    ) -> Tuple[FusedIntelligence, List[InjectCorrelation]]:
        """
        Process a new intelligence inject.
        Correlates with existing injects and generates fused intelligence.
        """
        
        log.info(f"Processing inject {inject.inject_id} from {inject.source_name} "
                f"(domain: {inject.domain.value}, confidence: {inject.confidence:.2f})")
        
        # Store inject
        self.active_injects[inject.inject_id] = inject
        self.inject_history.append(inject)
        
        # Find correlations with existing injects
        correlations = await self._find_correlations(inject)
        
        # Determine which injects to fuse together
        fusion_group = self._determine_fusion_group(inject, correlations)
        
        # Perform fusion
        fused_intelligence = await self._fuse_injects(fusion_group, correlations)
        
        # Store result
        self.fusion_results[fused_intelligence.fusion_id] = fused_intelligence
        
        log.info(f"Fusion complete: {len(fusion_group)} injects, "
                f"confidence {fused_intelligence.confidence:.2f}, "
                f"{len(correlations)} correlations")
        
        return fused_intelligence, correlations
    
    async def _find_correlations(
        self,
        new_inject: IntelligenceInject
    ) -> List[InjectCorrelation]:
        """Find correlations between new inject and existing injects"""
        
        correlations = []
        
        # Get recent injects within temporal window
        current_time = new_inject.timestamp
        recent_injects = [
            inj for inj in self.inject_history
            if abs(inj.timestamp - current_time) <= self.temporal_window
            and inj.inject_id != new_inject.inject_id
        ]
        
        for existing_inject in recent_injects:
            # Check temporal correlation
            temporal_corr = self._check_temporal_correlation(
                new_inject, existing_inject
            )
            
            # Check spatial correlation
            spatial_corr = self._check_spatial_correlation(
                new_inject, existing_inject
            )
            
            # Check semantic correlation
            semantic_corr = self._check_semantic_correlation(
                new_inject, existing_inject
            )
            
            # Check causal correlation
            causal_corr = self._check_causal_correlation(
                new_inject, existing_inject
            )
            
            # Determine overall correlation
            if any([temporal_corr, spatial_corr, semantic_corr, causal_corr]):
                correlation = self._create_correlation(
                    new_inject, existing_inject,
                    temporal_corr, spatial_corr, semantic_corr, causal_corr
                )
                correlations.append(correlation)
        
        return correlations
    
    def _check_temporal_correlation(
        self,
        inject1: IntelligenceInject,
        inject2: IntelligenceInject
    ) -> Optional[float]:
        """Check if two injects are temporally correlated"""
        
        time_diff = abs(inject1.timestamp - inject2.timestamp)
        
        if time_diff == 0:
            return 1.0  # Simultaneous
        elif time_diff < 300:  # 5 minutes
            return 0.95
        elif time_diff < 1800:  # 30 minutes
            return 0.85
        elif time_diff < 3600:  # 1 hour
            return 0.75
        elif time_diff < self.temporal_window:
            return 0.6
        else:
            return None
    
    def _check_spatial_correlation(
        self,
        inject1: IntelligenceInject,
        inject2: IntelligenceInject
    ) -> Optional[float]:
        """Check if two injects are spatially correlated"""
        
        # Extract location data
        loc1 = inject1.content.get("location") or inject1.metadata.get("location")
        loc2 = inject2.content.get("location") or inject2.metadata.get("location")
        
        if not loc1 or not loc2:
            return None
        
        # Simple distance calculation (would use proper geospatial in production)
        try:
            if isinstance(loc1, dict) and isinstance(loc2, dict):
                lat1, lon1 = loc1.get("lat", 0), loc1.get("lon", 0)
                lat2, lon2 = loc2.get("lat", 0), loc2.get("lon", 0)
                
                # Rough distance calculation
                distance = ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5 * 111000  # meters
                
                if distance < 1000:  # 1km
                    return 0.95
                elif distance < 5000:  # 5km
                    return 0.85
                elif distance < 20000:  # 20km
                    return 0.75
                elif distance < self.spatial_threshold:
                    return 0.6
        except:
            pass
        
        return None
    
    def _check_semantic_correlation(
        self,
        inject1: IntelligenceInject,
        inject2: IntelligenceInject
    ) -> Optional[float]:
        """Check if two injects are semantically related"""
        
        # Check for shared tags
        shared_tags = inject1.tags & inject2.tags
        if shared_tags:
            tag_overlap = len(shared_tags) / max(len(inject1.tags), len(inject2.tags), 1)
            if tag_overlap > 0.5:
                return min(0.8 + tag_overlap * 0.2, 1.0)
        
        # Check for shared entities/keywords
        content1_str = json.dumps(inject1.content).lower()
        content2_str = json.dumps(inject2.content).lower()
        
        # Simple keyword matching (would use NLP in production)
        keywords1 = set(content1_str.split())
        keywords2 = set(content2_str.split())
        
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
            if overlap > 0.3:
                return min(0.6 + overlap * 0.4, 0.9)
        
        # Check domain relationship
        if inject1.domain == inject2.domain:
            return 0.5  # Same domain suggests some relationship
        
        return None
    
    def _check_causal_correlation(
        self,
        inject1: IntelligenceInject,
        inject2: IntelligenceInject
    ) -> Optional[float]:
        """Check if one inject could have caused the other"""
        
        # Temporal ordering is necessary for causation
        if inject1.timestamp >= inject2.timestamp:
            return None
        
        time_diff = inject2.timestamp - inject1.timestamp
        
        # Check for causal keywords
        causal_keywords = {
            "caused", "resulted", "led to", "triggered", "initiated",
            "enabled", "preceded", "followed", "response to"
        }
        
        content2_str = json.dumps(inject2.content).lower()
        if any(keyword in content2_str for keyword in causal_keywords):
            # Check if inject1 is referenced
            if inject1.inject_id in content2_str or inject1.source_name.lower() in content2_str:
                return 0.85
        
        # Domain-specific causality
        # SIGINT -> CYBER (signal preceded cyber event)
        if inject1.domain == IntelligenceDomain.SIGINT and inject2.domain == IntelligenceDomain.CYBINT:
            if time_diff < 300:  # 5 minutes
                return 0.7
        
        # CYBER -> CYBER (cascading cyber events)
        if inject1.domain == IntelligenceDomain.CYBINT and inject2.domain == IntelligenceDomain.CYBINT:
            if time_diff < 600:  # 10 minutes
                return 0.65
        
        return None
    
    def _create_correlation(
        self,
        inject1: IntelligenceInject,
        inject2: IntelligenceInject,
        temporal_score: Optional[float],
        spatial_score: Optional[float],
        semantic_score: Optional[float],
        causal_score: Optional[float]
    ) -> InjectCorrelation:
        """Create correlation object from correlation scores"""
        
        # Determine correlation type
        scores = {
            "temporal": temporal_score or 0,
            "spatial": spatial_score or 0,
            "semantic": semantic_score or 0,
            "causal": causal_score or 0
        }
        
        correlation_type = max(scores, key=scores.get)
        correlation_strength = max(scores.values())
        
        # Generate reasoning
        reasoning = []
        if temporal_score and temporal_score > 0.7:
            time_diff = abs(inject1.timestamp - inject2.timestamp)
            reasoning.append(f"Events occurred within {time_diff:.0f} seconds")
        
        if spatial_score and spatial_score > 0.7:
            reasoning.append("Events occurred in close geographic proximity")
        
        if semantic_score and semantic_score > 0.6:
            shared_tags = inject1.tags & inject2.tags
            if shared_tags:
                reasoning.append(f"Shared indicators: {', '.join(list(shared_tags)[:3])}")
        
        if causal_score and causal_score > 0.6:
            reasoning.append(f"Event 1 may have caused or enabled Event 2")
        
        correlation_id = hashlib.sha256(
            f"{inject1.inject_id}:{inject2.inject_id}".encode()
        ).hexdigest()[:16]
        
        return InjectCorrelation(
            correlation_id=correlation_id,
            correlated_injects=[inject1.inject_id, inject2.inject_id],
            correlation_type=correlation_type,
            correlation_strength=correlation_strength,
            temporal_proximity=abs(inject1.timestamp - inject2.timestamp) if temporal_score else None,
            spatial_proximity=None,  # Would calculate actual distance
            semantic_similarity=semantic_score,
            reasoning=reasoning
        )
    
    def _determine_fusion_group(
        self,
        new_inject: IntelligenceInject,
        correlations: List[InjectCorrelation]
    ) -> List[IntelligenceInject]:
        """Determine which injects should be fused together"""
        
        fusion_group = [new_inject]
        
        # Add strongly correlated injects
        for correlation in correlations:
            if correlation.correlation_strength >= 0.7:
                for inject_id in correlation.correlated_injects:
                    if inject_id in self.active_injects and inject_id != new_inject.inject_id:
                        inject = self.active_injects[inject_id]
                        if inject not in fusion_group:
                            fusion_group.append(inject)
        
        return fusion_group
    
    async def _fuse_injects(
        self,
        injects: List[IntelligenceInject],
        correlations: List[InjectCorrelation]
    ) -> FusedIntelligence:
        """Fuse multiple injects into unified intelligence assessment"""
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        for inject in injects:
            credibility_weight = self.credibility_weights[inject.credibility]
            weight = inject.confidence * credibility_weight
            weighted_confidence += weight
            total_weight += 1.0
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Boost confidence for multi-source confirmation
        if len(injects) >= 3:
            final_confidence = min(final_confidence * 1.2, 1.0)
        elif len(injects) >= 2:
            final_confidence = min(final_confidence * 1.1, 1.0)
        
        # Determine overall credibility
        credibilities = [inj.credibility for inj in injects]
        if all(c == SourceCredibility.CONFIRMED for c in credibilities):
            overall_credibility = SourceCredibility.CONFIRMED
        elif len([c for c in credibilities if c in [SourceCredibility.CONFIRMED, SourceCredibility.PROBABLY_TRUE]]) >= 2:
            overall_credibility = SourceCredibility.PROBABLY_TRUE
        else:
            overall_credibility = SourceCredibility.POSSIBLY_TRUE
        
        # Collect all domains
        domains = set(inj.domain for inj in injects)
        
        # Build fused assessment
        fused_assessment = {
            "summary": self._generate_summary(injects, correlations),
            "key_findings": self._extract_key_findings(injects),
            "timeline": self._construct_timeline(injects),
            "entities": self._extract_entities(injects),
            "indicators": self._extract_indicators(injects),
            "source_count": len(injects),
            "domain_count": len(domains),
            "earliest_report": min(inj.timestamp for inj in injects),
            "latest_report": max(inj.timestamp for inj in injects)
        }
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(injects, correlations, final_confidence)
        
        # Generate alternative hypotheses
        alternatives = self._generate_alternative_hypotheses(injects, correlations)
        
        fusion_id = hashlib.sha256(
            f"{':'.join(inj.inject_id for inj in injects)}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        return FusedIntelligence(
            fusion_id=fusion_id,
            source_injects=[inj.inject_id for inj in injects],
            domains=domains,
            fused_assessment=fused_assessment,
            confidence=final_confidence,
            credibility=overall_credibility,
            correlations=correlations,
            created_at=time.time(),
            reasoning_chain=reasoning_chain,
            alternative_hypotheses=alternatives
        )
    
    def _generate_summary(
        self,
        injects: List[IntelligenceInject],
        correlations: List[InjectCorrelation]
    ) -> str:
        """Generate human-readable summary of fused intelligence"""
        
        domains_str = ", ".join(d.value for d in set(inj.domain for inj in injects))
        source_count = len(injects)
        
        if len(correlations) > 0:
            strongest_corr = max(correlations, key=lambda c: c.correlation_strength)
            corr_type = strongest_corr.correlation_type
            
            return (f"Fused intelligence from {source_count} sources across {domains_str}. "
                   f"Strong {corr_type} correlation detected (strength: {strongest_corr.correlation_strength:.2f}).")
        else:
            return f"Fused intelligence from {source_count} sources across {domains_str}."
    
    def _extract_key_findings(self, injects: List[IntelligenceInject]) -> List[str]:
        """Extract key findings from injects"""
        
        findings = []
        
        for inject in injects:
            # Extract key information from content
            if "finding" in inject.content:
                findings.append(inject.content["finding"])
            elif "observation" in inject.content:
                findings.append(inject.content["observation"])
            elif "description" in inject.content:
                findings.append(inject.content["description"])
        
        return findings[:5]  # Top 5 findings
    
    def _construct_timeline(self, injects: List[IntelligenceInject]) -> List[Dict[str, Any]]:
        """Construct timeline of events"""
        
        timeline = []
        
        for inject in sorted(injects, key=lambda i: i.timestamp):
            timeline.append({
                "timestamp": inject.timestamp,
                "source": inject.source_name,
                "domain": inject.domain.value,
                "event": inject.content.get("description", "Event recorded")
            })
        
        return timeline
    
    def _extract_entities(self, injects: List[IntelligenceInject]) -> Set[str]:
        """Extract entities mentioned in injects"""
        
        entities = set()
        
        for inject in injects:
            # Extract from tags
            entities.update(inject.tags)
            
            # Extract from content
            if "entities" in inject.content:
                entities.update(inject.content["entities"])
            
            if "target" in inject.content:
                entities.add(inject.content["target"])
        
        return entities
    
    def _extract_indicators(self, injects: List[IntelligenceInject]) -> List[str]:
        """Extract threat indicators"""
        
        indicators = []
        
        for inject in injects:
            if "indicators" in inject.content:
                indicators.extend(inject.content["indicators"])
            
            if "threat_indicators" in inject.content:
                indicators.extend(inject.content["threat_indicators"])
        
        return list(set(indicators))
    
    def _generate_reasoning_chain(
        self,
        injects: List[IntelligenceInject],
        correlations: List[InjectCorrelation],
        confidence: float
    ) -> List[str]:
        """Generate reasoning chain explaining the fusion"""
        
        chain = []
        
        chain.append(f"Received {len(injects)} intelligence injects from multiple sources")
        
        if len(correlations) > 0:
            chain.append(f"Identified {len(correlations)} correlations between injects")
            
            for corr in correlations[:3]:  # Top 3 correlations
                chain.extend(corr.reasoning)
        
        high_cred = len([i for i in injects if i.credibility in [
            SourceCredibility.CONFIRMED, SourceCredibility.PROBABLY_TRUE
        ]])
        
        if high_cred >= 2:
            chain.append(f"{high_cred} high-credibility sources confirm assessment")
        
        chain.append(f"Final confidence: {confidence:.2f} based on source credibility and correlation strength")
        
        return chain
    
    def _generate_alternative_hypotheses(
        self,
        injects: List[IntelligenceInject],
        correlations: List[InjectCorrelation]
    ) -> List[Dict[str, Any]]:
        """Generate alternative interpretations"""
        
        alternatives = []
        
        # Hypothesis 1: Coordinated operation
        if len(correlations) > 0 and any(c.correlation_type == "temporal" for c in correlations):
            alternatives.append({
                "hypothesis": "Coordinated multi-domain operation",
                "confidence": 0.75,
                "supporting_evidence": "Multiple temporally correlated events across domains",
                "counter_evidence": None
            })
        
        # Hypothesis 2: Coincidence
        if len(injects) >= 2:
            alternatives.append({
                "hypothesis": "Unrelated coincidental events",
                "confidence": 0.3,
                "supporting_evidence": "Events could occur independently",
                "counter_evidence": "Strong temporal/spatial correlation suggests connection"
            })
        
        return alternatives


# Global instance
multi_domain_fusion_system = MultiDomainIntelligenceFusion()


async def process_intelligence_inject(inject: IntelligenceInject) -> Tuple[FusedIntelligence, List[InjectCorrelation]]:
    """
    Main entry point: Process an intelligence inject.
    Returns fused intelligence and identified correlations.
    """
    return await multi_domain_fusion_system.process_inject(inject)

