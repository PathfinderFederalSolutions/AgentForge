"""
Neural Mesh Integration with Belief Revision and Source Credibility Assessment
Advanced cognitive fusion for intelligence analysis with contradictory evidence handling
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Callable
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
from scipy.special import softmax
from scipy.stats import entropy
import asyncio
import threading

log = logging.getLogger("neural-mesh-integration")

class BeliefRevisionStrategy(Enum):
    """Strategies for belief revision"""
    AGM_REVISION = "agm_revision"  # Alchourrón-Gärdenfors-Makinson
    COHERENTIST = "coherentist"
    FOUNDATIONALIST = "foundationalist"
    PROBABILISTIC = "probabilistic"
    DYNAMIC_EPISTEMIC = "dynamic_epistemic"

class CredibilityAssessmentMethod(Enum):
    """Methods for assessing source credibility"""
    TRACK_RECORD = "track_record"
    CONSENSUS_BASED = "consensus_based"
    EXPERTISE_WEIGHTED = "expertise_weighted"
    TEMPORAL_DECAY = "temporal_decay"
    CROSS_VALIDATION = "cross_validation"
    BAYESIAN_REPUTATION = "bayesian_reputation"

class EvidenceType(Enum):
    """Types of evidence in the neural mesh"""
    DIRECT_OBSERVATION = "direct_observation"
    INFERENCE = "inference"
    EXPERT_JUDGMENT = "expert_judgment"
    AUTOMATED_ANALYSIS = "automated_analysis"
    HISTORICAL_DATA = "historical_data"
    PREDICTIVE_MODEL = "predictive_model"

class ConflictResolutionMethod(Enum):
    """Methods for resolving conflicting evidence"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_OVERRIDE = "expert_override"
    TEMPORAL_PREFERENCE = "temporal_preference"
    CREDIBILITY_BASED = "credibility_based"
    DIALECTICAL_SYNTHESIS = "dialectical_synthesis"

@dataclass
class BeliefState:
    """Represents a belief state in the neural mesh"""
    proposition: str
    confidence: float  # 0.0 to 1.0
    evidence_support: float  # Strength of supporting evidence
    evidence_against: float  # Strength of contradicting evidence
    last_updated: float
    revision_count: int = 0
    justification: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate belief state"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.last_updated == 0:
            self.last_updated = time.time()
    
    def get_strength(self) -> float:
        """Calculate overall belief strength"""
        net_evidence = self.evidence_support - self.evidence_against
        return self.confidence * (1.0 + np.tanh(net_evidence))
    
    def is_coherent_with(self, other: 'BeliefState') -> float:
        """Calculate coherence with another belief (0=contradictory, 1=supportive)"""
        # Simplified coherence calculation
        # In production, would use sophisticated logical consistency checking
        
        # Check for explicit contradictions in propositions
        if "not " + self.proposition.lower() == other.proposition.lower():
            return 0.0
        if "not " + other.proposition.lower() == self.proposition.lower():
            return 0.0
        
        # Check for semantic similarity (simplified)
        common_words = set(self.proposition.lower().split()) & set(other.proposition.lower().split())
        total_words = set(self.proposition.lower().split()) | set(other.proposition.lower().split())
        
        if len(total_words) == 0:
            return 0.5
        
        similarity = len(common_words) / len(total_words)
        
        # Convert similarity to coherence
        if similarity > 0.7:
            return 0.8 + 0.2 * similarity  # High coherence
        elif similarity > 0.3:
            return 0.5 + 0.3 * similarity  # Moderate coherence
        else:
            return 0.5  # Neutral coherence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "proposition": self.proposition,
            "confidence": self.confidence,
            "evidence_support": self.evidence_support,
            "evidence_against": self.evidence_against,
            "last_updated": self.last_updated,
            "revision_count": self.revision_count,
            "justification": self.justification,
            "metadata": self.metadata,
            "strength": self.get_strength()
        }

@dataclass
class EvidenceItem:
    """Individual piece of evidence"""
    evidence_id: str
    content: str
    evidence_type: EvidenceType
    source_id: str
    credibility_score: float
    timestamp: float
    confidence: float
    supporting_propositions: List[str] = field(default_factory=list)
    contradicting_propositions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate evidence item"""
        self.credibility_score = max(0.0, min(1.0, self.credibility_score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not self.evidence_id:
            self.evidence_id = f"evidence_{int(time.time() * 1000)}"
    
    def get_weighted_impact(self) -> float:
        """Calculate weighted impact of evidence"""
        return self.credibility_score * self.confidence
    
    def supports_proposition(self, proposition: str) -> bool:
        """Check if evidence supports a proposition"""
        return proposition in self.supporting_propositions
    
    def contradicts_proposition(self, proposition: str) -> bool:
        """Check if evidence contradicts a proposition"""
        return proposition in self.contradicting_propositions

@dataclass
class SourceCredibility:
    """Credibility assessment for information sources"""
    source_id: str
    overall_credibility: float
    domain_expertise: Dict[str, float] = field(default_factory=dict)
    track_record: List[float] = field(default_factory=list)  # Historical accuracy
    last_assessment: float = field(default_factory=time.time)
    assessment_count: int = 0
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    reliability_trend: float = 0.0  # Positive = improving, negative = declining
    
    def __post_init__(self):
        """Validate source credibility"""
        self.overall_credibility = max(0.0, min(1.0, self.overall_credibility))
        
        # Ensure domain expertise values are valid
        for domain in self.domain_expertise:
            self.domain_expertise[domain] = max(0.0, min(1.0, self.domain_expertise[domain]))
    
    def get_domain_credibility(self, domain: str) -> float:
        """Get credibility for specific domain"""
        domain_cred = self.domain_expertise.get(domain, self.overall_credibility)
        
        # Apply reliability trend
        trend_factor = 1.0 + (self.reliability_trend * 0.1)  # Max 10% adjustment
        trend_factor = max(0.5, min(1.5, trend_factor))
        
        return max(0.0, min(1.0, domain_cred * trend_factor))
    
    def update_track_record(self, accuracy: float):
        """Update track record with new accuracy measurement"""
        self.track_record.append(accuracy)
        
        # Keep limited history
        if len(self.track_record) > 100:
            self.track_record.pop(0)
        
        # Update overall credibility based on recent track record
        if len(self.track_record) >= 5:
            recent_performance = np.mean(self.track_record[-10:])  # Last 10 assessments
            historical_performance = np.mean(self.track_record[:-10]) if len(self.track_record) > 10 else recent_performance
            
            # Calculate trend
            self.reliability_trend = recent_performance - historical_performance
            
            # Update overall credibility (weighted combination)
            self.overall_credibility = 0.7 * self.overall_credibility + 0.3 * recent_performance
        
        self.assessment_count += 1
        self.last_assessment = time.time()
    
    def calculate_bias_score(self) -> float:
        """Calculate overall bias score"""
        if not self.bias_indicators:
            return 0.0
        
        # Weight different types of bias
        bias_weights = {
            "confirmation_bias": 0.3,
            "selection_bias": 0.2,
            "temporal_bias": 0.15,
            "political_bias": 0.2,
            "commercial_bias": 0.15
        }
        
        weighted_bias = 0.0
        total_weight = 0.0
        
        for bias_type, bias_score in self.bias_indicators.items():
            weight = bias_weights.get(bias_type, 0.1)
            weighted_bias += weight * bias_score
            total_weight += weight
        
        return weighted_bias / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source_id": self.source_id,
            "overall_credibility": self.overall_credibility,
            "domain_expertise": self.domain_expertise,
            "track_record_length": len(self.track_record),
            "recent_accuracy": np.mean(self.track_record[-10:]) if self.track_record else 0.0,
            "assessment_count": self.assessment_count,
            "bias_score": self.calculate_bias_score(),
            "reliability_trend": self.reliability_trend,
            "last_assessment": self.last_assessment
        }

class BeliefRevisionEngine:
    """Advanced belief revision engine using multiple strategies"""
    
    def __init__(self, 
                 revision_strategy: BeliefRevisionStrategy = BeliefRevisionStrategy.PROBABILISTIC,
                 coherence_threshold: float = 0.7):
        
        self.revision_strategy = revision_strategy
        self.coherence_threshold = coherence_threshold
        self.belief_states: Dict[str, BeliefState] = {}
        self.revision_history: List[Dict[str, Any]] = []
        
        # Revision parameters
        self.max_revision_iterations = 10
        self.convergence_threshold = 0.01
        self.temporal_decay_rate = 0.1  # Per day
        
        log.info(f"Belief revision engine initialized with {revision_strategy.value} strategy")
    
    def add_belief(self, belief: BeliefState):
        """Add new belief to the system"""
        self.belief_states[belief.proposition] = belief
        log.debug(f"Added belief: {belief.proposition} (confidence: {belief.confidence:.3f})")
    
    def revise_beliefs(self, new_evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """Revise beliefs based on new evidence"""
        start_time = time.time()
        
        try:
            revision_log = {
                "timestamp": start_time,
                "strategy": self.revision_strategy.value,
                "evidence_count": len(new_evidence),
                "beliefs_before": len(self.belief_states),
                "changes": []
            }
            
            # Apply revision strategy
            if self.revision_strategy == BeliefRevisionStrategy.PROBABILISTIC:
                changes = self._probabilistic_revision(new_evidence)
            elif self.revision_strategy == BeliefRevisionStrategy.AGM_REVISION:
                changes = self._agm_revision(new_evidence)
            elif self.revision_strategy == BeliefRevisionStrategy.COHERENTIST:
                changes = self._coherentist_revision(new_evidence)
            elif self.revision_strategy == BeliefRevisionStrategy.FOUNDATIONALIST:
                changes = self._foundationalist_revision(new_evidence)
            else:
                changes = self._dynamic_epistemic_revision(new_evidence)
            
            revision_log["changes"] = changes
            revision_log["beliefs_after"] = len(self.belief_states)
            revision_log["processing_time_ms"] = (time.time() - start_time) * 1000
            
            # Store revision history
            self.revision_history.append(revision_log)
            if len(self.revision_history) > 1000:
                self.revision_history.pop(0)
            
            log.info(f"Belief revision completed: {len(changes)} changes in "
                    f"{revision_log['processing_time_ms']:.2f}ms")
            
            return revision_log
            
        except Exception as e:
            log.error(f"Belief revision failed: {e}")
            return {"error": str(e), "timestamp": start_time}
    
    def _probabilistic_revision(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Probabilistic belief revision"""
        changes = []
        
        # Group evidence by supported/contradicted propositions
        proposition_evidence = defaultdict(list)
        
        for item in evidence:
            for prop in item.supporting_propositions:
                proposition_evidence[prop].append((item, "support"))
            for prop in item.contradicting_propositions:
                proposition_evidence[prop].append((item, "contradict"))
        
        # Update beliefs based on evidence
        for proposition, evidence_list in proposition_evidence.items():
            if proposition in self.belief_states:
                old_belief = self.belief_states[proposition]
                new_belief = self._update_belief_probabilistic(old_belief, evidence_list)
                
                if abs(new_belief.confidence - old_belief.confidence) > self.convergence_threshold:
                    self.belief_states[proposition] = new_belief
                    changes.append({
                        "proposition": proposition,
                        "old_confidence": old_belief.confidence,
                        "new_confidence": new_belief.confidence,
                        "evidence_count": len(evidence_list)
                    })
        
        # Check for coherence and resolve conflicts
        changes.extend(self._resolve_coherence_conflicts())
        
        return changes
    
    def _update_belief_probabilistic(self, belief: BeliefState, evidence_list: List[Tuple[EvidenceItem, str]]) -> BeliefState:
        """Update belief using probabilistic methods"""
        # Calculate evidence impact
        support_strength = 0.0
        contradict_strength = 0.0
        
        for evidence_item, support_type in evidence_list:
            weighted_impact = evidence_item.get_weighted_impact()
            
            if support_type == "support":
                support_strength += weighted_impact
            else:
                contradict_strength += weighted_impact
        
        # Bayesian update (simplified)
        prior_confidence = belief.confidence
        
        # Calculate likelihood ratios
        if support_strength > 0 or contradict_strength > 0:
            support_ratio = support_strength / (support_strength + contradict_strength + 1e-6)
            
            # Update confidence using Bayesian approach
            posterior_confidence = (prior_confidence * support_ratio) / (
                prior_confidence * support_ratio + (1 - prior_confidence) * (1 - support_ratio) + 1e-6
            )
        else:
            posterior_confidence = prior_confidence
        
        # Apply temporal decay
        time_since_update = time.time() - belief.last_updated
        decay_factor = np.exp(-self.temporal_decay_rate * time_since_update / 86400)  # Per day
        posterior_confidence *= decay_factor
        
        # Create updated belief
        updated_belief = BeliefState(
            proposition=belief.proposition,
            confidence=posterior_confidence,
            evidence_support=belief.evidence_support + support_strength,
            evidence_against=belief.evidence_against + contradict_strength,
            last_updated=time.time(),
            revision_count=belief.revision_count + 1,
            justification=belief.justification + [f"Updated with {len(evidence_list)} evidence items"],
            metadata=belief.metadata.copy()
        )
        
        return updated_belief
    
    def _agm_revision(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """AGM (Alchourrón-Gärdenfors-Makinson) belief revision"""
        # Simplified AGM implementation
        # Full AGM would require sophisticated logical reasoning
        
        changes = []
        
        for evidence_item in evidence:
            # For each piece of evidence, determine what beliefs need revision
            for prop in evidence_item.supporting_propositions:
                if prop not in self.belief_states:
                    # Add new belief
                    new_belief = BeliefState(
                        proposition=prop,
                        confidence=evidence_item.confidence * evidence_item.credibility_score,
                        evidence_support=evidence_item.get_weighted_impact(),
                        evidence_against=0.0,
                        last_updated=time.time(),
                        justification=[f"Added based on evidence {evidence_item.evidence_id}"]
                    )
                    
                    self.belief_states[prop] = new_belief
                    changes.append({
                        "proposition": prop,
                        "action": "added",
                        "confidence": new_belief.confidence
                    })
                
                else:
                    # Revise existing belief
                    old_belief = self.belief_states[prop]
                    
                    # AGM contraction and expansion
                    revised_confidence = self._agm_expand_confidence(
                        old_belief.confidence,
                        evidence_item.confidence * evidence_item.credibility_score
                    )
                    
                    if abs(revised_confidence - old_belief.confidence) > self.convergence_threshold:
                        old_belief.confidence = revised_confidence
                        old_belief.revision_count += 1
                        old_belief.last_updated = time.time()
                        old_belief.justification.append(f"AGM revision with evidence {evidence_item.evidence_id}")
                        
                        changes.append({
                            "proposition": prop,
                            "action": "revised",
                            "old_confidence": old_belief.confidence,
                            "new_confidence": revised_confidence
                        })
        
        return changes
    
    def _agm_expand_confidence(self, current_confidence: float, evidence_confidence: float) -> float:
        """AGM expansion of confidence"""
        # Simplified expansion - in full AGM, would use logical consistency checking
        return min(1.0, current_confidence + evidence_confidence * (1.0 - current_confidence))
    
    def _coherentist_revision(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Coherentist belief revision - maximize overall coherence"""
        changes = []
        
        # Calculate current coherence
        current_coherence = self._calculate_system_coherence()
        
        # Try different revision strategies and pick the most coherent
        best_coherence = current_coherence
        best_changes = []
        
        # Strategy 1: Accept all evidence
        temp_beliefs = self.belief_states.copy()
        strategy1_changes = self._accept_all_evidence(evidence, temp_beliefs)
        strategy1_coherence = self._calculate_coherence_for_beliefs(temp_beliefs)
        
        if strategy1_coherence > best_coherence:
            best_coherence = strategy1_coherence
            best_changes = strategy1_changes
            self.belief_states = temp_beliefs
        
        # Strategy 2: Selective acceptance based on coherence
        temp_beliefs = self.belief_states.copy()
        strategy2_changes = self._selective_evidence_acceptance(evidence, temp_beliefs)
        strategy2_coherence = self._calculate_coherence_for_beliefs(temp_beliefs)
        
        if strategy2_coherence > best_coherence:
            best_coherence = strategy2_coherence
            best_changes = strategy2_changes
            self.belief_states = temp_beliefs
        
        return best_changes
    
    def _foundationalist_revision(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Foundationalist belief revision - prioritize foundational beliefs"""
        changes = []
        
        # Identify foundational beliefs (high confidence, well-supported)
        foundational_props = set()
        for prop, belief in self.belief_states.items():
            if belief.confidence > 0.8 and belief.evidence_support > belief.evidence_against * 2:
                foundational_props.add(prop)
        
        # Revise non-foundational beliefs first
        for evidence_item in evidence:
            for prop in evidence_item.supporting_propositions:
                if prop not in foundational_props:
                    # Safe to revise
                    changes.extend(self._update_belief_with_evidence(prop, evidence_item, "support"))
            
            for prop in evidence_item.contradicting_propositions:
                if prop not in foundational_props:
                    # Safe to revise
                    changes.extend(self._update_belief_with_evidence(prop, evidence_item, "contradict"))
        
        return changes
    
    def _dynamic_epistemic_revision(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Dynamic epistemic logic revision"""
        changes = []
        
        # Model belief updates as dynamic epistemic actions
        for evidence_item in evidence:
            # Public announcement of evidence
            announcement_effect = self._process_public_announcement(evidence_item)
            changes.extend(announcement_effect)
        
        return changes
    
    def _resolve_coherence_conflicts(self) -> List[Dict[str, Any]]:
        """Resolve conflicts to maintain coherence"""
        changes = []
        
        # Find conflicting beliefs
        conflicts = self._identify_conflicts()
        
        for conflict in conflicts:
            prop1, prop2, conflict_strength = conflict
            belief1 = self.belief_states[prop1]
            belief2 = self.belief_states[prop2]
            
            # Resolve based on confidence and evidence strength
            if belief1.get_strength() > belief2.get_strength():
                # Reduce confidence in weaker belief
                old_confidence = belief2.confidence
                belief2.confidence *= (1.0 - conflict_strength * 0.5)
                
                changes.append({
                    "proposition": prop2,
                    "action": "conflict_resolution",
                    "old_confidence": old_confidence,
                    "new_confidence": belief2.confidence,
                    "conflicted_with": prop1
                })
            else:
                # Reduce confidence in weaker belief
                old_confidence = belief1.confidence
                belief1.confidence *= (1.0 - conflict_strength * 0.5)
                
                changes.append({
                    "proposition": prop1,
                    "action": "conflict_resolution",
                    "old_confidence": old_confidence,
                    "new_confidence": belief1.confidence,
                    "conflicted_with": prop2
                })
        
        return changes
    
    def _calculate_system_coherence(self) -> float:
        """Calculate overall system coherence"""
        return self._calculate_coherence_for_beliefs(self.belief_states)
    
    def _calculate_coherence_for_beliefs(self, beliefs: Dict[str, BeliefState]) -> float:
        """Calculate coherence for a set of beliefs"""
        if len(beliefs) < 2:
            return 1.0
        
        total_coherence = 0.0
        pair_count = 0
        
        belief_list = list(beliefs.values())
        
        for i in range(len(belief_list)):
            for j in range(i + 1, len(belief_list)):
                coherence = belief_list[i].is_coherent_with(belief_list[j])
                
                # Weight by confidence of both beliefs
                weight = belief_list[i].confidence * belief_list[j].confidence
                total_coherence += coherence * weight
                pair_count += weight
        
        return total_coherence / pair_count if pair_count > 0 else 1.0
    
    def _identify_conflicts(self) -> List[Tuple[str, str, float]]:
        """Identify conflicting beliefs"""
        conflicts = []
        
        beliefs = list(self.belief_states.items())
        
        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                prop1, belief1 = beliefs[i]
                prop2, belief2 = beliefs[j]
                
                coherence = belief1.is_coherent_with(belief2)
                
                # Conflict if coherence is low and both beliefs are confident
                if coherence < self.coherence_threshold and belief1.confidence > 0.5 and belief2.confidence > 0.5:
                    conflict_strength = (1.0 - coherence) * min(belief1.confidence, belief2.confidence)
                    conflicts.append((prop1, prop2, conflict_strength))
        
        return conflicts
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get comprehensive belief system summary"""
        try:
            summary = {
                "total_beliefs": len(self.belief_states),
                "revision_count": len(self.revision_history),
                "system_coherence": self._calculate_system_coherence(),
                "high_confidence_beliefs": sum(1 for b in self.belief_states.values() if b.confidence > 0.8),
                "conflicted_beliefs": len(self._identify_conflicts()),
                "average_confidence": np.mean([b.confidence for b in self.belief_states.values()]) if self.belief_states else 0.0,
                "recent_revisions": len([r for r in self.revision_history if time.time() - r["timestamp"] < 3600])
            }
            
            # Top beliefs by strength
            sorted_beliefs = sorted(
                self.belief_states.items(),
                key=lambda x: x[1].get_strength(),
                reverse=True
            )
            
            summary["top_beliefs"] = [
                {
                    "proposition": prop,
                    "confidence": belief.confidence,
                    "strength": belief.get_strength()
                }
                for prop, belief in sorted_beliefs[:10]
            ]
            
            return summary
            
        except Exception as e:
            log.error(f"Belief summary generation failed: {e}")
            return {"error": str(e)}

class SourceCredibilityManager:
    """Advanced source credibility assessment and management"""
    
    def __init__(self, 
                 assessment_method: CredibilityAssessmentMethod = CredibilityAssessmentMethod.BAYESIAN_REPUTATION):
        
        self.assessment_method = assessment_method
        self.source_credibilities: Dict[str, SourceCredibility] = {}
        self.credibility_history: List[Dict[str, Any]] = []
        
        # Assessment parameters
        self.min_assessments_for_reliability = 5
        self.credibility_decay_rate = 0.05  # Per month
        self.bias_detection_threshold = 0.3
        
        # Cross-validation tracking
        self.validation_results: Dict[str, List[float]] = defaultdict(list)
        
        log.info(f"Source credibility manager initialized with {assessment_method.value} method")
    
    def register_source(self, source_id: str, initial_credibility: float = 0.5, 
                       domain_expertise: Optional[Dict[str, float]] = None):
        """Register new information source"""
        
        credibility = SourceCredibility(
            source_id=source_id,
            overall_credibility=initial_credibility,
            domain_expertise=domain_expertise or {},
            track_record=[],
            assessment_count=0
        )
        
        self.source_credibilities[source_id] = credibility
        log.info(f"Registered source: {source_id} with credibility {initial_credibility:.3f}")
    
    def assess_source_credibility(self, source_id: str, domain: str = "general") -> float:
        """Assess credibility of source for specific domain"""
        
        if source_id not in self.source_credibilities:
            log.warning(f"Unknown source: {source_id}, using default credibility")
            return 0.5
        
        credibility = self.source_credibilities[source_id]
        
        if self.assessment_method == CredibilityAssessmentMethod.TRACK_RECORD:
            return self._track_record_assessment(credibility)
        elif self.assessment_method == CredibilityAssessmentMethod.CONSENSUS_BASED:
            return self._consensus_based_assessment(credibility, domain)
        elif self.assessment_method == CredibilityAssessmentMethod.EXPERTISE_WEIGHTED:
            return self._expertise_weighted_assessment(credibility, domain)
        elif self.assessment_method == CredibilityAssessmentMethod.TEMPORAL_DECAY:
            return self._temporal_decay_assessment(credibility)
        elif self.assessment_method == CredibilityAssessmentMethod.CROSS_VALIDATION:
            return self._cross_validation_assessment(credibility, domain)
        elif self.assessment_method == CredibilityAssessmentMethod.BAYESIAN_REPUTATION:
            return self._bayesian_reputation_assessment(credibility, domain)
        else:
            return credibility.overall_credibility
    
    def update_source_performance(self, source_id: str, accuracy: float, domain: str = "general"):
        """Update source performance based on validation"""
        
        if source_id not in self.source_credibilities:
            log.warning(f"Cannot update unknown source: {source_id}")
            return
        
        credibility = self.source_credibilities[source_id]
        credibility.update_track_record(accuracy)
        
        # Update domain expertise
        if domain in credibility.domain_expertise:
            old_expertise = credibility.domain_expertise[domain]
            credibility.domain_expertise[domain] = 0.8 * old_expertise + 0.2 * accuracy
        else:
            credibility.domain_expertise[domain] = accuracy
        
        # Store validation result
        self.validation_results[source_id].append(accuracy)
        if len(self.validation_results[source_id]) > 100:
            self.validation_results[source_id].pop(0)
        
        # Detect bias patterns
        self._detect_bias_patterns(source_id)
        
        log.debug(f"Updated {source_id} performance: accuracy={accuracy:.3f}, "
                 f"new_credibility={credibility.overall_credibility:.3f}")
    
    def _track_record_assessment(self, credibility: SourceCredibility) -> float:
        """Assessment based on historical track record"""
        
        if len(credibility.track_record) < self.min_assessments_for_reliability:
            return credibility.overall_credibility
        
        # Recent performance weighted more heavily
        recent_weight = 0.7
        historical_weight = 0.3
        
        recent_performance = np.mean(credibility.track_record[-10:])
        historical_performance = np.mean(credibility.track_record[:-10]) if len(credibility.track_record) > 10 else recent_performance
        
        weighted_performance = recent_weight * recent_performance + historical_weight * historical_performance
        
        return max(0.0, min(1.0, weighted_performance))
    
    def _consensus_based_assessment(self, credibility: SourceCredibility, domain: str) -> float:
        """Assessment based on consensus with other sources"""
        
        # Find other sources in the same domain
        domain_sources = [
            c for c in self.source_credibilities.values()
            if domain in c.domain_expertise and c.source_id != credibility.source_id
        ]
        
        if len(domain_sources) < 2:
            return credibility.get_domain_credibility(domain)
        
        # Calculate consensus score (simplified)
        domain_credibilities = [s.get_domain_credibility(domain) for s in domain_sources]
        consensus_level = 1.0 - np.std(domain_credibilities)  # Higher std = lower consensus
        
        # Adjust credibility based on consensus
        base_credibility = credibility.get_domain_credibility(domain)
        consensus_adjustment = consensus_level * 0.2  # Max 20% adjustment
        
        return max(0.0, min(1.0, base_credibility + consensus_adjustment))
    
    def _expertise_weighted_assessment(self, credibility: SourceCredibility, domain: str) -> float:
        """Assessment weighted by domain expertise"""
        
        domain_expertise = credibility.domain_expertise.get(domain, 0.5)
        overall_credibility = credibility.overall_credibility
        
        # Weight expertise more heavily for specialized domains
        expertise_weight = 0.6 if domain != "general" else 0.3
        overall_weight = 1.0 - expertise_weight
        
        weighted_credibility = expertise_weight * domain_expertise + overall_weight * overall_credibility
        
        return max(0.0, min(1.0, weighted_credibility))
    
    def _temporal_decay_assessment(self, credibility: SourceCredibility) -> float:
        """Assessment with temporal decay of credibility"""
        
        time_since_assessment = time.time() - credibility.last_assessment
        decay_factor = np.exp(-self.credibility_decay_rate * time_since_assessment / (30 * 86400))  # Per month
        
        decayed_credibility = credibility.overall_credibility * decay_factor
        
        return max(0.1, min(1.0, decayed_credibility))  # Minimum 0.1 to avoid complete loss
    
    def _cross_validation_assessment(self, credibility: SourceCredibility, domain: str) -> float:
        """Assessment based on cross-validation with other sources"""
        
        if credibility.source_id not in self.validation_results:
            return credibility.get_domain_credibility(domain)
        
        validation_scores = self.validation_results[credibility.source_id]
        
        if len(validation_scores) < 3:
            return credibility.get_domain_credibility(domain)
        
        # Calculate validation-based credibility
        mean_validation = np.mean(validation_scores)
        validation_consistency = 1.0 - np.std(validation_scores)  # Lower std = higher consistency
        
        validation_credibility = mean_validation * validation_consistency
        
        # Combine with existing credibility
        base_credibility = credibility.get_domain_credibility(domain)
        combined_credibility = 0.6 * validation_credibility + 0.4 * base_credibility
        
        return max(0.0, min(1.0, combined_credibility))
    
    def _bayesian_reputation_assessment(self, credibility: SourceCredibility, domain: str) -> float:
        """Bayesian reputation assessment"""
        
        # Prior belief about credibility
        prior_credibility = credibility.get_domain_credibility(domain)
        
        if len(credibility.track_record) == 0:
            return prior_credibility
        
        # Bayesian update based on track record
        successes = sum(1 for score in credibility.track_record if score > 0.7)
        failures = sum(1 for score in credibility.track_record if score < 0.3)
        total_assessments = len(credibility.track_record)
        
        if total_assessments == 0:
            return prior_credibility
        
        # Beta distribution parameters (Bayesian approach)
        alpha = 1 + successes  # Prior alpha = 1
        beta = 1 + failures    # Prior beta = 1
        
        # Posterior mean of beta distribution
        posterior_credibility = alpha / (alpha + beta)
        
        # Weight by number of assessments (more assessments = more weight to posterior)
        assessment_weight = min(1.0, total_assessments / 20.0)  # Full weight at 20 assessments
        
        final_credibility = (1 - assessment_weight) * prior_credibility + assessment_weight * posterior_credibility
        
        return max(0.0, min(1.0, final_credibility))
    
    def _detect_bias_patterns(self, source_id: str):
        """Detect bias patterns in source behavior"""
        
        if source_id not in self.validation_results:
            return
        
        credibility = self.source_credibilities[source_id]
        validation_scores = self.validation_results[source_id]
        
        if len(validation_scores) < 10:
            return
        
        # Detect confirmation bias (consistently high or low scores)
        score_variance = np.var(validation_scores)
        if score_variance < 0.01:  # Very low variance
            mean_score = np.mean(validation_scores)
            if mean_score > 0.8:
                credibility.bias_indicators["confirmation_bias"] = 0.7  # High positive bias
            elif mean_score < 0.2:
                credibility.bias_indicators["confirmation_bias"] = 0.7  # High negative bias
        
        # Detect temporal bias (performance changes over time)
        if len(validation_scores) >= 20:
            early_scores = validation_scores[:10]
            late_scores = validation_scores[-10:]
            
            early_mean = np.mean(early_scores)
            late_mean = np.mean(late_scores)
            
            if abs(early_mean - late_mean) > 0.3:
                credibility.bias_indicators["temporal_bias"] = abs(early_mean - late_mean)
        
        # Overall bias score
        bias_score = credibility.calculate_bias_score()
        if bias_score > self.bias_detection_threshold:
            log.warning(f"Bias detected in source {source_id}: bias_score={bias_score:.3f}")
    
    def get_credibility_rankings(self, domain: str = "general", limit: int = 10) -> List[Dict[str, Any]]:
        """Get ranked list of sources by credibility"""
        
        rankings = []
        
        for source_id, credibility in self.source_credibilities.items():
            domain_credibility = self.assess_source_credibility(source_id, domain)
            
            rankings.append({
                "source_id": source_id,
                "credibility": domain_credibility,
                "assessments": credibility.assessment_count,
                "bias_score": credibility.calculate_bias_score(),
                "reliability_trend": credibility.reliability_trend
            })
        
        # Sort by credibility
        rankings.sort(key=lambda x: x["credibility"], reverse=True)
        
        return rankings[:limit]
    
    def get_credibility_statistics(self) -> Dict[str, Any]:
        """Get comprehensive credibility statistics"""
        
        if not self.source_credibilities:
            return {"error": "No sources registered"}
        
        credibilities = [c.overall_credibility for c in self.source_credibilities.values()]
        bias_scores = [c.calculate_bias_score() for c in self.source_credibilities.values()]
        
        stats = {
            "total_sources": len(self.source_credibilities),
            "average_credibility": np.mean(credibilities),
            "credibility_std": np.std(credibilities),
            "high_credibility_sources": sum(1 for c in credibilities if c > 0.8),
            "low_credibility_sources": sum(1 for c in credibilities if c < 0.3),
            "average_bias_score": np.mean(bias_scores),
            "biased_sources": sum(1 for b in bias_scores if b > self.bias_detection_threshold),
            "total_assessments": sum(c.assessment_count for c in self.source_credibilities.values())
        }
        
        return stats

class NeuralMeshIntegrator:
    """Main neural mesh integration system"""
    
    def __init__(self,
                 revision_strategy: BeliefRevisionStrategy = BeliefRevisionStrategy.PROBABILISTIC,
                 credibility_method: CredibilityAssessmentMethod = CredibilityAssessmentMethod.BAYESIAN_REPUTATION,
                 conflict_resolution: ConflictResolutionMethod = ConflictResolutionMethod.CREDIBILITY_BASED):
        
        self.belief_engine = BeliefRevisionEngine(revision_strategy)
        self.credibility_manager = SourceCredibilityManager(credibility_method)
        self.conflict_resolution = conflict_resolution
        
        # Integration components
        self.evidence_buffer: deque = deque(maxlen=10000)
        self.fusion_results: Dict[str, Any] = {}
        self.integration_history: List[Dict[str, Any]] = []
        
        # Processing control
        self._processing_lock = threading.Lock()
        
        log.info(f"Neural mesh integrator initialized: revision={revision_strategy.value}, "
                f"credibility={credibility_method.value}, conflict_resolution={conflict_resolution.value}")
    
    def integrate_fusion_result(self, 
                               fusion_result: Dict[str, Any],
                               source_sensors: List[str],
                               domain: str = "general") -> Dict[str, Any]:
        """Integrate fusion result into neural mesh"""
        
        start_time = time.time()
        
        try:
            with self._processing_lock:
                # Extract propositions from fusion result
                propositions = self._extract_propositions(fusion_result)
                
                # Create evidence items
                evidence_items = []
                
                for sensor_id in source_sensors:
                    # Assess sensor credibility
                    credibility = self.credibility_manager.assess_source_credibility(sensor_id, domain)
                    
                    # Create evidence item
                    evidence = EvidenceItem(
                        evidence_id=f"fusion_{int(time.time() * 1000)}_{sensor_id}",
                        content=json.dumps(fusion_result),
                        evidence_type=EvidenceType.AUTOMATED_ANALYSIS,
                        source_id=sensor_id,
                        credibility_score=credibility,
                        timestamp=fusion_result.get("timestamp", time.time()),
                        confidence=fusion_result.get("confidence", 0.5),
                        supporting_propositions=propositions
                    )
                    
                    evidence_items.append(evidence)
                    self.evidence_buffer.append(evidence)
                
                # Revise beliefs
                revision_result = self.belief_engine.revise_beliefs(evidence_items)
                
                # Handle conflicts if any
                conflict_resolution_result = self._resolve_conflicts(evidence_items)
                
                # Update fusion results
                self.fusion_results[fusion_result.get("task_id", "unknown")] = {
                    "fusion_result": fusion_result,
                    "evidence_items": [e.evidence_id for e in evidence_items],
                    "belief_changes": revision_result.get("changes", []),
                    "conflicts_resolved": conflict_resolution_result,
                    "integration_timestamp": time.time()
                }
                
                # Create integration summary
                integration_summary = {
                    "timestamp": start_time,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "fusion_task_id": fusion_result.get("task_id", "unknown"),
                    "source_sensors": source_sensors,
                    "domain": domain,
                    "evidence_created": len(evidence_items),
                    "beliefs_changed": len(revision_result.get("changes", [])),
                    "conflicts_resolved": len(conflict_resolution_result),
                    "system_coherence": self.belief_engine._calculate_system_coherence()
                }
                
                self.integration_history.append(integration_summary)
                if len(self.integration_history) > 1000:
                    self.integration_history.pop(0)
                
                log.info(f"Fusion result integrated: {len(evidence_items)} evidence items, "
                        f"{len(revision_result.get('changes', []))} belief changes in "
                        f"{integration_summary['processing_time_ms']:.2f}ms")
                
                return integration_summary
                
        except Exception as e:
            log.error(f"Neural mesh integration failed: {e}")
            return {"error": str(e), "timestamp": start_time}
    
    def _extract_propositions(self, fusion_result: Dict[str, Any]) -> List[str]:
        """Extract propositions from fusion result"""
        propositions = []
        
        # Extract key findings
        if "fused_value" in fusion_result:
            value = fusion_result["fused_value"]
            if isinstance(value, (int, float)):
                if value > 0.7:
                    propositions.append("high_confidence_detection")
                elif value > 0.3:
                    propositions.append("moderate_confidence_detection")
                else:
                    propositions.append("low_confidence_detection")
        
        # Extract algorithm-specific propositions
        algorithm = fusion_result.get("algorithm", "unknown")
        if algorithm != "unknown":
            propositions.append(f"fusion_algorithm_{algorithm}_applied")
        
        # Extract quality indicators
        confidence = fusion_result.get("confidence", 0.0)
        if confidence > 0.8:
            propositions.append("high_quality_fusion")
        elif confidence < 0.3:
            propositions.append("low_quality_fusion")
        
        return propositions
    
    def _resolve_conflicts(self, evidence_items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Resolve conflicts between evidence items"""
        
        conflicts_resolved = []
        
        if self.conflict_resolution == ConflictResolutionMethod.CREDIBILITY_BASED:
            conflicts_resolved = self._credibility_based_resolution(evidence_items)
        elif self.conflict_resolution == ConflictResolutionMethod.MAJORITY_VOTE:
            conflicts_resolved = self._majority_vote_resolution(evidence_items)
        elif self.conflict_resolution == ConflictResolutionMethod.WEIGHTED_AVERAGE:
            conflicts_resolved = self._weighted_average_resolution(evidence_items)
        elif self.conflict_resolution == ConflictResolutionMethod.TEMPORAL_PREFERENCE:
            conflicts_resolved = self._temporal_preference_resolution(evidence_items)
        
        return conflicts_resolved
    
    def _credibility_based_resolution(self, evidence_items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """Resolve conflicts based on source credibility"""
        
        resolutions = []
        
        # Group evidence by propositions they support
        proposition_evidence = defaultdict(list)
        
        for item in evidence_items:
            for prop in item.supporting_propositions:
                proposition_evidence[prop].append(item)
        
        # Check for conflicts within each proposition
        for proposition, supporting_evidence in proposition_evidence.items():
            if len(supporting_evidence) > 1:
                # Check if evidence items conflict (simplified)
                credibilities = [e.credibility_score for e in supporting_evidence]
                
                if max(credibilities) - min(credibilities) > 0.3:  # Significant credibility difference
                    # Prefer higher credibility source
                    best_evidence = max(supporting_evidence, key=lambda e: e.credibility_score)
                    
                    resolutions.append({
                        "proposition": proposition,
                        "resolution_method": "credibility_based",
                        "selected_evidence": best_evidence.evidence_id,
                        "rejected_evidence": [e.evidence_id for e in supporting_evidence if e != best_evidence]
                    })
        
        return resolutions
    
    def get_neural_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive neural mesh status"""
        
        try:
            belief_summary = self.belief_engine.get_belief_summary()
            credibility_stats = self.credibility_manager.get_credibility_statistics()
            
            status = {
                "timestamp": time.time(),
                "belief_system": belief_summary,
                "credibility_system": credibility_stats,
                "evidence_buffer_size": len(self.evidence_buffer),
                "fusion_results_stored": len(self.fusion_results),
                "integration_history_length": len(self.integration_history),
                "recent_integrations": len([h for h in self.integration_history if time.time() - h["timestamp"] < 3600])
            }
            
            # Performance metrics
            if self.integration_history:
                processing_times = [h["processing_time_ms"] for h in self.integration_history[-100:]]
                status["performance"] = {
                    "average_processing_time_ms": np.mean(processing_times),
                    "processing_time_std_ms": np.std(processing_times),
                    "max_processing_time_ms": max(processing_times),
                    "min_processing_time_ms": min(processing_times)
                }
            
            return status
            
        except Exception as e:
            log.error(f"Neural mesh status generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

# Utility functions
def create_intelligence_neural_mesh(
    domain: str = "intelligence_analysis"
) -> NeuralMeshIntegrator:
    """Create neural mesh optimized for intelligence analysis"""
    
    # Select strategies based on domain
    if domain == "real_time_intelligence":
        revision_strategy = BeliefRevisionStrategy.PROBABILISTIC
        credibility_method = CredibilityAssessmentMethod.TRACK_RECORD
        conflict_resolution = ConflictResolutionMethod.CREDIBILITY_BASED
    elif domain == "strategic_analysis":
        revision_strategy = BeliefRevisionStrategy.COHERENTIST
        credibility_method = CredibilityAssessmentMethod.BAYESIAN_REPUTATION
        conflict_resolution = ConflictResolutionMethod.DIALECTICAL_SYNTHESIS
    elif domain == "tactical_operations":
        revision_strategy = BeliefRevisionStrategy.FOUNDATIONALIST
        credibility_method = CredibilityAssessmentMethod.CONSENSUS_BASED
        conflict_resolution = ConflictResolutionMethod.TEMPORAL_PREFERENCE
    else:
        revision_strategy = BeliefRevisionStrategy.PROBABILISTIC
        credibility_method = CredibilityAssessmentMethod.BAYESIAN_REPUTATION
        conflict_resolution = ConflictResolutionMethod.CREDIBILITY_BASED
    
    integrator = NeuralMeshIntegrator(
        revision_strategy=revision_strategy,
        credibility_method=credibility_method,
        conflict_resolution=conflict_resolution
    )
    
    log.info(f"Intelligence neural mesh created for domain: {domain}")
    
    return integrator
