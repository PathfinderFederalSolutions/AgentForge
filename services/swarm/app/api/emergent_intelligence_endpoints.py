"""
Emergent Intelligence Endpoints - Phase 3 Implementation
Advanced pattern recognition, learning, and adaptive behavior systems
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict, deque

log = logging.getLogger("emergent-intelligence-api")

# Data Models
class UserInteractionPattern(BaseModel):
    user_id: str
    pattern_type: str  # 'behavioral', 'preference', 'temporal', 'capability_usage'
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    last_observed: datetime
    trend_direction: str  # 'increasing', 'decreasing', 'stable'

class ConversationContext(BaseModel):
    conversation_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    context_embeddings: List[float] = []
    topics_discussed: List[str] = []
    capabilities_used: List[str] = []
    sentiment_progression: List[float] = []
    complexity_progression: List[float] = []

class EmergentInsight(BaseModel):
    insight_id: str
    type: str  # 'user_preference', 'system_optimization', 'capability_gap', 'behavioral_pattern'
    title: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    actionable_recommendations: List[str]
    impact_score: float
    created_at: datetime

class AdaptiveRecommendation(BaseModel):
    recommendation_id: str
    user_id: str
    type: str  # 'capability', 'workflow', 'ui_adjustment', 'content'
    title: str
    description: str
    reasoning: str
    confidence: float
    priority: str  # 'low', 'medium', 'high', 'critical'
    implementation_complexity: str  # 'simple', 'moderate', 'complex'
    expected_benefit: str

# Router
router = APIRouter(prefix="/v1/intelligence", tags=["emergent-intelligence"])

# In-memory storage for pattern analysis (in production, use persistent storage)
user_patterns: Dict[str, List[UserInteractionPattern]] = defaultdict(list)
conversation_contexts: Dict[str, ConversationContext] = {}
emergent_insights: List[EmergentInsight] = []
user_recommendations: Dict[str, List[AdaptiveRecommendation]] = defaultdict(list)

# Pattern analysis engines
class PatternRecognitionEngine:
    """Advanced pattern recognition for user behavior and system optimization"""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=10000)  # Keep last 10k patterns
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
    
    async def analyze_user_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> List[UserInteractionPattern]:
        """Analyze user interaction for emerging patterns"""
        patterns = []
        
        # Behavioral pattern analysis
        behavioral_pattern = await self._analyze_behavioral_pattern(user_id, interaction_data)
        if behavioral_pattern:
            patterns.append(behavioral_pattern)
        
        # Preference pattern analysis
        preference_pattern = await self._analyze_preference_pattern(user_id, interaction_data)
        if preference_pattern:
            patterns.append(preference_pattern)
        
        # Temporal pattern analysis
        temporal_pattern = await self._analyze_temporal_pattern(user_id, interaction_data)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        # Capability usage pattern analysis
        capability_pattern = await self._analyze_capability_usage_pattern(user_id, interaction_data)
        if capability_pattern:
            patterns.append(capability_pattern)
        
        return patterns
    
    async def _analyze_behavioral_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[UserInteractionPattern]:
        """Analyze behavioral patterns in user interactions"""
        try:
            # Extract behavioral indicators
            message_length = len(interaction_data.get('message', ''))
            response_time = interaction_data.get('response_time', 0)
            complexity_level = interaction_data.get('complexity_level', 0)
            capability_count = len(interaction_data.get('capabilities_used', []))
            
            # Calculate behavioral score
            behavioral_score = (
                min(message_length / 100, 1.0) * 0.3 +  # Message complexity
                min(capability_count / 5, 1.0) * 0.4 +   # Capability diversity
                min(complexity_level, 1.0) * 0.3         # Request complexity
            )
            
            if behavioral_score > 0.5:  # Significant behavioral pattern
                return UserInteractionPattern(
                    user_id=user_id,
                    pattern_type='behavioral',
                    pattern_data={
                        'average_message_length': message_length,
                        'average_response_time': response_time,
                        'complexity_preference': complexity_level,
                        'capability_diversity': capability_count,
                        'behavioral_score': behavioral_score
                    },
                    confidence=min(behavioral_score * 1.2, 1.0),
                    frequency=1,
                    last_observed=datetime.now(),
                    trend_direction='increasing' if behavioral_score > 0.7 else 'stable'
                )
        except Exception as e:
            log.error(f"Error analyzing behavioral pattern: {e}")
        
        return None
    
    async def _analyze_preference_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[UserInteractionPattern]:
        """Analyze user preference patterns"""
        try:
            preferences = interaction_data.get('preferences', {})
            capabilities_used = interaction_data.get('capabilities_used', [])
            output_formats = interaction_data.get('output_formats_requested', [])
            
            # Build preference profile
            preference_profile = {
                'preferred_capabilities': capabilities_used,
                'preferred_output_formats': output_formats,
                'interaction_style': preferences.get('interaction_style', 'detailed'),
                'complexity_preference': preferences.get('complexity', 'medium'),
                'response_format': preferences.get('response_format', 'comprehensive')
            }
            
            # Calculate preference strength
            preference_strength = len([v for v in preference_profile.values() if v]) / len(preference_profile)
            
            if preference_strength > 0.6:
                return UserInteractionPattern(
                    user_id=user_id,
                    pattern_type='preference',
                    pattern_data=preference_profile,
                    confidence=preference_strength,
                    frequency=1,
                    last_observed=datetime.now(),
                    trend_direction='stable'
                )
        except Exception as e:
            log.error(f"Error analyzing preference pattern: {e}")
        
        return None
    
    async def _analyze_temporal_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[UserInteractionPattern]:
        """Analyze temporal usage patterns"""
        try:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            session_duration = interaction_data.get('session_duration', 0)
            
            temporal_data = {
                'preferred_hour': current_hour,
                'preferred_day': current_day,
                'average_session_duration': session_duration,
                'usage_frequency': 'regular' if session_duration > 300 else 'brief'
            }
            
            # Temporal patterns are more reliable with more data points
            confidence = min(session_duration / 1800, 0.8)  # Max confidence after 30 min session
            
            if confidence > 0.3:
                return UserInteractionPattern(
                    user_id=user_id,
                    pattern_type='temporal',
                    pattern_data=temporal_data,
                    confidence=confidence,
                    frequency=1,
                    last_observed=datetime.now(),
                    trend_direction='stable'
                )
        except Exception as e:
            log.error(f"Error analyzing temporal pattern: {e}")
        
        return None
    
    async def _analyze_capability_usage_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[UserInteractionPattern]:
        """Analyze capability usage patterns"""
        try:
            capabilities_used = interaction_data.get('capabilities_used', [])
            success_rate = interaction_data.get('success_rate', 0.8)
            user_satisfaction = interaction_data.get('user_satisfaction', 0.7)
            
            if capabilities_used:
                capability_data = {
                    'most_used_capabilities': capabilities_used,
                    'success_rate': success_rate,
                    'user_satisfaction': user_satisfaction,
                    'capability_diversity': len(set(capabilities_used)),
                    'advanced_usage': len([cap for cap in capabilities_used if 'quantum' in cap or 'neural' in cap]) > 0
                }
                
                usage_strength = (success_rate + user_satisfaction) / 2
                
                return UserInteractionPattern(
                    user_id=user_id,
                    pattern_type='capability_usage',
                    pattern_data=capability_data,
                    confidence=usage_strength,
                    frequency=len(capabilities_used),
                    last_observed=datetime.now(),
                    trend_direction='increasing' if usage_strength > 0.8 else 'stable'
                )
        except Exception as e:
            log.error(f"Error analyzing capability usage pattern: {e}")
        
        return None

class EmergentInsightEngine:
    """Generate emergent insights from pattern analysis"""
    
    def __init__(self):
        self.insight_threshold = 0.75
        self.pattern_correlation_threshold = 0.6
    
    async def generate_insights(self, patterns: List[UserInteractionPattern]) -> List[EmergentInsight]:
        """Generate emergent insights from user patterns"""
        insights = []
        
        # User preference insights
        preference_insight = await self._generate_preference_insight(patterns)
        if preference_insight:
            insights.append(preference_insight)
        
        # System optimization insights
        optimization_insight = await self._generate_optimization_insight(patterns)
        if optimization_insight:
            insights.append(optimization_insight)
        
        # Capability gap insights
        capability_insight = await self._generate_capability_gap_insight(patterns)
        if capability_insight:
            insights.append(capability_insight)
        
        # Behavioral pattern insights
        behavioral_insight = await self._generate_behavioral_insight(patterns)
        if behavioral_insight:
            insights.append(behavioral_insight)
        
        return insights
    
    async def _generate_preference_insight(self, patterns: List[UserInteractionPattern]) -> Optional[EmergentInsight]:
        """Generate insights about user preferences"""
        preference_patterns = [p for p in patterns if p.pattern_type == 'preference']
        
        if preference_patterns:
            # Aggregate preference data
            all_capabilities = []
            all_formats = []
            
            for pattern in preference_patterns:
                all_capabilities.extend(pattern.pattern_data.get('preferred_capabilities', []))
                all_formats.extend(pattern.pattern_data.get('preferred_output_formats', []))
            
            # Find most common preferences
            capability_counts = defaultdict(int)
            format_counts = defaultdict(int)
            
            for cap in all_capabilities:
                capability_counts[cap] += 1
            for fmt in all_formats:
                format_counts[fmt] += 1
            
            top_capabilities = sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_formats = sorted(format_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_capabilities:
                return EmergentInsight(
                    insight_id=str(uuid.uuid4()),
                    type='user_preference',
                    title='User Capability Preferences Identified',
                    description=f'User shows strong preference for {", ".join([cap[0] for cap in top_capabilities])} capabilities',
                    confidence=min(sum([p.confidence for p in preference_patterns]) / len(preference_patterns), 1.0),
                    supporting_evidence=[
                        f'Used {cap[0]} capability {cap[1]} times' for cap in top_capabilities
                    ],
                    actionable_recommendations=[
                        f'Prioritize {cap[0]} in capability suggestions' for cap in top_capabilities[:2]
                    ],
                    impact_score=0.8,
                    created_at=datetime.now()
                )
        
        return None
    
    async def _generate_optimization_insight(self, patterns: List[UserInteractionPattern]) -> Optional[EmergentInsight]:
        """Generate system optimization insights"""
        behavioral_patterns = [p for p in patterns if p.pattern_type == 'behavioral']
        
        if behavioral_patterns:
            avg_complexity = sum([p.pattern_data.get('complexity_preference', 0) for p in behavioral_patterns]) / len(behavioral_patterns)
            avg_capability_diversity = sum([p.pattern_data.get('capability_diversity', 0) for p in behavioral_patterns]) / len(behavioral_patterns)
            
            if avg_complexity > 0.7 and avg_capability_diversity > 3:
                return EmergentInsight(
                    insight_id=str(uuid.uuid4()),
                    type='system_optimization',
                    title='High-Complexity User Detected',
                    description='User consistently requests complex operations with diverse capabilities',
                    confidence=0.85,
                    supporting_evidence=[
                        f'Average complexity preference: {avg_complexity:.2f}',
                        f'Average capability diversity: {avg_capability_diversity:.1f}'
                    ],
                    actionable_recommendations=[
                        'Enable quantum coordination by default',
                        'Pre-load advanced capabilities',
                        'Increase default agent allocation'
                    ],
                    impact_score=0.9,
                    created_at=datetime.now()
                )
        
        return None
    
    async def _generate_capability_gap_insight(self, patterns: List[UserInteractionPattern]) -> Optional[EmergentInsight]:
        """Generate insights about capability gaps"""
        capability_patterns = [p for p in patterns if p.pattern_type == 'capability_usage']
        
        if capability_patterns:
            # Analyze success rates and satisfaction
            low_satisfaction_capabilities = []
            
            for pattern in capability_patterns:
                success_rate = pattern.pattern_data.get('success_rate', 1.0)
                satisfaction = pattern.pattern_data.get('user_satisfaction', 1.0)
                capabilities = pattern.pattern_data.get('most_used_capabilities', [])
                
                if success_rate < 0.7 or satisfaction < 0.6:
                    low_satisfaction_capabilities.extend(capabilities)
            
            if low_satisfaction_capabilities:
                capability_counts = defaultdict(int)
                for cap in low_satisfaction_capabilities:
                    capability_counts[cap] += 1
                
                problematic_caps = sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                
                return EmergentInsight(
                    insight_id=str(uuid.uuid4()),
                    type='capability_gap',
                    title='Capability Performance Issues Detected',
                    description=f'Low satisfaction with {", ".join([cap[0] for cap in problematic_caps])} capabilities',
                    confidence=0.75,
                    supporting_evidence=[
                        f'{cap[0]} capability had {cap[1]} low-satisfaction instances' for cap in problematic_caps
                    ],
                    actionable_recommendations=[
                        f'Improve {cap[0]} capability performance' for cap in problematic_caps
                    ],
                    impact_score=0.7,
                    created_at=datetime.now()
                )
        
        return None
    
    async def _generate_behavioral_insight(self, patterns: List[UserInteractionPattern]) -> Optional[EmergentInsight]:
        """Generate behavioral insights"""
        temporal_patterns = [p for p in patterns if p.pattern_type == 'temporal']
        
        if temporal_patterns:
            # Analyze temporal usage
            hours = [p.pattern_data.get('preferred_hour', 12) for p in temporal_patterns]
            session_durations = [p.pattern_data.get('average_session_duration', 0) for p in temporal_patterns]
            
            avg_hour = sum(hours) / len(hours)
            avg_duration = sum(session_durations) / len(session_durations)
            
            if avg_duration > 1800:  # 30+ minute sessions
                return EmergentInsight(
                    insight_id=str(uuid.uuid4()),
                    type='behavioral_pattern',
                    title='Extended Session User Profile',
                    description=f'User typically engages in extended sessions (avg: {avg_duration/60:.1f} minutes)',
                    confidence=0.8,
                    supporting_evidence=[
                        f'Average session duration: {avg_duration/60:.1f} minutes',
                        f'Preferred time: {int(avg_hour):02d}:00'
                    ],
                    actionable_recommendations=[
                        'Enable advanced features by default',
                        'Provide session progress saving',
                        'Offer complex workflow templates'
                    ],
                    impact_score=0.6,
                    created_at=datetime.now()
                )
        
        return None

class AdaptiveRecommendationEngine:
    """Generate adaptive recommendations based on patterns and insights"""
    
    def __init__(self):
        self.recommendation_threshold = 0.6
    
    async def generate_recommendations(self, user_id: str, patterns: List[UserInteractionPattern], insights: List[EmergentInsight]) -> List[AdaptiveRecommendation]:
        """Generate adaptive recommendations for user"""
        recommendations = []
        
        # Capability recommendations
        capability_recs = await self._generate_capability_recommendations(user_id, patterns)
        recommendations.extend(capability_recs)
        
        # Workflow recommendations
        workflow_recs = await self._generate_workflow_recommendations(user_id, patterns)
        recommendations.extend(workflow_recs)
        
        # UI adjustment recommendations
        ui_recs = await self._generate_ui_recommendations(user_id, insights)
        recommendations.extend(ui_recs)
        
        # Content recommendations
        content_recs = await self._generate_content_recommendations(user_id, patterns)
        recommendations.extend(content_recs)
        
        return recommendations
    
    async def _generate_capability_recommendations(self, user_id: str, patterns: List[UserInteractionPattern]) -> List[AdaptiveRecommendation]:
        """Generate capability-based recommendations"""
        recommendations = []
        
        capability_patterns = [p for p in patterns if p.pattern_type == 'capability_usage']
        
        for pattern in capability_patterns:
            if pattern.confidence > 0.7:
                advanced_usage = pattern.pattern_data.get('advanced_usage', False)
                
                if advanced_usage:
                    recommendations.append(AdaptiveRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        user_id=user_id,
                        type='capability',
                        title='Enable Quantum Coordination by Default',
                        description='Based on your advanced capability usage, quantum coordination could improve performance',
                        reasoning='User consistently uses advanced capabilities with high success rate',
                        confidence=pattern.confidence,
                        priority='high',
                        implementation_complexity='simple',
                        expected_benefit='20-30% faster processing for complex requests'
                    ))
        
        return recommendations
    
    async def _generate_workflow_recommendations(self, user_id: str, patterns: List[UserInteractionPattern]) -> List[AdaptiveRecommendation]:
        """Generate workflow-based recommendations"""
        recommendations = []
        
        behavioral_patterns = [p for p in patterns if p.pattern_type == 'behavioral']
        
        for pattern in behavioral_patterns:
            complexity_pref = pattern.pattern_data.get('complexity_preference', 0)
            
            if complexity_pref > 0.8:
                recommendations.append(AdaptiveRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    type='workflow',
                    title='Create Custom Workflow Templates',
                    description='Pre-configured templates for your complex, multi-step processes',
                    reasoning='User frequently engages in complex, multi-capability workflows',
                    confidence=pattern.confidence,
                    priority='medium',
                    implementation_complexity='moderate',
                    expected_benefit='Faster workflow initiation and reduced setup time'
                ))
        
        return recommendations
    
    async def _generate_ui_recommendations(self, user_id: str, insights: List[EmergentInsight]) -> List[AdaptiveRecommendation]:
        """Generate UI adjustment recommendations"""
        recommendations = []
        
        for insight in insights:
            if insight.type == 'behavioral_pattern' and 'Extended Session' in insight.title:
                recommendations.append(AdaptiveRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    type='ui_adjustment',
                    title='Enable Advanced UI Mode',
                    description='Show advanced controls and detailed information by default',
                    reasoning='User engages in extended sessions and prefers detailed interactions',
                    confidence=insight.confidence,
                    priority='medium',
                    implementation_complexity='simple',
                    expected_benefit='Better access to advanced features and information'
                ))
        
        return recommendations
    
    async def _generate_content_recommendations(self, user_id: str, patterns: List[UserInteractionPattern]) -> List[AdaptiveRecommendation]:
        """Generate content-based recommendations"""
        recommendations = []
        
        preference_patterns = [p for p in patterns if p.pattern_type == 'preference']
        
        for pattern in preference_patterns:
            preferred_caps = pattern.pattern_data.get('preferred_capabilities', [])
            
            if 'neural_mesh_analysis' in preferred_caps:
                recommendations.append(AdaptiveRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    type='content',
                    title='Advanced Analytics Tutorials',
                    description='Personalized tutorials for advanced neural mesh analytics features',
                    reasoning='User shows strong preference for neural mesh analysis capabilities',
                    confidence=pattern.confidence,
                    priority='low',
                    implementation_complexity='simple',
                    expected_benefit='Improved capability utilization and user satisfaction'
                ))
        
        return recommendations

# Initialize engines
pattern_engine = PatternRecognitionEngine()
insight_engine = EmergentInsightEngine()
recommendation_engine = AdaptiveRecommendationEngine()

# API Endpoints
@router.post("/analyze-interaction")
async def analyze_user_interaction(
    user_id: str,
    interaction_data: Dict[str, Any]
):
    """Analyze user interaction for pattern recognition"""
    try:
        # Analyze patterns
        patterns = await pattern_engine.analyze_user_interaction(user_id, interaction_data)
        
        # Store patterns
        user_patterns[user_id].extend(patterns)
        
        # Keep only recent patterns (last 100 per user)
        if len(user_patterns[user_id]) > 100:
            user_patterns[user_id] = user_patterns[user_id][-100:]
        
        # Generate insights
        insights = await insight_engine.generate_insights(patterns)
        emergent_insights.extend(insights)
        
        # Generate recommendations
        recommendations = await recommendation_engine.generate_recommendations(user_id, patterns, insights)
        user_recommendations[user_id].extend(recommendations)
        
        return {
            "patterns_detected": len(patterns),
            "insights_generated": len(insights),
            "recommendations_created": len(recommendations),
            "patterns": [p.dict() for p in patterns],
            "insights": [i.dict() for i in insights],
            "recommendations": [r.dict() for r in recommendations]
        }
        
    except Exception as e:
        log.error(f"Error analyzing user interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-patterns/{user_id}")
async def get_user_patterns(user_id: str):
    """Get user's interaction patterns"""
    try:
        patterns = user_patterns.get(user_id, [])
        return {
            "user_id": user_id,
            "pattern_count": len(patterns),
            "patterns": [p.dict() for p in patterns[-20:]]  # Last 20 patterns
        }
    except Exception as e:
        log.error(f"Error getting user patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights")
async def get_emergent_insights(limit: int = 10):
    """Get recent emergent insights"""
    try:
        recent_insights = sorted(emergent_insights, key=lambda x: x.created_at, reverse=True)[:limit]
        return {
            "insight_count": len(emergent_insights),
            "recent_insights": [i.dict() for i in recent_insights]
        }
    except Exception as e:
        log.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    """Get adaptive recommendations for user"""
    try:
        recommendations = user_recommendations.get(user_id, [])
        active_recommendations = [r for r in recommendations if r.priority in ['high', 'critical']]
        
        return {
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "active_recommendations": [r.dict() for r in active_recommendations]
        }
    except Exception as e:
        log.error(f"Error getting user recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_user_feedback(
    user_id: str,
    interaction_id: str,
    feedback_data: Dict[str, Any]
):
    """Submit user feedback for learning"""
    try:
        # Process feedback for pattern learning
        satisfaction_score = feedback_data.get('satisfaction', 0.7)
        usefulness_score = feedback_data.get('usefulness', 0.7)
        accuracy_score = feedback_data.get('accuracy', 0.7)
        
        # Create feedback interaction data
        feedback_interaction = {
            'message': feedback_data.get('original_request', ''),
            'capabilities_used': feedback_data.get('capabilities_used', []),
            'success_rate': accuracy_score,
            'user_satisfaction': satisfaction_score,
            'response_time': feedback_data.get('response_time', 1.0),
            'complexity_level': feedback_data.get('complexity_level', 0.5),
            'session_duration': feedback_data.get('session_duration', 300)
        }
        
        # Analyze feedback patterns
        patterns = await pattern_engine.analyze_user_interaction(user_id, feedback_interaction)
        user_patterns[user_id].extend(patterns)
        
        return {
            "feedback_processed": True,
            "patterns_updated": len(patterns),
            "learning_impact": "User preferences and satisfaction patterns updated"
        }
        
    except Exception as e:
        log.error(f"Error processing user feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@router.get("/health")
async def emergent_intelligence_health():
    """Health check for emergent intelligence system"""
    return {
        "status": "healthy",
        "total_patterns": sum(len(patterns) for patterns in user_patterns.values()),
        "total_insights": len(emergent_insights),
        "total_recommendations": sum(len(recs) for recs in user_recommendations.values()),
        "active_users": len(user_patterns),
        "timestamp": time.time()
    }
