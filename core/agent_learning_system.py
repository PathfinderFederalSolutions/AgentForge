#!/usr/bin/env python3
"""
Agent Learning and Continuous Improvement System for AgentForge
Feedback loops, performance metrics, A/B testing, and behavior analytics
"""

import asyncio
import json
import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

log = logging.getLogger("agent-learning-system")

class FeedbackType(Enum):
    """Types of feedback"""
    HUMAN = "human"
    AUTOMATED = "automated"
    PEER_AGENT = "peer_agent"
    SYSTEM = "system"

class MetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    COST = "cost"
    USER_SATISFACTION = "user_satisfaction"
    TASK_COMPLETION = "task_completion"
    COLLABORATION = "collaboration"

@dataclass
class FeedbackRecord:
    """Record of feedback received"""
    feedback_id: str
    agent_id: str
    task_id: str
    feedback_type: FeedbackType
    feedback_source: str
    rating: float  # 0.0 to 1.0
    comments: str
    specific_aspects: Dict[str, float] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerformanceMetric:
    """Performance metric for an agent"""
    agent_id: str
    metric_type: MetricType
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    measurement_period: str = "instant"  # instant, hourly, daily, weekly

@dataclass
class BehaviorPattern:
    """Identified behavior pattern"""
    pattern_id: str
    agent_id: str
    pattern_type: str
    description: str
    frequency: float
    impact_score: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)

@dataclass
class ImprovementAction:
    """Action taken for improvement"""
    action_id: str
    agent_id: str
    improvement_type: str
    description: str
    implemented_at: float
    expected_impact: float
    actual_impact: Optional[float] = None
    success: Optional[bool] = None

class AgentLearningSystem:
    """Comprehensive agent learning and improvement system"""
    
    def __init__(self):
        self.feedback_records: List[FeedbackRecord] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.behavior_patterns: Dict[str, BehaviorPattern] = {}
        self.improvement_actions: List[ImprovementAction] = []
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Learning models
        self.performance_predictors: Dict[str, Any] = {}
        self.behavior_analyzers: Dict[str, Any] = {}
        
        # Neural mesh integration
        self.neural_mesh = None
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components"""
        try:
            # Neural mesh integration
            try:
                from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
                self.neural_mesh = EnhancedNeuralMesh()
                await self.neural_mesh.initialize()
            except ImportError:
                log.warning("Neural mesh not available for learning system")
            
            # Load historical data
            await self._load_historical_data()
            
            # Start continuous learning processes
            asyncio.create_task(self._continuous_behavior_analysis())
            asyncio.create_task(self._continuous_improvement_monitoring())
            
            log.info("✅ Agent learning system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize learning system: {e}")
    
    async def record_feedback(
        self,
        agent_id: str,
        task_id: str,
        feedback_type: FeedbackType,
        feedback_source: str,
        rating: float,
        comments: str = "",
        specific_aspects: Dict[str, float] = None,
        improvement_suggestions: List[str] = None
    ) -> str:
        """Record feedback for an agent"""
        
        feedback_id = self._generate_feedback_id()
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            agent_id=agent_id,
            task_id=task_id,
            feedback_type=feedback_type,
            feedback_source=feedback_source,
            rating=rating,
            comments=comments,
            specific_aspects=specific_aspects or {},
            improvement_suggestions=improvement_suggestions or []
        )
        
        self.feedback_records.append(feedback)
        
        # Store in neural mesh
        if self.neural_mesh:
            await self.neural_mesh.store_knowledge(
                agent_id=agent_id,
                knowledge_type="feedback",
                data={
                    "feedback_id": feedback_id,
                    "rating": rating,
                    "comments": comments,
                    "specific_aspects": specific_aspects,
                    "improvement_suggestions": improvement_suggestions,
                    "feedback_source": feedback_source
                },
                memory_tier="L3"
            )
        
        # Trigger immediate learning if feedback is significant
        if rating < 0.5 or improvement_suggestions:
            await self._trigger_immediate_learning(agent_id, feedback)
        
        log.info(f"Recorded feedback {feedback_id} for agent {agent_id}")
        return feedback_id
    
    async def record_performance_metric(
        self,
        agent_id: str,
        metric_type: MetricType,
        value: float,
        context: Dict[str, Any] = None,
        measurement_period: str = "instant"
    ):
        """Record performance metric for an agent"""
        
        metric = PerformanceMetric(
            agent_id=agent_id,
            metric_type=metric_type,
            value=value,
            context=context or {},
            measurement_period=measurement_period
        )
        
        self.performance_metrics.append(metric)
        
        # Store in neural mesh
        if self.neural_mesh:
            await self.neural_mesh.store_knowledge(
                agent_id=agent_id,
                knowledge_type="performance_metric",
                data={
                    "metric_type": metric_type.value,
                    "value": value,
                    "context": context,
                    "measurement_period": measurement_period
                },
                memory_tier="L2"
            )
        
        # Check for performance anomalies
        await self._check_performance_anomalies(agent_id, metric)
    
    async def analyze_agent_behavior(
        self,
        agent_id: str,
        analysis_window: int = 86400  # 24 hours
    ) -> Dict[str, Any]:
        """Analyze agent behavior patterns"""
        
        try:
            # Get recent metrics and feedback
            cutoff_time = time.time() - analysis_window
            
            recent_metrics = [
                m for m in self.performance_metrics
                if m.agent_id == agent_id and m.timestamp > cutoff_time
            ]
            
            recent_feedback = [
                f for f in self.feedback_records
                if f.agent_id == agent_id and f.timestamp > cutoff_time
            ]
            
            # Analyze patterns
            behavior_analysis = {
                "agent_id": agent_id,
                "analysis_period": analysis_window,
                "total_metrics": len(recent_metrics),
                "total_feedback": len(recent_feedback),
                "patterns": [],
                "trends": {},
                "anomalies": [],
                "recommendations": []
            }
            
            # Performance trends
            if recent_metrics:
                behavior_analysis["trends"] = self._analyze_performance_trends(recent_metrics)
            
            # Feedback analysis
            if recent_feedback:
                behavior_analysis["feedback_analysis"] = self._analyze_feedback_patterns(recent_feedback)
            
            # Identify behavior patterns
            patterns = await self._identify_behavior_patterns(agent_id, recent_metrics, recent_feedback)
            behavior_analysis["patterns"] = patterns
            
            # Generate recommendations
            recommendations = await self._generate_improvement_recommendations(
                agent_id, behavior_analysis
            )
            behavior_analysis["recommendations"] = recommendations
            
            return behavior_analysis
            
        except Exception as e:
            log.error(f"Error analyzing agent behavior: {e}")
            return {"error": str(e), "agent_id": agent_id}
    
    async def start_ab_test(
        self,
        test_name: str,
        agent_ids: List[str],
        test_variants: Dict[str, Any],
        success_metric: str = "user_satisfaction",
        test_duration: int = 86400  # 24 hours
    ) -> str:
        """Start A/B test for agent improvements"""
        
        test_id = self._generate_test_id(test_name)
        
        # Randomly assign agents to variants
        import random
        variant_names = list(test_variants.keys())
        agents_per_variant = len(agent_ids) // len(variant_names)
        
        agent_assignments = {}
        for i, agent_id in enumerate(agent_ids):
            variant_index = i // agents_per_variant
            if variant_index >= len(variant_names):
                variant_index = len(variant_names) - 1
            
            variant = variant_names[variant_index]
            agent_assignments[agent_id] = variant
        
        # Store test configuration
        self.ab_tests[test_id] = {
            "test_name": test_name,
            "test_variants": test_variants,
            "agent_assignments": agent_assignments,
            "success_metric": success_metric,
            "start_time": time.time(),
            "end_time": time.time() + test_duration,
            "status": "active",
            "results": {}
        }
        
        # Apply variants to agents
        for agent_id, variant in agent_assignments.items():
            await self._apply_test_variant(agent_id, variant, test_variants[variant])
        
        log.info(f"Started A/B test {test_id} with {len(agent_ids)} agents")
        return test_id
    
    async def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test = self.ab_tests[test_id]
        
        # Collect metrics for each variant
        variant_results = {}
        
        for variant in test["test_variants"].keys():
            variant_agents = [
                agent_id for agent_id, assigned_variant in test["agent_assignments"].items()
                if assigned_variant == variant
            ]
            
            # Get metrics for variant agents during test period
            variant_metrics = [
                m for m in self.performance_metrics
                if m.agent_id in variant_agents
                and test["start_time"] <= m.timestamp <= test.get("end_time", time.time())
                and m.metric_type.value == test["success_metric"]
            ]
            
            if variant_metrics:
                values = [m.value for m in variant_metrics]
                variant_results[variant] = {
                    "agent_count": len(variant_agents),
                    "metric_count": len(variant_metrics),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values)
                }
        
        # Statistical analysis
        analysis = {
            "test_id": test_id,
            "test_name": test["test_name"],
            "status": test["status"],
            "duration": time.time() - test["start_time"],
            "variant_results": variant_results,
            "statistical_significance": self._calculate_statistical_significance(variant_results),
            "winner": self._determine_test_winner(variant_results),
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["statistical_significance"] > 0.95:
            winner = analysis["winner"]
            if winner:
                analysis["recommendations"].append(f"Deploy variant '{winner}' - statistically significant improvement")
        elif analysis["statistical_significance"] > 0.8:
            analysis["recommendations"].append("Results promising but need more data for confidence")
        else:
            analysis["recommendations"].append("No significant difference detected - continue testing or try new variants")
        
        return analysis
    
    async def collect_fine_tuning_data(
        self,
        agent_id: str,
        data_type: str = "successful_interactions",
        min_quality_score: float = 0.8,
        max_samples: int = 1000
    ) -> List[Dict[str, Any]]:
        """Collect data for fine-tuning agent models"""
        
        try:
            training_data = []
            
            if data_type == "successful_interactions":
                # Get high-quality interactions from neural mesh
                if self.neural_mesh:
                    context = await self.neural_mesh.get_context(
                        agent_id=agent_id,
                        query=f"agent:{agent_id} success:true quality:>{min_quality_score}",
                        memory_tiers=["L2", "L3"]
                    )
                    
                    if context and context.get("relevant_knowledge"):
                        for knowledge in context["relevant_knowledge"][:max_samples]:
                            if knowledge.get("quality_score", 0) >= min_quality_score:
                                training_sample = {
                                    "input": knowledge.get("input", ""),
                                    "output": knowledge.get("output", ""),
                                    "context": knowledge.get("context", {}),
                                    "quality_score": knowledge.get("quality_score", 0),
                                    "timestamp": knowledge.get("timestamp", time.time())
                                }
                                training_data.append(training_sample)
            
            elif data_type == "error_corrections":
                # Get error correction examples
                error_corrections = [
                    f for f in self.feedback_records
                    if f.agent_id == agent_id and f.improvement_suggestions
                ]
                
                for feedback in error_corrections[:max_samples]:
                    training_sample = {
                        "original_response": feedback.comments,
                        "improved_response": feedback.improvement_suggestions[0] if feedback.improvement_suggestions else "",
                        "error_type": "correction",
                        "quality_improvement": feedback.rating,
                        "timestamp": feedback.timestamp
                    }
                    training_data.append(training_sample)
            
            # Store collected data
            if self.neural_mesh and training_data:
                await self.neural_mesh.store_knowledge(
                    agent_id=agent_id,
                    knowledge_type="fine_tuning_data",
                    data={
                        "data_type": data_type,
                        "sample_count": len(training_data),
                        "quality_threshold": min_quality_score,
                        "collection_timestamp": time.time()
                    },
                    memory_tier="L4"
                )
            
            log.info(f"Collected {len(training_data)} fine-tuning samples for agent {agent_id}")
            return training_data
            
        except Exception as e:
            log.error(f"Error collecting fine-tuning data: {e}")
            return []
    
    async def implement_continuous_improvement(
        self,
        agent_id: str,
        improvement_threshold: float = 0.1,
        analysis_window: int = 86400
    ) -> Dict[str, Any]:
        """Implement continuous improvement for an agent"""
        
        try:
            # Analyze current performance
            behavior_analysis = await self.analyze_agent_behavior(agent_id, analysis_window)
            
            # Identify improvement opportunities
            improvements = []
            
            # Check performance trends
            if "trends" in behavior_analysis:
                for metric, trend in behavior_analysis["trends"].items():
                    if trend.get("slope", 0) < -improvement_threshold:
                        improvements.append({
                            "type": "performance_decline",
                            "metric": metric,
                            "decline_rate": trend["slope"],
                            "suggested_action": "investigate_and_retrain"
                        })
            
            # Check feedback patterns
            if "feedback_analysis" in behavior_analysis:
                avg_rating = behavior_analysis["feedback_analysis"].get("average_rating", 0.5)
                if avg_rating < 0.6:
                    improvements.append({
                        "type": "low_satisfaction",
                        "rating": avg_rating,
                        "suggested_action": "analyze_feedback_and_adjust"
                    })
            
            # Implement improvements
            implemented_actions = []
            for improvement in improvements:
                action = await self._implement_improvement_action(agent_id, improvement)
                if action:
                    implemented_actions.append(action)
            
            # Schedule follow-up analysis
            asyncio.create_task(
                self._schedule_improvement_followup(agent_id, implemented_actions, 3600)  # 1 hour
            )
            
            return {
                "agent_id": agent_id,
                "improvements_identified": len(improvements),
                "actions_implemented": len(implemented_actions),
                "next_analysis": time.time() + analysis_window,
                "improvements": improvements,
                "actions": implemented_actions
            }
            
        except Exception as e:
            log.error(f"Error implementing continuous improvement: {e}")
            return {"error": str(e), "agent_id": agent_id}
    
    async def _trigger_immediate_learning(self, agent_id: str, feedback: FeedbackRecord):
        """Trigger immediate learning from significant feedback"""
        
        try:
            # Analyze what went wrong
            if feedback.rating < 0.5:
                # Poor performance - immediate analysis needed
                analysis = await self._analyze_poor_performance(agent_id, feedback)
                
                # Implement quick fixes
                for suggestion in feedback.improvement_suggestions:
                    await self._implement_quick_fix(agent_id, suggestion)
            
            # Update agent's learning state in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=agent_id,
                    knowledge_type="immediate_learning",
                    data={
                        "trigger_feedback": feedback.feedback_id,
                        "learning_actions": feedback.improvement_suggestions,
                        "urgency": "high" if feedback.rating < 0.3 else "medium",
                        "timestamp": time.time()
                    },
                    memory_tier="L1"  # Immediate memory
                )
            
        except Exception as e:
            log.error(f"Error in immediate learning: {e}")
    
    async def _continuous_behavior_analysis(self):
        """Continuously analyze agent behaviors"""
        
        while True:
            try:
                # Analyze behavior for all agents
                active_agents = set(m.agent_id for m in self.performance_metrics[-1000:])  # Recent agents
                
                for agent_id in active_agents:
                    await self._detect_behavior_patterns(agent_id)
                
                # Sleep for analysis interval
                await asyncio.sleep(3600)  # Analyze every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in continuous behavior analysis: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _continuous_improvement_monitoring(self):
        """Monitor improvement actions and their effectiveness"""
        
        while True:
            try:
                # Check improvement actions that need evaluation
                for action in self.improvement_actions:
                    if (action.actual_impact is None and 
                        time.time() - action.implemented_at > 3600):  # 1 hour after implementation
                        
                        await self._evaluate_improvement_action(action)
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in improvement monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _detect_behavior_patterns(self, agent_id: str):
        """Detect behavior patterns for an agent"""
        
        try:
            # Get recent performance data
            recent_metrics = [
                m for m in self.performance_metrics
                if m.agent_id == agent_id and m.timestamp > time.time() - 86400
            ]
            
            if len(recent_metrics) < 10:  # Need minimum data
                return
            
            # Analyze patterns by metric type
            for metric_type in MetricType:
                type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                
                if len(type_metrics) >= 5:
                    pattern = await self._analyze_metric_pattern(agent_id, metric_type, type_metrics)
                    if pattern:
                        self.behavior_patterns[pattern.pattern_id] = pattern
            
        except Exception as e:
            log.error(f"Error detecting behavior patterns: {e}")
    
    async def _analyze_metric_pattern(
        self,
        agent_id: str,
        metric_type: MetricType,
        metrics: List[PerformanceMetric]
    ) -> Optional[BehaviorPattern]:
        """Analyze pattern in specific metric type"""
        
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Calculate trend
        if len(values) >= 3:
            # Simple linear regression for trend
            x = np.array(range(len(values)))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            
            # Detect significant patterns
            if abs(slope) > 0.05:  # Significant trend
                pattern_type = "improving" if slope > 0 else "declining"
                
                pattern = BehaviorPattern(
                    pattern_id=f"{agent_id}_{metric_type.value}_{pattern_type}",
                    agent_id=agent_id,
                    pattern_type=f"{metric_type.value}_{pattern_type}",
                    description=f"Agent shows {pattern_type} trend in {metric_type.value}",
                    frequency=len(metrics) / (max(timestamps) - min(timestamps)) * 86400,  # per day
                    impact_score=abs(slope),
                    examples=[
                        {"value": m.value, "timestamp": m.timestamp, "context": m.context}
                        for m in metrics[-3:]  # Last 3 examples
                    ],
                    first_observed=min(timestamps),
                    last_observed=max(timestamps)
                )
                
                return pattern
        
        return None
    
    def _analyze_performance_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends"""
        
        trends = {}
        
        # Group by metric type
        by_type = {}
        for metric in metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric)
        
        # Analyze each type
        for metric_type, type_metrics in by_type.items():
            if len(type_metrics) >= 3:
                values = [m.value for m in sorted(type_metrics, key=lambda x: x.timestamp)]
                
                # Calculate trend
                x = np.array(range(len(values)))
                y = np.array(values)
                slope, intercept = np.polyfit(x, y, 1)
                
                trends[metric_type.value] = {
                    "slope": slope,
                    "direction": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
                    "current_value": values[-1],
                    "change_rate": slope,
                    "sample_count": len(values)
                }
        
        return trends
    
    def _analyze_feedback_patterns(self, feedback: List[FeedbackRecord]) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        
        if not feedback:
            return {}
        
        ratings = [f.rating for f in feedback]
        
        # Feedback by type
        by_type = {}
        for f in feedback:
            if f.feedback_type not in by_type:
                by_type[f.feedback_type] = []
            by_type[f.feedback_type].append(f.rating)
        
        type_analysis = {}
        for feedback_type, type_ratings in by_type.items():
            type_analysis[feedback_type.value] = {
                "count": len(type_ratings),
                "average": statistics.mean(type_ratings),
                "trend": "positive" if statistics.mean(type_ratings) > 0.7 else "negative" if statistics.mean(type_ratings) < 0.5 else "neutral"
            }
        
        return {
            "total_feedback": len(feedback),
            "average_rating": statistics.mean(ratings),
            "rating_trend": "improving" if ratings[-1] > ratings[0] else "declining",
            "by_type": type_analysis,
            "improvement_suggestions_count": sum(len(f.improvement_suggestions) for f in feedback)
        }
    
    async def _generate_improvement_recommendations(
        self,
        agent_id: str,
        behavior_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations based on behavior analysis"""
        
        recommendations = []
        
        # Check trends
        if "trends" in behavior_analysis:
            for metric, trend in behavior_analysis["trends"].items():
                if trend["direction"] == "declining" and trend["slope"] < -0.1:
                    recommendations.append(f"Address declining {metric} performance")
        
        # Check feedback
        if "feedback_analysis" in behavior_analysis:
            feedback = behavior_analysis["feedback_analysis"]
            if feedback.get("average_rating", 0.5) < 0.6:
                recommendations.append("Improve response quality based on user feedback")
            
            if feedback.get("improvement_suggestions_count", 0) > 5:
                recommendations.append("Review and implement common improvement suggestions")
        
        # Check patterns
        for pattern in behavior_analysis.get("patterns", []):
            if "declining" in pattern.get("pattern_type", ""):
                recommendations.append(f"Address {pattern['pattern_type']} pattern")
        
        return recommendations
    
    async def _implement_improvement_action(
        self,
        agent_id: str,
        improvement: Dict[str, Any]
    ) -> Optional[ImprovementAction]:
        """Implement specific improvement action"""
        
        try:
            action_id = self._generate_action_id()
            
            action = ImprovementAction(
                action_id=action_id,
                agent_id=agent_id,
                improvement_type=improvement["type"],
                description=improvement["suggested_action"],
                implemented_at=time.time(),
                expected_impact=0.1  # Default expected improvement
            )
            
            # Implement based on improvement type
            if improvement["type"] == "performance_decline":
                await self._implement_performance_improvement(agent_id, improvement)
            elif improvement["type"] == "low_satisfaction":
                await self._implement_satisfaction_improvement(agent_id, improvement)
            
            self.improvement_actions.append(action)
            
            # Store in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=agent_id,
                    knowledge_type="improvement_action",
                    data={
                        "action_id": action_id,
                        "improvement_type": improvement["type"],
                        "description": improvement["suggested_action"],
                        "expected_impact": action.expected_impact
                    },
                    memory_tier="L3"
                )
            
            return action
            
        except Exception as e:
            log.error(f"Error implementing improvement action: {e}")
            return None
    
    async def _apply_test_variant(
        self,
        agent_id: str,
        variant_name: str,
        variant_config: Dict[str, Any]
    ):
        """Apply A/B test variant to an agent"""
        
        # Store variant configuration in neural mesh
        if self.neural_mesh:
            await self.neural_mesh.store_knowledge(
                agent_id=agent_id,
                knowledge_type="ab_test_variant",
                data={
                    "variant_name": variant_name,
                    "variant_config": variant_config,
                    "applied_at": time.time()
                },
                memory_tier="L1"
            )
        
        log.info(f"Applied A/B test variant '{variant_name}' to agent {agent_id}")
    
    def _calculate_statistical_significance(
        self,
        variant_results: Dict[str, Any]
    ) -> float:
        """Calculate statistical significance of A/B test"""
        
        if len(variant_results) < 2:
            return 0.0
        
        # Simplified statistical significance calculation
        # In production, would use proper statistical tests (t-test, chi-square, etc.)
        
        means = [result["mean"] for result in variant_results.values()]
        sample_sizes = [result["metric_count"] for result in variant_results.values()]
        
        # Effect size
        effect_size = (max(means) - min(means)) / (sum(means) / len(means))
        
        # Sample size factor
        min_sample_size = min(sample_sizes)
        sample_factor = min(min_sample_size / 30.0, 1.0)  # 30 is minimum for decent confidence
        
        # Combined significance
        significance = min(effect_size * sample_factor * 2, 1.0)
        
        return significance
    
    def _determine_test_winner(self, variant_results: Dict[str, Any]) -> Optional[str]:
        """Determine winner of A/B test"""
        
        if not variant_results:
            return None
        
        # Find variant with highest mean performance
        best_variant = max(
            variant_results.keys(),
            key=lambda v: variant_results[v]["mean"]
        )
        
        return best_variant
    
    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID"""
        return f"feedback_{int(time.time())}_{hash(time.time()) % 10000}"
    
    def _generate_test_id(self, test_name: str) -> str:
        """Generate unique test ID"""
        return f"test_{test_name}_{int(time.time())}"
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        return f"action_{int(time.time())}_{hash(time.time()) % 10000}"
    
    async def get_agent_learning_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive learning summary for an agent"""
        
        # Get recent feedback
        recent_feedback = [
            f for f in self.feedback_records
            if f.agent_id == agent_id and f.timestamp > time.time() - 86400
        ]
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.performance_metrics
            if m.agent_id == agent_id and m.timestamp > time.time() - 86400
        ]
        
        # Get behavior patterns
        agent_patterns = [
            p for p in self.behavior_patterns.values()
            if p.agent_id == agent_id
        ]
        
        # Get improvement actions
        agent_improvements = [
            a for a in self.improvement_actions
            if a.agent_id == agent_id
        ]
        
        return {
            "agent_id": agent_id,
            "learning_summary": {
                "feedback_received": len(recent_feedback),
                "average_feedback_rating": statistics.mean([f.rating for f in recent_feedback]) if recent_feedback else 0.0,
                "performance_metrics_recorded": len(recent_metrics),
                "behavior_patterns_identified": len(agent_patterns),
                "improvement_actions_taken": len(agent_improvements),
                "learning_velocity": len(recent_feedback) + len(recent_metrics),
                "last_learning_event": max([f.timestamp for f in recent_feedback] + [m.timestamp for m in recent_metrics]) if (recent_feedback or recent_metrics) else 0
            },
            "current_patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "impact_score": p.impact_score
                }
                for p in agent_patterns
            ],
            "recent_improvements": [
                {
                    "improvement_type": a.improvement_type,
                    "description": a.description,
                    "expected_impact": a.expected_impact,
                    "actual_impact": a.actual_impact,
                    "success": a.success
                }
                for a in agent_improvements[-5:]  # Last 5 improvements
            ]
        }

# Global instance
learning_system = AgentLearningSystem()
