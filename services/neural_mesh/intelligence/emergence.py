"""
Emergent Intelligence Engine - DEPRECATED
This file is deprecated. Use streaming_analytics.py for new implementations.
Provides backward compatibility for existing code.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import numpy as np
from collections import defaultdict, deque

# Import from new consolidated types
from ..core.memory_types import PatternType, PatternStrength, Interaction, Pattern, Knowledge

# Optional imports with fallbacks
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
except ImportError:
    DBSCAN = None
    KMeans = None
    cosine_similarity = None
    PCA = None

try:
    import networkx as nx
except ImportError:
    nx = None

log = logging.getLogger("emergent-intelligence-deprecated")

@dataclass
class EmergenceMetrics:
    """Metrics for tracking emergent intelligence"""
    pattern_count: int = 0
    knowledge_synthesis_rate: float = 0.0
    emergence_score: float = 0.0
    complexity_level: float = 0.0
    coherence_score: float = 0.0
    novelty_score: float = 0.0

def _deprecation_warning(func_name: str):
    """Issue deprecation warning"""
    warnings.warn(
        f"{func_name} is deprecated. Use streaming_analytics.py for new implementations.",
        DeprecationWarning,
        stacklevel=3
    )

class AdvancedPatternDetector:
    """Advanced ML-based pattern detection system"""
    
    def __init__(self, embedding_dimension: int = 768):
        self.embedding_dimension = embedding_dimension
        self.interaction_history: deque = deque(maxlen=10000)  # Keep last 10k interactions
        self.pattern_cache: Dict[str, Pattern] = {}
        self.temporal_windows = [60, 300, 900, 3600]  # 1min, 5min, 15min, 1hour
        
    async def analyze_interactions(self, interactions: List[Interaction]) -> List[Pattern]:
        """Analyze interactions to detect complex patterns"""
        if not interactions:
            return []
            
        # Add to history
        for interaction in interactions:
            self.interaction_history.append(interaction)
            
        patterns = []
        
        # Run different pattern detection algorithms
        patterns.extend(await self._detect_temporal_patterns(interactions))
        patterns.extend(await self._detect_collaboration_patterns(interactions))
        patterns.extend(await self._detect_performance_patterns(interactions))
        patterns.extend(await self._detect_learning_patterns(interactions))
        patterns.extend(await self._detect_anomaly_patterns(interactions))
        
        if cosine_similarity and DBSCAN:
            patterns.extend(await self._detect_clustering_patterns(interactions))
            
        if nx:
            patterns.extend(await self._detect_network_patterns(interactions))
            
        # Filter and deduplicate patterns
        return self._filter_patterns(patterns)
        
    async def _detect_temporal_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect time-based patterns"""
        patterns = []
        
        for window_size in self.temporal_windows:
            current_time = time.time()
            window_start = current_time - window_size
            
            # Get interactions in time window
            window_interactions = [
                i for i in interactions 
                if i.timestamp >= window_start
            ]
            
            if len(window_interactions) < 3:
                continue
                
            # Analyze interaction frequency
            agent_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            for interaction in window_interactions:
                agent_counts[interaction.agent_id] += 1
                type_counts[interaction.interaction_type] += 1
                
            # Detect high-frequency patterns
            max_agent_count = max(agent_counts.values()) if agent_counts else 0
            max_type_count = max(type_counts.values()) if type_counts else 0
            
            if max_agent_count > len(window_interactions) * 0.5:  # One agent dominates
                dominant_agent = max(agent_counts, key=agent_counts.get)
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("temporal_dominant", dominant_agent, window_size),
                    pattern_type=PatternType.TEMPORAL,
                    strength=PatternStrength.STRONG,
                    confidence=0.8,
                    agents_involved={dominant_agent},
                    interactions=[i.interaction_id for i in window_interactions if i.agent_id == dominant_agent],
                    metadata={
                        "window_size": window_size,
                        "frequency": max_agent_count,
                        "pattern_subtype": "dominant_agent"
                    }
                )
                patterns.append(pattern)
                
            if max_type_count > len(window_interactions) * 0.6:  # One type dominates
                dominant_type = max(type_counts, key=type_counts.get)
                involved_agents = {i.agent_id for i in window_interactions if i.interaction_type == dominant_type}
                
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("temporal_type", dominant_type, window_size),
                    pattern_type=PatternType.TEMPORAL,
                    strength=PatternStrength.MODERATE,
                    confidence=0.7,
                    agents_involved=involved_agents,
                    interactions=[i.interaction_id for i in window_interactions if i.interaction_type == dominant_type],
                    metadata={
                        "window_size": window_size,
                        "frequency": max_type_count,
                        "pattern_subtype": "dominant_type",
                        "interaction_type": dominant_type
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    async def _detect_collaboration_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect agent collaboration patterns"""
        patterns = []
        
        # Group interactions by time windows
        time_window = 300  # 5 minutes
        current_time = time.time()
        
        # Get recent interactions
        recent_interactions = [
            i for i in interactions 
            if i.timestamp >= current_time - time_window
        ]
        
        if len(recent_interactions) < 2:
            return patterns
            
        # Analyze agent co-occurrence
        agent_pairs = defaultdict(int)
        agent_interactions = defaultdict(list)
        
        for interaction in recent_interactions:
            agent_interactions[interaction.agent_id].append(interaction)
            
        # Find agents that interact frequently together
        for agent1 in agent_interactions:
            for agent2 in agent_interactions:
                if agent1 != agent2:
                    # Check for temporal proximity of interactions
                    agent1_times = [i.timestamp for i in agent_interactions[agent1]]
                    agent2_times = [i.timestamp for i in agent_interactions[agent2]]
                    
                    # Count interactions within 30 seconds of each other
                    close_interactions = 0
                    for t1 in agent1_times:
                        for t2 in agent2_times:
                            if abs(t1 - t2) <= 30:  # 30 second window
                                close_interactions += 1
                                
                    if close_interactions >= 2:
                        pair_key = tuple(sorted([agent1, agent2]))
                        agent_pairs[pair_key] += close_interactions
                        
        # Create collaboration patterns
        for (agent1, agent2), count in agent_pairs.items():
            if count >= 3:  # Threshold for collaboration
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("collaboration", f"{agent1}_{agent2}"),
                    pattern_type=PatternType.COLLABORATION,
                    strength=PatternStrength.STRONG if count >= 5 else PatternStrength.MODERATE,
                    confidence=min(0.9, 0.5 + count * 0.1),
                    agents_involved={agent1, agent2},
                    interactions=[
                        i.interaction_id for i in recent_interactions 
                        if i.agent_id in {agent1, agent2}
                    ],
                    metadata={
                        "collaboration_count": count,
                        "time_window": time_window,
                        "pattern_subtype": "pair_collaboration"
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    async def _detect_performance_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect performance optimization patterns"""
        patterns = []
        
        # Analyze task execution times
        task_executions = [
            i for i in interactions 
            if i.interaction_type == "task_execution" and "duration" in i.data
        ]
        
        if len(task_executions) < 5:
            return patterns
            
        # Group by agent
        agent_performance = defaultdict(list)
        for interaction in task_executions:
            duration = interaction.data.get("duration", 0)
            agent_performance[interaction.agent_id].append(duration)
            
        # Find performance trends
        for agent_id, durations in agent_performance.items():
            if len(durations) < 3:
                continue
                
            # Calculate trend (simple linear regression)
            x = np.arange(len(durations))
            y = np.array(durations)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Detect improving performance (negative slope)
                if slope < -0.1:  # Improving by 0.1 units per task
                    pattern = Pattern(
                        pattern_id=self._generate_pattern_id("performance", agent_id, "improving"),
                        pattern_type=PatternType.PERFORMANCE_OPTIMIZATION,
                        strength=PatternStrength.STRONG if slope < -0.5 else PatternStrength.MODERATE,
                        confidence=0.8,
                        agents_involved={agent_id},
                        interactions=[i.interaction_id for i in task_executions if i.agent_id == agent_id],
                        metadata={
                            "trend_slope": slope,
                            "improvement_rate": -slope,
                            "pattern_subtype": "performance_improvement",
                            "sample_size": len(durations)
                        }
                    )
                    patterns.append(pattern)
                    
                # Detect degrading performance (positive slope)
                elif slope > 0.1:  # Degrading by 0.1 units per task
                    pattern = Pattern(
                        pattern_id=self._generate_pattern_id("performance", agent_id, "degrading"),
                        pattern_type=PatternType.ANOMALY,
                        strength=PatternStrength.CRITICAL if slope > 0.5 else PatternStrength.MODERATE,
                        confidence=0.7,
                        agents_involved={agent_id},
                        interactions=[i.interaction_id for i in task_executions if i.agent_id == agent_id],
                        metadata={
                            "trend_slope": slope,
                            "degradation_rate": slope,
                            "pattern_subtype": "performance_degradation",
                            "sample_size": len(durations)
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_learning_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect learning and adaptation patterns"""
        patterns = []
        
        # Analyze memory access patterns for learning indicators
        memory_accesses = [
            i for i in interactions 
            if i.interaction_type == "memory_access" and "query" in i.data
        ]
        
        if len(memory_accesses) < 5:
            return patterns
            
        # Group by agent
        agent_queries = defaultdict(list)
        for interaction in memory_accesses:
            query = interaction.data.get("query", "")
            agent_queries[interaction.agent_id].append((query, interaction.timestamp))
            
        # Detect query evolution (learning)
        for agent_id, queries in agent_queries.items():
            if len(queries) < 3:
                continue
                
            # Sort by timestamp
            queries.sort(key=lambda x: x[1])
            
            # Analyze query complexity over time
            query_lengths = [len(q[0]) for q in queries]
            
            if len(query_lengths) > 2:
                # Check if queries are getting more complex (learning)
                recent_avg = np.mean(query_lengths[-3:])
                early_avg = np.mean(query_lengths[:3])
                
                if recent_avg > early_avg * 1.5:  # 50% increase in complexity
                    pattern = Pattern(
                        pattern_id=self._generate_pattern_id("learning", agent_id, "complexity"),
                        pattern_type=PatternType.LEARNING,
                        strength=PatternStrength.MODERATE,
                        confidence=0.7,
                        agents_involved={agent_id},
                        interactions=[i.interaction_id for i in memory_accesses if i.agent_id == agent_id],
                        metadata={
                            "complexity_increase": recent_avg / early_avg,
                            "early_avg_length": early_avg,
                            "recent_avg_length": recent_avg,
                            "pattern_subtype": "query_complexity_growth",
                            "sample_size": len(queries)
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_anomaly_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect anomalous behavior patterns"""
        patterns = []
        
        if len(interactions) < 10:
            return patterns
            
        # Analyze interaction frequency anomalies
        agent_frequencies = defaultdict(int)
        type_frequencies = defaultdict(int)
        
        for interaction in interactions:
            agent_frequencies[interaction.agent_id] += 1
            type_frequencies[interaction.interaction_type] += 1
            
        # Statistical analysis for anomalies
        agent_counts = list(agent_frequencies.values())
        type_counts = list(type_frequencies.values())
        
        if len(agent_counts) > 2:
            agent_mean = np.mean(agent_counts)
            agent_std = np.std(agent_counts)
            
            # Detect agents with anomalous activity levels
            for agent_id, count in agent_frequencies.items():
                z_score = (count - agent_mean) / (agent_std + 1e-8)
                
                if abs(z_score) > 2.5:  # More than 2.5 standard deviations
                    anomaly_type = "hyperactive" if z_score > 0 else "inactive"
                    
                    pattern = Pattern(
                        pattern_id=self._generate_pattern_id("anomaly", agent_id, anomaly_type),
                        pattern_type=PatternType.ANOMALY,
                        strength=PatternStrength.CRITICAL if abs(z_score) > 3 else PatternStrength.STRONG,
                        confidence=min(0.9, abs(z_score) / 3.0),
                        agents_involved={agent_id},
                        interactions=[i.interaction_id for i in interactions if i.agent_id == agent_id],
                        metadata={
                            "z_score": z_score,
                            "activity_count": count,
                            "expected_count": agent_mean,
                            "anomaly_type": anomaly_type,
                            "pattern_subtype": "activity_anomaly"
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_clustering_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect patterns using clustering algorithms (requires scikit-learn)"""
        patterns = []
        
        if not interactions or not DBSCAN:
            return patterns
            
        # Prepare embeddings for clustering
        embeddings = []
        interaction_map = {}
        
        for i, interaction in enumerate(interactions):
            if interaction.embedding is not None:
                embeddings.append(interaction.embedding)
                interaction_map[i] = interaction
                
        if len(embeddings) < 5:
            return patterns
            
        embeddings = np.array(embeddings)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=3)
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
            if len(cluster_indices) < 3:
                continue
                
            cluster_interactions = [interaction_map[i] for i in cluster_indices]
            cluster_agents = {i.agent_id for i in cluster_interactions}
            
            if len(cluster_agents) > 1:  # Multi-agent cluster
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("clustering", f"cluster_{label}"),
                    pattern_type=PatternType.COLLABORATION,
                    strength=PatternStrength.MODERATE,
                    confidence=0.6,
                    agents_involved=cluster_agents,
                    interactions=[i.interaction_id for i in cluster_interactions],
                    metadata={
                        "cluster_label": int(label),
                        "cluster_size": len(cluster_interactions),
                        "agent_count": len(cluster_agents),
                        "pattern_subtype": "embedding_cluster"
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    async def _detect_network_patterns(self, interactions: List[Interaction]) -> List[Pattern]:
        """Detect patterns using network analysis (requires networkx)"""
        patterns = []
        
        if not interactions or not nx:
            return patterns
            
        # Build interaction graph
        G = nx.Graph()
        
        # Add nodes (agents)
        agents = {i.agent_id for i in interactions}
        G.add_nodes_from(agents)
        
        # Add edges (interactions between agents in time windows)
        time_window = 60  # 1 minute
        
        for i, interaction1 in enumerate(interactions):
            for j, interaction2 in enumerate(interactions[i+1:], i+1):
                if (interaction1.agent_id != interaction2.agent_id and
                    abs(interaction1.timestamp - interaction2.timestamp) <= time_window):
                    
                    if G.has_edge(interaction1.agent_id, interaction2.agent_id):
                        G[interaction1.agent_id][interaction2.agent_id]['weight'] += 1
                    else:
                        G.add_edge(interaction1.agent_id, interaction2.agent_id, weight=1)
                        
        if G.number_of_edges() == 0:
            return patterns
            
        # Analyze network properties
        try:
            # Find highly connected nodes (hubs)
            centrality = nx.degree_centrality(G)
            max_centrality = max(centrality.values()) if centrality else 0
            
            if max_centrality > 0.7:  # Highly central agent
                hub_agent = max(centrality, key=centrality.get)
                
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("network", hub_agent, "hub"),
                    pattern_type=PatternType.COLLABORATION,
                    strength=PatternStrength.STRONG,
                    confidence=0.8,
                    agents_involved={hub_agent},
                    interactions=[i.interaction_id for i in interactions if i.agent_id == hub_agent],
                    metadata={
                        "centrality_score": centrality[hub_agent],
                        "connected_agents": len([n for n in G.neighbors(hub_agent)]),
                        "pattern_subtype": "network_hub"
                    }
                )
                patterns.append(pattern)
                
            # Find communities
            if len(G.nodes()) > 3:
                try:
                    communities = list(nx.community.greedy_modularity_communities(G))
                    
                    for i, community in enumerate(communities):
                        if len(community) >= 3:  # Significant community
                            pattern = Pattern(
                                pattern_id=self._generate_pattern_id("network", f"community_{i}"),
                                pattern_type=PatternType.COLLABORATION,
                                strength=PatternStrength.MODERATE,
                                confidence=0.7,
                                agents_involved=set(community),
                                interactions=[
                                    inter.interaction_id for inter in interactions 
                                    if inter.agent_id in community
                                ],
                                metadata={
                                    "community_size": len(community),
                                    "community_id": i,
                                    "pattern_subtype": "network_community"
                                }
                            )
                            patterns.append(pattern)
                            
                except Exception as e:
                    log.debug(f"Community detection failed: {e}")
                    
        except Exception as e:
            log.debug(f"Network analysis failed: {e}")
            
        return patterns
        
    def _filter_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Filter and deduplicate patterns"""
        # Remove duplicates based on pattern_id
        unique_patterns = {}
        for pattern in patterns:
            if pattern.pattern_id not in unique_patterns:
                unique_patterns[pattern.pattern_id] = pattern
            else:
                # Keep the one with higher confidence
                existing = unique_patterns[pattern.pattern_id]
                if pattern.confidence > existing.confidence:
                    unique_patterns[pattern.pattern_id] = pattern
                    
        # Filter by minimum confidence
        filtered_patterns = [
            p for p in unique_patterns.values() 
            if p.confidence >= 0.5
        ]
        
        # Sort by confidence
        filtered_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_patterns
        
    def _generate_pattern_id(self, pattern_type: str, *args) -> str:
        """Generate unique pattern ID"""
        components = [pattern_type] + [str(arg) for arg in args]
        content = ":".join(components)
        return hashlib.md5(content.encode()).hexdigest()[:12]

class AdvancedKnowledgeSynthesizer:
    """Advanced knowledge synthesis from detected patterns"""
    
    def __init__(self):
        self.synthesis_rules = self._load_synthesis_rules()
        self.knowledge_cache: Dict[str, Knowledge] = {}
        
    async def synthesize(self, patterns: List[Pattern]) -> List[Knowledge]:
        """Synthesize knowledge from multiple patterns"""
        if not patterns:
            return []
            
        knowledge_items = []
        
        # Group patterns by type for synthesis
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)
            
        # Apply synthesis rules
        for rule in self.synthesis_rules:
            knowledge = await self._apply_synthesis_rule(rule, patterns, pattern_groups)
            if knowledge:
                knowledge_items.extend(knowledge)
                
        # Cross-pattern synthesis
        cross_knowledge = await self._cross_pattern_synthesis(patterns)
        knowledge_items.extend(cross_knowledge)
        
        return knowledge_items
        
    async def _apply_synthesis_rule(
        self, 
        rule: Dict[str, Any], 
        all_patterns: List[Pattern],
        pattern_groups: Dict[PatternType, List[Pattern]]
    ) -> List[Knowledge]:
        """Apply a specific synthesis rule"""
        knowledge_items = []
        
        applicable_patterns = []
        for pattern_type in rule.get("applicable_patterns", []):
            if isinstance(pattern_type, str):
                try:
                    pattern_type = PatternType(pattern_type)
                except ValueError:
                    continue
            applicable_patterns.extend(pattern_groups.get(pattern_type, []))
            
        if len(applicable_patterns) < rule.get("min_patterns", 1):
            return knowledge_items
            
        # Filter by confidence threshold
        high_confidence_patterns = [
            p for p in applicable_patterns 
            if p.confidence >= rule.get("min_confidence", 0.6)
        ]
        
        if len(high_confidence_patterns) < rule.get("min_patterns", 1):
            return knowledge_items
            
        # Generate knowledge
        knowledge_id = self._generate_knowledge_id(rule["name"], high_confidence_patterns)
        
        # Check if already synthesized
        if knowledge_id in self.knowledge_cache:
            return [self.knowledge_cache[knowledge_id]]
            
        # Create knowledge content
        content = await self._generate_knowledge_content(rule, high_confidence_patterns)
        
        knowledge = Knowledge(
            knowledge_id=knowledge_id,
            content=content,
            knowledge_type=rule["knowledge_type"],
            source_patterns=[p.pattern_id for p in high_confidence_patterns],
            confidence=np.mean([p.confidence for p in high_confidence_patterns]),
            applicable_contexts=rule.get("contexts", ["general"]),
            metadata={
                "synthesis_rule": rule["name"],
                "pattern_count": len(high_confidence_patterns),
                "agent_count": len(set().union(*[p.agents_involved for p in high_confidence_patterns]))
            }
        )
        
        self.knowledge_cache[knowledge_id] = knowledge
        knowledge_items.append(knowledge)
        
        return knowledge_items
        
    async def _cross_pattern_synthesis(self, patterns: List[Pattern]) -> List[Knowledge]:
        """Synthesize knowledge from cross-pattern relationships"""
        knowledge_items = []
        
        # Find patterns involving same agents
        agent_patterns = defaultdict(list)
        for pattern in patterns:
            for agent in pattern.agents_involved:
                agent_patterns[agent].append(pattern)
                
        # Synthesize agent-specific insights
        for agent_id, agent_pattern_list in agent_patterns.items():
            if len(agent_pattern_list) >= 2:
                knowledge = await self._synthesize_agent_insights(agent_id, agent_pattern_list)
                if knowledge:
                    knowledge_items.append(knowledge)
                    
        # Find temporal correlations
        temporal_knowledge = await self._synthesize_temporal_insights(patterns)
        knowledge_items.extend(temporal_knowledge)
        
        return knowledge_items
        
    async def _synthesize_agent_insights(self, agent_id: str, patterns: List[Pattern]) -> Optional[Knowledge]:
        """Synthesize insights about specific agent behavior"""
        if len(patterns) < 2:
            return None
            
        # Analyze pattern types for this agent
        pattern_types = [p.pattern_type for p in patterns]
        type_counts = defaultdict(int)
        for pt in pattern_types:
            type_counts[pt] += 1
            
        # Generate insight based on dominant patterns
        dominant_type = max(type_counts, key=type_counts.get)
        
        knowledge_id = self._generate_knowledge_id("agent_insight", [agent_id, dominant_type.value])
        
        if dominant_type == PatternType.COLLABORATION:
            content = f"Agent {agent_id} demonstrates strong collaborative behavior patterns, frequently engaging in coordinated activities with other agents."
        elif dominant_type == PatternType.PERFORMANCE_OPTIMIZATION:
            content = f"Agent {agent_id} shows consistent performance optimization patterns, indicating adaptive learning capabilities."
        elif dominant_type == PatternType.ANOMALY:
            content = f"Agent {agent_id} exhibits anomalous behavior patterns that may require attention or investigation."
        else:
            content = f"Agent {agent_id} shows distinctive {dominant_type.value} behavior patterns across multiple interactions."
            
        return Knowledge(
            knowledge_id=knowledge_id,
            content=content,
            knowledge_type="agent_behavior",
            source_patterns=[p.pattern_id for p in patterns],
            confidence=np.mean([p.confidence for p in patterns]),
            applicable_contexts=["agent_management", "performance_monitoring"],
            metadata={
                "agent_id": agent_id,
                "pattern_types": [pt.value for pt in pattern_types],
                "dominant_pattern": dominant_type.value
            }
        )
        
    async def _synthesize_temporal_insights(self, patterns: List[Pattern]) -> List[Knowledge]:
        """Synthesize temporal insights from patterns"""
        knowledge_items = []
        
        if len(patterns) < 3:
            return knowledge_items
            
        # Sort patterns by discovery time
        sorted_patterns = sorted(patterns, key=lambda p: p.discovered_at)
        
        # Analyze temporal trends
        time_window = 3600  # 1 hour
        current_time = time.time()
        
        recent_patterns = [p for p in sorted_patterns if p.discovered_at >= current_time - time_window]
        
        if len(recent_patterns) >= 3:
            # Analyze pattern evolution
            pattern_types = [p.pattern_type for p in recent_patterns]
            
            if PatternType.PERFORMANCE_OPTIMIZATION in pattern_types:
                knowledge = Knowledge(
                    knowledge_id=self._generate_knowledge_id("temporal", "performance_trend"),
                    content="Recent patterns indicate system-wide performance optimization trends, suggesting effective learning and adaptation across the agent swarm.",
                    knowledge_type="system_trend",
                    source_patterns=[p.pattern_id for p in recent_patterns if p.pattern_type == PatternType.PERFORMANCE_OPTIMIZATION],
                    confidence=0.8,
                    applicable_contexts=["system_optimization", "performance_monitoring"],
                    metadata={
                        "time_window": time_window,
                        "pattern_count": len(recent_patterns)
                    }
                )
                knowledge_items.append(knowledge)
                
        return knowledge_items
        
    async def _generate_knowledge_content(self, rule: Dict[str, Any], patterns: List[Pattern]) -> str:
        """Generate knowledge content from rule and patterns"""
        template = rule.get("template", "Detected {pattern_count} patterns involving {agent_count} agents.")
        
        # Extract template variables
        all_agents = set()
        for pattern in patterns:
            all_agents.update(pattern.agents_involved)
            
        variables = {
            "pattern_count": len(patterns),
            "agent_count": len(all_agents),
            "agents": ", ".join(sorted(all_agents)),
            "avg_confidence": np.mean([p.confidence for p in patterns]),
            "pattern_types": ", ".join(set(p.pattern_type.value for p in patterns))
        }
        
        try:
            return template.format(**variables)
        except KeyError as e:
            log.warning(f"Template variable {e} not available, using default content")
            return f"Synthesized knowledge from {len(patterns)} patterns involving {len(all_agents)} agents."
            
    def _generate_knowledge_id(self, prefix: str, source_data: Any) -> str:
        """Generate unique knowledge ID"""
        if isinstance(source_data, list):
            if source_data and hasattr(source_data[0], 'pattern_id'):
                # List of patterns
                content = f"{prefix}:{':'.join(p.pattern_id for p in source_data)}"
            else:
                # List of strings or other data
                content = f"{prefix}:{':'.join(str(item) for item in source_data)}"
        else:
            content = f"{prefix}:{source_data}"
            
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def _load_synthesis_rules(self) -> List[Dict[str, Any]]:
        """Load knowledge synthesis rules"""
        return [
            {
                "name": "multi_agent_collaboration",
                "knowledge_type": "collaboration_pattern",
                "applicable_patterns": ["collaboration", "agent_communication"],
                "min_patterns": 2,
                "min_confidence": 0.6,
                "template": "Detected collaborative behavior among {agent_count} agents with {pattern_count} coordination patterns. Average confidence: {avg_confidence:.2f}",
                "contexts": ["collaboration", "teamwork", "coordination"]
            },
            {
                "name": "performance_optimization_trend",
                "knowledge_type": "performance_insight",
                "applicable_patterns": ["performance_optimization"],
                "min_patterns": 2,
                "min_confidence": 0.7,
                "template": "Identified performance optimization trends across {agent_count} agents, indicating systematic improvement in task execution efficiency.",
                "contexts": ["optimization", "performance", "learning"]
            },
            {
                "name": "anomaly_cluster",
                "knowledge_type": "anomaly_insight",
                "applicable_patterns": ["anomaly"],
                "min_patterns": 2,
                "min_confidence": 0.6,
                "template": "Detected anomalous behavior cluster involving {agent_count} agents, requiring investigation and potential intervention.",
                "contexts": ["anomaly_detection", "system_health", "monitoring"]
            },
            {
                "name": "learning_progression",
                "knowledge_type": "learning_insight",
                "applicable_patterns": ["learning"],
                "min_patterns": 1,
                "min_confidence": 0.7,
                "template": "Observed learning progression patterns in {agent_count} agents, demonstrating adaptive capability improvement over time.",
                "contexts": ["learning", "adaptation", "development"]
            },
            {
                "name": "temporal_coordination",
                "knowledge_type": "temporal_insight",
                "applicable_patterns": ["temporal", "collaboration"],
                "min_patterns": 2,
                "min_confidence": 0.6,
                "template": "Identified temporal coordination patterns suggesting synchronized behavior across {agent_count} agents.",
                "contexts": ["synchronization", "temporal_patterns", "coordination"]
            }
        ]

class EmergentIntelligence:
    """DEPRECATED: Main emergent intelligence engine - Use StreamingEmergentIntelligence instead"""
    
    def __init__(self, neural_mesh: Any):  # Any to avoid circular import
        _deprecation_warning("EmergentIntelligence")
        self.neural_mesh = neural_mesh
        self.pattern_detector = AdvancedPatternDetector()
        self.knowledge_synthesizer = AdvancedKnowledgeSynthesizer()
        self.interaction_buffer: deque = deque(maxlen=10000)
        self.knowledge_history: List[Knowledge] = []
        self.analysis_interval = 60  # Analyze every minute
        self.last_analysis = 0
        
    async def record_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Record agent interaction for pattern analysis"""
        # Create interaction object
        interaction = Interaction(
            interaction_id=interaction_data.get("interaction_id", self._generate_interaction_id()),
            agent_id=interaction_data["agent_id"],
            interaction_type=interaction_data["type"],
            timestamp=interaction_data.get("timestamp", time.time()),
            data=interaction_data.get("data", {}),
            metadata=interaction_data.get("metadata", {})
        )
        
        # Generate embedding if content available
        if "content" in interaction_data:
            try:
                # Use neural mesh embedder if available
                if hasattr(self.neural_mesh, 'embedder'):
                    embedding_result = await self.neural_mesh.embedder.encode(
                        interaction_data["content"], 
                        "text"
                    )
                    interaction.embedding = embedding_result.embedding
            except Exception as e:
                log.debug(f"Failed to generate interaction embedding: {e}")
                
        self.interaction_buffer.append(interaction)
        
        # Trigger analysis if enough time has passed
        current_time = time.time()
        if current_time - self.last_analysis >= self.analysis_interval:
            asyncio.create_task(self.analyze_and_synthesize())
            
    async def analyze_and_synthesize(self) -> List[Knowledge]:
        """Analyze patterns and synthesize new knowledge"""
        self.last_analysis = time.time()
        
        if len(self.interaction_buffer) < 10:
            return []
            
        try:
            # Convert buffer to list for analysis
            interactions = list(self.interaction_buffer)
            
            # Detect patterns
            log.info(f"Analyzing {len(interactions)} interactions for pattern detection")
            patterns = await self.pattern_detector.analyze_interactions(interactions)
            
            log.info(f"Detected {len(patterns)} patterns")
            
            # Synthesize knowledge from patterns
            if patterns:
                knowledge_items = await self.knowledge_synthesizer.synthesize(patterns)
                
                log.info(f"Synthesized {len(knowledge_items)} knowledge items")
                
                # Store knowledge in history
                self.knowledge_history.extend(knowledge_items)
                
                # Propagate new knowledge through the mesh
                for knowledge in knowledge_items:
                    try:
                        await self.neural_mesh.propagate_knowledge(knowledge)
                    except Exception as e:
                        log.error(f"Failed to propagate knowledge {knowledge.knowledge_id}: {e}")
                        
                return knowledge_items
            else:
                return []
                
        except Exception as e:
            log.error(f"Error in emergent intelligence analysis: {e}")
            return []
            
    async def get_insights(self, context: Optional[str] = None) -> List[Knowledge]:
        """Get relevant insights for a given context"""
        if not context:
            return self.knowledge_history[-10:]  # Return recent insights
            
        # Filter by applicable contexts
        relevant_insights = [
            k for k in self.knowledge_history
            if context in k.applicable_contexts or "general" in k.applicable_contexts
        ]
        
        # Sort by confidence and recency
        relevant_insights.sort(
            key=lambda x: (x.confidence, x.created_at), 
            reverse=True
        )
        
        return relevant_insights[:10]
        
    async def get_agent_insights(self, agent_id: str) -> List[Knowledge]:
        """Get insights specific to an agent"""
        agent_insights = [
            k for k in self.knowledge_history
            if agent_id in k.metadata.get("agent_id", "") or
               agent_id in k.content
        ]
        
        agent_insights.sort(key=lambda x: x.created_at, reverse=True)
        return agent_insights[:5]
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health insights from emergent intelligence"""
        current_time = time.time()
        recent_cutoff = current_time - 3600  # Last hour
        
        recent_knowledge = [
            k for k in self.knowledge_history
            if k.created_at >= recent_cutoff
        ]
        
        # Analyze knowledge types
        knowledge_types = defaultdict(int)
        for k in recent_knowledge:
            knowledge_types[k.knowledge_type] += 1
            
        # Check for anomalies
        anomaly_count = knowledge_types.get("anomaly_insight", 0)
        performance_count = knowledge_types.get("performance_insight", 0)
        collaboration_count = knowledge_types.get("collaboration_pattern", 0)
        
        health_score = min(100, max(0, 100 - (anomaly_count * 10) + (performance_count * 5) + (collaboration_count * 3)))
        
        return {
            "health_score": health_score,
            "recent_insights": len(recent_knowledge),
            "anomaly_alerts": anomaly_count,
            "performance_improvements": performance_count,
            "collaboration_patterns": collaboration_count,
            "knowledge_types": dict(knowledge_types),
            "buffer_utilization": len(self.interaction_buffer) / self.interaction_buffer.maxlen,
            "last_analysis": self.last_analysis
        }
        
    def _generate_interaction_id(self) -> str:
        """Generate unique interaction ID"""
        return hashlib.sha256(f"{time.time()}:{np.random.random()}".encode()).hexdigest()[:12]
