"""
Streaming Analytics Engine - Scalable Pattern Detection for Million-Agent Deployments
Replaces O(n²) algorithms with streaming analytics and sampling strategies
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import heapq
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterator
from enum import Enum
import numpy as np
from collections import defaultdict, deque, Counter
import threading
from concurrent.futures import ThreadPoolExecutor

# Import base classes
from ..core.memory_types import Interaction, Pattern, Knowledge, PatternType, PatternStrength

# Optional imports for advanced streaming
try:
    from scipy import sparse
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    sparse = None
    chi2_contingency = None
    SCIPY_AVAILABLE = False

try:
    import mmh3  # MurmurHash for consistent hashing
    MMH3_AVAILABLE = True
except ImportError:
    mmh3 = None
    MMH3_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = lambda *args, **kwargs: None

log = logging.getLogger("streaming-analytics")

class StreamingWindow(Enum):
    """Time window types for streaming analysis"""
    SLIDING = "sliding"
    TUMBLING = "tumbling"
    SESSION = "session"
    LANDMARK = "landmark"

@dataclass
class StreamingSample:
    """Sample for reservoir sampling"""
    interaction: Interaction
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
class ReservoirSampler:
    """Reservoir sampling for streaming data"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.samples: List[StreamingSample] = []
        self.count = 0
        self.lock = threading.Lock()
    
    def add_sample(self, interaction: Interaction, weight: float = 1.0):
        """Add sample using reservoir sampling algorithm"""
        with self.lock:
            self.count += 1
            
            if len(self.samples) < self.capacity:
                # Reservoir not full, add directly
                self.samples.append(StreamingSample(interaction, weight))
            else:
                # Replace random element with probability k/n
                j = random.randint(1, self.count)
                if j <= self.capacity:
                    self.samples[j - 1] = StreamingSample(interaction, weight)
    
    def get_samples(self) -> List[Interaction]:
        """Get current samples"""
        with self.lock:
            return [sample.interaction for sample in self.samples]
    
    def get_weighted_samples(self) -> List[Tuple[Interaction, float]]:
        """Get samples with weights"""
        with self.lock:
            return [(sample.interaction, sample.weight) for sample in self.samples]
    
    def clear(self):
        """Clear all samples"""
        with self.lock:
            self.samples.clear()
            self.count = 0

class CountMinSketch:
    """Count-Min Sketch for frequency estimation"""
    
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[callable]:
        """Generate hash functions for the sketch"""
        functions = []
        for i in range(self.depth):
            # Use different seeds for each hash function
            seed = random.randint(1, 2**31 - 1)
            if MMH3_AVAILABLE:
                functions.append(lambda x, s=seed: mmh3.hash(str(x), s) % self.width)
            else:
                functions.append(lambda x, s=seed: hash(f"{x}:{s}") % self.width)
        return functions
    
    def update(self, item: str, count: int = 1):
        """Update count for an item"""
        for i, hash_func in enumerate(self.hash_functions):
            j = hash_func(item)
            self.table[i, j] += count
    
    def query(self, item: str) -> int:
        """Query count for an item"""
        counts = []
        for i, hash_func in enumerate(self.hash_functions):
            j = hash_func(item)
            counts.append(self.table[i, j])
        return min(counts)  # Return minimum estimate
    
    def heavy_hitters(self, threshold: int) -> List[Tuple[str, int]]:
        """Get items with count above threshold (approximate)"""
        # This is a simplified version - production would need more sophisticated approach
        candidates = set()
        
        # Find potential heavy hitters from sketch
        for i in range(self.depth):
            for j in range(self.width):
                if self.table[i, j] >= threshold:
                    # This is approximate - would need to track actual items
                    candidates.add(f"candidate_{i}_{j}")
        
        # Return with estimated counts
        return [(item, self.query(item)) for item in candidates]

class HyperLogLog:
    """HyperLogLog for cardinality estimation"""
    
    def __init__(self, precision: int = 12):
        self.precision = precision
        self.m = 2 ** precision
        self.buckets = np.zeros(self.m, dtype=np.int8)
        self.alpha = self._get_alpha()
    
    def _get_alpha(self) -> float:
        """Get alpha constant for bias correction"""
        if self.m >= 128:
            return 0.7213 / (1 + 1.079 / self.m)
        elif self.m >= 64:
            return 0.709
        elif self.m >= 32:
            return 0.697
        else:
            return 0.673
    
    def add(self, item: str):
        """Add item to the set"""
        if MMH3_AVAILABLE:
            hash_val = mmh3.hash(str(item), signed=False)
        else:
            hash_val = hash(str(item)) & 0xFFFFFFFF
        
        # Get bucket index (first p bits)
        bucket = hash_val & (self.m - 1)
        
        # Get remaining bits and count leading zeros
        w = hash_val >> self.precision
        leading_zeros = self._count_leading_zeros(w) + 1
        
        # Update bucket with maximum
        self.buckets[bucket] = max(self.buckets[bucket], leading_zeros)
    
    def _count_leading_zeros(self, w: int) -> int:
        """Count leading zeros in binary representation"""
        if w == 0:
            return 32 - self.precision
        
        count = 0
        for i in range(32 - self.precision - 1, -1, -1):
            if (w >> i) & 1:
                break
            count += 1
        return count
    
    def cardinality(self) -> int:
        """Estimate cardinality"""
        raw_estimate = self.alpha * (self.m ** 2) / np.sum(2 ** (-self.buckets))
        
        # Apply bias correction for small ranges
        if raw_estimate <= 2.5 * self.m:
            zeros = np.sum(self.buckets == 0)
            if zeros != 0:
                return int(self.m * np.log(self.m / zeros))
        
        # Apply correction for large ranges
        if raw_estimate <= (1.0/30.0) * (2 ** 32):
            return int(raw_estimate)
        else:
            return int(-2 ** 32 * np.log(1 - raw_estimate / (2 ** 32)))

class StreamingPatternDetector:
    """Streaming pattern detector with O(1) per-item complexity"""
    
    def __init__(self, sample_size: int = 10000, sketch_width: int = 10000):
        self.sample_size = sample_size
        self.sketch_width = sketch_width
        
        # Sampling and sketching components
        self.reservoir_sampler = ReservoirSampler(sample_size)
        self.frequency_sketch = CountMinSketch(sketch_width, 7)
        self.cardinality_estimator = HyperLogLog(14)
        
        # Time-based windows
        self.sliding_windows = {
            60: deque(maxlen=1000),    # 1 minute
            300: deque(maxlen=1000),   # 5 minutes  
            900: deque(maxlen=1000),   # 15 minutes
            3600: deque(maxlen=1000)   # 1 hour
        }
        
        # Pattern state tracking
        self.agent_activity = defaultdict(lambda: {"count": 0, "last_seen": 0})
        self.interaction_pairs = defaultdict(int)
        self.temporal_buckets = defaultdict(list)
        
        # Anomaly detection state
        self.baseline_stats = {}
        self.anomaly_threshold = 3.0  # Standard deviations
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        if METRICS_AVAILABLE:
            self.streaming_operations = Counter(
                'streaming_pattern_operations_total',
                'Streaming pattern detection operations',
                ['operation', 'status']
            )
            self.pattern_detection_latency = Histogram(
                'streaming_pattern_detection_latency_seconds',
                'Pattern detection latency',
                ['pattern_type']
            )
            self.sample_reservoir_size = Gauge(
                'streaming_sample_reservoir_size',
                'Current size of sample reservoir'
            )
    
    async def process_interaction(self, interaction: Interaction) -> List[Pattern]:
        """Process single interaction with streaming algorithms"""
        start_time = time.time()
        patterns = []
        
        try:
            # Update streaming data structures
            await self._update_streaming_state(interaction)
            
            # Detect patterns using streaming algorithms
            patterns.extend(await self._detect_frequency_patterns(interaction))
            patterns.extend(await self._detect_temporal_patterns(interaction))
            patterns.extend(await self._detect_collaboration_patterns(interaction))
            patterns.extend(await self._detect_anomaly_patterns(interaction))
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.streaming_operations.labels(
                    operation="process_interaction",
                    status="success"
                ).inc()
                
                for pattern in patterns:
                    self.pattern_detection_latency.labels(
                        pattern_type=pattern.pattern_type.value
                    ).observe(time.time() - start_time)
            
            return patterns
            
        except Exception as e:
            log.error(f"Error processing interaction {interaction.interaction_id}: {e}")
            
            if METRICS_AVAILABLE:
                self.streaming_operations.labels(
                    operation="process_interaction",
                    status="error"
                ).inc()
            
            return []
    
    async def _update_streaming_state(self, interaction: Interaction):
        """Update streaming data structures"""
        current_time = time.time()
        
        with self.lock:
            # Add to reservoir sample
            self.reservoir_sampler.add_sample(interaction)
            
            # Update frequency sketch
            agent_key = f"agent:{interaction.agent_id}"
            type_key = f"type:{interaction.interaction_type}"
            self.frequency_sketch.update(agent_key)
            self.frequency_sketch.update(type_key)
            
            # Update cardinality estimator
            self.cardinality_estimator.add(interaction.agent_id)
            
            # Update time windows
            for window_size in self.sliding_windows:
                window = self.sliding_windows[window_size]
                
                # Add current interaction
                window.append(interaction)
                
                # Remove old interactions (sliding window)
                while window and current_time - window[0].timestamp > window_size:
                    window.popleft()
            
            # Update agent activity
            self.agent_activity[interaction.agent_id]["count"] += 1
            self.agent_activity[interaction.agent_id]["last_seen"] = current_time
            
            # Update temporal buckets (for temporal pattern detection)
            time_bucket = int(current_time // 60)  # 1-minute buckets
            self.temporal_buckets[time_bucket].append(interaction)
            
            # Clean old temporal buckets
            old_buckets = [
                bucket for bucket in self.temporal_buckets.keys()
                if bucket < time_bucket - 60  # Keep last hour
            ]
            for bucket in old_buckets:
                del self.temporal_buckets[bucket]
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.sample_reservoir_size.set(len(self.reservoir_sampler.samples))
    
    async def _detect_frequency_patterns(self, interaction: Interaction) -> List[Pattern]:
        """Detect frequency-based patterns using Count-Min Sketch"""
        patterns = []
        
        try:
            agent_key = f"agent:{interaction.agent_id}"
            type_key = f"type:{interaction.interaction_type}"
            
            agent_freq = self.frequency_sketch.query(agent_key)
            type_freq = self.frequency_sketch.query(type_key)
            
            # Detect high-frequency agents
            if agent_freq > 100:  # Threshold for high activity
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("frequency_agent", interaction.agent_id),
                    pattern_type=PatternType.ANOMALY,
                    strength=PatternStrength.STRONG if agent_freq > 500 else PatternStrength.MODERATE,
                    confidence=min(0.9, agent_freq / 1000),
                    agents_involved={interaction.agent_id},
                    interactions=[interaction.interaction_id],
                    metadata={
                        "frequency": agent_freq,
                        "pattern_subtype": "high_frequency_agent",
                        "detection_method": "count_min_sketch"
                    }
                )
                patterns.append(pattern)
            
            # Detect high-frequency interaction types
            if type_freq > 200:  # Threshold for type frequency
                pattern = Pattern(
                    pattern_id=self._generate_pattern_id("frequency_type", interaction.interaction_type),
                    pattern_type=PatternType.TEMPORAL,
                    strength=PatternStrength.MODERATE,
                    confidence=min(0.8, type_freq / 500),
                    agents_involved={interaction.agent_id},
                    interactions=[interaction.interaction_id],
                    metadata={
                        "frequency": type_freq,
                        "interaction_type": interaction.interaction_type,
                        "pattern_subtype": "high_frequency_type",
                        "detection_method": "count_min_sketch"
                    }
                )
                patterns.append(pattern)
                
        except Exception as e:
            log.debug(f"Error in frequency pattern detection: {e}")
        
        return patterns
    
    async def _detect_temporal_patterns(self, interaction: Interaction) -> List[Pattern]:
        """Detect temporal patterns using sliding windows"""
        patterns = []
        
        try:
            current_time = time.time()
            
            # Analyze different time windows
            for window_size, window in self.sliding_windows.items():
                if len(window) < 10:  # Need minimum interactions
                    continue
                
                # Count interactions by agent in this window
                agent_counts = defaultdict(int)
                for inter in window:
                    agent_counts[inter.agent_id] += 1
                
                # Detect temporal bursts
                total_interactions = len(window)
                expected_per_agent = total_interactions / max(1, len(agent_counts))
                
                for agent_id, count in agent_counts.items():
                    if count > expected_per_agent * 3:  # 3x above average
                        pattern = Pattern(
                            pattern_id=self._generate_pattern_id("temporal_burst", agent_id, window_size),
                            pattern_type=PatternType.TEMPORAL,
                            strength=PatternStrength.STRONG if count > expected_per_agent * 5 else PatternStrength.MODERATE,
                            confidence=min(0.9, count / (expected_per_agent * 5)),
                            agents_involved={agent_id},
                            interactions=[i.interaction_id for i in window if i.agent_id == agent_id],
                            metadata={
                                "window_size": window_size,
                                "burst_count": count,
                                "expected_count": expected_per_agent,
                                "burst_ratio": count / expected_per_agent,
                                "pattern_subtype": "temporal_burst",
                                "detection_method": "sliding_window"
                            }
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            log.debug(f"Error in temporal pattern detection: {e}")
        
        return patterns
    
    async def _detect_collaboration_patterns(self, interaction: Interaction) -> List[Pattern]:
        """Detect collaboration patterns using approximate algorithms"""
        patterns = []
        
        try:
            current_time = time.time()
            
            # Use 5-minute window for collaboration detection
            window = self.sliding_windows.get(300, deque())
            if len(window) < 5:
                return patterns
            
            # Build co-occurrence matrix (approximate)
            agent_cooccurrence = defaultdict(int)
            
            # Sample interactions to reduce complexity
            sample_size = min(100, len(window))
            sampled_interactions = random.sample(list(window), sample_size)
            
            # Find interactions that occur close in time
            for i, inter1 in enumerate(sampled_interactions):
                for j, inter2 in enumerate(sampled_interactions[i+1:], i+1):
                    if (inter1.agent_id != inter2.agent_id and 
                        abs(inter1.timestamp - inter2.timestamp) <= 30):  # 30 second window
                        
                        agent_pair = tuple(sorted([inter1.agent_id, inter2.agent_id]))
                        agent_cooccurrence[agent_pair] += 1
            
            # Detect significant collaborations
            for (agent1, agent2), count in agent_cooccurrence.items():
                if count >= 3:  # Threshold for collaboration
                    pattern = Pattern(
                        pattern_id=self._generate_pattern_id("collaboration", f"{agent1}_{agent2}"),
                        pattern_type=PatternType.COLLABORATION,
                        strength=PatternStrength.STRONG if count >= 5 else PatternStrength.MODERATE,
                        confidence=min(0.8, count / 10),
                        agents_involved={agent1, agent2},
                        interactions=[inter.interaction_id for inter in sampled_interactions 
                                    if inter.agent_id in {agent1, agent2}],
                        metadata={
                            "collaboration_count": count,
                            "sample_size": sample_size,
                            "pattern_subtype": "temporal_collaboration",
                            "detection_method": "sampled_cooccurrence"
                        }
                    )
                    patterns.append(pattern)
                    
        except Exception as e:
            log.debug(f"Error in collaboration pattern detection: {e}")
        
        return patterns
    
    async def _detect_anomaly_patterns(self, interaction: Interaction) -> List[Pattern]:
        """Detect anomalies using streaming statistics"""
        patterns = []
        
        try:
            current_time = time.time()
            agent_id = interaction.agent_id
            
            # Update baseline statistics
            if agent_id not in self.baseline_stats:
                self.baseline_stats[agent_id] = {
                    "interaction_times": deque(maxlen=100),
                    "mean_interval": 0,
                    "std_interval": 0,
                    "last_update": current_time
                }
            
            stats = self.baseline_stats[agent_id]
            
            # Add current interaction time
            if stats["interaction_times"]:
                interval = current_time - stats["interaction_times"][-1]
                stats["interaction_times"].append(current_time)
                
                # Update running statistics
                if len(stats["interaction_times"]) >= 10:
                    intervals = [
                        stats["interaction_times"][i] - stats["interaction_times"][i-1]
                        for i in range(1, len(stats["interaction_times"]))
                    ]
                    
                    stats["mean_interval"] = np.mean(intervals)
                    stats["std_interval"] = np.std(intervals)
                    
                    # Detect anomalous intervals
                    if stats["std_interval"] > 0:
                        z_score = abs(interval - stats["mean_interval"]) / stats["std_interval"]
                        
                        if z_score > self.anomaly_threshold:
                            anomaly_type = "unusually_fast" if interval < stats["mean_interval"] else "unusually_slow"
                            
                            pattern = Pattern(
                                pattern_id=self._generate_pattern_id("anomaly", agent_id, anomaly_type),
                                pattern_type=PatternType.ANOMALY,
                                strength=PatternStrength.CRITICAL if z_score > 4 else PatternStrength.STRONG,
                                confidence=min(0.95, z_score / 5),
                                agents_involved={agent_id},
                                interactions=[interaction.interaction_id],
                                metadata={
                                    "z_score": z_score,
                                    "interval": interval,
                                    "mean_interval": stats["mean_interval"],
                                    "std_interval": stats["std_interval"],
                                    "anomaly_type": anomaly_type,
                                    "pattern_subtype": "timing_anomaly",
                                    "detection_method": "streaming_statistics"
                                }
                            )
                            patterns.append(pattern)
            else:
                stats["interaction_times"].append(current_time)
                
        except Exception as e:
            log.debug(f"Error in anomaly pattern detection: {e}")
        
        return patterns
    
    def _generate_pattern_id(self, pattern_type: str, *args) -> str:
        """Generate unique pattern ID"""
        components = [pattern_type] + [str(arg) for arg in args]
        content = ":".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming analytics statistics"""
        with self.lock:
            # Calculate window statistics
            window_stats = {}
            for window_size, window in self.sliding_windows.items():
                if window:
                    agent_counts = defaultdict(int)
                    type_counts = defaultdict(int)
                    
                    for interaction in window:
                        agent_counts[interaction.agent_id] += 1
                        type_counts[interaction.interaction_type] += 1
                    
                    window_stats[f"{window_size}s"] = {
                        "total_interactions": len(window),
                        "unique_agents": len(agent_counts),
                        "unique_types": len(type_counts),
                        "avg_per_agent": len(window) / max(1, len(agent_counts)),
                        "most_active_agent": max(agent_counts, key=agent_counts.get) if agent_counts else None,
                        "most_common_type": max(type_counts, key=type_counts.get) if type_counts else None
                    }
            
            return {
                "reservoir_sample_size": len(self.reservoir_sampler.samples),
                "reservoir_capacity": self.reservoir_sampler.capacity,
                "total_interactions_processed": self.reservoir_sampler.count,
                "unique_agents_estimated": self.cardinality_estimator.cardinality(),
                "active_agents": len(self.agent_activity),
                "temporal_buckets": len(self.temporal_buckets),
                "baseline_stats_agents": len(self.baseline_stats),
                "window_statistics": window_stats
            }
    
    async def batch_process_interactions(self, interactions: List[Interaction]) -> List[Pattern]:
        """Process multiple interactions efficiently"""
        all_patterns = []
        
        # Process in parallel batches
        batch_size = 100
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            
            # Process batch in parallel
            tasks = [self.process_interaction(interaction) for interaction in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, list):
                    all_patterns.extend(result)
                elif isinstance(result, Exception):
                    log.error(f"Batch processing error: {result}")
        
        return all_patterns

class StreamingKnowledgeSynthesizer:
    """Streaming knowledge synthesis with incremental updates"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.knowledge_cache = {}
        self.synthesis_rules = self._load_synthesis_rules()
        self.lock = threading.RLock()
        
        # Incremental synthesis state
        self.pattern_groups = defaultdict(list)
        self.last_synthesis = 0
        self.synthesis_interval = 60  # Synthesize every minute
        
        # Metrics
        if METRICS_AVAILABLE:
            self.synthesis_operations = Counter(
                'streaming_synthesis_operations_total',
                'Streaming synthesis operations',
                ['operation', 'status']
            )
            self.knowledge_cache_size = Gauge(
                'streaming_knowledge_cache_size',
                'Knowledge cache size'
            )
    
    async def add_patterns(self, patterns: List[Pattern]) -> List[Knowledge]:
        """Add patterns and trigger incremental synthesis"""
        new_knowledge = []
        
        with self.lock:
            # Add patterns to groups
            for pattern in patterns:
                self.pattern_groups[pattern.pattern_type].append(pattern)
                self.pattern_cache[pattern.pattern_id] = pattern
            
            # Check if synthesis should be triggered
            current_time = time.time()
            if current_time - self.last_synthesis >= self.synthesis_interval:
                new_knowledge = await self._incremental_synthesis()
                self.last_synthesis = current_time
        
        return new_knowledge
    
    async def _incremental_synthesis(self) -> List[Knowledge]:
        """Perform incremental knowledge synthesis"""
        knowledge_items = []
        
        try:
            # Process each pattern group
            for pattern_type, patterns in self.pattern_groups.items():
                if len(patterns) >= 2:  # Need minimum patterns for synthesis
                    
                    # Find applicable synthesis rules
                    applicable_rules = [
                        rule for rule in self.synthesis_rules
                        if pattern_type.value in rule.get("applicable_patterns", [])
                    ]
                    
                    for rule in applicable_rules:
                        knowledge = await self._apply_incremental_rule(rule, patterns)
                        if knowledge:
                            knowledge_items.append(knowledge)
                            self.knowledge_cache[knowledge.knowledge_id] = knowledge
            
            # Clear processed patterns (keep recent ones)
            current_time = time.time()
            for pattern_type in self.pattern_groups:
                # Keep patterns from last 10 minutes
                self.pattern_groups[pattern_type] = [
                    p for p in self.pattern_groups[pattern_type]
                    if current_time - p.discovered_at <= 600
                ]
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.synthesis_operations.labels(
                    operation="incremental_synthesis",
                    status="success"
                ).inc()
                
                self.knowledge_cache_size.set(len(self.knowledge_cache))
            
        except Exception as e:
            log.error(f"Incremental synthesis error: {e}")
            
            if METRICS_AVAILABLE:
                self.synthesis_operations.labels(
                    operation="incremental_synthesis",
                    status="error"
                ).inc()
        
        return knowledge_items
    
    async def _apply_incremental_rule(self, rule: Dict[str, Any], patterns: List[Pattern]) -> Optional[Knowledge]:
        """Apply synthesis rule to patterns incrementally"""
        try:
            # Filter patterns by confidence
            high_confidence_patterns = [
                p for p in patterns
                if p.confidence >= rule.get("min_confidence", 0.6)
            ]
            
            if len(high_confidence_patterns) < rule.get("min_patterns", 2):
                return None
            
            # Generate knowledge ID
            pattern_ids = sorted([p.pattern_id for p in high_confidence_patterns])
            knowledge_id = hashlib.sha256(
                f"{rule['name']}:{':'.join(pattern_ids)}".encode()
            ).hexdigest()[:12]
            
            # Check if already synthesized
            if knowledge_id in self.knowledge_cache:
                return None
            
            # Create knowledge content
            all_agents = set()
            for pattern in high_confidence_patterns:
                all_agents.update(pattern.agents_involved)
            
            content = rule["template"].format(
                pattern_count=len(high_confidence_patterns),
                agent_count=len(all_agents),
                agents=", ".join(sorted(all_agents)[:5]),  # Limit for readability
                avg_confidence=np.mean([p.confidence for p in high_confidence_patterns])
            )
            
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
                    "agent_count": len(all_agents),
                    "synthesis_method": "incremental_streaming"
                }
            )
            
            return knowledge
            
        except Exception as e:
            log.debug(f"Rule application error: {e}")
            return None
    
    def _load_synthesis_rules(self) -> List[Dict[str, Any]]:
        """Load synthesis rules for streaming processing"""
        return [
            {
                "name": "streaming_collaboration",
                "knowledge_type": "collaboration_pattern",
                "applicable_patterns": ["collaboration"],
                "min_patterns": 2,
                "min_confidence": 0.6,
                "template": "Detected {pattern_count} collaboration patterns among {agent_count} agents with average confidence {avg_confidence:.2f}",
                "contexts": ["collaboration", "teamwork"]
            },
            {
                "name": "streaming_anomaly_cluster",
                "knowledge_type": "anomaly_insight",
                "applicable_patterns": ["anomaly"],
                "min_patterns": 3,
                "min_confidence": 0.7,
                "template": "Identified anomaly cluster with {pattern_count} patterns affecting {agent_count} agents - requires investigation",
                "contexts": ["anomaly_detection", "monitoring"]
            },
            {
                "name": "streaming_temporal_trend",
                "knowledge_type": "temporal_insight",
                "applicable_patterns": ["temporal"],
                "min_patterns": 2,
                "min_confidence": 0.6,
                "template": "Observed temporal activity patterns across {agent_count} agents suggesting coordinated behavior",
                "contexts": ["temporal_patterns", "coordination"]
            }
        ]

class StreamingEmergentIntelligence:
    """Main streaming emergent intelligence engine"""
    
    def __init__(self, neural_mesh: Any):
        self.neural_mesh = neural_mesh
        self.pattern_detector = StreamingPatternDetector()
        self.knowledge_synthesizer = StreamingKnowledgeSynthesizer()
        
        # Streaming processing queue
        self.interaction_queue = asyncio.Queue(maxsize=10000)
        self.processing_tasks = []
        self.is_running = False
        
        # Metrics
        if METRICS_AVAILABLE:
            self.queue_size = Gauge(
                'streaming_intelligence_queue_size',
                'Size of interaction processing queue'
            )
            self.processing_rate = Counter(
                'streaming_intelligence_processing_rate_total',
                'Rate of interaction processing'
            )
    
    async def start(self):
        """Start streaming processing"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start processing tasks
        for i in range(4):  # 4 parallel processors
            task = asyncio.create_task(self._processing_loop(f"processor_{i}"))
            self.processing_tasks.append(task)
        
        # Start knowledge synthesis task
        synthesis_task = asyncio.create_task(self._synthesis_loop())
        self.processing_tasks.append(synthesis_task)
        
        log.info("Streaming emergent intelligence engine started")
    
    async def stop(self):
        """Stop streaming processing"""
        self.is_running = False
        
        # Cancel all tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        log.info("Streaming emergent intelligence engine stopped")
    
    async def record_interaction(self, interaction_data: Dict[str, Any]):
        """Record interaction for streaming processing"""
        try:
            interaction = Interaction(
                interaction_id=interaction_data.get("interaction_id", self._generate_interaction_id()),
                agent_id=interaction_data["agent_id"],
                interaction_type=interaction_data["type"],
                timestamp=interaction_data.get("timestamp", time.time()),
                data=interaction_data.get("data", {}),
                metadata=interaction_data.get("metadata", {})
            )
            
            # Add to processing queue (non-blocking)
            try:
                self.interaction_queue.put_nowait(interaction)
                
                if METRICS_AVAILABLE:
                    self.queue_size.set(self.interaction_queue.qsize())
                    
            except asyncio.QueueFull:
                log.warning("Interaction queue full, dropping interaction")
                
        except Exception as e:
            log.error(f"Error recording interaction: {e}")
    
    async def _processing_loop(self, processor_id: str):
        """Main processing loop for interactions"""
        log.info(f"Started processing loop: {processor_id}")
        
        while self.is_running:
            try:
                # Get interaction from queue
                interaction = await asyncio.wait_for(
                    self.interaction_queue.get(), 
                    timeout=1.0
                )
                
                # Process interaction
                patterns = await self.pattern_detector.process_interaction(interaction)
                
                if patterns:
                    # Add patterns to synthesizer
                    knowledge_items = await self.knowledge_synthesizer.add_patterns(patterns)
                    
                    # Propagate new knowledge
                    for knowledge in knowledge_items:
                        try:
                            await self.neural_mesh.propagate_knowledge(knowledge)
                        except Exception as e:
                            log.error(f"Failed to propagate knowledge: {e}")
                
                # Update metrics
                if METRICS_AVAILABLE:
                    self.processing_rate.inc()
                    self.queue_size.set(self.interaction_queue.qsize())
                
            except asyncio.TimeoutError:
                continue  # No interaction available, continue loop
            except Exception as e:
                log.error(f"Processing loop error in {processor_id}: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _synthesis_loop(self):
        """Background knowledge synthesis loop"""
        log.info("Started knowledge synthesis loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Trigger synthesis of accumulated patterns
                knowledge_items = await self.knowledge_synthesizer._incremental_synthesis()
                
                # Propagate synthesized knowledge
                for knowledge in knowledge_items:
                    try:
                        await self.neural_mesh.propagate_knowledge(knowledge)
                    except Exception as e:
                        log.error(f"Failed to propagate synthesized knowledge: {e}")
                
            except Exception as e:
                log.error(f"Synthesis loop error: {e}")
    
    def _generate_interaction_id(self) -> str:
        """Generate unique interaction ID"""
        return hashlib.sha256(f"{time.time()}:{random.random()}".encode()).hexdigest()[:12]
    
    async def get_streaming_insights(self) -> Dict[str, Any]:
        """Get streaming analytics insights"""
        pattern_stats = await self.pattern_detector.get_streaming_stats()
        
        return {
            "streaming_analytics": pattern_stats,
            "queue_size": self.interaction_queue.qsize(),
            "processing_tasks": len(self.processing_tasks),
            "is_running": self.is_running,
            "knowledge_cache_size": len(self.knowledge_synthesizer.knowledge_cache),
            "pattern_cache_size": len(self.knowledge_synthesizer.pattern_cache)
        }
