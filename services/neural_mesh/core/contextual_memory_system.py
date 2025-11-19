#!/usr/bin/env python3
"""
Contextual Memory System for Neural Mesh
Implements different memory types: short-term, long-term, episodic, semantic, and working memory
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid
import numpy as np

log = logging.getLogger("contextual-memory-system")

class MemoryRetrievalStrategy(Enum):
    """Memory retrieval strategies"""
    RECENCY = "recency"          # Most recent first
    RELEVANCE = "relevance"      # Most relevant first
    FREQUENCY = "frequency"      # Most frequently accessed first
    IMPORTANCE = "importance"    # Most important first
    HYBRID = "hybrid"           # Combination of strategies

class MemoryAccessPattern(Enum):
    """Memory access patterns"""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"

@dataclass
class ConversationContext:
    """Short-term conversation memory"""
    conversation_id: str
    agent_id: str
    messages: deque = field(default_factory=lambda: deque(maxlen=50))
    context_summary: str = ""
    active_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

@dataclass
class EpisodicMemory:
    """Episodic memory for event sequences"""
    episode_id: str
    agent_id: str
    event_sequence: List[Dict[str, Any]] = field(default_factory=list)
    episode_summary: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    importance_score: float = 0.5
    emotional_valence: float = 0.0  # -1 to 1
    tags: List[str] = field(default_factory=list)
    related_episodes: List[str] = field(default_factory=list)

@dataclass
class SemanticMemory:
    """Semantic memory for facts and relationships"""
    fact_id: str
    agent_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    evidence: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_frequency: int = 0

@dataclass
class WorkingMemory:
    """Working memory for active task state"""
    task_id: str
    agent_id: str
    active_goals: List[str] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    attention_focus: List[str] = field(default_factory=list)
    cognitive_load: float = 0.0
    working_set_size: int = 0
    last_updated: float = field(default_factory=time.time)

@dataclass
class MemoryAccessRecord:
    """Record of memory access for analytics"""
    access_id: str
    memory_id: str
    agent_id: str
    access_pattern: MemoryAccessPattern
    retrieval_strategy: MemoryRetrievalStrategy
    access_time: float
    retrieval_time: float
    success: bool
    context: Dict[str, Any] = field(default_factory=dict)

class ContextualMemorySystem:
    """Advanced contextual memory system with multiple memory types"""
    
    def __init__(self):
        # Memory stores
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.working_memories: Dict[str, WorkingMemory] = {}
        
        # Long-term persistent memory (integrated with distributed store)
        self.distributed_store = None
        
        # Access tracking
        self.access_records: List[MemoryAccessRecord] = []
        self.access_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Memory retrieval optimization
        self.retrieval_cache: Dict[str, Tuple[List[Any], float]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize contextual memory system"""
        if self._initialized:
            return
            
        try:
            # Initialize distributed store integration
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            self.distributed_store = distributed_memory_store
            
            # Start maintenance tasks
            asyncio.create_task(self._memory_maintenance_worker())
            asyncio.create_task(self._access_pattern_analyzer())
            
            self._initialized = True
            log.info("✅ Contextual memory system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize contextual memory system: {e}")
    
    async def ensure_initialized(self):
        """Ensure the system is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def create_conversation_context(
        self,
        agent_id: str,
        conversation_id: Optional[str] = None,
        initial_context: Dict[str, Any] = None
    ) -> str:
        """Create new conversation context"""
        
        if conversation_id is None:
            conversation_id = f"conv_{agent_id}_{str(uuid.uuid4())[:8]}"
        
        context = ConversationContext(
            conversation_id=conversation_id,
            agent_id=agent_id,
            conversation_state=initial_context or {}
        )
        
        self.conversation_contexts[conversation_id] = context
        
        # Store in distributed memory
        if self.distributed_store:
            await self.distributed_store.store_memory(
                agent_id=agent_id,
                memory_type="conversational",
                memory_tier="L1",  # Short-term, high-speed access
                content=asdict(context),
                metadata={"conversation_id": conversation_id}
            )
        
        log.debug(f"Created conversation context {conversation_id} for agent {agent_id}")
        return conversation_id
    
    async def update_conversation_context(
        self,
        conversation_id: str,
        message: Dict[str, Any],
        context_updates: Dict[str, Any] = None
    ):
        """Update conversation context with new message"""
        
        if conversation_id not in self.conversation_contexts:
            raise ValueError(f"Conversation context {conversation_id} not found")
        
        context = self.conversation_contexts[conversation_id]
        
        # Add message to conversation
        context.messages.append({
            "timestamp": time.time(),
            "message": message,
            "context": context_updates or {}
        })
        
        # Update context state
        if context_updates:
            context.conversation_state.update(context_updates)
        
        # Update access tracking
        context.last_accessed = time.time()
        context.access_count += 1
        
        # Update context summary periodically
        if len(context.messages) % 10 == 0:  # Every 10 messages
            await self._update_context_summary(context)
        
        # Sync with distributed store
        if self.distributed_store:
            await self.distributed_store.update_memory(
                memory_id=f"conv_{conversation_id}",
                agent_id=context.agent_id,
                updates=asdict(context)
            )
    
    async def create_episodic_memory(
        self,
        agent_id: str,
        event_sequence: List[Dict[str, Any]],
        episode_summary: str = "",
        importance_score: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Create episodic memory from event sequence"""
        
        episode_id = f"episode_{agent_id}_{str(uuid.uuid4())[:8]}"
        
        episode = EpisodicMemory(
            episode_id=episode_id,
            agent_id=agent_id,
            event_sequence=event_sequence,
            episode_summary=episode_summary,
            importance_score=importance_score,
            tags=tags or []
        )
        
        # Calculate emotional valence from events
        episode.emotional_valence = self._calculate_emotional_valence(event_sequence)
        
        self.episodic_memories[episode_id] = episode
        
        # Store in distributed memory (L3 for important episodes, L4 for others)
        memory_tier = "L3" if importance_score > 0.7 else "L4"
        
        if self.distributed_store:
            await self.distributed_store.store_memory(
                agent_id=agent_id,
                memory_type="episodic",
                memory_tier=memory_tier,
                content=asdict(episode),
                metadata={
                    "episode_id": episode_id,
                    "importance_score": importance_score,
                    "tags": tags or []
                }
            )
        
        log.debug(f"Created episodic memory {episode_id} for agent {agent_id}")
        return episode_id
    
    async def add_semantic_fact(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: str = "",
        evidence: List[str] = None
    ) -> str:
        """Add semantic fact to memory"""
        
        fact_id = self._generate_fact_id(subject, predicate, object)
        
        # Check if fact already exists
        if fact_id in self.semantic_memories:
            # Update existing fact
            existing_fact = self.semantic_memories[fact_id]
            existing_fact.confidence = max(existing_fact.confidence, confidence)
            existing_fact.evidence.extend(evidence or [])
            existing_fact.updated_at = time.time()
            existing_fact.access_frequency += 1
        else:
            # Create new fact
            fact = SemanticMemory(
                fact_id=fact_id,
                agent_id=agent_id,
                subject=subject,
                predicate=predicate,
                object=object,
                confidence=confidence,
                source=source,
                evidence=evidence or []
            )
            
            self.semantic_memories[fact_id] = fact
        
        # Store in distributed memory (L4 for long-term semantic knowledge)
        if self.distributed_store:
            await self.distributed_store.store_memory(
                agent_id=agent_id,
                memory_type="semantic",
                memory_tier="L4",
                content=asdict(self.semantic_memories[fact_id]),
                metadata={
                    "fact_id": fact_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }
            )
        
        log.debug(f"Added semantic fact {fact_id}")
        return fact_id
    
    async def create_working_memory(
        self,
        agent_id: str,
        task_id: str,
        initial_goals: List[str] = None,
        initial_state: Dict[str, Any] = None
    ) -> str:
        """Create working memory for active task"""
        
        working_memory = WorkingMemory(
            task_id=task_id,
            agent_id=agent_id,
            active_goals=initial_goals or [],
            current_state=initial_state or {},
            cognitive_load=0.0,
            working_set_size=len(initial_goals or [])
        )
        
        self.working_memories[task_id] = working_memory
        
        # Store in distributed memory (L1 for immediate access)
        if self.distributed_store:
            await self.distributed_store.store_memory(
                agent_id=agent_id,
                memory_type="working",
                memory_tier="L1",
                content=asdict(working_memory),
                metadata={"task_id": task_id},
                ttl=3600  # 1 hour TTL for working memory
            )
        
        log.debug(f"Created working memory for task {task_id}")
        return task_id
    
    async def retrieve_memories(
        self,
        agent_id: str,
        query: str,
        memory_types: List[str] = None,
        strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.HYBRID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories using specified strategy"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(agent_id, query, memory_types, strategy)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                await self._record_access(agent_id, "cache_hit", strategy, time.time() - start_time)
                return cached_result
            
            # Retrieve from different memory types
            results = []
            
            if not memory_types or "conversational" in memory_types:
                conv_results = await self._retrieve_conversational_memories(agent_id, query, limit)
                results.extend(conv_results)
            
            if not memory_types or "episodic" in memory_types:
                episodic_results = await self._retrieve_episodic_memories(agent_id, query, limit)
                results.extend(episodic_results)
            
            if not memory_types or "semantic" in memory_types:
                semantic_results = await self._retrieve_semantic_memories(agent_id, query, limit)
                results.extend(semantic_results)
            
            if not memory_types or "working" in memory_types:
                working_results = await self._retrieve_working_memories(agent_id, query, limit)
                results.extend(working_results)
            
            # Apply retrieval strategy
            if strategy == MemoryRetrievalStrategy.RECENCY:
                results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            elif strategy == MemoryRetrievalStrategy.RELEVANCE:
                results = await self._rank_by_relevance(query, results)
            elif strategy == MemoryRetrievalStrategy.FREQUENCY:
                results.sort(key=lambda x: x.get("access_count", 0), reverse=True)
            elif strategy == MemoryRetrievalStrategy.IMPORTANCE:
                results.sort(key=lambda x: x.get("importance_score", 0.5), reverse=True)
            elif strategy == MemoryRetrievalStrategy.HYBRID:
                results = await self._apply_hybrid_ranking(query, results)
            
            # Limit results
            final_results = results[:limit]
            
            # Cache results
            self._cache_result(cache_key, final_results)
            
            # Record access
            await self._record_access(
                agent_id, "retrieval_success", strategy, time.time() - start_time
            )
            
            return final_results
            
        except Exception as e:
            log.error(f"Error retrieving memories: {e}")
            await self._record_access(
                agent_id, "retrieval_error", strategy, time.time() - start_time
            )
            return []
    
    async def _retrieve_conversational_memories(
        self,
        agent_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve conversational memories"""
        
        results = []
        
        # Search conversation contexts
        for conv_id, context in self.conversation_contexts.items():
            if context.agent_id == agent_id:
                # Search messages for query terms
                query_terms = set(query.lower().split())
                
                for message in context.messages:
                    message_text = json.dumps(message).lower()
                    message_terms = set(message_text.split())
                    
                    # Calculate relevance
                    relevance = len(query_terms.intersection(message_terms)) / len(query_terms)
                    
                    if relevance > 0.1:  # Minimum relevance threshold
                        results.append({
                            "memory_type": "conversational",
                            "conversation_id": conv_id,
                            "message": message,
                            "relevance": relevance,
                            "timestamp": message.get("timestamp", context.created_at),
                            "access_count": context.access_count,
                            "importance_score": 0.3  # Default importance for conversations
                        })
        
        return results
    
    async def _retrieve_episodic_memories(
        self,
        agent_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve episodic memories"""
        
        results = []
        query_terms = set(query.lower().split())
        
        for episode_id, episode in self.episodic_memories.items():
            if episode.agent_id == agent_id:
                # Search episode content
                episode_text = (episode.episode_summary + " " + 
                               " ".join(json.dumps(event) for event in episode.event_sequence)).lower()
                episode_terms = set(episode_text.split())
                
                # Calculate relevance
                relevance = len(query_terms.intersection(episode_terms)) / len(query_terms)
                
                if relevance > 0.1:
                    results.append({
                        "memory_type": "episodic",
                        "episode_id": episode_id,
                        "episode": episode,
                        "relevance": relevance,
                        "timestamp": episode.start_time,
                        "importance_score": episode.importance_score,
                        "emotional_valence": episode.emotional_valence
                    })
        
        return results
    
    async def _retrieve_semantic_memories(
        self,
        agent_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve semantic memories (facts and relationships)"""
        
        results = []
        query_terms = set(query.lower().split())
        
        for fact_id, fact in self.semantic_memories.items():
            if fact.agent_id == agent_id:
                # Search fact content
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                fact_terms = set(fact_text.split())
                
                # Calculate relevance
                relevance = len(query_terms.intersection(fact_terms)) / len(query_terms)
                
                if relevance > 0.1:
                    results.append({
                        "memory_type": "semantic",
                        "fact_id": fact_id,
                        "fact": fact,
                        "relevance": relevance,
                        "timestamp": fact.updated_at,
                        "access_count": fact.access_frequency,
                        "importance_score": fact.confidence,
                        "confidence": fact.confidence
                    })
        
        return results
    
    async def _retrieve_working_memories(
        self,
        agent_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve working memories"""
        
        results = []
        query_terms = set(query.lower().split())
        
        for task_id, working_mem in self.working_memories.items():
            if working_mem.agent_id == agent_id:
                # Search working memory content
                working_text = (
                    " ".join(working_mem.active_goals) + " " +
                    json.dumps(working_mem.current_state) + " " +
                    " ".join(working_mem.attention_focus)
                ).lower()
                
                working_terms = set(working_text.split())
                
                # Calculate relevance
                relevance = len(query_terms.intersection(working_terms)) / len(query_terms)
                
                if relevance > 0.1:
                    results.append({
                        "memory_type": "working",
                        "task_id": task_id,
                        "working_memory": working_mem,
                        "relevance": relevance,
                        "timestamp": working_mem.last_updated,
                        "importance_score": 0.8,  # Working memory is highly important
                        "cognitive_load": working_mem.cognitive_load
                    })
        
        return results
    
    async def _rank_by_relevance(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank results by relevance to query"""
        
        # Enhanced relevance calculation
        query_terms = set(query.lower().split())
        
        for result in results:
            base_relevance = result.get("relevance", 0.0)
            
            # Boost relevance based on memory type importance
            memory_type = result.get("memory_type", "")
            type_boost = {
                "working": 0.3,
                "conversational": 0.1,
                "episodic": 0.2,
                "semantic": 0.25
            }.get(memory_type, 0.0)
            
            # Boost based on recency
            timestamp = result.get("timestamp", 0)
            recency_boost = max(0, 1.0 - (time.time() - timestamp) / 86400) * 0.1  # Decay over 24 hours
            
            # Boost based on access frequency
            access_count = result.get("access_count", 0)
            frequency_boost = min(access_count / 100.0, 0.1)  # Cap at 0.1
            
            # Calculate final relevance
            result["final_relevance"] = (
                base_relevance * 0.6 +
                type_boost +
                recency_boost +
                frequency_boost
            )
        
        # Sort by final relevance
        results.sort(key=lambda x: x.get("final_relevance", 0), reverse=True)
        return results
    
    async def _apply_hybrid_ranking(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply hybrid ranking strategy"""
        
        # First rank by relevance
        results = await self._rank_by_relevance(query, results)
        
        # Apply additional ranking factors
        for i, result in enumerate(results):
            relevance_score = result.get("final_relevance", 0.0)
            recency_score = self._calculate_recency_score(result.get("timestamp", 0))
            frequency_score = self._calculate_frequency_score(result.get("access_count", 0))
            importance_score = result.get("importance_score", 0.5)
            
            # Weighted hybrid score
            hybrid_score = (
                relevance_score * 0.4 +
                recency_score * 0.2 +
                frequency_score * 0.2 +
                importance_score * 0.2
            )
            
            result["hybrid_score"] = hybrid_score
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return results
    
    def _calculate_recency_score(self, timestamp: float) -> float:
        """Calculate recency score (0-1)"""
        if timestamp == 0:
            return 0.0
        
        age_hours = (time.time() - timestamp) / 3600
        return max(0.0, 1.0 - age_hours / 168)  # Decay over 1 week
    
    def _calculate_frequency_score(self, access_count: int) -> float:
        """Calculate frequency score (0-1)"""
        return min(access_count / 50.0, 1.0)  # Normalize to 50 accesses
    
    def _calculate_emotional_valence(self, event_sequence: List[Dict[str, Any]]) -> float:
        """Calculate emotional valence of episode"""
        
        # Simple sentiment analysis based on keywords
        positive_keywords = ["success", "achievement", "good", "excellent", "positive", "win"]
        negative_keywords = ["failure", "error", "bad", "negative", "problem", "issue"]
        
        positive_count = 0
        negative_count = 0
        
        for event in event_sequence:
            event_text = json.dumps(event).lower()
            
            for keyword in positive_keywords:
                if keyword in event_text:
                    positive_count += 1
            
            for keyword in negative_keywords:
                if keyword in event_text:
                    negative_count += 1
        
        total_sentiment = positive_count + negative_count
        if total_sentiment == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / total_sentiment
    
    async def _update_context_summary(self, context: ConversationContext):
        """Update conversation context summary"""
        
        try:
            # Get recent messages
            recent_messages = list(context.messages)[-20:]  # Last 20 messages
            
            # Create summary using LLM
            from core.enhanced_llm_integration import get_llm_integration, LLMRequest
            
            llm_integration = await get_llm_integration()
            
            messages_text = "\n".join([
                f"Message {i+1}: {json.dumps(msg['message'])}"
                for i, msg in enumerate(recent_messages)
            ])
            
            summary_prompt = f"""Summarize this conversation context:

{messages_text}

Provide a concise summary that captures:
1. Main topics discussed
2. Key decisions or conclusions
3. Current conversation state
4. User preferences or patterns

Summary:"""
            
            request = LLMRequest(
                agent_id=context.agent_id,
                task_type="context_summarization",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
            
            response = await llm_integration.generate_response(request)
            context.context_summary = response.content
            
            # Extract topics
            context.active_topics = self._extract_topics(response.content)
            
        except Exception as e:
            log.error(f"Error updating context summary: {e}")
    
    def _extract_topics(self, summary: str) -> List[str]:
        """Extract topics from conversation summary"""
        
        # Simple topic extraction (in production, would use NLP)
        import re
        
        # Look for topic indicators
        topic_patterns = [
            r"topic[s]?[:\s]+([^.]+)",
            r"discuss[ed]?[:\s]+([^.]+)",
            r"about[:\s]+([^.]+)"
        ]
        
        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, summary.lower())
            for match in matches:
                # Clean and split topics
                topic_words = [word.strip() for word in match.split(",")]
                topics.extend(topic_words)
        
        # Remove duplicates and empty topics
        unique_topics = list(set(topic.strip() for topic in topics if topic.strip()))
        return unique_topics[:10]  # Limit to 10 topics
    
    async def _memory_maintenance_worker(self):
        """Worker for memory maintenance tasks"""
        
        while True:
            try:
                # Clean up expired conversation contexts
                await self._cleanup_expired_conversations()
                
                # Consolidate episodic memories
                await self._consolidate_episodic_memories()
                
                # Update semantic memory relationships
                await self._update_semantic_relationships()
                
                # Clean up working memories for completed tasks
                await self._cleanup_completed_working_memories()
                
                # Sleep for maintenance interval
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in memory maintenance: {e}")
                await asyncio.sleep(300)
    
    async def _access_pattern_analyzer(self):
        """Analyze memory access patterns for optimization"""
        
        while True:
            try:
                # Analyze access patterns
                current_time = time.time()
                
                # Get recent access records
                recent_accesses = [
                    record for record in self.access_records
                    if current_time - record.access_time < 3600  # Last hour
                ]
                
                # Analyze patterns by agent
                agent_patterns = defaultdict(list)
                for access in recent_accesses:
                    agent_patterns[access.agent_id].append(access)
                
                # Generate optimization recommendations
                for agent_id, accesses in agent_patterns.items():
                    await self._generate_access_optimizations(agent_id, accesses)
                
                await asyncio.sleep(900)  # 15 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in access pattern analysis: {e}")
                await asyncio.sleep(300)
    
    async def _generate_access_optimizations(
        self,
        agent_id: str,
        accesses: List[MemoryAccessRecord]
    ):
        """Generate memory access optimizations for agent"""
        
        try:
            # Analyze access patterns
            strategy_usage = defaultdict(int)
            retrieval_times = defaultdict(list)
            
            for access in accesses:
                strategy_usage[access.retrieval_strategy.value] += 1
                retrieval_times[access.retrieval_strategy.value].append(access.retrieval_time)
            
            # Find optimal strategy
            if retrieval_times:
                avg_times = {
                    strategy: sum(times) / len(times)
                    for strategy, times in retrieval_times.items()
                }
                
                optimal_strategy = min(avg_times.keys(), key=lambda s: avg_times[s])
                
                # Store optimization recommendation
                if self.distributed_store:
                    await self.distributed_store.store_memory(
                        agent_id=agent_id,
                        memory_type="optimization",
                        memory_tier="L2",
                        content={
                            "optimal_retrieval_strategy": optimal_strategy,
                            "average_retrieval_times": avg_times,
                            "access_patterns": dict(strategy_usage),
                            "recommendation_confidence": 0.8
                        },
                        metadata={"optimization_type": "memory_access"}
                    )
            
        except Exception as e:
            log.error(f"Error generating access optimizations: {e}")
    
    def _generate_cache_key(
        self,
        agent_id: str,
        query: str,
        memory_types: List[str],
        strategy: MemoryRetrievalStrategy
    ) -> str:
        """Generate cache key for retrieval results"""
        
        cache_data = {
            "agent_id": agent_id,
            "query": query,
            "memory_types": sorted(memory_types) if memory_types else [],
            "strategy": strategy.value
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval result"""
        
        if cache_key in self.retrieval_cache:
            result, timestamp = self.retrieval_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired cache
                del self.retrieval_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache retrieval result"""
        
        self.retrieval_cache[cache_key] = (result, time.time())
        
        # Limit cache size
        if len(self.retrieval_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.retrieval_cache.keys(),
                key=lambda k: self.retrieval_cache[k][1]
            )[:100]
            
            for key in oldest_keys:
                del self.retrieval_cache[key]
    
    async def _record_access(
        self,
        agent_id: str,
        access_type: str,
        strategy: MemoryRetrievalStrategy,
        retrieval_time: float,
        memory_id: str = "unknown"
    ):
        """Record memory access for analytics"""
        
        access_record = MemoryAccessRecord(
            access_id=str(uuid.uuid4()),
            memory_id=memory_id,
            agent_id=agent_id,
            access_pattern=MemoryAccessPattern.READ,  # Default to read
            retrieval_strategy=strategy,
            access_time=time.time(),
            retrieval_time=retrieval_time,
            success=access_type != "retrieval_error"
        )
        
        self.access_records.append(access_record)
        
        # Update access patterns
        self.access_patterns[agent_id][strategy.value] += 1
        
        # Limit access records
        if len(self.access_records) > 10000:
            self.access_records = self.access_records[-5000:]  # Keep last 5000
    
    def _generate_fact_id(self, subject: str, predicate: str, object: str) -> str:
        """Generate unique fact ID"""
        fact_string = f"{subject}|{predicate}|{object}"
        return hashlib.md5(fact_string.encode()).hexdigest()[:16]
    
    async def get_memory_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive memory analytics"""
        
        analytics = {
            "timestamp": time.time(),
            "memory_counts": {},
            "access_patterns": {},
            "performance_metrics": {},
            "optimization_suggestions": []
        }
        
        # Memory counts by type
        if agent_id:
            # Agent-specific analytics
            analytics["memory_counts"] = {
                "conversational": len([c for c in self.conversation_contexts.values() if c.agent_id == agent_id]),
                "episodic": len([e for e in self.episodic_memories.values() if e.agent_id == agent_id]),
                "semantic": len([s for s in self.semantic_memories.values() if s.agent_id == agent_id]),
                "working": len([w for w in self.working_memories.values() if w.agent_id == agent_id])
            }
            
            # Access patterns for agent
            analytics["access_patterns"] = dict(self.access_patterns.get(agent_id, {}))
        else:
            # System-wide analytics
            analytics["memory_counts"] = {
                "conversational": len(self.conversation_contexts),
                "episodic": len(self.episodic_memories),
                "semantic": len(self.semantic_memories),
                "working": len(self.working_memories)
            }
            
            # Aggregate access patterns
            all_patterns = defaultdict(int)
            for agent_patterns in self.access_patterns.values():
                for strategy, count in agent_patterns.items():
                    all_patterns[strategy] += count
            
            analytics["access_patterns"] = dict(all_patterns)
        
        # Performance metrics
        recent_accesses = [
            record for record in self.access_records
            if (not agent_id or record.agent_id == agent_id) and
               time.time() - record.access_time < 3600
        ]
        
        if recent_accesses:
            analytics["performance_metrics"] = {
                "total_accesses": len(recent_accesses),
                "success_rate": sum(1 for r in recent_accesses if r.success) / len(recent_accesses),
                "average_retrieval_time": sum(r.retrieval_time for r in recent_accesses) / len(recent_accesses),
                "cache_hit_rate": len([r for r in recent_accesses if "cache_hit" in str(r.context)]) / len(recent_accesses)
            }
        
        return analytics
    
    async def export_memory_state(
        self,
        agent_id: str,
        export_path: str = "exports/memory"
    ):
        """Export complete memory state for agent"""
        
        from pathlib import Path
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export all memory types
        memory_export = {
            "agent_id": agent_id,
            "export_timestamp": time.time(),
            "conversational_contexts": [
                asdict(context) for context in self.conversation_contexts.values()
                if context.agent_id == agent_id
            ],
            "episodic_memories": [
                asdict(episode) for episode in self.episodic_memories.values()
                if episode.agent_id == agent_id
            ],
            "semantic_memories": [
                asdict(fact) for fact in self.semantic_memories.values()
                if fact.agent_id == agent_id
            ],
            "working_memories": [
                asdict(working) for working in self.working_memories.values()
                if working.agent_id == agent_id
            ]
        }
        
        # Save to file
        export_file = export_dir / f"memory_export_{agent_id}_{int(time.time())}.json"
        with open(export_file, 'w') as f:
            json.dump(memory_export, f, indent=2, default=str)
        
        log.info(f"Exported memory state for agent {agent_id} to {export_file}")

# Global instance
contextual_memory = ContextualMemorySystem()
