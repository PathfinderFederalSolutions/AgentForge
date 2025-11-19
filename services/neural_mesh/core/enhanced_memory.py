"""
Enhanced Neural Mesh Memory System - Fully Integrated Implementation
Complete neural mesh with distributed memory, synchronization, and collective intelligence
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import numpy as np

# Import consolidated types
from .memory_types import (
    MemoryItem, Query, Knowledge, Pattern, MemoryTier, 
    NeuralMeshLayer, PatternType, PatternStrength
)

# Import enhanced neural mesh systems
try:
    from .distributed_memory_store import distributed_memory_store
    from .contextual_memory_system import contextual_memory
    from .inter_agent_communication import inter_agent_comm
    from .memory_synchronization_protocol import memory_sync_protocol
    from .memory_versioning_system import memory_versioning
    ENHANCED_NEURAL_MESH_AVAILABLE = True
except ImportError:
    ENHANCED_NEURAL_MESH_AVAILABLE = False

# Optional imports with fallbacks
try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

log = logging.getLogger("neural-mesh")

class MultiModalEmbedder:
    """Multi-modal embedding system for universal content types"""
    
    def __init__(self):
        self.text_encoder = None
        self.dimensions = 384
        self._init_encoders()
        
    def _init_encoders(self):
        """Initialize embedding models"""
        if SentenceTransformer:
            try:
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimensions = self.text_encoder.get_sentence_embedding_dimension()
            except Exception as e:
                log.warning(f"Failed to load SentenceTransformer: {e}")
                
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding"""
        if self.text_encoder:
            return self.text_encoder.encode([text])[0]
        else:
            # Fallback to hash-based embedding
            return self._hash_embedding(text)
            
    def encode_content(self, content: Any, content_type: str) -> np.ndarray:
        """Encode any content type to unified embedding space"""
        if content_type == "text" or isinstance(content, str):
            return self.encode_text(str(content))
        elif content_type == "json":
            return self.encode_text(json.dumps(content))
        else:
            # Fallback for unsupported types
            return self.encode_text(str(content))
            
    def _hash_embedding(self, text: str) -> np.ndarray:
        """Generate hash-based embedding as fallback"""
        # Create multiple hashes for better distribution
        hashes = []
        for i in range(self.dimensions // 32 + 1):
            hash_input = f"{text}:{i}".encode('utf-8')
            hash_obj = hashlib.sha256(hash_input)
            hash_bytes = hash_obj.digest()
            
            # Convert bytes to floats
            for j in range(0, len(hash_bytes), 4):
                if len(hashes) >= self.dimensions:
                    break
                chunk = hash_bytes[j:j+4].ljust(4, b'\x00')
                val = int.from_bytes(chunk, byteorder='big', signed=False)
                # Normalize to [-1, 1] range
                normalized = (val / (2**32 - 1)) * 2 - 1
                hashes.append(normalized)
                
        # Ensure exact dimensions
        vec = np.array(hashes[:self.dimensions])
        if len(vec) < self.dimensions:
            vec = np.pad(vec, (0, self.dimensions - len(vec)))
            
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec

class L1AgentMemory(NeuralMeshLayer):
    """L1: Local agent working memory with vector cache"""
    
    def __init__(self, agent_id: str, max_items: int = 1000):
        self.agent_id = agent_id
        self.max_items = max_items
        self.items: Dict[str, MemoryItem] = {}
        # Use the basic embedder for L1 and L2
        self.embedder = MultiModalEmbedder()
        
    async def store(self, item: MemoryItem) -> bool:
        """Store item in local memory"""
        try:
            # Generate embedding if not provided
            if item.embedding is None:
                item.embedding = self.embedder.encode_content(
                    item.value, 
                    item.metadata.get("content_type", "text")
                )
                
            # Add agent context
            item.context["agent_id"] = self.agent_id
            item.tier = MemoryTier.L1_AGENT
            
            # Store item
            self.items[item.key] = item
            
            # Prune if necessary
            await self._maybe_prune()
            
            log.debug(f"Stored item {item.key} in L1 memory for agent {self.agent_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to store item {item.key}: {e}")
            return False
            
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items matching query"""
        try:
            # Generate query embedding if not provided
            if query.embedding is None:
                query.embedding = self.embedder.encode_text(query.text)
                
            results = []
            
            for item in self.items.values():
                if item.embedding is None:
                    continue
                    
                # Calculate similarity
                similarity = self._cosine_similarity(query.embedding, item.embedding)
                
                if similarity >= query.min_score:
                    # Update access statistics
                    item.access_count += 1
                    item.last_accessed = time.time()
                    
                    results.append((similarity, item))
                    
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in results[:query.top_k]]
            
        except Exception as e:
            log.error(f"Failed to retrieve items for query '{query.text}': {e}")
            return []
            
    async def propagate(self, knowledge: Knowledge) -> bool:
        """Propagate knowledge (L1 just stores locally)"""
        # At L1, we just store the knowledge as a memory item
        item = MemoryItem(
            key=f"knowledge:{knowledge.knowledge_id}",
            value=knowledge.content,
            context={"type": "knowledge", "source": "propagation"},
            metadata={
                "knowledge_id": knowledge.knowledge_id,
                "confidence": knowledge.confidence,
                "applicable_contexts": knowledge.applicable_contexts
            }
        )
        return await self.store(item)
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get L1 memory statistics"""
        total_items = len(self.items)
        total_accesses = sum(item.access_count for item in self.items.values())
        
        return {
            "tier": "L1_AGENT",
            "agent_id": self.agent_id,
            "total_items": total_items,
            "max_items": self.max_items,
            "utilization": total_items / self.max_items if self.max_items > 0 else 0,
            "total_accesses": total_accesses,
            "avg_accesses_per_item": total_accesses / total_items if total_items > 0 else 0
        }
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
            
    async def _maybe_prune(self):
        """Prune old items if memory is full"""
        if len(self.items) <= self.max_items:
            return
            
        # Sort by last accessed time (LRU)
        items_by_access = sorted(
            self.items.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest items
        items_to_remove = items_by_access[:len(self.items) - self.max_items]
        for key, _ in items_to_remove:
            del self.items[key]
            
        log.info(f"Pruned {len(items_to_remove)} items from L1 memory")

class L2SwarmMemory(NeuralMeshLayer):
    """L2: Distributed cluster memory with CRDT sync"""
    
    def __init__(self, swarm_id: str, redis_url: Optional[str] = None):
        self.swarm_id = swarm_id
        self.redis_client = None
        # Use the basic embedder for L1 and L2
        self.embedder = MultiModalEmbedder()
        self._init_redis(redis_url)
        
    def _init_redis(self, redis_url: Optional[str]):
        """Initialize Redis connection"""
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                log.info(f"Connected to Redis for L2 memory: {redis_url}")
            except Exception as e:
                log.warning(f"Failed to connect to Redis: {e}")
                
    async def store(self, item: MemoryItem) -> bool:
        """Store item in distributed swarm memory"""
        try:
            item.tier = MemoryTier.L2_SWARM
            item.context["swarm_id"] = self.swarm_id
            
            if self.redis_client:
                # Store in Redis with TTL
                key = f"swarm:{self.swarm_id}:memory:{item.key}"
                value = json.dumps(item.to_dict())
                await self.redis_client.setex(key, 3600, value)  # 1 hour TTL
                
                # Store embedding separately for efficient search
                if item.embedding is not None:
                    embedding_key = f"swarm:{self.swarm_id}:embeddings:{item.key}"
                    embedding_data = {
                        "embedding": item.embedding.tolist(),
                        "metadata": item.metadata
                    }
                    await self.redis_client.setex(
                        embedding_key, 3600, json.dumps(embedding_data)
                    )
                    
                log.debug(f"Stored item {item.key} in L2 swarm memory")
                return True
            else:
                log.warning("Redis not available for L2 storage")
                return False
                
        except Exception as e:
            log.error(f"Failed to store item {item.key} in L2: {e}")
            return False
            
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items from distributed swarm memory"""
        if not self.redis_client:
            return []
            
        try:
            # Generate query embedding
            if query.embedding is None:
                query.embedding = self.embedder.encode_text(query.text)
                
            # Get all embedding keys for this swarm
            pattern = f"swarm:{self.swarm_id}:embeddings:*"
            embedding_keys = await self.redis_client.keys(pattern)
            
            results = []
            
            for embedding_key in embedding_keys:
                try:
                    # Get embedding data
                    embedding_data = await self.redis_client.get(embedding_key)
                    if not embedding_data:
                        continue
                        
                    data = json.loads(embedding_data)
                    embedding = np.array(data["embedding"])
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query.embedding, embedding)
                    
                    if similarity >= query.min_score:
                        # Get the actual memory item
                        item_key = embedding_key.replace(
                            f"swarm:{self.swarm_id}:embeddings:", ""
                        )
                        memory_key = f"swarm:{self.swarm_id}:memory:{item_key}"
                        item_data = await self.redis_client.get(memory_key)
                        
                        if item_data:
                            item = MemoryItem.from_dict(json.loads(item_data))
                            results.append((similarity, item))
                            
                except Exception as e:
                    log.debug(f"Error processing embedding key {embedding_key}: {e}")
                    continue
                    
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in results[:query.top_k]]
            
        except Exception as e:
            log.error(f"Failed to retrieve from L2 memory: {e}")
            return []
            
    async def propagate(self, knowledge: Knowledge) -> bool:
        """Propagate knowledge across swarm"""
        # Store knowledge in swarm memory for all agents to access
        item = MemoryItem(
            key=f"knowledge:{knowledge.knowledge_id}",
            value=knowledge.content,
            context={"type": "knowledge", "source": "propagation", "swarm_id": self.swarm_id},
            metadata={
                "knowledge_id": knowledge.knowledge_id,
                "confidence": knowledge.confidence,
                "applicable_contexts": knowledge.applicable_contexts
            }
        )
        
        success = await self.store(item)
        
        if success and self.redis_client:
            # Publish notification to swarm agents
            notification = {
                "type": "knowledge_propagation",
                "knowledge_id": knowledge.knowledge_id,
                "swarm_id": self.swarm_id,
                "timestamp": time.time()
            }
            
            await self.redis_client.publish(
                f"swarm:{self.swarm_id}:knowledge",
                json.dumps(notification)
            )
            
        return success
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get L2 memory statistics"""
        if not self.redis_client:
            return {"tier": "L2_SWARM", "error": "Redis not available"}
            
        try:
            # Count items in swarm memory
            pattern = f"swarm:{self.swarm_id}:memory:*"
            memory_keys = await self.redis_client.keys(pattern)
            
            # Count embeddings
            embedding_pattern = f"swarm:{self.swarm_id}:embeddings:*"
            embedding_keys = await self.redis_client.keys(embedding_pattern)
            
            return {
                "tier": "L2_SWARM",
                "swarm_id": self.swarm_id,
                "memory_items": len(memory_keys),
                "embeddings": len(embedding_keys),
                "redis_connected": True
            }
            
        except Exception as e:
            return {
                "tier": "L2_SWARM",
                "error": str(e),
                "redis_connected": False
            }
            
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0

class PatternDetector:
    """Detect patterns in agent interactions and memory access"""
    
    def __init__(self):
        self.interaction_history = []
        self.pattern_cache = {}
        
    async def analyze_interactions(self, interactions: List[Dict[str, Any]]) -> List[Pattern]:
        """Analyze interactions to detect patterns"""
        patterns = []
        
        # Simple pattern detection - can be enhanced with ML
        for interaction_type in ["memory_access", "agent_communication", "task_execution"]:
            pattern = await self._detect_pattern_type(interactions, interaction_type)
            if pattern:
                patterns.append(pattern)
                
        return patterns
        
    async def _detect_pattern_type(
        self, 
        interactions: List[Dict[str, Any]], 
        pattern_type: str
    ) -> Optional[Pattern]:
        """Detect specific pattern type"""
        relevant_interactions = [
            i for i in interactions 
            if i.get("type") == pattern_type
        ]
        
        if len(relevant_interactions) < 3:  # Need minimum interactions
            return None
            
        # Simple frequency-based pattern detection
        agents_involved = set()
        memory_keys = set()
        
        for interaction in relevant_interactions:
            agents_involved.add(interaction.get("agent_id"))
            if "memory_keys" in interaction:
                memory_keys.update(interaction["memory_keys"])
                
        if len(agents_involved) >= 2:  # Multi-agent pattern
            pattern_id = hashlib.sha256(
                f"{pattern_type}:{sorted(agents_involved)}".encode()
            ).hexdigest()[:8]
            
            return Pattern(
                pattern_id=pattern_id,
                pattern_type=PatternType.MEMORY_ACCESS,  # Use enum
                strength=PatternStrength.MODERATE,
                confidence=0.8,  # Simple confidence score
                agents_involved=set(agents_involved),
                interactions=[],  # Would need interaction IDs
                metadata={"interaction_count": len(relevant_interactions)}
            )
            
        return None

class KnowledgeSynthesizer:
    """Synthesize new knowledge from detected patterns"""
    
    def __init__(self):
        self.synthesis_rules = self._load_synthesis_rules()
        
    async def synthesize(self, patterns: List[Pattern]) -> Optional[Knowledge]:
        """Synthesize knowledge from patterns"""
        if not patterns:
            return None
            
        # Simple rule-based synthesis - can be enhanced with ML
        for rule in self.synthesis_rules:
            knowledge = await self._apply_rule(rule, patterns)
            if knowledge:
                return knowledge
                
        return None
        
    async def _apply_rule(self, rule: Dict[str, Any], patterns: List[Pattern]) -> Optional[Knowledge]:
        """Apply synthesis rule to patterns"""
        matching_patterns = [
            p for p in patterns 
            if p.pattern_type in rule.get("applicable_patterns", [])
        ]
        
        if len(matching_patterns) >= rule.get("min_patterns", 1):
            knowledge_id = hashlib.sha256(
                f"knowledge:{time.time()}:{len(matching_patterns)}".encode()
            ).hexdigest()[:8]
            
            content = rule["template"].format(
                pattern_count=len(matching_patterns),
                agents=", ".join(set().union(*[p.agents_involved for p in matching_patterns]))
            )
            
            return Knowledge(
                knowledge_id=knowledge_id,
                content=content,
                knowledge_type="pattern_synthesis",
                source_patterns=[p.pattern_id for p in matching_patterns],
                confidence=min(p.confidence for p in matching_patterns),
                applicable_contexts=rule.get("contexts", ["general"])
            )
            
        return None
        
    def _load_synthesis_rules(self) -> List[Dict[str, Any]]:
        """Load knowledge synthesis rules"""
        return [
            {
                "name": "multi_agent_collaboration",
                "applicable_patterns": ["memory_access", "agent_communication"],
                "min_patterns": 2,
                "template": "Detected collaboration pattern among {agents} with {pattern_count} interaction types",
                "contexts": ["collaboration", "teamwork"]
            },
            {
                "name": "task_execution_optimization",
                "applicable_patterns": ["task_execution"],
                "min_patterns": 3,
                "template": "Identified optimization opportunity in task execution involving {agents}",
                "contexts": ["optimization", "performance"]
            }
        ]

class EmergentIntelligence:
    """Emergent intelligence engine for swarm behaviors"""
    
    def __init__(self, neural_mesh: 'EnhancedNeuralMesh'):
        self.neural_mesh = neural_mesh
        self.pattern_detector = PatternDetector()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.interaction_buffer = []
        self.max_buffer_size = 1000
        
    async def record_interaction(self, interaction: Dict[str, Any]) -> None:
        """Record agent interaction for pattern analysis"""
        interaction["timestamp"] = time.time()
        self.interaction_buffer.append(interaction)
        
        # Prune buffer if too large
        if len(self.interaction_buffer) > self.max_buffer_size:
            self.interaction_buffer = self.interaction_buffer[-self.max_buffer_size:]
            
    async def analyze_and_synthesize(self) -> List[Knowledge]:
        """Analyze patterns and synthesize new knowledge"""
        if len(self.interaction_buffer) < 10:  # Need minimum interactions
            return []
            
        try:
            # Detect patterns
            patterns = await self.pattern_detector.analyze_interactions(
                self.interaction_buffer
            )
            
            # Synthesize knowledge from patterns
            knowledge_items = []
            for pattern_group in self._group_patterns(patterns):
                knowledge = await self.knowledge_synthesizer.synthesize(pattern_group)
                if knowledge:
                    knowledge_items.append(knowledge)
                    
            # Propagate new knowledge through the mesh
            for knowledge in knowledge_items:
                await self.neural_mesh.propagate_knowledge(knowledge)
                
            return knowledge_items
            
        except Exception as e:
            log.error(f"Error in emergent intelligence analysis: {e}")
            return []
            
    def _group_patterns(self, patterns: List[Pattern]) -> List[List[Pattern]]:
        """Group related patterns for synthesis"""
        # Simple grouping by pattern type - can be enhanced
        groups = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append(pattern)
            
        return list(groups.values())

class EnhancedNeuralMesh:
    """Enhanced neural mesh memory system with complete 4-tier architecture"""
    
    def __init__(
        self, 
        agent_id: str, 
        swarm_id: Optional[str] = None,
        redis_url: Optional[str] = None,
        org_config: Optional[Any] = None,
        postgres_url: Optional[str] = None,
        global_sources: Optional[List[Any]] = None
    ):
        self.agent_id = agent_id
        self.swarm_id = swarm_id or "default"
        
        # Initialize memory layers
        self.l1_memory = L1AgentMemory(agent_id)
        self.l2_memory = L2SwarmMemory(self.swarm_id, redis_url) if swarm_id else None
        
        # Initialize L3 and L4 if configurations provided
        self.l3_memory = None
        self.l4_memory = None
        
        if org_config:
            try:
                from .l3_l4_memory import L3OrganizationMemory, VectorStoreType
                self.l3_memory = L3OrganizationMemory(
                    org_config=org_config,
                    postgres_url=postgres_url,
                    vector_store_type=VectorStoreType.MEMORY  # Default to memory for now
                )
            except ImportError as e:
                log.warning(f"L3 memory not available: {e}")
        
        if global_sources:
            try:
                from .l3_l4_memory import L4GlobalMemory
                self.l4_memory = L4GlobalMemory(global_sources)
            except ImportError as e:
                log.warning(f"L4 memory not available: {e}")
        
        # Initialize emergent intelligence
        self.emergent_intelligence = EmergentIntelligence(self)
        
        log.info(f"Initialized enhanced neural mesh for agent {agent_id} with {self._count_active_layers()} active layers")
    
    def _count_active_layers(self) -> int:
        """Count number of active memory layers"""
        count = 1  # L1 is always active
        if self.l2_memory:
            count += 1
        if self.l3_memory:
            count += 1
        if self.l4_memory:
            count += 1
        return count
        
    async def store(self, key: str, value: Any, **kwargs) -> bool:
        """Store value across all available memory tiers"""
        # Create memory item
        item = MemoryItem(
            key=key,
            value=value,
            context=kwargs.get("context", {}),
            metadata=kwargs.get("metadata", {})
        )
        
        # Determine storage strategy based on context
        storage_tier = kwargs.get("tier", "auto")
        store_globally = kwargs.get("global", False)
        store_organizationally = kwargs.get("organizational", False)
        
        results = {}
        
        # Always store in L1 (local) memory
        results["l1"] = await self.l1_memory.store(item)
        
        # Store in L2 (swarm) memory if available
        if self.l2_memory:
            results["l2"] = await self.l2_memory.store(item)
        
        # Store in L3 (organizational) memory if configured and appropriate
        if self.l3_memory and (store_organizationally or storage_tier in ["auto", "organizational", "global"]):
            results["l3"] = await self.l3_memory.store(item)
        
        # Store in L4 (global) memory if configured and appropriate
        if self.l4_memory and (store_globally or storage_tier in ["global"]):
            results["l4"] = await self.l4_memory.store(item)
            
        # Record interaction for pattern analysis
        await self.emergent_intelligence.record_interaction({
            "type": "memory_access",
            "action": "store",
            "agent_id": self.agent_id,
            "memory_keys": [key],
            "storage_results": results,
            "layers_used": list(results.keys()),
            "success": all(results.values())
        })
        
        # Return success if at least L1 succeeds (resilient to external service failures)
        l1_success = results.get("l1", False)
        return l1_success
        
    async def retrieve(self, query_text: str, **kwargs) -> List[MemoryItem]:
        """Retrieve items matching query from all available tiers"""
        query = Query(
            text=query_text,
            top_k=kwargs.get("top_k", 5),
            min_score=kwargs.get("min_score", 0.7),
            context=kwargs.get("context", {}),
            filters=kwargs.get("filters", {})
        )
        
        # Determine search scope
        search_scope = kwargs.get("scope", "auto")  # auto, local, swarm, organizational, global
        
        results = []
        layers_searched = []
        
        # Search L1 memory - always search unless scope is very specific
        if search_scope in ["auto", "local", "swarm", "organizational", "global"]:
            l1_results = await self.l1_memory.retrieve(query)
            results.extend(l1_results)
            layers_searched.append("l1")
        
        # Search L2 memory if available and in scope
        if (self.l2_memory and 
            search_scope in ["auto", "swarm", "organizational", "global"] and 
            len(results) < query.top_k):
            l2_results = await self.l2_memory.retrieve(query)
            results.extend(l2_results)
            layers_searched.append("l2")
        
        # Search L3 memory if available and in scope
        if (self.l3_memory and 
            search_scope in ["auto", "organizational", "global"] and 
            len(results) < query.top_k):
            l3_results = await self.l3_memory.retrieve(query)
            results.extend(l3_results)
            layers_searched.append("l3")
        
        # Search L4 memory if available and in scope
        if (self.l4_memory and 
            search_scope in ["global"] and 
            len(results) < query.top_k):
            l4_results = await self.l4_memory.retrieve(query)
            results.extend(l4_results)
            layers_searched.append("l4")
            
        # Intelligent result fusion and deduplication
        final_results = await self._fuse_memory_results(results, query)
        
        # Record interaction
        await self.emergent_intelligence.record_interaction({
            "type": "memory_access",
            "action": "retrieve",
            "agent_id": self.agent_id,
            "query": query_text,
            "results_count": len(final_results),
            "memory_keys": [item.key for item in final_results],
            "layers_searched": layers_searched,
            "search_scope": search_scope,
            "total_candidates": len(results)
        })
        
        return final_results
    
    async def _fuse_memory_results(self, results: List[MemoryItem], query: Query) -> List[MemoryItem]:
        """Intelligently fuse and rank results from multiple memory tiers"""
        if not results:
            return []
        
        # Deduplicate by key, keeping the best version from each tier
        unique_results = {}
        
        for item in results:
            if item.key not in unique_results:
                unique_results[item.key] = item
            else:
                existing = unique_results[item.key]
                
                # Tier-based priority: L4 > L3 > L2 > L1 for authoritative content
                tier_priority = {
                    MemoryTier.L4_GLOBAL: 4,
                    MemoryTier.L3_ORGANIZATION: 3,
                    MemoryTier.L2_SWARM: 2,
                    MemoryTier.L1_AGENT: 1
                }
                
                # Choose based on tier priority, then recency, then access count
                if (tier_priority.get(item.tier, 0) > tier_priority.get(existing.tier, 0) or
                    (item.tier == existing.tier and item.last_accessed > existing.last_accessed) or
                    (item.tier == existing.tier and item.last_accessed == existing.last_accessed and 
                     item.access_count > existing.access_count)):
                    unique_results[item.key] = item
        
        # Calculate relevance scores
        scored_results = []
        for item in unique_results.values():
            score = await self._calculate_relevance_score(item, query)
            if score >= query.min_score:
                item.metadata["relevance_score"] = score
                scored_results.append(item)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        
        return scored_results
    
    async def _calculate_relevance_score(self, item: MemoryItem, query: Query) -> float:
        """Calculate relevance score for memory item"""
        try:
            # Base similarity score
            if item.embedding is not None and query.embedding is not None:
                similarity = self._cosine_similarity(item.embedding, query.embedding)
            else:
                # Fallback to text matching
                similarity = self._text_similarity(str(item.value), query.text)
            
            # Tier-based bonus (higher tiers get slight bonus for authority)
            tier_bonus = {
                MemoryTier.L4_GLOBAL: 0.1,
                MemoryTier.L3_ORGANIZATION: 0.05,
                MemoryTier.L2_SWARM: 0.02,
                MemoryTier.L1_AGENT: 0.0
            }.get(item.tier, 0.0)
            
            # Recency bonus (more recent items get slight bonus)
            recency_bonus = min(0.1, (time.time() - item.last_accessed) / 86400)  # Up to 0.1 for recent items
            
            # Access frequency bonus
            frequency_bonus = min(0.05, item.access_count / 100)  # Up to 0.05 for frequently accessed
            
            # Context matching bonus
            context_bonus = self._calculate_context_bonus(item, query)
            
            # Combine scores
            total_score = similarity + tier_bonus + recency_bonus + frequency_bonus + context_bonus
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            log.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Simple word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def _calculate_context_bonus(self, item: MemoryItem, query: Query) -> float:
        """Calculate context matching bonus"""
        if not query.context or not item.context:
            return 0.0
        
        # Simple context key matching
        query_keys = set(query.context.keys())
        item_keys = set(item.context.keys())
        
        overlap = len(query_keys.intersection(item_keys))
        total = len(query_keys.union(item_keys))
        
        return (overlap / total * 0.1) if total > 0 else 0.0
        
    async def propagate_knowledge(self, knowledge: Knowledge) -> bool:
        """Propagate knowledge through appropriate memory tiers"""
        success = True
        
        # Propagate through L1
        if not await self.l1_memory.propagate(knowledge):
            success = False
            
        # Propagate through L2 if available
        if self.l2_memory:
            if not await self.l2_memory.propagate(knowledge):
                success = False
        
        # Propagate through L3 if available
        if self.l3_memory:
            if not await self.l3_memory.propagate(knowledge):
                success = False
        
        # Propagate through L4 if available
        if self.l4_memory:
            if not await self.l4_memory.propagate(knowledge):
                success = False
                
        return success
        
    async def analyze_emergence(self) -> List[Knowledge]:
        """Trigger emergent intelligence analysis"""
        return await self.emergent_intelligence.analyze_and_synthesize()
        
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory tiers"""
        stats = {
            "agent_id": self.agent_id,
            "swarm_id": self.swarm_id,
            "tiers": {}
        }
        
        # L1 stats
        stats["tiers"]["L1"] = await self.l1_memory.get_stats()
        
        # L2 stats if available
        if self.l2_memory:
            stats["tiers"]["L2"] = await self.l2_memory.get_stats()
        
        # L3 stats if available
        if self.l3_memory:
            stats["tiers"]["L3"] = await self.l3_memory.get_stats()
        
        # L4 stats if available
        if self.l4_memory:
            stats["tiers"]["L4"] = await self.l4_memory.get_stats()
            
        # Emergent intelligence stats
        stats["emergent_intelligence"] = {
            "interaction_buffer_size": len(self.emergent_intelligence.interaction_buffer),
            "max_buffer_size": self.emergent_intelligence.max_buffer_size
        }
        
        # Summary statistics
        total_items = sum(
            tier_stats.get("total_items", 0) 
            for tier_stats in stats["tiers"].values()
        )
        
        stats["summary"] = {
            "total_items": total_items,
            "active_tiers": len(stats["tiers"]),
            "l3_available": self.l3_memory is not None,
            "l4_available": self.l4_memory is not None,
            "timestamp": time.time()
        }
        
        return stats
    
    async def register_agent(self, agent_id: str, capabilities: List[str]) -> bool:
        """Register an agent with the neural mesh"""
        try:
            # Store agent capabilities in L1 memory
            agent_info = MemoryItem(
                key=f"agent_registration_{agent_id}",
                value=f"Agent {agent_id} with capabilities: {', '.join(capabilities)}",
                context={"type": "agent_registration"},
                metadata={
                    "agent_id": agent_id,
                    "capabilities": capabilities,
                    "registered_at": time.time()
                }
            )
            
            success = await self.l1_memory.store(agent_info)
            if success:
                log.info(f"✅ Agent {agent_id} registered with neural mesh")
            return success
            
        except Exception as e:
            log.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    async def share_knowledge(self, knowledge_data) -> bool:
        """Share knowledge with the neural mesh"""
        try:
            # Handle both AgentKnowledge objects and dictionaries
            if hasattr(knowledge_data, 'content'):
                # It's an AgentKnowledge object
                content = knowledge_data.content
                agent_id = getattr(knowledge_data, 'agent_id', 'unknown')
                context = getattr(knowledge_data, 'context', {})
                confidence = getattr(knowledge_data, 'confidence', 0.8)
            else:
                # It's a dictionary
                content = knowledge_data.get("content", "")
                agent_id = knowledge_data.get("agent_id", "unknown")
                context = knowledge_data.get("context", {})
                confidence = knowledge_data.get("confidence", 0.8)
            
            # Store as MemoryItem in L1 memory
            memory_item = MemoryItem(
                key=f"knowledge_{agent_id}_{int(time.time())}",
                value=content,
                context={"type": "knowledge", **context},
                metadata={
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "shared_at": time.time()
                }
            )
            success = await self.l1_memory.store(memory_item)
            
            if success:
                log.info(f"✅ Knowledge shared with neural mesh from agent {agent_id}")
            return success
            
        except Exception as e:
            log.error(f"Failed to share knowledge: {e}")
            return False
    
    async def update_goal_progress(self, goal_id: str, agent_id: str, progress_data: Dict[str, Any]) -> bool:
        """Update goal progress in the neural mesh"""
        try:
            progress_item = MemoryItem(
                key=f"goal_progress_{goal_id}_{agent_id}",
                value=f"Goal {goal_id} progress by agent {agent_id}: {progress_data.get('status', 'unknown')}",
                context={"type": "goal_progress"},
                metadata={
                    "goal_id": goal_id,
                    "agent_id": agent_id,
                    "progress": progress_data.get("progress", 0.0),
                    "status": progress_data.get("status", "unknown"),
                    "updated_at": time.time(),
                    **progress_data.get("metadata", {})
                }
            )
            
            success = await self.l1_memory.store(progress_item)
            if success:
                log.info(f"✅ Goal {goal_id} progress updated in neural mesh")
            return success
            
        except Exception as e:
            log.error(f"Failed to update goal progress: {e}")
            return False
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through the neural mesh"""
        try:
            # Store coordination request
            coord_item = MemoryItem(
                key=f"coordination_{int(time.time())}",
                value=f"Agent coordination: {coordination_request.get('objective', 'unknown')}",
                context={"type": "coordination"},
                metadata={
                    "objective": coordination_request.get("objective", ""),
                    "agents": coordination_request.get("agents", []),
                    "priority": coordination_request.get("priority", "normal"),
                    "created_at": time.time()
                }
            )
            
            await self.l1_memory.store(coord_item)
            
            return {
                "status": "coordinated",
                "coordination_id": f"coord_{int(time.time())}",
                "agents_involved": coordination_request.get("agents", []),
                "objective": coordination_request.get("objective", "")
            }
            
        except Exception as e:
            log.error(f"Failed to coordinate agents: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search in the neural mesh"""
        try:
            # Create query object
            search_query = Query(
                content=query,
                query_type="semantic_search",
                metadata={"limit": limit, "timestamp": time.time()}
            )
            
            # Search in L1 memory
            results = await self.l1_memory.retrieve(search_query)
            
            # Convert to dict format
            search_results = []
            for item in results[:limit]:
                search_results.append({
                    "content": item.content,
                    "type": item.item_type,
                    "metadata": item.metadata,
                    "confidence": item.metadata.get("confidence", 0.8)
                })
            
            return search_results
            
        except Exception as e:
            log.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get neural mesh system status"""
        try:
            base_stats = await self.get_stats()
            
            return {
                "neural_mesh_status": "operational",
                "agent_count": len(base_stats.get("agents", [])),
                "knowledge_items": base_stats.get("total_items", 0),
                "memory_tiers": base_stats.get("active_tiers", 0),
                "last_update": time.time(),
                "capabilities": ["agent_registration", "knowledge_sharing", "coordination", "semantic_search"]
            }
            
        except Exception as e:
            log.error(f"Failed to get system status: {e}")
            return {"neural_mesh_status": "error", "error": str(e)}

# Backward compatibility wrapper
class MemoryMesh(EnhancedNeuralMesh):
    """Backward compatibility wrapper for existing code"""
    
    def __init__(self, scope: str, actor: str = "gateway", **kwargs):
        # Extract agent_id and swarm_id from scope
        if ":" in scope:
            scope_type, scope_id = scope.split(":", 1)
            if scope_type == "agent":
                agent_id = scope_id
                swarm_id = kwargs.get("swarm_id", "default")
            elif scope_type == "swarm":
                agent_id = actor
                swarm_id = scope_id
            else:
                agent_id = actor
                swarm_id = scope_id
        else:
            agent_id = actor
            swarm_id = scope
            
        super().__init__(
            agent_id=agent_id,
            swarm_id=swarm_id,
            redis_url=kwargs.get("redis_url")
        )
        
        # Store original parameters for compatibility
        self.scope = scope
        self.actor = actor
        
    def key_ns(self, key: str) -> str:
        """Namespaced key for backward compatibility"""
        return f"{self.scope}:{key}"
        
    def set(self, key: str, value: Any) -> Any:
        """Synchronous set for backward compatibility"""
        # This is a simplified sync wrapper - in production, use async
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.store(key, value))
        except Exception:
            # Fallback for environments without event loop
            return True
            
    def get(self, key: str, default: Any = None) -> Any:
        """Synchronous get for backward compatibility"""
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self.retrieve(key, top_k=1))
            return results[0].value if results else default
        except Exception:
            return default
            
    def search(self, query: str, top_k: int = 5, min_score: float = 0.7, 
               scopes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search with backward compatible interface"""
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                self.retrieve(query, top_k=top_k, min_score=min_score)
            )
            
            # Convert to expected format
            return [
                {
                    "key": item.key,
                    "score": 1.0,  # Simplified score
                    "text": str(item.value),
                    "metadata": item.metadata
                }
                for item in results
            ]
        except Exception:
            return []
