"""
Consolidated Memory System for AgentForge
Combines functionality from multiple memory implementations into a unified system
"""
from __future__ import annotations

import os
import time
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Optional imports with fallbacks
try:
    import redis
except ImportError:
    redis = None

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

try:
    from ..memory.crdt import LWWMap, Op
except ImportError:
    # Fallback CRDT implementation
    @dataclass
    class Op:
        key: str
        value: Any
        ts: float
        actor: str
    
    class LWWMap:
        def __init__(self):
            self.data = {}
            
        def set(self, key: str, value: Any, actor: str) -> Op:
            op = Op(key=key, value=value, ts=time.time(), actor=actor)
            self.data[key] = (value, op.ts, actor)
            return op
            
        def get(self, key: str, default: Any = None) -> Any:
            if key in self.data:
                return self.data[key][0]
            return default

log = logging.getLogger("memory")

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        pass

class HashEmbeddingProvider(EmbeddingProvider):
    """Simple hash-based embedding provider for development/testing"""
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        
    def encode(self, text: str) -> List[float]:
        """Create deterministic embedding from text hash"""
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
        vec = hashes[:self.dimensions]
        if len(vec) < self.dimensions:
            vec.extend([0.0] * (self.dimensions - len(vec)))
            
        # L2 normalize
        norm = sum(x*x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
            
        return vec

class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence transformer embedding provider"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                log.warning("sentence-transformers not available, falling back to hash embeddings")
                self._model = HashEmbeddingProvider()
        return self._model
        
    def encode(self, text: str) -> List[float]:
        if hasattr(self.model, 'encode'):
            # Sentence transformer
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        else:
            # Fallback to hash
            return self.model.encode(text)

class MemoryMesh:
    """
    Unified memory mesh combining local CRDT with optional distributed features.
    Supports multiple storage backends and embedding providers.
    """
    
    def __init__(
        self, 
        scope: str, 
        actor: str = "gateway",
        ttl_seconds: Optional[int] = None,
        embedding_provider: Optional[str] = None
    ):
        self.scope = scope
        self.actor = actor
        self.ttl = ttl_seconds or int(os.getenv("MEMORY_TTL_SECONDS", "604800"))  # 7 days
        
        # Initialize CRDT
        self.crdt = LWWMap()
        
        # Initialize storage backends
        self._init_redis()
        self._init_pinecone()
        self._init_embedding_provider(embedding_provider)
        
        # Local storage fallbacks
        self._kv_fallback: Dict[str, Any] = {}
        self._vector_fallback: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        
        # Pruning configuration
        self.prune_threshold = int(os.getenv("MEMORY_PRUNE_THRESHOLD", "1000"))
        
        log.info(f"Initialized memory mesh for scope '{scope}' with actor '{actor}'")

    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        self.redis = None
        if redis:
            try:
                redis_url = os.getenv("REDIS_URL")
                if redis_url:
                    self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
                else:
                    host = os.getenv("REDIS_HOST", "localhost")
                    port = int(os.getenv("REDIS_PORT", "6379"))
                    db = int(os.getenv("REDIS_DB", "0"))
                    self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                
                # Test connection
                self.redis.ping()
                log.info("Redis connection established")
                
            except Exception as e:
                log.warning(f"Redis connection failed: {e}")
                self.redis = None

    def _init_pinecone(self) -> None:
        """Initialize Pinecone vector database"""
        self.pinecone_index = None
        self._pinecone_available = False
        
        if Pinecone and os.getenv("AF_ENABLE_PINECONE", "0") == "1":
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                try:
                    pc = Pinecone(api_key=api_key)
                    
                    index_name = os.getenv("PINECONE_INDEX", "agentforge-memory")
                    
                    # Check if index exists
                    existing_indexes = set()
                    try:
                        indexes = pc.list_indexes()
                        existing_indexes = {idx.name for idx in indexes.indexes}
                    except:
                        pass
                    
                    if index_name in existing_indexes:
                        self.pinecone_index = pc.Index(index_name)
                        self._pinecone_available = True
                        log.info(f"Connected to Pinecone index '{index_name}'")
                    else:
                        log.warning(f"Pinecone index '{index_name}' not found")
                        
                except Exception as e:
                    log.warning(f"Pinecone initialization failed: {e}")

    def _init_embedding_provider(self, provider: Optional[str] = None) -> None:
        """Initialize embedding provider"""
        provider = provider or os.getenv("EMBEDDINGS_BACKEND", "hash")
        
        if provider == "st" or provider == "sentence-transformers":
            self.embedder = SentenceTransformerProvider()
        elif provider == "hash":
            self.embedder = HashEmbeddingProvider()
        else:
            log.warning(f"Unknown embedding provider '{provider}', using hash")
            self.embedder = HashEmbeddingProvider()
            
        log.info(f"Using embedding provider: {type(self.embedder).__name__}")

    def key_ns(self, key: str) -> str:
        """Get namespaced key"""
        return f"{self.scope}:{key}"

    def set(self, key: str, value: Any) -> Op:
        """Set a value in the memory mesh"""
        namespaced_key = self.key_ns(key)
        
        # Update CRDT
        op = self.crdt.set(namespaced_key, value, self.actor)
        
        # Store in Redis if available
        if self.redis:
            try:
                serialized = json.dumps(value) if not isinstance(value, str) else value
                if self.ttl:
                    self.redis.setex(namespaced_key, self.ttl, serialized)
                else:
                    self.redis.set(namespaced_key, serialized)
            except Exception as e:
                log.debug(f"Redis storage failed for key {namespaced_key}: {e}")
                
        # Fallback to local storage
        self._kv_fallback[namespaced_key] = value
        
        # Store embedding if text value
        if isinstance(value, str):
            self._store_embedding(namespaced_key, value)
            
        # Prune if needed
        self._maybe_prune()
        
        return op

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the memory mesh"""
        namespaced_key = self.key_ns(key)
        
        # Try CRDT first
        value = self.crdt.get(namespaced_key, None)
        if value is not None:
            return value
            
        # Try Redis
        if self.redis:
            try:
                redis_value = self.redis.get(namespaced_key)
                if redis_value is not None:
                    try:
                        return json.loads(redis_value)
                    except json.JSONDecodeError:
                        return redis_value
            except Exception as e:
                log.debug(f"Redis retrieval failed for key {namespaced_key}: {e}")
                
        # Fallback to local storage
        return self._kv_fallback.get(namespaced_key, default)

    def delete(self, key: str) -> None:
        """Delete a key from the memory mesh"""
        namespaced_key = self.key_ns(key)
        
        # Remove from CRDT (tombstone)
        self.crdt.set(namespaced_key, None, self.actor)
        
        # Remove from Redis
        if self.redis:
            try:
                self.redis.delete(namespaced_key)
            except Exception as e:
                log.debug(f"Redis deletion failed for key {namespaced_key}: {e}")
                
        # Remove from local storage
        self._kv_fallback.pop(namespaced_key, None)
        self._vector_fallback.pop(namespaced_key, None)

    def _store_embedding(self, key: str, text: str) -> None:
        """Store text embedding for semantic search"""
        try:
            embedding = self.embedder.encode(text)
            metadata = {
                "text": text,
                "key": key,
                "scope": self.scope,
                "timestamp": time.time()
            }
            
            # Store in Pinecone if available
            if self._pinecone_available and self.pinecone_index:
                try:
                    self.pinecone_index.upsert([(key, embedding, metadata)])
                except Exception as e:
                    log.debug(f"Pinecone upsert failed for key {key}: {e}")
                    
            # Store in local fallback
            self._vector_fallback[key] = (embedding, metadata)
            
        except Exception as e:
            log.debug(f"Embedding storage failed for key {key}: {e}")

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.7,
        scopes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search across stored embeddings"""
        try:
            query_embedding = self.embedder.encode(query)
            results = []
            
            # Search Pinecone if available
            if self._pinecone_available and self.pinecone_index:
                try:
                    filter_dict = None
                    if scopes:
                        filter_dict = {"scope": {"$in": scopes}}
                        
                    search_results = self.pinecone_index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter_dict
                    )
                    
                    for match in search_results.matches:
                        if match.score >= min_score:
                            results.append({
                                "key": match.id,
                                "score": match.score,
                                "text": match.metadata.get("text", ""),
                                "metadata": match.metadata
                            })
                            
                except Exception as e:
                    log.debug(f"Pinecone search failed: {e}")
                    
            # Fallback to local vector search
            if not results:
                results = self._local_vector_search(query_embedding, top_k, min_score, scopes)
                
            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            
        except Exception as e:
            log.error(f"Search failed for query '{query}': {e}")
            return []

    def _local_vector_search(
        self, 
        query_embedding: List[float], 
        top_k: int, 
        min_score: float,
        scopes: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Local vector similarity search"""
        results = []
        
        for key, (embedding, metadata) in self._vector_fallback.items():
            # Filter by scope if specified
            if scopes and metadata.get("scope") not in scopes:
                continue
                
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, embedding)
            
            if score >= min_score:
                results.append({
                    "key": key,
                    "score": score,
                    "text": metadata.get("text", ""),
                    "metadata": metadata
                })
                
        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0
            
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    def _maybe_prune(self) -> None:
        """Prune old entries if threshold exceeded"""
        total_entries = len(self._kv_fallback) + len(self._vector_fallback)
        
        if total_entries > self.prune_threshold:
            # Simple LRU-like pruning based on timestamps
            cutoff_time = time.time() - self.ttl
            
            # Prune KV entries (need to add timestamps)
            keys_to_remove = []
            for key in list(self._kv_fallback.keys()):
                # For now, remove oldest entries (would need better timestamp tracking)
                if len(keys_to_remove) < total_entries - self.prune_threshold:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                self._kv_fallback.pop(key, None)
                
            # Prune vector entries based on timestamp
            vector_keys_to_remove = [
                key for key, (_, metadata) in self._vector_fallback.items()
                if metadata.get("timestamp", 0) < cutoff_time
            ]
            
            for key in vector_keys_to_remove:
                self._vector_fallback.pop(key, None)
                
            log.info(f"Pruned {len(keys_to_remove) + len(vector_keys_to_remove)} old memory entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory mesh statistics"""
        return {
            "scope": self.scope,
            "actor": self.actor,
            "kv_entries": len(self._kv_fallback),
            "vector_entries": len(self._vector_fallback),
            "redis_available": self.redis is not None,
            "pinecone_available": self._pinecone_available,
            "embedding_provider": type(self.embedder).__name__,
            "ttl_seconds": self.ttl,
            "prune_threshold": self.prune_threshold
        }

    # Backward compatibility methods
    def store(self, key: str, value: Any, scope: str = "task") -> None:
        """Store value with scope context (backward compatibility)"""
        scoped_key = f"{scope}:{key}"
        self.set(scoped_key, value)
        
    def add(self, key: str, value: str, scopes: Optional[List[str]] = None) -> None:
        """Add value to memory (backward compatibility)"""
        scope = scopes[0] if scopes else "task"
        self.store(key, value, scope)
        
    def retrieve(self, key: str) -> Any:
        """Retrieve value from memory (backward compatibility)"""
        return self.get(key)
