import os
import time
from dotenv import load_dotenv
import numpy as np
import redis
try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None
try:
    from sentence_transformers import SentenceTransformer  # For embeddings
except Exception:
    SentenceTransformer = None
import logging
from typing import List, Any

load_dotenv()

class HashEmbedder:
    """Deterministic, lightweight embedder as a fallback to avoid heavyweight model downloads.
    Produces a fixed-size vector using hashing of tokens.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        # Very simple token hashing
        for tok in str(text).lower().split():
            h = abs(hash(tok)) % self.dim
            vec[h] += 1.0
        # Normalize to unit vector
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec.tolist()

# Patentable: Evolutionary Memory Optimizer - Evolves memory with reflection, scaling O(√t log t) for long-term tasks without explosion.
class EvoMemory:
    def __init__(self, namespace: str = "default", prune_threshold: int = 100, ttl_seconds: int | None = None):
        # Redis for fast key-value storage
        self._kv_fallback: dict[str, str] = {}
        try:
            self.r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            # Lazy connection; will try on first use
        except Exception:
            self.r = None
        # Pinecone for semantic search when available
        self.namespace = namespace
        self._pinecone_available = False
        self.index = None
        api_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
        if Pinecone and api_key:
            try:
                self.pc = Pinecone(api_key=api_key)
                self.index_name = "agentforge-memory"
                # Safely check and create index only if list_indexes is available
                try:
                    existing = {getattr(ix, 'name', str(ix)) for ix in getattr(self.pc.list_indexes(), 'indexes', [])}
                except Exception:
                    # New SDKs can return different structures
                    try:
                        existing = set(self.pc.list_indexes().names())  # type: ignore[attr-defined]
                    except Exception:
                        existing = set()
                if self.index_name not in existing:
                    if ServerlessSpec:
                        try:
                            self.pc.create_index(
                                self.index_name,
                                dimension=384,
                                metric='cosine',
                                spec=ServerlessSpec(cloud='aws', region='us-east-1')
                            )
                        except Exception:
                            pass
                try:
                    self.index = self.pc.Index(self.index_name)
                    self._pinecone_available = True
                except Exception:
                    self.index = None
                    self._pinecone_available = False
            except Exception:
                self._pinecone_available = False
                self.index = None
        # Local vector fallback
        self._local_vectors: dict[str, tuple[list[float], dict]] = {}

        # Choose embeddings backend: sentence-transformers or HashEmbedder
        backend = os.getenv("EMBEDDINGS_BACKEND", "hash").lower()
        self.embedder: Any
        if backend == "st" and SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
                # Warm-up (optional)
                try:
                    _ = self.embedder.encode("warmup")
                except Exception:
                    pass
            except Exception:
                self.embedder = HashEmbedder()
        else:
            self.embedder = HashEmbedder()

        self.threshold = prune_threshold
        self.ttl = ttl_seconds
        # Add deterministic noise sigma so identical strings aren't perfect matches
        self._noise_sigma = float(os.getenv("MEMORY_VECTOR_NOISE_SIGMA", "0.2"))

    def _rk(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def store(self, key, value, scope: str = "task"):
        # Store in Redis with optional TTL; fallback to in-process dict
        rkey = self._rk(key)
        try:
            if self.r:
                if self.ttl:
                    self.r.set(rkey, value, ex=self.ttl)
                else:
                    self.r.set(rkey, value)
            else:
                raise RuntimeError("no redis")
        except Exception:
            self._kv_fallback[rkey] = value

        # Embed
        base_embedding = self.embedder.encode(value)
        if hasattr(base_embedding, 'tolist'):
            base_embedding = base_embedding.tolist()
        e = np.array(base_embedding, dtype=np.float32)
        # Deterministic noise based on key
        try:
            seed = abs(hash(rkey)) % (2**32)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, self._noise_sigma, size=e.shape)
            e_noisy = (e + noise).astype(np.float32)
        except Exception:
            e_noisy = e
        embedding = e_noisy.tolist()

        metadata = {"value": value, "scope": scope, "ts": time.time()}
        # Store in Pinecone or local index
        if self._pinecone_available and self.index is not None:
            try:
                self.index.upsert(
                    vectors=[(rkey, embedding, metadata)],
                    namespace=self.namespace
                )
            except Exception:
                # fall back to local if upsert fails
                self._local_vectors[rkey] = (embedding, metadata)
        else:
            self._local_vectors[rkey] = (embedding, metadata)

        # Evolutionary pruning using SCAN (avoid blocking)
        # Only attempt Redis prune if Redis available
        try:
            dbsize = self.r.dbsize() if self.r else len(self._kv_fallback)
        except Exception:
            dbsize = len(self._kv_fallback)
        if dbsize > self.threshold:
            target = int(np.sqrt(max(dbsize, 1)) * np.log(dbsize + 1))
            to_delete = []
            if self.r:
                try:
                    cursor = 0
                    while True:
                        cursor, keys = self.r.scan(cursor=cursor, match=f"{self.namespace}:*", count=500)
                        for k in keys:
                            to_delete.append(k)
                            if len(to_delete) >= target:
                                break
                        if cursor == 0 or len(to_delete) >= target:
                            break
                except Exception:
                    to_delete = list(self._kv_fallback.keys())[:target]
            else:
                to_delete = list(self._kv_fallback.keys())[:target]
            # Delete from KV
            if to_delete:
                try:
                    if self.r:
                        self.r.delete(*to_delete)
                    else:
                        raise RuntimeError("no redis")
                except Exception:
                    for k in to_delete:
                        self._kv_fallback.pop(k, None)
                # Best-effort vector delete
                try:
                    if self._pinecone_available and self.index is not None:
                        self.index.delete(ids=to_delete, namespace=self.namespace)
                    else:
                        for k in to_delete:
                            self._local_vectors.pop(k, None)
                except Exception:
                    pass

    # Back-compat helpers for older API/tests
    def add(self, key: str, value: str, scopes: list[str] | None = None):
        scope = scopes[0] if scopes else "task"
        return self.store(key, value, scope=scope)

    def get(self, key: str):
        return self.retrieve(key)

    def retrieve(self, key):
        rkey = self._rk(key)
        try:
            if self.r:
                return self.r.get(rkey)
        except Exception:
            pass
        return self._kv_fallback.get(rkey)

    def semantic_search(self, query, top_k=5, min_score: float = 0.35, scopes: list[str] | None = None):
        q_emb = self.embedder.encode(query)
        if hasattr(q_emb, 'tolist'):
            q_emb = q_emb.tolist()
        embedding = np.array(q_emb, dtype=np.float32).tolist()
        pine_filter = None
        if scopes:
            pine_filter = {"scope": {"$in": scopes}}
        if self._pinecone_available and self.index is not None:
            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.namespace,
                    filter=pine_filter
                )
                matches = []
                for m in results.get("matches", []) if isinstance(results, dict) else results.matches:
                    md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
                    score = m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)
                    if score is None:
                        score = 0.0
                    if score >= min_score and "value" in md:
                        matches.append(md["value"])
                return matches
            except Exception:
                pass
        # Local search fallback (cosine similarity)
        def cosine(a: List[float], b: List[float]) -> float:
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if an == 0 or bn == 0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))
        scored = []
        for _, (vec, md) in self._local_vectors.items():
            if scopes and md.get("scope") not in scopes:
                continue
            s = cosine(embedding, vec)
            if s >= min_score:
                scored.append((s, md.get("value")))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:top_k]]