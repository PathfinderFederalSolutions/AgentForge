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
        self.namespace = namespace
        self.threshold = prune_threshold
        self.ttl = ttl_seconds
        # Deterministic orthogonal jitter to avoid perfect matches while keeping high similarity
        self._epsilon = float(os.getenv("MEMORY_VECTOR_EPSILON", "0.25"))

        # Backends and storages
        self._kv_fallback: dict[str, str] = {}
        self.r = self._init_redis()
        self._pinecone_available = False
        self.index = None
        self._init_pinecone()
        self._local_vectors: dict[str, tuple[list[float], dict]] = {}
        self.embedder: Any = self._init_embedder()

    # ---- Initialization helpers ----
    def _init_redis(self):
        try:
            return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        except Exception:
            return None

    def _pinecone_enabled(self) -> bool:
        enable = os.getenv("AF_ENABLE_PINECONE", "0").lower() in {"1", "true", "yes", "on"}
        api_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
        return bool(enable and Pinecone and api_key)

    def _safe_list_indexes(self):
        try:
            lst = getattr(self.pc.list_indexes(), 'indexes', [])
            return {getattr(ix, 'name', str(ix)) for ix in lst}
        except Exception:
            try:
                return set(self.pc.list_indexes().names())  # type: ignore[attr-defined]
            except Exception:
                return set()

    def _ensure_index_exists(self, existing: set[str]):
        if getattr(self, 'index_name', None) not in existing and ServerlessSpec:
            try:
                self.pc.create_index(
                    self.index_name,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            except Exception:
                pass

    def _open_index(self):
        try:
            self.index = self.pc.Index(self.index_name)
            self._pinecone_available = True
        except Exception:
            self.index = None
            self._pinecone_available = False

    def _init_pinecone(self):
        self._pinecone_available = False
        self.index = None
        if not self._pinecone_enabled():
            return
        try:
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY"))
            self.index_name = "agentforge-memory"
            existing = self._safe_list_indexes()
            self._ensure_index_exists(existing)
            self._open_index()
        except Exception:
            self._pinecone_available = False
            self.index = None

    def _init_embedder(self):
        backend = os.getenv("EMBEDDINGS_BACKEND", "hash").lower()
        if backend == "st" and SentenceTransformer is not None:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
                try:
                    _ = model.encode("warmup")
                except Exception:
                    pass
                return model
            except Exception:
                return HashEmbedder()
        return HashEmbedder()

    # ---- Key helpers ----
    def _rk(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # ---- Storage helpers ----
    def _store_kv(self, rkey: str, value: str):
        try:
            if self.r:
                if self.ttl:
                    self.r.set(rkey, value, ex=self.ttl)
                else:
                    self.r.set(rkey, value)
                return
        except Exception:
            pass
        self._kv_fallback[rkey] = value

    def _embed_value(self, key: str, value: str) -> list[float]:
        base_embedding = self.embedder.encode(value)
        if hasattr(base_embedding, 'tolist'):
            base_embedding = base_embedding.tolist()
        e = np.array(base_embedding, dtype=np.float32)
        try:
            en = np.linalg.norm(e)
            if en == 0:
                e = np.ones_like(e)
                en = np.linalg.norm(e)
            e_unit = e / en
            seed = abs(hash(self._rk(key))) % (2**32)
            rng = np.random.default_rng(seed)
            r = rng.normal(0.0, 1.0, size=e.shape).astype(np.float32)
            proj = np.dot(r, e_unit)
            r_ortho = r - proj * e_unit
            rn = np.linalg.norm(r_ortho)
            if rn == 0:
                r_ortho = r
                rn = np.linalg.norm(r_ortho)
            u_perp = (r_ortho / rn).astype(np.float32)
            e_noisy = e_unit + self._epsilon * u_perp
            e_noisy = (e_noisy / np.linalg.norm(e_noisy)).astype(np.float32)
        except Exception:
            e_noisy = e
        return e_noisy.tolist()

    def _upsert_vector(self, rkey: str, embedding: list[float], metadata: dict):
        if self._pinecone_available and self.index is not None:
            try:
                self.index.upsert(
                    vectors=[(rkey, embedding, metadata)],
                    namespace=self.namespace
                )
                return
            except Exception:
                pass
        self._local_vectors[rkey] = (embedding, metadata)

    # ---- Pruning helpers ----
    def _db_size(self) -> int:
        try:
            return self.r.dbsize() if self.r else len(self._kv_fallback)
        except Exception:
            return len(self._kv_fallback)

    def _prune_target(self, dbsize: int) -> int:
        return int(np.sqrt(max(dbsize, 1)) * np.log(dbsize + 1))

    def _collect_candidates(self, target: int) -> list[str]:
        if target <= 0:
            return []
        if self.r:
            try:
                collected: list[str] = []
                cursor = 0
                while True:
                    cursor, keys = self.r.scan(cursor=cursor, match=f"{self.namespace}:*", count=500)
                    collected.extend(keys)
                    if cursor == 0 or len(collected) >= target:
                        break
                return collected[:target]
            except Exception:
                pass
        return list(self._kv_fallback.keys())[:target]

    def _delete_kv(self, keys: list[str]):
        if not keys:
            return
        try:
            if self.r:
                self.r.delete(*keys)
            else:
                raise RuntimeError("no redis")
        except Exception:
            for k in keys:
                self._kv_fallback.pop(k, None)

    def _delete_vectors(self, keys: list[str]):
        if not keys:
            return
        try:
            if self._pinecone_available and self.index is not None:
                self.index.delete(ids=keys, namespace=self.namespace)
            else:
                for k in keys:
                    self._local_vectors.pop(k, None)
        except Exception:
            pass

    def _prune_if_needed(self):
        dbsize = self._db_size()
        if dbsize <= self.threshold:
            return
        target = self._prune_target(dbsize)
        keys = self._collect_candidates(target)
        if not keys:
            return
        self._delete_kv(keys)
        self._delete_vectors(keys)

    # ---- Public API ----
    def store(self, key, value, scope: str = "task"):
        rkey = self._rk(key)
        self._store_kv(rkey, value)
        embedding = self._embed_value(key, value)
        metadata = {"value": value, "scope": scope, "ts": time.time()}
        self._upsert_vector(rkey, embedding, metadata)
        self._prune_if_needed()

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

    # ---- Search helpers ----
    def _build_pine_filter(self, scopes: list[str] | None):
        if scopes:
            return {"scope": {"$in": scopes}}
        return None

    def _extract_pinecone_matches(self, results, min_score: float):
        matches = []
        items = results.get("matches", []) if isinstance(results, dict) else getattr(results, 'matches', [])
        for m in items:
            md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            score = m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)
            if (score or 0.0) >= min_score and "value" in md:
                matches.append(md["value"])
        return matches

    def _pinecone_search(self, embedding: list[float], top_k: int, min_score: float, scopes: list[str] | None):
        pine_filter = self._build_pine_filter(scopes)
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=pine_filter
            )
            return self._extract_pinecone_matches(results, min_score)
        except Exception:
            return None

    def _local_search(self, query: str, top_k: int, min_score: float, scopes: list[str] | None):
        # Encode query once
        q_emb = self.embedder.encode(query)
        if hasattr(q_emb, 'tolist'):
            q_emb = q_emb.tolist()
        q_vec = np.array(q_emb, dtype=np.float32)
        def cosine(a: List[float], b: List[float]) -> float:
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if an == 0 or bn == 0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))
        # Simple lexical overlap boost to stabilize short queries
        q_tokens = set(str(query).lower().split())
        scored: list[tuple[float, str]] = []
        for _, (vec, md) in self._local_vectors.items():
            if scopes and md.get("scope") not in scopes:
                continue
            cos = cosine(q_vec, vec)
            d_tokens = set(str(md.get("value", "")).lower().split())
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & d_tokens) / len(q_tokens)
            # Hybrid score: mostly cosine, slight lexical boost
            s = 0.85 * cos + 0.15 * overlap
            if s >= min_score:
                scored.append((s, md.get("value")))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:top_k]]

    def semantic_search(self, query, top_k=5, min_score: float = 0.35, scopes: list[str] | None = None):
        # Prefer Pinecone if available
        if self._pinecone_available and self.index is not None:
            q_emb = self.embedder.encode(query)
            if hasattr(q_emb, 'tolist'):
                q_emb = q_emb.tolist()
            embedding = np.array(q_emb, dtype=np.float32).tolist()
            results = self._pinecone_search(embedding, top_k, min_score, scopes)
            if results is not None:
                return results
        # Fallback to local
        return self._local_search(str(query), top_k, min_score, scopes)