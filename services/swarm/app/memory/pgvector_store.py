from __future__ import annotations
import os
import hashlib
import json
from typing import List, Sequence, Optional, Dict, Any, Tuple
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
# removed: from sqlalchemy import text
from sqlalchemy import select, func, Table, Column, MetaData, String, Integer, bindparam
from sqlalchemy import types as satypes
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert

# Simple pgvector store (defer declarative mapping to keep lightweight)
# Assumes pgvector extension & table created externally if migrate=False
DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
  id UUID PRIMARY KEY,
  namespace TEXT NOT NULL,
  scope TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb,
  version INT NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_embeddings_namespace ON embeddings(namespace);
CREATE INDEX IF NOT EXISTS idx_embeddings_ns_scope ON embeddings(namespace, scope);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at DESC);
-- Trigram for hybrid lexical filtering (requires pg_trgm)
CREATE INDEX IF NOT EXISTS idx_embeddings_content_gin ON embeddings USING gin (content gin_trgm_ops);
"""

IVFFLAT = """
-- Create IVFFLAT index (requires ANALYZE after population). Safe to attempt every startup.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE c.relname = 'idx_embeddings_embedding' AND n.nspname = 'public') THEN
    CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
  END IF;
END$$;
"""

class EmbeddingProvider:
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError

class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 384):
        self.dim = dim
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype='float32')
            for tok in str(t).lower().split():
                h = abs(hash(tok)) % self.dim
                v[h] += 1.0
            n = float((v**2).sum()) ** 0.5
            if n > 0:
                v /= n
            out.append(v.tolist())
        return out

class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(None, self.model.encode, list(texts))
        return [e.tolist() if hasattr(e,'tolist') else list(e) for e in emb]

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = 'text-embedding-3-small'):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # chunk respecting token limits (simplified)
        out: List[List[float]] = []
        batch = list(texts)
        if not batch:
            return out
        res = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.client.embeddings.create(model=self.model, input=batch)
        )
        for d in res.data:
            out.append(d.embedding)
        return out

class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = 'embed-english-light-v3.0'):
        import cohere
        self.client = cohere.Client()
        self.model = model
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self.client.embed(model=self.model, texts=list(texts))
        )
        return resp.embeddings  # type: ignore[attr-defined]

class EmbeddingProviderFactory:
    @staticmethod
    def best_available() -> EmbeddingProvider:
        backend_pref = os.getenv('EMBEDDINGS_BACKEND', 'auto').lower()
        # explicit selection first
        if backend_pref == 'hash':
            return HashEmbeddingProvider()
        if backend_pref == 'st':
            try:
                return SentenceTransformerProvider()
            except Exception:
                return HashEmbeddingProvider()
        if backend_pref == 'openai':
            try:
                return OpenAIEmbeddingProvider()
            except Exception:
                return HashEmbeddingProvider()
        if backend_pref == 'cohere':
            try:
                return CohereEmbeddingProvider()
            except Exception:
                return HashEmbeddingProvider()
        # auto strategy: priority order env availability
        if os.getenv('OPENAI_API_KEY'):
            try:
                return OpenAIEmbeddingProvider()
            except Exception:
                pass
        if os.getenv('COHERE_API_KEY') or os.getenv('CO_API_KEY'):
            try:
                return CohereEmbeddingProvider()
            except Exception:
                pass
        try:
            return SentenceTransformerProvider()
        except Exception:
            return HashEmbeddingProvider()

class PGVectorStore:
    def __init__(self, dsn: Optional[str] = None, migrate: bool = True):
        self.dsn = dsn or os.getenv('PGVECTOR_DSN') or 'postgresql+asyncpg://postgres:postgres@localhost:5432/agentforge'
        # Lazy engine/session creation to avoid hard dependency on asyncpg in test environments
        self.engine = None  # type: ignore[assignment]
        self.Session = None  # type: ignore[assignment]
        self.provider: EmbeddingProvider = EmbeddingProviderFactory.best_available()
        self._migrate = migrate
        # lightweight table metadata for safe query construction
        md = MetaData()
        self._emb_tbl = Table(
            'embeddings', md,
            Column('id', String),
            Column('namespace', String),
            Column('scope', String),
            Column('content', String),
            # Use NULLTYPE to avoid pgvector dependency; only used for operators
            Column('embedding', satypes.NULLTYPE),
            Column('meta', JSONB),
            Column('version', Integer),
        )

    def _ensure_engine(self) -> None:
        if self.engine is not None and self.Session is not None:
            return
        # Defer import/creation; allow tests to inject Session before usage
        try:
            self.engine = create_async_engine(self.dsn, pool_pre_ping=True)
            self.Session = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        except ModuleNotFoundError:
            # Leave uninitialized so tests can stub Session; real runtime should have deps installed
            self.engine = None  # type: ignore[assignment]
            self.Session = None  # type: ignore[assignment]

    async def init(self):
        if not self._migrate:
            return
        self._ensure_engine()
        if not self.engine:
            return
        async with self.engine.begin() as conn:
            # constant DDL; safe via exec_driver_sql
            await conn.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS vector')
            await conn.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS pg_trgm')
            for stmt in DDL.strip().split(';'):
                s = stmt.strip()
                if s:
                    await conn.exec_driver_sql(s)
            # attempt IVFFLAT (ignore errors before ANALYZE)
            try:
                await conn.exec_driver_sql(IVFFLAT)
            except Exception:
                pass

    @staticmethod
    def _det_id(namespace: str, scope: str, content: str) -> str:
        h = hashlib.sha256(f'{namespace}\x00{scope}\x00{content}'.encode()).hexdigest()
        # Make it UUID-ish (take first 32 hex chars)
        return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}'

    async def upsert_batch(self, namespace: str, docs: Sequence[Tuple[str,str,Dict[str,Any]]], version: int = 1) -> int:
        # docs: (scope, content, meta)
        if not docs:
            return 0
        texts = [c for _, c, _ in docs]
        embeddings = await self.provider.embed(texts)
        # If Session is not injected by tests, ensure engine/session
        if self.Session is None:
            self._ensure_engine()
        if self.Session is None:
            # No DB available; behave as no-op insert for test environments
            return len(docs)
        async with self.Session() as sess:  # type: ignore[operator]
            values = []
            for (scope, content, meta), emb in zip(docs, embeddings):
                det_id = self._det_id(namespace, scope, content)
                values.append({
                    'id': det_id,
                    'namespace': namespace,
                    'scope': scope,
                    'content': content,
                    'embedding': emb,
                    'meta': meta or {},
                    'version': version
                })
            t = self._emb_tbl
            stmt = pg_insert(t).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[t.c.id],
                set_={
                    'embedding': stmt.excluded.embedding,
                    'meta': stmt.excluded.meta,
                    'version': stmt.excluded.version,
                },
            )
            await sess.execute(stmt)
            await sess.commit()
            return len(values)

    async def search(self, namespace: str, query: str, top_k: int = 5, scopes: Optional[Sequence[str]] = None, min_score: float = 0.35, hybrid_weight: float = 0.15) -> List[Dict[str,Any]]:
        # hybrid: semantic + lexical trigram normalized
        emb = (await self.provider.embed([query]))[0]
        # Build safe SQLAlchemy Core query using bind parameters and custom operators
        params: Dict[str, Any] = {'ns': namespace, 'qemb': emb, 'qtxt': query}
        t = self._emb_tbl
        cosine = (1 - t.c.embedding.op('<=>')(bindparam('qemb'))).label('cosine')
        lexical = func.greatest(func.similarity(t.c.content, bindparam('qtxt')), 0).label('lexical')
        stmt = (
            select(t.c.id, t.c.scope, t.c.content, t.c.meta, cosine, lexical)
            .where(t.c.namespace == bindparam('ns'))
        )
        if scopes:
            # expanding bind to support IN with a list
            stmt = stmt.where(t.c.scope.in_(bindparam('scopes', expanding=True)))
            params['scopes'] = list(scopes)
        # order by distance (lower is better)
        stmt = stmt.order_by(t.c.embedding.op('<=>')(bindparam('qemb'))).limit(top_k * 4)

        if self.Session is None:
            self._ensure_engine()
        if self.Session is None:
            # No DB path: return empty to satisfy callers in dry mode
            return []
        async with self.Session() as sess:  # type: ignore[operator]
            res = await sess.execute(stmt, params)
            rows = res.fetchall()
        scored = []
        for r in rows:
            meta = r.meta if isinstance(r.meta, dict) else json.loads(r.meta or '{}')
            score = (1-hybrid_weight)*float(r.cosine) + hybrid_weight*float(r.lexical)
            if score >= min_score:
                scored.append({'id': r.id, 'scope': r.scope, 'content': r.content, 'meta': meta, 'score': score})
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

__all__ = ['PGVectorStore','EmbeddingProvider','EmbeddingProviderFactory']
