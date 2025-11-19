from __future__ import annotations
import asyncio
import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover
    class Vector:  # type: ignore
        def __init__(self, *a, **k):
            pass
from sqlalchemy import String, Integer, JSON as SA_JSON, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import select

from ..config import settings

PII_RE = re.compile(r"(\b\d{3}-\d{2}-\d{4}\b|\b\d{16}\b|[\w\.-]+@[\w\.-]+)")


class Base(DeclarativeBase):
    pass


class VectorMem(Base):
    __tablename__ = "vector_mem"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(256), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))  # default dim
    meta: Mapped[dict] = mapped_column(SA_JSON)
    expire_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


_engine = create_engine(os.getenv("VECTOR_DATABASE_URL") or os.getenv("VECTOR_DB_URL") or settings.db_url, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)

_init_done = False


def init_db():
    global _init_done
    if not _init_done:
        Base.metadata.create_all(_engine)
        try:
            with _engine.begin() as conn:
                conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            # Extension may not be available (e.g., SQLite during tests)
            pass
        _init_done = True


def scrub(text: str) -> str:
    return PII_RE.sub("[REDACTED]", text)


async def embed_async(texts: List[str], dim: int = 1536) -> List[List[float]]:
    """Production embedding using OpenAI or fallback to deterministic hash"""
    import os
    
    # Try OpenAI embeddings first
    try:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            client = AsyncOpenAI(api_key=api_key)
            
            # Use text-embedding-3-large for production quality
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=texts,
                dimensions=min(dim, 3072)  # Max dimensions for this model
            )
            
            embeddings = []
            for embedding_obj in response.data:
                embedding = embedding_obj.embedding
                # Resize to requested dimensions if needed
                if len(embedding) != dim:
                    import numpy as np
                    embedding_array = np.array(embedding)
                    if len(embedding) > dim:
                        # Truncate
                        embedding = embedding_array[:dim].tolist()
                    else:
                        # Pad with zeros
                        padding = np.zeros(dim - len(embedding))
                        embedding = np.concatenate([embedding_array, padding]).tolist()
                embeddings.append(embedding)
            
            return embeddings
            
    except Exception as e:
        print(f"OpenAI embeddings failed, using fallback: {e}")
    
    # Try Sentence Transformers as fallback
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = model.encode(texts, convert_to_tensor=False)
        
        # Resize embeddings to requested dimensions
        import numpy as np
        resized_embeddings = []
        for embedding in embeddings:
            if len(embedding) != dim:
                if len(embedding) > dim:
                    # Truncate
                    embedding = embedding[:dim]
                else:
                    # Pad with normalized values
                    padding = np.random.normal(0, 0.1, dim - len(embedding))
                    embedding = np.concatenate([embedding, padding])
            resized_embeddings.append(embedding.tolist())
        
        return resized_embeddings
        
    except Exception as e:
        print(f"Sentence Transformers failed, using deterministic fallback: {e}")
    
    # Deterministic fallback for development/testing
    out: List[List[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.resize(arr / 255.0, dim)
        out.append(vec.tolist())
    
    await asyncio.sleep(0)
    return out


def upsert(scope: str, key: str, content: str, meta: Dict[str, Any], ttl_seconds: Optional[int] = None, dim: int = 1536) -> None:
    init_db()
    content = scrub(content)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    expire_at = int(time.time()) + ttl_seconds if ttl_seconds else None
    emb = asyncio.get_event_loop().run_until_complete(embed_async([content], dim=dim))[0]
    with SessionLocal.begin() as s:
        row = VectorMem(scope=scope, key=key, sha256=sha, dim=dim, embedding=emb, meta=meta, expire_at=expire_at)
        s.add(row)


def search(scope: str, query: str, top_k: int = 5, dim: int = 1536) -> List[Dict[str, Any]]:
    init_db()
    try:
        qvec = asyncio.get_event_loop().run_until_complete(embed_async([scrub(query)], dim=dim))[0]
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            qvec = loop.run_until_complete(embed_async([scrub(query)], dim=dim))[0]
        finally:
            loop.close()
    with SessionLocal() as s:
        try:
            t = VectorMem.__table__
            stmt = select(
                t.c.id,
                t.c.scope,
                t.c.key,
                t.c.sha256,
                t.c.meta,
                t.c.expire_at,
            )
            stmt = stmt.where(t.c.scope == scope)
            # Prefer pgvector comparator if available; otherwise degrade to stable ordering
            distance_expr = None
            emb_col = getattr(t.c, "embedding", None)
            if emb_col is not None:
                l2 = getattr(emb_col, "l2_distance", None)
                if callable(l2):
                    try:
                        distance_expr = l2(qvec)
                    except Exception:
                        distance_expr = None
            if distance_expr is not None:
                stmt = stmt.order_by(distance_expr)
            else:
                stmt = stmt.order_by(t.c.id.desc())
            stmt = stmt.limit(int(top_k))
            res = s.execute(stmt).mappings().all()
            return [dict(r) for r in res]
        except Exception:
            return []