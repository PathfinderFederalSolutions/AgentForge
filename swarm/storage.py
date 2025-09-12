from __future__ import annotations
import os
import uuid
import shutil
import hashlib
from typing import BinaryIO, Dict, Optional, Tuple, Any, List
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from swarm.config import settings

# Optional PGVector store for track state embeddings (best-effort)
try:  # pragma: no cover
    from swarm.memory.pgvector_store import PGVectorStore  # type: ignore
except Exception:  # pragma: no cover
    PGVectorStore = None  # type: ignore

class ArtifactStore:
    def __init__(self) -> None:
        self._client: Optional[Minio] = None
        self._bucket = settings.artifact_bucket

        # Prepare local dir fallback
        os.makedirs(settings.local_artifacts_dir, exist_ok=True)

        # Initialize MinIO/S3 client if configured
        try:
            parsed = urlparse(settings.s3_endpoint)
            endpoint = parsed.netloc or settings.s3_endpoint.replace("http://", "").replace("https://", "")
            self._client = Minio(
                endpoint=endpoint,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                secure=(parsed.scheme == "https") if parsed.scheme else settings.s3_secure,
            )
            # Ensure bucket exists
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
        except Exception:
            # Fall back to local only
            self._client = None

    def _hash_and_size(self, path: str) -> Tuple[str, int]:
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                size += len(chunk)
                h.update(chunk)
        return h.hexdigest(), size

    def save_file(self, fileobj: BinaryIO, filename: str, content_type: Optional[str] = None) -> Dict:
        # Guard size
        max_bytes = settings.max_upload_mb * 1024 * 1024
        tmp_dir = os.path.join(settings.local_artifacts_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"up-{uuid.uuid4().hex}.bin")

        written = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    os.remove(tmp_path)
                    raise ValueError(f"Upload exceeds limit of {settings.max_upload_mb}MB")
                out.write(chunk)

        sha256, size = self._hash_and_size(tmp_path)
        artifact_id = uuid.uuid4().hex
        object_name = f"{artifact_id}/{filename}"

        # Try S3 first
        if self._client:
            try:
                self._client.fput_object(
                    bucket_name=self._bucket,
                    object_name=object_name,
                    file_path=tmp_path,
                    content_type=content_type or "application/octet-stream",
                )
                os.remove(tmp_path)
                return {
                    "artifact_id": artifact_id,
                    "backend": "s3",
                    "bucket": self._bucket,
                    "object": object_name,
                    "sha256": sha256,
                    "size": size,
                    "filename": filename,
                    "content_type": content_type or "application/octet-stream",
                }
            except S3Error:
                # Fall through to local
                pass

        # Local fallback
        target_dir = os.path.join(settings.local_artifacts_dir, artifact_id)
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, filename)
        shutil.move(tmp_path, local_path)
        return {
            "artifact_id": artifact_id,
            "backend": "local",
            "path": local_path,
            "sha256": sha256,
            "size": size,
            "filename": filename,
            "content_type": content_type or "application/octet-stream",
        }

    def presign(self, artifact_meta: Dict, expires_seconds: int = 3600) -> Optional[str]:
        if artifact_meta.get("backend") != "s3" or not self._client:
            return None
        try:
            return self._client.presigned_get_object(
                bucket_name=artifact_meta["bucket"],
                object_name=artifact_meta["object"],
                expires=expires_seconds,
            )
        except S3Error:
            return None

store = ArtifactStore()

# --- Fused Track Persistence -------------------------------------------------
# Lightweight local JSON persistence (future: Postgres + PostGIS table). For now we
# keep minimal schema: track_id, state, covariance, confidence, evidence, created_at.
import json
import datetime as _dt

_FUSED_TRACK_DIR = os.path.join("var", "fused_tracks")
os.makedirs(_FUSED_TRACK_DIR, exist_ok=True)

# In-memory index for tests (track_id -> path)
_fused_index: Dict[str, str] = {}

# Optional singleton pgvector store (lazy)
_pgvector_store: Optional[PGVectorStore] = None

def _get_pgvector_store() -> Optional[PGVectorStore]:  # pragma: no cover - best effort
    global _pgvector_store
    if _pgvector_store is not None:
        return _pgvector_store
    if PGVectorStore is None:
        return None
    try:
        _pgvector_store = PGVectorStore(migrate=False)
    except Exception:
        _pgvector_store = None
    return _pgvector_store

def _embed_track(state: Dict[str, Any]) -> Optional[List[float]]:  # pragma: no cover
    store_impl = _get_pgvector_store()
    if not store_impl:
        return None
    try:
        # Represent state as text for embedding provider
        text = json.dumps(state, sort_keys=True)
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In tests runtime may be sync; run threadpool blocking call
            emb = loop.run_until_complete(store_impl.provider.embed([text]))  # type: ignore
        else:
            emb = asyncio.run(store_impl.provider.embed([text]))
        return emb[0] if emb else None
    except Exception:
        return None

def persist_fused_track(
    state: Dict[str, Any],
    covariance: List[List[float]],
    confidence: float,
    evidence: List[Dict[str, Any]],
    track_id: Optional[str] = None,
) -> str:
    track_id = track_id or uuid.uuid4().hex
    payload = {
        "track_id": track_id,
        "state": state,
        "covariance": covariance,
        "confidence": confidence,
        "evidence": evidence,
        "created_at": _dt.datetime.utcnow().isoformat() + "Z",
    }
    path = os.path.join(_FUSED_TRACK_DIR, f"{track_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    _fused_index[track_id] = path

    # Best-effort embedding indexing
    emb = _embed_track(state)
    if emb is not None:
        try:
            store_impl = _get_pgvector_store()
            if store_impl:
                import asyncio
                loop = asyncio.get_event_loop()
                docs = [("track", json.dumps(state, sort_keys=True), {"track_id": track_id})]
                if loop.is_running():
                    loop.run_until_complete(store_impl.upsert_batch("tracks", docs))  # type: ignore
                else:
                    asyncio.run(store_impl.upsert_batch("tracks", docs))
        except Exception:
            pass
    return track_id

def load_fused_track(track_id: str) -> Optional[Dict[str, Any]]:
    path = _fused_index.get(track_id) or os.path.join(_FUSED_TRACK_DIR, f"{track_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

__all__ = [
    'ArtifactStore',
    'store',
    'persist_fused_track',
    'load_fused_track'
]