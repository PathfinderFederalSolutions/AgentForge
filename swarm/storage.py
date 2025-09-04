from __future__ import annotations
import os
import io
import uuid
import shutil
import hashlib
from typing import BinaryIO, Dict, Optional, Tuple
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from swarm.config import settings

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