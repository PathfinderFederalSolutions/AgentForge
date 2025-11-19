from __future__ import annotations
import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Core
    env: str = os.getenv("ENV", "development")
    service_name: str = "agentforge-swarm-gateway"

    # Temporal
    temporal_address: str = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    temporal_namespace: str = os.getenv("TEMPORAL_NAMESPACE", "default")
    temporal_task_queue: str = os.getenv("TEMPORAL_TASK_QUEUE", "swarm-tq")

    # NATS
    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    nats_topic_jobs: str = os.getenv("NATS_TOPIC_JOBS", "swarm.jobs")
    nats_topic_results: str = os.getenv("NATS_TOPIC_RESULTS", "swarm.results")

    # Dispatch
    dispatch_mode: str = os.getenv("DISPATCH_MODE", "sync")  # sync | nats | temporal

    # Artifacts (MinIO/S3)
    artifact_bucket: str = os.getenv("ARTIFACT_BUCKET", "swarm-artifacts")
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_secure: bool = bool(int(os.getenv("S3_SECURE", "0")))
    local_artifacts_dir: str = os.getenv("LOCAL_ARTIFACTS_DIR", "./var/artifacts")

    # Upload limits
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))

    # DB (lineage/provenance)
    db_url: str = os.getenv("SWARM_DB_URL", "sqlite:///./var/swarm.db")

    # Enforcement thresholds
    error_rate_max: float = float(os.getenv("SLA_ERROR_RATE_MAX", "0.0"))
    completeness_min: float = float(os.getenv("SLA_COMPLETENESS_MIN", "0.95"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra="ignore", env_ignore_empty=True)

# Initialize settings with error handling for missing .env
try:
    settings = Settings()
except (PermissionError, FileNotFoundError, Exception) as e:
    # Fallback to settings without .env file
    import warnings
    warnings.warn(f"Could not load .env file ({e}), using environment variables and defaults")
    settings = Settings(_env_file=None)