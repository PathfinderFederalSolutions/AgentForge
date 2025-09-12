from __future__ import annotations
import os
import uuid
import datetime as dt
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, create_engine, JSON as SA_JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from swarm.config import settings
from swarm.protocol.messages import DAGSpec
import json

os.makedirs("./var", exist_ok=True)

class Base(DeclarativeBase):
    pass

class Artifact(Base):
    __tablename__ = "artifacts"
    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    backend: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(128))
    size: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # s3 object or local path
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))

    jobs: Mapped[list["JobArtifact"]] = relationship(back_populates="artifact")

class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="created")  # created|running|completed|failed
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    decision: Mapped[Optional[dict]] = mapped_column(SA_JSON, nullable=True)
    results: Mapped[Optional[dict]] = mapped_column(SA_JSON, nullable=True)
    dag_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    artifacts: Mapped[list["JobArtifact"]] = relationship(back_populates="job")
    events: Mapped[list["LineageEvent"]] = relationship(back_populates="job")

class JobArtifact(Base):
    __tablename__ = "job_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"))
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.artifact_id"))

    job: Mapped["Job"] = relationship(back_populates="artifacts")
    artifact: Mapped["Artifact"] = relationship(back_populates="jobs")

class LineageEvent(Base):
    __tablename__ = "lineage_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[Optional[str]] = mapped_column(ForeignKey("jobs.job_id"), nullable=True)
    artifact_id: Mapped[Optional[str]] = mapped_column(ForeignKey("artifacts.artifact_id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64))
    data: Mapped[Optional[dict]] = mapped_column(SA_JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))

    job: Mapped[Optional["Job"]] = relationship(back_populates="events")
    artifact: Mapped[Optional["Artifact"]] = relationship()

_engine = create_engine(settings.db_url, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)

# Ensure metadata is created before any DB operation (important for tests)
_db_initialized = False
_migrated = False

def _run_migrations():
    """Lightweight in-place migrations for sqlite (add missing columns)."""
    global _migrated
    if _migrated:
        return
    try:
        if _engine.url.get_backend_name().startswith("sqlite"):
            with _engine.connect() as conn:  # type: ignore
                res = conn.exec_driver_sql("PRAGMA table_info('jobs')")
                cols = {row[1] for row in res.fetchall()}
                if 'dag_hash' not in cols:
                    conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN dag_hash VARCHAR(128)")
    except Exception:
        pass  # best-effort; tests will surface issues if still missing
    _migrated = True

def init_db() -> None:
    global _db_initialized
    Base.metadata.create_all(_engine)
    _run_migrations()
    _db_initialized = True

def _ensure_db() -> None:
    if not _db_initialized:
        init_db()

def record_artifact(meta: Dict[str, Any]) -> str:
    """
    Persist an uploaded artifact's metadata.
    """
    _ensure_db()
    art_id = meta.get("artifact_id") or uuid.uuid4().hex
    uri = None
    if meta.get("backend") == "s3":
        uri = f"s3://{meta.get('bucket')}/{meta.get('object')}"
    elif meta.get("backend") == "local":
        uri = meta.get("path")
    with SessionLocal.begin() as s:
        a = s.get(Artifact, art_id)
        if not a:
            a = Artifact(
                artifact_id=art_id,
                filename=meta.get("filename", "artifact"),
                backend=meta.get("backend", "unknown"),
                sha256=meta.get("sha256", ""),
                size=int(meta.get("size", 0)),
                content_type=meta.get("content_type"),
                uri=uri,
            )
            s.add(a)
        s.add(LineageEvent(event_type="artifact_recorded", artifact_id=art_id, data=meta))
    return art_id

def start_job(goal: str, artifacts: Optional[Iterable[Dict[str, Any]]] = None, dag_hash: Optional[str] = None) -> str:
    _ensure_db()
    job_id = uuid.uuid4().hex
    with SessionLocal.begin() as s:
        j = Job(job_id=job_id, goal=goal, status="running", dag_hash=dag_hash)
        s.add(j)
        if artifacts:
            for meta in artifacts:
                aid = meta.get("artifact_id") or record_artifact(meta)
                s.add(JobArtifact(job_id=job_id, artifact_id=aid))
        s.add(LineageEvent(job_id=job_id, event_type="job_started", data={"goal": goal, "dag_hash": dag_hash}))
    return job_id

def complete_job(job_id: str, decision: Dict[str, Any], results: Any) -> None:
    _ensure_db()
    with SessionLocal.begin() as s:
        j = s.get(Job, job_id)
        if not j:
            return
        j.status = "completed" if decision.get("approved", False) else "failed"
        j.completed_at = dt.datetime.now(dt.timezone.utc)
        j.decision = decision
        # Keep results small: store summary; full results may live in object storage
        j.results = {"approved": decision.get("approved", False), "metrics": decision.get("metrics", {}), "count": len(results) if isinstance(results, list) else 1}
        s.add(LineageEvent(job_id=job_id, event_type="job_completed", data={"decision": decision}))

def get_job(job_id: str) -> Optional[Job]:
    _ensure_db()
    with SessionLocal() as s:
        return s.get(Job, job_id)

def list_job_events(job_id: str) -> list[dict]:
    _ensure_db()
    with SessionLocal() as s:
        rows = s.query(LineageEvent).filter(LineageEvent.job_id == job_id).order_by(LineageEvent.id.asc()).all()
        return [{"id": r.id, "event_type": r.event_type, "data": r.data, "created_at": r.created_at.isoformat()} for r in rows]

def record_event(event_type: str, data: Optional[Dict[str, Any]] = None, job_id: Optional[str] = None, artifact_id: Optional[str] = None) -> None:
    """Record a lineage event with optional job or artifact association."""
    _ensure_db()
    with SessionLocal.begin() as s:
        s.add(LineageEvent(job_id=job_id, artifact_id=artifact_id, event_type=event_type, data=data))

def persist_dag(dag: DAGSpec, job_id: Optional[str] = None) -> str:
    """Persist a DAGSpec JSON under var/artifacts/<hash>.dag.json and record lineage event.
    If job_id supplied, associate event with that job for lineage queries.
    Also registers the DAG JSON as an Artifact (backend=local) and links it to the job.
    """
    _ensure_db()
    dag_hash = dag.compute_hash()
    artifacts_dir = os.path.join("var", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"{dag_hash}.dag.json")
    payload = dag.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    # Register artifact + lineage
    meta = {
        'filename': os.path.basename(path),
        'backend': 'local',
        'sha256': dag_hash,
        'size': len(json.dumps(payload)),
        'content_type': 'application/json',
        'path': path,
        'artifact_id': f"dag-{dag_hash}"
    }
    art_id = record_artifact(meta)
    with SessionLocal.begin() as s:
        # Link to job if provided
        if job_id:
            exists = s.query(JobArtifact).filter(JobArtifact.job_id==job_id, JobArtifact.artifact_id==art_id).first()
            if not exists:
                s.add(JobArtifact(job_id=job_id, artifact_id=art_id))
        s.add(LineageEvent(job_id=job_id, artifact_id=art_id, event_type="dag_persisted", data={"hash": dag_hash, "path": path, "goal": dag.goal, "seed": dag.seed}))
    return path

def set_job_dag_hash(job_id: str, dag_hash: str) -> None:
    _ensure_db()
    with SessionLocal.begin() as s:
        j = s.get(Job, job_id)
        if not j:
            return
        # Only set if empty to preserve first deterministic hash
        if not j.dag_hash:
            j.dag_hash = dag_hash
        s.add(LineageEvent(job_id=job_id, event_type="dag_hash_set", data={"dag_hash": dag_hash}))

def build_evidence_bundle(job_id: str) -> dict | None:
    """Assemble a reproducible evidence bundle for a completed (or in‑flight) job.

    Bundle contents:
      job_id, goal, dag_hash, decision, results_summary
      artifacts: list of artifact metadata (filename, sha256, size, uri, content_type, artifact_id)
      events: ordered lineage event list
      citations: decision.citations if present
      confidence: decision.metrics.confidence or decision.confidence if present
      reproducibility: { dag_path_exists: bool, dag_path: str | None }
    """
    _ensure_db()
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        if not job:
            return None
        # Gather artifact metadata
        arts: list[dict] = []
        for ja in job.artifacts:
            a = ja.artifact
            if not a:
                continue
            arts.append({
                'artifact_id': a.artifact_id,
                'filename': a.filename,
                'sha256': a.sha256,
                'size': a.size,
                'content_type': a.content_type,
                'uri': a.uri,
                'backend': a.backend,
                'created_at': a.created_at.isoformat(),
            })
        events = list_job_events(job_id)
        decision = job.decision or {}
        citations = []
        try:
            citations = decision.get('citations', []) or []
        except Exception:
            citations = []
        confidence = None
        try:
            metrics = decision.get('metrics', {}) or {}
            confidence = metrics.get('confidence') or decision.get('confidence')
        except Exception:
            confidence = None
        dag_path = None
        if job.dag_hash:
            candidate = os.path.join('var','artifacts', f"{job.dag_hash}.dag.json")
            if os.path.exists(candidate):
                dag_path = candidate
        bundle = {
            'job_id': job.job_id,
            'goal': job.goal,
            'dag_hash': job.dag_hash,
            'decision': decision,
            'results_summary': job.results,
            'artifacts': arts,
            'events': events,
            'citations': citations,
            'confidence': confidence,
            'reproducibility': {
                'dag_path': dag_path,
                'dag_path_exists': bool(dag_path),
            }
        }
        return bundle

__all__ = [
    'record_artifact',
    'start_job',
    'complete_job',
    'get_job',
    'list_job_events',
    'record_event',
    'persist_dag',
    'set_job_dag_hash',
    'build_evidence_bundle'
]