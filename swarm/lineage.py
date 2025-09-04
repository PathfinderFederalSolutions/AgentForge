from __future__ import annotations
import os
import uuid
import datetime as dt
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, create_engine, JSON as SA_JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from swarm.config import settings

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

def init_db() -> None:
    global _db_initialized
    Base.metadata.create_all(_engine)
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

def start_job(goal: str, artifacts: Optional[Iterable[Dict[str, Any]]] = None) -> str:
    _ensure_db()
    job_id = uuid.uuid4().hex
    with SessionLocal.begin() as s:
        j = Job(job_id=job_id, goal=goal, status="running")
        s.add(j)
        if artifacts:
            for meta in artifacts:
                aid = meta.get("artifact_id") or record_artifact(meta)
                s.add(JobArtifact(job_id=job_id, artifact_id=aid))
        s.add(LineageEvent(job_id=job_id, event_type="job_started", data={"goal": goal}))
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