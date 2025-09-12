from __future__ import annotations
import json
import os
import datetime as _dt

from sqlalchemy import create_engine, Integer, String, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from swarm.config import settings

DB_URL = os.getenv("LEARNING_DB_URL", settings.db_url)

class Base(DeclarativeBase):
    pass

class CapStats(Base):
    __tablename__ = "cap_stats"
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    approvals: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)  # learned boost [-1..+1]

class Feedback(Base):
    __tablename__ = "feedback"
    subject_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    decision: Mapped[str] = mapped_column(String(32))
    rationale: Mapped[str] = mapped_column(String(2048))
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evidence_refs: Mapped[str] = mapped_column(String(4096), default="[]")  # JSON serialized list
    # Use timezone-aware UTC timestamp, replacing deprecated utcnow usage
    created_at: Mapped[str] = mapped_column(
        String(64),
        default=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )

_engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
Base.metadata.create_all(_engine)

def record_feedback(goal: str, results: list[dict], decision: dict) -> None:
    caps = [r.get("capability") for r in results if r.get("capability")]
    if not caps:
        return
    with SessionLocal.begin() as s:
        for c in caps:
            row = s.get(CapStats, c)
            if not row:
                row = CapStats(name=c, approvals=0, failures=0, score=0.0)
                s.add(row)
            if decision.get("approved"):
                row.approvals += 1
                row.score = min(1.0, row.score + 0.05)
            else:
                row.failures += 1
                row.score = max(-1.0, row.score - 0.05)

def get_boost(name: str) -> float:
    with SessionLocal() as s:
        rec = s.get(CapStats, name)
        return float(rec.score) if rec else 0.0

def write_feedback(subject_id: str, decision: str, rationale: str, evidence_refs: list[str] | None = None, user_id: str | None = None) -> None:
    rec = Feedback(
        subject_id=subject_id,
        decision=decision,
        rationale=rationale or "",
        user_id=user_id,
        evidence_refs=json.dumps(evidence_refs or []),
    )
    with SessionLocal.begin() as s:
        s.merge(rec)

def list_feedback(limit: int = 100) -> list[dict]:
    with SessionLocal() as s:
        rows = s.query(Feedback).order_by(Feedback.created_at.desc()).limit(limit).all()  # type: ignore
        out: list[dict] = []
        for r in rows:
            try:
                evid = json.loads(r.evidence_refs) if r.evidence_refs else []
            except Exception:
                evid = []
            out.append({
                "subject_id": r.subject_id,
                "decision": r.decision,
                "rationale": r.rationale,
                "user_id": r.user_id,
                "evidence_refs": evid,
                "created_at": r.created_at,
            })
        return out