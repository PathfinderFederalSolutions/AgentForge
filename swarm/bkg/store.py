from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Optional

from sqlalchemy import String, Integer, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from swarm.config import settings

class Base(DeclarativeBase):
    pass

class BestKnownGood(Base):
    __tablename__ = "best_known_good"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # hash of goal
    goal: Mapped[str] = mapped_column(Text)
    decision_json: Mapped[str] = mapped_column(Text)
    results_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[int] = mapped_column(Integer)

_engine = create_engine(settings.db_url, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)

def init_db() -> None:
    Base.metadata.create_all(_engine)

def key_for_goal(goal: str) -> str:
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()

def get(goal: str) -> Optional[dict]:
    init_db()
    k = key_for_goal(goal)
    with SessionLocal() as s:
        row = s.get(BestKnownGood, k)
        if not row:
            return None
        return {
            "goal": row.goal,
            "decision": json.loads(row.decision_json),
            "results": json.loads(row.results_json),
            "updated_at": row.updated_at,
        }

def update(goal: str, decision: dict, results: Any) -> None:
    init_db()
    k = key_for_goal(goal)
    now = int(time.time())
    with SessionLocal.begin() as s:
        row = s.get(BestKnownGood, k)
        blob_dec = json.dumps(decision)
        blob_res = json.dumps(list(results) if not isinstance(results, str) else [results])
        if not row:
            row = BestKnownGood(key=k, goal=goal, decision_json=blob_dec, results_json=blob_res, updated_at=now)
            s.add(row)
        else:
            row.goal = goal
            row.decision_json = blob_dec
            row.results_json = blob_res
            row.updated_at = now