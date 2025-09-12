from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sqlite3
from swarm.learning.feedback import write_feedback  # type: ignore
from typing import List

# Allow overriding storage dir; fall back to local project 'var' (test-friendly)
BASE_DIR = os.getenv("HITL_DATA_DIR", "var")
os.makedirs(BASE_DIR, exist_ok=True)
DB_FILE = os.path.join(BASE_DIR, "hitl.sqlite")

app = FastAPI(title="AgentForge HITL")

class Review(BaseModel):
    job_id: str
    mission: str
    verdict: str  # approve|reject
    notes: str | None = None

class Adjudication(BaseModel):
    subject_id: str
    decision: str  # approve|reject|revise
    rationale: str | None = None
    user_id: str | None = None
    evidence_refs: List[str] | None = None

def _conn():
    # Simple SQLite for staging; swap to Postgres by changing driver
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reviews(job_id TEXT, mission TEXT, verdict TEXT, notes TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/reviews")
def create_review(r: Review):
    conn = _conn()
    with conn:
        conn.execute(
            "INSERT INTO reviews(job_id, mission, verdict, notes) VALUES (?,?,?,?)",
            (r.job_id, r.mission, r.verdict, r.notes),
        )
    return {"status": "ok"}

@app.get("/reviews")
def list_reviews(mission: str | None = None, limit: int = 50):
    conn = _conn()
    cur = conn.cursor()
    if mission:
        cur.execute("SELECT job_id, mission, verdict, notes, ts FROM reviews WHERE mission=? ORDER BY ts DESC LIMIT ?", (mission, limit))
    else:
        cur.execute("SELECT job_id, mission, verdict, notes, ts FROM reviews ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"job_id": r[0], "mission": r[1], "verdict": r[2], "notes": r[3], "ts": r[4]} for r in rows]

@app.post("/adjudications")
def submit_adjudication(a: Adjudication):
    try:
        write_feedback(
            subject_id=a.subject_id,
            decision=a.decision,
            rationale=a.rationale or "",
            evidence_refs=a.evidence_refs or [],
            user_id=a.user_id,
        )
        return {"status": "ok", "subject_id": a.subject_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to persist feedback: {e}")