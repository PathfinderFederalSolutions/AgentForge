from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sqlite3
from swarm.learning.feedback import write_feedback  # type: ignore
from typing import List
import json
import asyncio

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

async def _publish_soldier_alert(subject: str, payload: dict) -> None:
    # Publish via NATS if available; otherwise no-op
    try:
        import importlib
        nats = importlib.import_module("nats")
    except Exception:
        return
    nats_url = os.getenv("NATS_URL", "nats://nats.agentforge-staging.svc.cluster.local:4222")
    try:
        nc = await nats.connect(servers=[nats_url])
        await nc.publish(subject, json.dumps(payload).encode("utf-8"))
        await nc.drain()
    except Exception:
        pass

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
        # Only emit soldier alerts for approved high-priority items; policy can be refined later
        if a.decision in {"approve", "approved"}:
            payload = {
                "subject_id": a.subject_id,
                "decision": a.decision,
                "rationale": a.rationale,
                "evidence_refs": a.evidence_refs or [],
                "ts": __import__("time").time(),
            }
            try:
                asyncio.get_event_loop().create_task(_publish_soldier_alert("soldier.alert", payload))
            except RuntimeError:
                # If no running loop (sync context), run quickly in a new loop
                asyncio.run(_publish_soldier_alert("soldier.alert", payload))
        return {"status": "ok", "subject_id": a.subject_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to persist feedback: {e}")

def require_dual_approval(request, packet_id):
    # Simulate dual approval (WebAuthn/YubiKey)
    # In production, validate two independent approvals
    return getattr(request, "dual_approved", False)

def log_evidence_dag(packet_id, evidence_list):
    # Log to evidence DAG (stub)
    pass

def emit_authorized_action(packet_id):
    # Emit authorized action event (stub)
    pass