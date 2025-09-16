# services/engagement/app/main.py
import os
import uuid
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from services.hitl.app import require_dual_approval, log_evidence_dag, emit_authorized_action

app = FastAPI()

ARTIFACTS_DIR = "var/artifacts/engagement"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class EngagementPacket(BaseModel):
    target_metadata: Dict[str, Any]
    recommended_coa: str
    roe_checks: Dict[str, Any]
    evidence_list: List[str]
    packet_id: str = None
    signed: bool = False
    approved: bool = False
    approval_time: float = None

@app.post("/engage")
def build_engagement_packet(packet: EngagementPacket, request: Request):
    packet.packet_id = str(uuid.uuid4())
    start = time.time()
    # Require dual approval (2FA)
    if not require_dual_approval(request, packet.packet_id):
        raise HTTPException(status_code=403, detail="Dual approval required")
    packet.approved = True
    packet.approval_time = time.time() - start
    # Sign packet (Cosign stub)
    packet.signed = True
    # Store signed packet
    out_dir = os.path.join(ARTIFACTS_DIR, packet.packet_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "packet.json"), "w") as f:
        f.write(packet.json())
    # Log to evidence DAG
    log_evidence_dag(packet.packet_id, packet.evidence_list)
    # Emit authorized action event
    emit_authorized_action(packet.packet_id)
    # Metrics
    return {"packet_id": packet.packet_id, "engagement_time_to_decision_seconds": packet.approval_time, "signed": packet.signed, "approved": packet.approved}
