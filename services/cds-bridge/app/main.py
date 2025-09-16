# services/cds-bridge/app/main.py
import os
import time
import json
import hashlib
import random
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

LOG_DIR = "services/cds-bridge/logs/transfers"
os.makedirs(LOG_DIR, exist_ok=True)

class CDSAdapter:
    def send(self, bundle: Dict[str, Any], banner: str) -> Dict[str, Any]:
        # Stub: simulate transfer, randomly fail for test
        if random.random() < 0.2:
            raise Exception("CDS transfer failed")
        return {"status": "success", "sha256": hashlib.sha256(json.dumps(bundle).encode()).hexdigest()}

cds_adapter = CDSAdapter()

@app.post("/transfer")
def transfer_evidence_bundle(request: Request):
    data = request.json() if hasattr(request, "json") else request.body
    bundle = data.get("bundle")
    banner = data.get("banner")
    attempt = 0
    max_attempts = 5
    delay = 2
    while attempt < max_attempts:
        try:
            resp = cds_adapter.send(bundle, banner)
            # Log transfer receipt
            receipt = {
                "timestamp": time.time(),
                "bundle_id": bundle.get("id"),
                "sha256": resp["sha256"],
                "banner": banner,
                "status": resp["status"],
                "attempt": attempt + 1,
            }
            with open(os.path.join(LOG_DIR, f"{bundle.get('id')}.jsonl"), "a") as f:
                f.write(json.dumps(receipt) + "\n")
            # Export metric
            # (stub) cds_transfer_success_total += 1
            return receipt
        except Exception as e:
            attempt += 1
            time.sleep(delay)
            delay *= 2
    raise HTTPException(status_code=500, detail="CDS transfer failed after retries")

@app.post("/verify")
def verify_bundle(request: Request):
    data = request.json() if hasattr(request, "json") else request.body
    bundle = data.get("bundle")
    expected_sha = data.get("sha256")
    actual_sha = hashlib.sha256(json.dumps(bundle).encode()).hexdigest()
    if actual_sha != expected_sha:
        # Log alert
        alert = {
            "timestamp": time.time(),
            "bundle_id": bundle.get("id"),
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "alert": "SHA256 mismatch: possible corruption or tampering"
        }
        with open(os.path.join(LOG_DIR, f"{bundle.get('id')}_alert.jsonl"), "a") as f:
            f.write(json.dumps(alert) + "\n")
        raise HTTPException(status_code=400, detail="SHA256 verification failed")
    return {"status": "verified", "sha256": actual_sha}
