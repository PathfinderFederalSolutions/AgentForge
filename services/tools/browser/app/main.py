from fastapi import FastAPI
import httpx, os
import hashlib, time
from pydantic import BaseModel
app = FastAPI(title="browser.fetch")
CACHE: dict[str, dict] = {}
ALLOWED = set(os.getenv("BROWSER_ALLOW","example.com wikipedia.org").split())
class FetchRequest(BaseModel):
    url: str
    request_id: str
    task_id: str
class FetchResult(BaseModel):
    request_id: str
    task_id: str
    status: str
    artifacts: list
    output: dict
@app.post("/fetch", response_model=FetchResult)
async def fetch(fr: FetchRequest):
    host = fr.url.split("/")[2] if "://" in fr.url else fr.url
    if ALLOWED and all(not host.endswith(a) for a in ALLOWED):
        return FetchResult(request_id=fr.request_id, task_id=fr.task_id, status="error", artifacts=[], output={"error":"domain not allowed"})
    key = hashlib.sha256(fr.url.encode()).hexdigest()[:16]
    if key in CACHE:
        d = CACHE[key]
    else:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(fr.url)
            d = {"status_code": r.status_code, "text": r.text[:50000], "fetched_at": time.time()}
            CACHE[key] = d
    return FetchResult(
        request_id=fr.request_id,
        task_id=fr.task_id,
        status="ok",
        artifacts=[{"type":"text","uri":f"mem://cache/{key}.txt"}],
        output=d
    )