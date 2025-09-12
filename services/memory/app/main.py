from fastapi import FastAPI
from pydantic import BaseModel
import hashlib
app = FastAPI(title="memory.rag")
EMBED_INDEX: dict[str, dict] = {}
class Ingest(BaseModel):
    request_id: str
    text: str
class Search(BaseModel):
    query: str
    top_k: int = 3
def embed(txt: str):
    return hashlib.sha256(txt.encode()).digest()
@app.post("/ingest")
def ingest(doc: Ingest):
    emb = embed(doc.text)
    # Use the SHA-256 digest hex directly rather than hashing again with SHA-1
    key = emb.hex()
    EMBED_INDEX[key] = {"text": doc.text, "vector": emb}
    return {"status":"ok","count":len(EMBED_INDEX)}
@app.post("/search")
def search(q: Search):
    qv = embed(q.query)
    scored = []
    for v in EMBED_INDEX.values():
        score = sum(a==b for a,b in zip(qv,v["vector"]))  # toy similarity
        scored.append((score,v["text"]))
    scored.sort(reverse=True)
    return {"results":[{"score":s,"text":t[:500]} for s,t in scored[:q.top_k]]}