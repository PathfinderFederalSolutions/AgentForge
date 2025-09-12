from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, tempfile
app = FastAPI(title="file.ingest")
class IngestResponse(BaseModel):
    artifacts: list
    text_preview: str
@app.post("/ingest", response_model=IngestResponse)
async def ingest(f: UploadFile = File(...)):
    data = await f.read()
    txt = data[:5000].decode(errors="ignore")
    # Placeholder: add real PDF/Docx/OCR extraction later
    return IngestResponse(
        artifacts=[{"type":"raw","name":f.filename,"bytes":len(data)}],
        text_preview=txt
    )