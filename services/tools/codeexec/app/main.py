from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, tempfile, textwrap, uuid, os, json, time
app = FastAPI(title="code.exec")
MAX_SECONDS = int(os.getenv("CODEEXEC_TIMEOUT","8"))
class ExecRequest(BaseModel):
    code: str
    request_id: str
    task_id: str
class ExecResult(BaseModel):
    request_id: str
    task_id: str
    status: str
    artifacts: list
    output: dict
@app.post("/run", response_model=ExecResult)
def run_code(er: ExecRequest):
    code = textwrap.dedent(er.code)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td,"main.py")
        open(p,"w").write(code)
        start = time.time()
        try:
            res = subprocess.run(["python","-u",p], capture_output=True, text=True, timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            return ExecResult(request_id=er.request_id, task_id=er.task_id, status="error", artifacts=[], output={"error":"timeout"})
        dur = time.time()-start
    return ExecResult(
        request_id=er.request_id,
        task_id=er.task_id,
        status="ok" if res.returncode==0 else "error",
        artifacts=[],
        output={"stdout":res.stdout, "stderr":res.stderr, "returncode":res.returncode, "duration_s":dur}
    )