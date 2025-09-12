from .models import SwarmJob, TaskSpec, ToolInvocation
import hashlib
def make_invocations(job: SwarmJob, completed: set[str] | None = None):
    completed = completed or set()
    ready = []
    for t in job.plan:
        if t.id in completed: continue
        if all(dep in completed for dep in t.dependsOn):
            key_basis = f"{job.request_id}:{t.id}:{t.type}"
            idempotency_key = hashlib.sha256(key_basis.encode()).hexdigest()[:16]
            tool = infer_tool_for_task(t)
            ready.append(ToolInvocation(
                request_id=job.request_id,
                task_id=t.id,
                tool=tool,
                payload=t.args,
                idempotency_key=idempotency_key
            ))
    return ready
def infer_tool_for_task(task: TaskSpec) -> str:
    mapping = {
        "gather":"browser.fetch",
        "analyze":"code.exec",
        "synthesize":"rag.search",
        "review":"review.validate",
        "map":"code.exec",
        "reduce":"code.exec"
    }
    return mapping.get(task.type,"code.exec")