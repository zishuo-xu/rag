import json
from datetime import datetime, timezone

from redis import Redis

from app.core.config import get_settings


settings = get_settings()
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
QA_PROGRESS_TTL_SECONDS = 60 * 30
DOCUMENT_TASK_QUEUE = "document:tasks"


def ping_redis() -> bool:
    return bool(redis_client.ping())


def set_qa_progress(
    request_id: str,
    *,
    status: str,
    stage: str,
    message: str,
    progress_percent: int,
    error_message: str | None = None,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
) -> None:
    payload = get_qa_progress(request_id) or {
        "request_id": request_id,
        "steps": [],
    }
    step_payload = {
        "stage": stage,
        "message": message,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    steps = [step for step in payload.get("steps", []) if step.get("stage") != stage]
    steps.append(step_payload)
    payload.update({
        "status": status,
        "stage": stage,
        "message": message,
        "progress_percent": progress_percent,
        "error_message": error_message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
    })
    redis_client.setex(_qa_progress_key(request_id), QA_PROGRESS_TTL_SECONDS, json.dumps(payload, ensure_ascii=True))


def get_qa_progress(request_id: str) -> dict | None:
    payload = redis_client.get(_qa_progress_key(request_id))
    if not payload:
        return None
    return json.loads(payload)


def _qa_progress_key(request_id: str) -> str:
    return f"qa:progress:{request_id}"


def enqueue_document_task(document_id: int) -> None:
    payload = {
        "document_id": document_id,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.lpush(DOCUMENT_TASK_QUEUE, json.dumps(payload, ensure_ascii=True))


def blocking_pop_document_task(timeout: int = 5) -> dict | None:
    item = redis_client.brpop(DOCUMENT_TASK_QUEUE, timeout=timeout)
    if not item:
        return None
    _, payload = item
    return json.loads(payload)
