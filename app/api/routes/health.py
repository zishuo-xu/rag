from fastapi import APIRouter
from sqlalchemy import text

from app.db.session import SessionLocal
from app.services.redis_service import get_document_worker_heartbeat, ping_redis


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/deps")
def health_deps() -> dict[str, str]:
    db_status = "ok"
    redis_status = "ok"
    worker_status = "unknown"

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    try:
        ping_redis()
        heartbeat = get_document_worker_heartbeat()
        if heartbeat and heartbeat.get("updated_at"):
            worker_status = heartbeat.get("status") or "ok"
        else:
            worker_status = "offline"
    except Exception:
        redis_status = "error"
        worker_status = "error"

    return {"database": db_status, "redis": redis_status, "document_worker": worker_status}
