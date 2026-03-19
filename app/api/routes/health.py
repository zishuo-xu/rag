from fastapi import APIRouter
from sqlalchemy import text

from app.db.session import SessionLocal
from app.services.redis_service import ping_redis


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/deps")
def health_deps() -> dict[str, str]:
    db_status = "ok"
    redis_status = "ok"

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    try:
        ping_redis()
    except Exception:
        redis_status = "error"

    return {"database": db_status, "redis": redis_status}
