from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db.models.document import Document
from app.db.session import SessionLocal
from app.services.ingest_service import IngestService
from app.services.redis_service import (
    blocking_pop_document_task,
    enqueue_document_task,
    set_document_worker_heartbeat,
)


class DocumentTaskService:
    @staticmethod
    def enqueue(document_id: int) -> None:
        enqueue_document_task(document_id)

    @staticmethod
    def _process_document(document_id: int) -> None:
        set_document_worker_heartbeat(status="processing", document_id=document_id, stage="started")
        db = SessionLocal()
        try:
            document = db.get(Document, document_id)
            if not document or document.status == "DELETED":
                set_document_worker_heartbeat(status="idle")
                return
            IngestService(db).process_document(document)
        finally:
            db.close()
            set_document_worker_heartbeat(status="idle")

    @staticmethod
    def run_forever() -> None:
        DocumentTaskService.recover_stale_documents()
        set_document_worker_heartbeat(status="idle")
        while True:
            task = blocking_pop_document_task(timeout=5)
            if not task:
                set_document_worker_heartbeat(status="idle")
                continue
            document_id = int(task["document_id"])
            DocumentTaskService._process_document(document_id)

    @staticmethod
    def recover_stale_documents() -> None:
        threshold = datetime.now(timezone.utc) - timedelta(minutes=15)
        with SessionLocal() as db:
            stale_documents = (
                db.query(Document)
                .filter(Document.status == "PROCESSING")
                .filter(Document.updated_time < threshold)
                .all()
            )
            for document in stale_documents:
                document.status = "FAILED"
                document.processing_stage = "failed"
                document.processing_message = "检测到陈旧任务，worker 启动时已自动恢复。"
                document.error_message = "stale task recovered on worker startup"
            db.commit()
