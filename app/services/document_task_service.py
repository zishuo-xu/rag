from __future__ import annotations

from app.db.models.document import Document
from app.db.session import SessionLocal
from app.services.ingest_service import IngestService
from app.services.redis_service import blocking_pop_document_task, enqueue_document_task


class DocumentTaskService:
    @staticmethod
    def enqueue(document_id: int) -> None:
        enqueue_document_task(document_id)

    @staticmethod
    def _process_document(document_id: int) -> None:
        db = SessionLocal()
        try:
            document = db.get(Document, document_id)
            if not document or document.status == "DELETED":
                return
            IngestService(db).process_document(document)
        finally:
            db.close()

    @staticmethod
    def run_forever() -> None:
        while True:
            task = blocking_pop_document_task(timeout=5)
            if not task:
                continue
            document_id = int(task["document_id"])
            DocumentTaskService._process_document(document_id)
