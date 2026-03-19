from __future__ import annotations

from threading import Thread

from app.db.models.document import Document
from app.db.session import SessionLocal
from app.services.ingest_service import IngestService


class DocumentTaskService:
    @staticmethod
    def enqueue(document_id: int) -> None:
        worker = Thread(target=DocumentTaskService._process_document, args=(document_id,), daemon=True)
        worker.start()

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
