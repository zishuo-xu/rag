from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.document import Document
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.document_task_repository import DocumentTaskRepository
from app.services.document_task_service import DocumentTaskService


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class DocumentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = DocumentRepository(db)
        self.task_repo = DocumentTaskRepository(db)
        self.settings = get_settings()

    def list_documents(self) -> list[Document]:
        return self.repo.list()

    def get_document(self, document_id: int) -> Document:
        document = self.repo.get(document_id)
        if not document or document.status == "DELETED":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
        return document

    def save_upload(self, upload: UploadFile) -> Document:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unsupported file type")

        raw = upload.file.read()
        if not raw:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty file")

        max_size = self.settings.max_file_size_mb * 1024 * 1024
        if len(raw) > max_size:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file too large")

        stored_name = f"{uuid4().hex}{suffix}"
        stored_path = self.settings.upload_dir / stored_name
        stored_path.write_bytes(raw)

        try:
            document = self.repo.create(
                file_name=upload.filename or stored_name,
                file_type=suffix.lstrip("."),
                file_path=str(stored_path),
                status="QUEUED",
            )
            self.db.add(document)
            document.processing_stage = "queued"
            document.processing_message = "文档已入队，等待后台处理。"
            self.db.commit()
            self.db.refresh(document)
            task = self.task_repo.create(
                document_id=document.id,
                trigger_source="UPLOAD",
                status="QUEUED",
                processing_stage="queued",
                processing_message="文档已入队，等待后台处理。",
            )
            DocumentTaskService.enqueue(document.id, task.id)
            document.latest_task = task
            document.task_id = task.id
            return document
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

    def delete_document(self, document_id: int) -> None:
        document = self.get_document(document_id)
        file_path = Path(document.file_path)
        self.repo.delete_with_chunks(document)
        file_path.unlink(missing_ok=True)

    def reprocess_document(self, document_id: int) -> Document:
        document = self.get_document(document_id)
        if document.status in {"QUEUED", "PROCESSING"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="document is already being processed")

        self.db.add(document)
        document.status = "QUEUED"
        document.processing_stage = "queued"
        document.processing_message = "文档已重新入队，等待后台处理。"
        document.processing_started_time = None
        document.processing_finished_time = None
        document.processing_duration_ms = None
        document.stage_durations_json = None
        document.error_message = None
        document.chunk_count = 0
        self.db.commit()
        self.db.refresh(document)
        task = self.task_repo.create(
            document_id=document.id,
            trigger_source="REPROCESS",
            status="QUEUED",
            processing_stage="queued",
            processing_message="文档已重新入队，等待后台处理。",
        )
        DocumentTaskService.enqueue(document.id, task.id)
        document.latest_task = task
        document.task_id = task.id
        return document
