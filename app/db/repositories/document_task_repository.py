from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.document_task import DocumentTask


class DocumentTaskRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        document_id: int,
        task_type: str = "INGEST",
        trigger_source: str = "SYSTEM",
        status: str = "QUEUED",
        processing_stage: str | None = None,
        processing_message: str | None = None,
    ) -> DocumentTask:
        task = DocumentTask(
            document_id=document_id,
            task_type=task_type,
            trigger_source=trigger_source,
            status=status,
            processing_stage=processing_stage,
            processing_message=processing_message,
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get(self, task_id: int) -> DocumentTask | None:
        stmt = select(DocumentTask).where(DocumentTask.id == task_id).limit(1)
        return self.db.scalar(stmt)

    def list_recent_by_document(self, document_id: int, limit: int = 10) -> list[DocumentTask]:
        stmt = (
            select(DocumentTask)
            .where(DocumentTask.document_id == document_id)
            .order_by(DocumentTask.created_time.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt))
