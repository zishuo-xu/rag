from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.document_task import DocumentTask


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, *, file_name: str, file_type: str, file_path: str, status: str) -> Document:
        document = Document(
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            status=status,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def list(self) -> list[Document]:
        stmt = select(Document).where(Document.status != "DELETED").order_by(Document.created_time.desc())
        documents = list(self.db.scalars(stmt))
        self._attach_latest_tasks(documents)
        return documents

    def get(self, document_id: int) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        document = self.db.scalar(stmt)
        if document:
            self._attach_latest_tasks([document], include_recent=True)
        return document

    def delete_with_chunks(self, document: Document) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        self.db.delete(document)
        self.db.commit()

    def clear_chunks(self, document_id: int) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        self.db.flush()

    def _attach_latest_tasks(self, documents: list[Document], include_recent: bool = False) -> None:
        if not documents:
            return
        document_ids = [document.id for document in documents]
        tasks = list(
            self.db.scalars(
                select(DocumentTask)
                .where(DocumentTask.document_id.in_(document_ids))
                .order_by(DocumentTask.created_time.desc())
            )
        )
        grouped: dict[int, list[DocumentTask]] = {}
        for task in tasks:
            grouped.setdefault(task.document_id, []).append(task)
        for document in documents:
            recent = grouped.get(document.id, [])
            latest = recent[0] if recent else None
            document.latest_task = latest
            if include_recent:
                document.recent_tasks = recent[:10]
