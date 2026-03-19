from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk


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
        return list(self.db.scalars(stmt))

    def get(self, document_id: int) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        return self.db.scalar(stmt)

    def delete_with_chunks(self, document: Document) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        self.db.delete(document)
        self.db.commit()

    def clear_chunks(self, document_id: int) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        self.db.flush()
