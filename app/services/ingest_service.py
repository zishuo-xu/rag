from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.providers.embedding.provider import EmbeddingProvider
from app.utils.file_parser import parse_file
from app.utils.semantic_tags import derive_semantic_tags
from app.utils.text_cleaner import clean_text
from app.utils.text_splitter import split_text_with_metadata


class IngestService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.embedding_provider = EmbeddingProvider()

    def process_document(self, document: Document) -> Document:
        try:
            self.db.add(document)
            document.status = "PROCESSING"
            document.error_message = None
            document.chunk_count = 0
            self.repo_clear_chunks(document.id)

            raw_text, metadata = parse_file(document.file_path, document.file_type)
            cleaned = clean_text(raw_text)
            if not cleaned:
                raise ValueError("parsed content is empty")

            chunks = split_text_with_metadata(cleaned)
            if not chunks:
                raise ValueError("no chunks produced")

            embeddings = self.embedding_provider.embed_documents([item.text for item in chunks])
            for index, chunk in enumerate(chunks):
                chunk_metadata = dict(metadata or {})
                if chunk.section_title:
                    chunk_metadata["section_title"] = chunk.section_title
                semantic_tags = derive_semantic_tags(chunk.text, chunk.section_title)
                if semantic_tags:
                    chunk_metadata["semantic_tags"] = semantic_tags
                self.db.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=index,
                        chunk_text=chunk.text,
                        section_title=chunk.section_title,
                        embedding_json=embeddings[index] if index < len(embeddings) else None,
                        embedding_vector=embeddings[index] if index < len(embeddings) else None,
                        metadata_json=chunk_metadata or None,
                    )
                )

            document.status = "SUCCESS"
            document.chunk_count = len(chunks)
            document.error_message = None
            self.db.commit()
            self.db.refresh(document)
            return document
        except Exception as exc:
            self.db.rollback()
            self.db.add(document)
            document.status = "FAILED"
            document.error_message = str(exc)
            document.chunk_count = 0
            self.db.commit()
            self.db.refresh(document)
            return document

    def repo_clear_chunks(self, document_id: int) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
