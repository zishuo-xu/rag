import logging
import time

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.session import SessionLocal
from app.providers.embedding.provider import EmbeddingProvider
from app.utils.file_parser import parse_file
from app.utils.semantic_tags import derive_semantic_tags
from app.utils.text_cleaner import clean_text
from app.utils.text_splitter import split_text_with_metadata


logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self, db=None) -> None:
        self.embedding_provider = EmbeddingProvider()

    def process_document(self, document: Document) -> Document:
        document_id = document.id
        file_path = document.file_path
        file_type = document.file_type
        try:
            self._update_processing(document_id, status="PROCESSING", stage="preparing", message="正在准备处理文档。")
            self.repo_clear_chunks(document_id)

            self._update_processing(document_id, stage="parsing", message="正在解析原始文件。")
            raw_text, metadata = parse_file(file_path, file_type)

            self._update_processing(document_id, stage="cleaning", message="正在清洗文本内容。")
            cleaned = clean_text(raw_text)
            if not cleaned:
                raise ValueError("parsed content is empty")

            self._update_processing(document_id, stage="splitting", message="正在切分文本片段。")
            chunks = split_text_with_metadata(cleaned)
            if not chunks:
                raise ValueError("no chunks produced")

            self._update_processing(document_id, stage="embedding", message="正在生成向量表示。")
            embeddings = self.embedding_provider.embed_documents([item.text for item in chunks])

            self._update_processing(document_id, stage="persisting", message="正在写入文本块和向量。")
            self._persist_chunks(document_id, chunks, embeddings, metadata)
            self._mark_success(document_id, len(chunks))
            return self._get_document(document_id)
        except Exception as exc:
            self._mark_failed(document_id, str(exc))
            return self._get_document(document_id)

    def repo_clear_chunks(self, document_id: int) -> None:
        with SessionLocal() as db:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            db.commit()

    def _update_processing(
        self,
        document_id: int,
        *,
        stage: str,
        message: str,
        status: str | None = None,
    ) -> None:
        for attempt in range(2):
            try:
                with SessionLocal() as db:
                    document = db.get(Document, document_id)
                    if not document:
                        return
                    if status:
                        document.status = status
                    document.processing_stage = stage
                    document.processing_message = message
                    if stage != "persisting":
                        document.error_message = None
                    if stage == "preparing":
                        document.chunk_count = 0
                    db.commit()
                    return
            except Exception as exc:
                logger.warning(
                    "failed to update document progress",
                    extra={"document_id": document_id, "stage": stage, "attempt": attempt + 1, "error": str(exc)},
                )
                time.sleep(0.5)
        logger.warning("skip document progress update after retries: document_id=%s stage=%s", document_id, stage)

    def _persist_chunks(self, document_id: int, chunks, embeddings, metadata: dict | None) -> None:
        with SessionLocal() as db:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            for index, chunk in enumerate(chunks):
                chunk_metadata = dict(metadata or {})
                if chunk.section_title:
                    chunk_metadata["section_title"] = chunk.section_title
                semantic_tags = derive_semantic_tags(chunk.text, chunk.section_title)
                if semantic_tags:
                    chunk_metadata["semantic_tags"] = semantic_tags
                db.add(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=index,
                        chunk_text=chunk.text,
                        section_title=chunk.section_title,
                        embedding_json=embeddings[index] if index < len(embeddings) else None,
                        embedding_vector=embeddings[index] if index < len(embeddings) else None,
                        metadata_json=chunk_metadata or None,
                    )
                )
            db.commit()

    def _mark_success(self, document_id: int, chunk_count: int) -> None:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                return
            document.status = "SUCCESS"
            document.processing_stage = "completed"
            document.processing_message = "文档处理完成。"
            document.chunk_count = chunk_count
            document.error_message = None
            db.commit()

    def _mark_failed(self, document_id: int, error_message: str) -> None:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                return
            document.status = "FAILED"
            document.processing_stage = "failed"
            document.processing_message = "文档处理失败。"
            document.error_message = error_message
            document.chunk_count = 0
            db.commit()

    def _get_document(self, document_id: int) -> Document:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                raise ValueError(f"document {document_id} not found")
            db.expunge(document)
            return document
