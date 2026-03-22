import logging
import math
import time
from datetime import datetime, timezone

from app.core.config import get_settings
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
        self.settings = get_settings()
        self.embedding_provider = EmbeddingProvider()

    def process_document(self, document: Document) -> Document:
        document_id = document.id
        file_path = document.file_path
        file_type = document.file_type
        started_time = datetime.now(timezone.utc)
        stage_durations: dict[str, int] = {}
        overall_started_at = time.perf_counter()
        try:
            stage_started_at = time.perf_counter()
            self._update_processing(
                document_id,
                status="PROCESSING",
                stage="preparing",
                message="正在准备处理文档。",
                started_time=started_time,
                stage_durations=stage_durations,
                reset_metrics=True,
            )
            self.repo_clear_chunks(document_id)
            stage_durations["preparing"] = self._duration_ms(stage_started_at)

            stage_started_at = time.perf_counter()
            self._update_processing(
                document_id,
                stage="parsing",
                message="正在解析原始文件。",
                started_time=started_time,
                stage_durations=stage_durations,
            )
            raw_text, metadata = parse_file(file_path, file_type)
            stage_durations["parsing"] = self._duration_ms(stage_started_at)

            stage_started_at = time.perf_counter()
            self._update_processing(
                document_id,
                stage="cleaning",
                message="正在清洗文本内容。",
                started_time=started_time,
                stage_durations=stage_durations,
            )
            cleaned = clean_text(raw_text)
            if not cleaned:
                raise ValueError("parsed content is empty")
            stage_durations["cleaning"] = self._duration_ms(stage_started_at)

            stage_started_at = time.perf_counter()
            self._update_processing(
                document_id,
                stage="splitting",
                message="正在切分文本片段。",
                started_time=started_time,
                stage_durations=stage_durations,
            )
            chunks = split_text_with_metadata(cleaned)
            if not chunks:
                raise ValueError("no chunks produced")
            stage_durations["splitting"] = self._duration_ms(stage_started_at)

            embeddings = self._embed_in_batches(
                document_id=document_id,
                chunks=chunks,
                started_time=started_time,
                stage_durations=stage_durations,
            )

            stage_started_at = time.perf_counter()
            self._persist_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
                started_time=started_time,
                stage_durations=stage_durations,
            )
            stage_durations["persisting"] = self._duration_ms(stage_started_at)
            self._mark_success(
                document_id,
                len(chunks),
                started_time=started_time,
                finished_time=datetime.now(timezone.utc),
                total_duration_ms=self._duration_ms(overall_started_at),
                stage_durations=stage_durations,
            )
            return self._get_document(document_id)
        except Exception as exc:
            self._mark_failed(
                document_id,
                str(exc),
                started_time=started_time,
                finished_time=datetime.now(timezone.utc),
                total_duration_ms=self._duration_ms(overall_started_at),
                stage_durations=stage_durations,
            )
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
        started_time: datetime | None = None,
        stage_durations: dict | None = None,
        reset_metrics: bool = False,
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
                    if started_time is not None and document.processing_started_time is None:
                        document.processing_started_time = started_time
                    if stage_durations is not None:
                        document.stage_durations_json = dict(stage_durations)
                    if stage != "persisting":
                        document.error_message = None
                    if reset_metrics:
                        document.chunk_count = 0
                        document.processing_finished_time = None
                        document.processing_duration_ms = None
                        document.stage_durations_json = {}
                    db.commit()
                    return
            except Exception as exc:
                logger.warning(
                    "failed to update document progress",
                    extra={"document_id": document_id, "stage": stage, "attempt": attempt + 1, "error": str(exc)},
                )
                time.sleep(0.5)
        logger.warning("skip document progress update after retries: document_id=%s stage=%s", document_id, stage)

    def _persist_chunks(
        self,
        *,
        document_id: int,
        chunks,
        embeddings,
        metadata: dict | None,
        started_time: datetime,
        stage_durations: dict,
    ) -> None:
        batch_size = max(1, int(self.settings.persist_batch_size))
        total_chunks = len(chunks)
        total_batches = max(1, math.ceil(total_chunks / batch_size))
        with SessionLocal() as db:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            for batch_index in range(total_batches):
                start = batch_index * batch_size
                end = min(start + batch_size, total_chunks)
                self._update_processing(
                    document_id,
                    stage="persisting",
                    message=f"正在写入文本块和向量（第 {batch_index + 1}/{total_batches} 批，chunks {start + 1}-{end}）。",
                    started_time=started_time,
                    stage_durations=stage_durations,
                )
                for index in range(start, end):
                    chunk = chunks[index]
                    chunk_metadata = dict(metadata or {})
                    if chunk.section_title:
                        chunk_metadata["section_title"] = chunk.section_title
                    if chunk.page_start is not None:
                        chunk_metadata["page_start"] = chunk.page_start
                    if chunk.page_end is not None:
                        chunk_metadata["page_end"] = chunk.page_end
                    semantic_tags = derive_semantic_tags(chunk.text, chunk.section_title)
                    if semantic_tags:
                        chunk_metadata["semantic_tags"] = semantic_tags
                    db.add(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=index,
                            chunk_text=chunk.text,
                            page_no=chunk.page_start,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            section_title=chunk.section_title,
                            semantic_tags_json=semantic_tags or None,
                            embedding_json=embeddings[index] if index < len(embeddings) else None,
                            embedding_vector=embeddings[index] if index < len(embeddings) else None,
                            metadata_json=chunk_metadata or None,
                        )
                    )
                db.commit()

    def _mark_success(
        self,
        document_id: int,
        chunk_count: int,
        *,
        started_time: datetime,
        finished_time: datetime,
        total_duration_ms: int,
        stage_durations: dict,
    ) -> None:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                return
            document.status = "SUCCESS"
            document.processing_stage = "completed"
            document.processing_message = "文档处理完成。"
            document.chunk_count = chunk_count
            document.processing_started_time = started_time
            document.processing_finished_time = finished_time
            document.processing_duration_ms = total_duration_ms
            document.stage_durations_json = dict(stage_durations)
            document.error_message = None
            db.commit()

    def _mark_failed(
        self,
        document_id: int,
        error_message: str,
        *,
        started_time: datetime,
        finished_time: datetime,
        total_duration_ms: int,
        stage_durations: dict,
    ) -> None:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                return
            document.status = "FAILED"
            document.processing_stage = "failed"
            document.processing_message = "文档处理失败。"
            document.processing_started_time = started_time
            document.processing_finished_time = finished_time
            document.processing_duration_ms = total_duration_ms
            document.stage_durations_json = dict(stage_durations)
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

    def _duration_ms(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def _embed_in_batches(
        self,
        *,
        document_id: int,
        chunks,
        started_time: datetime,
        stage_durations: dict,
    ) -> list[list[float]]:
        batch_size = max(1, int(self.settings.embedding_batch_size))
        total_chunks = len(chunks)
        total_batches = max(1, math.ceil(total_chunks / batch_size))
        embeddings: list[list[float]] = []
        stage_started_at = time.perf_counter()

        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, total_chunks)
            self._update_processing(
                document_id,
                stage="embedding",
                message=f"正在生成向量表示（第 {batch_index + 1}/{total_batches} 批，chunks {start + 1}-{end}）。",
                started_time=started_time,
                stage_durations=stage_durations,
            )
            batch_embeddings = self.embedding_provider.embed_documents([item.text for item in chunks[start:end]])
            embeddings.extend(batch_embeddings)

        stage_durations["embedding"] = self._duration_ms(stage_started_at)
        return embeddings
