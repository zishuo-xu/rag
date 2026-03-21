from sqlalchemy import cast, func, select
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import HALFVEC, HalfVector

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk, EMBEDDING_DIMENSION
from app.db.models.qa_record import QARecord


class QARepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_history(self) -> list[QARecord]:
        stmt = select(QARecord).order_by(QARecord.created_time.desc()).limit(50)
        return list(self.db.scalars(stmt))

    def get_history_detail(self, request_id: str) -> QARecord | None:
        stmt = select(QARecord).where(QARecord.request_id == request_id).limit(1)
        return self.db.scalar(stmt)

    def list_searchable_chunks(self, document_ids: list[int] | None = None) -> list[tuple[DocumentChunk, Document]]:
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "SUCCESS")
            .order_by(Document.created_time.desc(), DocumentChunk.chunk_index.asc())
        )
        if document_ids:
            stmt = stmt.where(Document.id.in_(document_ids))
        return list(self.db.execute(stmt).all())

    def count_chunks_with_embeddings(self) -> int:
        stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.embedding_vector.is_not(None))
        return int(self.db.scalar(stmt) or 0)

    def list_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 20,
        document_ids: list[int] | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        indexed_vector = cast(DocumentChunk.embedding_vector, HALFVEC(EMBEDDING_DIMENSION))
        distance = indexed_vector.cosine_distance(HalfVector(query_embedding))
        stmt = (
            select(DocumentChunk, Document, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "SUCCESS")
            .where(DocumentChunk.embedding_vector.is_not(None))
            .order_by(distance.asc())
            .limit(top_k)
        )
        if document_ids:
            stmt = stmt.where(Document.id.in_(document_ids))
        return list(self.db.execute(stmt).all())

    def list_fulltext_chunks(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        document_ids: list[int] | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        normalized_query = " ".join(part.strip() for part in query_text.split() if part.strip())
        if not normalized_query:
            return []

        searchable_text = func.concat(func.coalesce(DocumentChunk.section_title, ""), " ", DocumentChunk.chunk_text)
        vector = func.to_tsvector("simple", searchable_text)
        ts_query = func.websearch_to_tsquery("simple", normalized_query)
        rank = func.ts_rank_cd(vector, ts_query)
        stmt = (
            select(DocumentChunk, Document, rank.label("rank"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "SUCCESS")
            .where(rank > 0)
            .order_by(rank.desc(), DocumentChunk.chunk_index.asc())
            .limit(top_k)
        )
        if document_ids:
            stmt = stmt.where(Document.id.in_(document_ids))
        return list(self.db.execute(stmt).all())

    def create_record(
        self,
        *,
        request_id: str,
        question: str,
        answer: str,
        citations_json: list[dict] | None,
        top_chunks_json: list[dict] | None,
        llm_input_text: str | None,
        llm_output_text: str | None,
        llm_provider_status: str | None,
        llm_fallback_reason: str | None,
        model_name: str,
        response_time_ms: int,
        status: str,
    ) -> QARecord:
        record = QARecord(
            request_id=request_id,
            question=question,
            answer=answer,
            citations_json=citations_json,
            top_chunks_json=top_chunks_json,
            llm_input_text=llm_input_text,
            llm_output_text=llm_output_text,
            llm_provider_status=llm_provider_status,
            llm_fallback_reason=llm_fallback_reason,
            model_name=model_name,
            response_time_ms=response_time_ms,
            status=status,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
