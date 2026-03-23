from pathlib import Path
import sys

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import engine
from app.db.models.document_chunk import EMBEDDING_DIMENSION


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(64)"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS processing_message TEXT"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS processing_started_time TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS processing_finished_time TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS processing_duration_ms INTEGER"))
        conn.execute(text("ALTER TABLE IF EXISTS document ADD COLUMN IF NOT EXISTS stage_durations_json JSON"))
        conn.execute(text("ALTER TABLE IF EXISTS document_chunk ADD COLUMN IF NOT EXISTS embedding_json JSON"))
        conn.execute(text("ALTER TABLE IF EXISTS document_chunk ADD COLUMN IF NOT EXISTS page_start INTEGER"))
        conn.execute(text("ALTER TABLE IF EXISTS document_chunk ADD COLUMN IF NOT EXISTS page_end INTEGER"))
        conn.execute(text("ALTER TABLE IF EXISTS document_chunk ADD COLUMN IF NOT EXISTS semantic_tags_json JSON"))
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS document_task ("
                "id BIGSERIAL PRIMARY KEY, "
                "document_id BIGINT NOT NULL REFERENCES document(id), "
                "task_type VARCHAR(32) NOT NULL DEFAULT 'INGEST', "
                "trigger_source VARCHAR(32) NOT NULL DEFAULT 'SYSTEM', "
                "status VARCHAR(32) NOT NULL DEFAULT 'QUEUED', "
                "processing_stage VARCHAR(64), "
                "processing_message TEXT, "
                "processing_started_time TIMESTAMPTZ, "
                "processing_finished_time TIMESTAMPTZ, "
                "processing_duration_ms INTEGER, "
                "stage_durations_json JSON, "
                "error_message TEXT, "
                "created_time TIMESTAMPTZ DEFAULT now(), "
                "updated_time TIMESTAMPTZ DEFAULT now())"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE IF EXISTS document_chunk "
                f"ADD COLUMN IF NOT EXISTS embedding_vector vector({EMBEDDING_DIMENSION})"
            )
        )
        conn.execute(text("ALTER TABLE IF EXISTS qa_record ADD COLUMN IF NOT EXISTS llm_input_text TEXT"))
        conn.execute(text("ALTER TABLE IF EXISTS qa_record ADD COLUMN IF NOT EXISTS llm_output_text TEXT"))
        conn.execute(text("ALTER TABLE IF EXISTS qa_record ADD COLUMN IF NOT EXISTS llm_provider_status VARCHAR(64)"))
        conn.execute(text("ALTER TABLE IF EXISTS qa_record ADD COLUMN IF NOT EXISTS llm_fallback_reason TEXT"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_status_created_time "
                "ON document (status, created_time DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_task_document_id_created_time "
                "ON document_task (document_id, created_time DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_task_status_updated_time "
                "ON document_task (status, updated_time DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunk_document_id_chunk_index "
                "ON document_chunk (document_id, chunk_index)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunk_document_id_page_range "
                "ON document_chunk (document_id, page_start, page_end)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunk_section_title "
                "ON document_chunk (section_title)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunk_fulltext_gin "
                "ON document_chunk USING gin "
                "(to_tsvector('simple', coalesce(section_title, '') || ' ' || chunk_text))"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunk_embedding_halfvec_hnsw "
                f"ON document_chunk USING hnsw ((CAST(embedding_vector AS halfvec({EMBEDDING_DIMENSION}))) "
                "halfvec_cosine_ops)"
            )
        )
    print("database initialized")


if __name__ == "__main__":
    main()
