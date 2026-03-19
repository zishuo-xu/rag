import argparse
import shutil
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.qa_record import QARecord
from app.db.session import SessionLocal
from app.services.ingest_service import IngestService
from app.services.redis_service import redis_client


def reset_database() -> dict[str, int]:
    db = SessionLocal()
    try:
        chunk_count = db.query(DocumentChunk).delete()
        doc_count = db.query(Document).delete()
        qa_count = db.query(QARecord).delete()
        db.commit()
        return {
            "document_chunk": chunk_count,
            "document": doc_count,
            "qa_record": qa_count,
        }
    finally:
        db.close()


def clear_upload_dir(upload_dir: Path) -> int:
    removed = 0
    upload_dir.mkdir(parents=True, exist_ok=True)
    for child in upload_dir.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
            removed += 1
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def clear_progress_cache() -> int:
    removed = 0
    for key in redis_client.scan_iter("qa:progress:*"):
        redis_client.delete(key)
        removed += 1
    return removed


def seed_documents(corpus_dir: Path) -> list[tuple[int, str, str, int]]:
    settings = get_settings()
    seeded: list[tuple[int, str, str, int]] = []
    db = SessionLocal()
    try:
        ingest_service = IngestService(db)
        for source_path in sorted(corpus_dir.glob("*.md")):
            stored_name = f"{uuid4().hex}{source_path.suffix.lower()}"
            stored_path = settings.upload_dir / stored_name
            shutil.copy2(source_path, stored_path)

            document = Document(
                file_name=source_path.name,
                file_type=source_path.suffix.lstrip("."),
                file_path=str(stored_path),
                status="PROCESSING",
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            processed = ingest_service.process_document(document)
            seeded.append((processed.id, processed.file_name, processed.status, processed.chunk_count))
        return seeded
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset current RAG documents and seed a local evaluation corpus.")
    parser.add_argument(
        "--corpus-dir",
        default="evals/industry_corpus",
        help="Directory containing markdown documents to ingest.",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    deleted = reset_database()
    removed_files = clear_upload_dir(settings.upload_dir)
    removed_progress = clear_progress_cache()
    seeded = seed_documents(corpus_dir)

    print("Reset complete:")
    print(f"- deleted document rows: {deleted['document']}")
    print(f"- deleted document_chunk rows: {deleted['document_chunk']}")
    print(f"- deleted qa_record rows: {deleted['qa_record']}")
    print(f"- removed upload entries: {removed_files}")
    print(f"- removed redis progress keys: {removed_progress}")
    print("")
    print("Seeded documents:")
    for document_id, file_name, status, chunk_count in seeded:
        print(f"- id={document_id}, status={status}, chunks={chunk_count}, file={file_name}")


if __name__ == "__main__":
    main()
