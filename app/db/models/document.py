from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PROCESSING")
    processing_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_started_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_finished_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage_durations_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
