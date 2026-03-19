from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class QARecord(Base):
    __tablename__ = "qa_record"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    top_chunks_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    llm_input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    llm_fallback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
