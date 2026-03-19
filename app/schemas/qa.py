from datetime import datetime

from pydantic import BaseModel, Field


class QAHistoryItem(BaseModel):
    request_id: str
    question: str
    status: str
    model_name: str
    response_time_ms: int
    created_time: datetime | None = None

    model_config = {"from_attributes": True}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[int] | None = None
    request_id: str | None = None


class CitationItem(BaseModel):
    citation_id: int
    document_id: int
    file_name: str
    chunk_index: int
    chunk_span: str | None = None
    section_title: str | None = None
    content: str
    score: int


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[CitationItem]
    elapsed_time_ms: int
    model_name: str
    generation_mode: str | None = None
    llm_provider_status: str | None = None
    llm_fallback_reason: str | None = None


class QAProgressResponse(BaseModel):
    request_id: str
    status: str
    stage: str
    message: str
    progress_percent: int
    error_message: str | None = None
    updated_at: datetime | None = None
    steps: list[dict] = Field(default_factory=list)


class QAHistoryDetailResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[CitationItem]
    llm_input_text: str | None = None
    llm_output_text: str | None = None
    llm_provider_status: str | None = None
    llm_fallback_reason: str | None = None
    model_name: str
    response_time_ms: int
    status: str
    created_time: datetime | None = None
