from datetime import datetime

from pydantic import BaseModel


class DocumentTaskInfo(BaseModel):
    id: int
    task_type: str
    trigger_source: str
    status: str
    processing_stage: str | None = None
    processing_message: str | None = None
    processing_started_time: datetime | None = None
    processing_finished_time: datetime | None = None
    processing_duration_ms: int | None = None
    stage_durations_json: dict | None = None
    error_message: str | None = None
    created_time: datetime | None = None
    updated_time: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: int
    file_name: str
    file_type: str
    status: str
    processing_stage: str | None = None
    processing_message: str | None = None
    processing_started_time: datetime | None = None
    processing_finished_time: datetime | None = None
    processing_duration_ms: int | None = None
    stage_durations_json: dict | None = None
    error_message: str | None = None
    chunk_count: int = 0
    created_time: datetime | None = None
    latest_task: DocumentTaskInfo | None = None

    model_config = {"from_attributes": True}


class DocumentDetail(DocumentListItem):
    file_path: str
    error_message: str | None = None
    updated_time: datetime | None = None
    recent_tasks: list[DocumentTaskInfo] | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    status: str
    task_id: int | None = None

    model_config = {"from_attributes": True}
