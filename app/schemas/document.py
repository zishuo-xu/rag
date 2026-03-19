from datetime import datetime

from pydantic import BaseModel


class DocumentListItem(BaseModel):
    id: int
    file_name: str
    file_type: str
    status: str
    chunk_count: int = 0
    created_time: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentDetail(DocumentListItem):
    file_path: str
    error_message: str | None = None
    updated_time: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    status: str

    model_config = {"from_attributes": True}
