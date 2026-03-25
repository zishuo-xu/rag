from uuid import uuid4

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.qa import (
    AskRequest,
    AskResponse,
    DemoAskRequest,
    DemoAskResponse,
    QAHistoryDetailResponse,
    QAHistoryItem,
    QAProgressResponse,
)
from app.services.qa_service import QAService
from app.services.redis_service import set_qa_progress


router = APIRouter()


@router.get("/history", response_model=list[QAHistoryItem])
def list_qa_history(db: Session = Depends(get_db)) -> list[QAHistoryItem]:
    return QAService(db).list_history()


@router.get("/history/{request_id}", response_model=QAHistoryDetailResponse)
def get_qa_history_detail(request_id: str, db: Session = Depends(get_db)) -> QAHistoryDetailResponse:
    return QAService(db).get_history_detail(request_id)


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    request_id = request.request_id or f"req_{uuid4().hex[:12]}"
    set_qa_progress(
        request_id,
        status="RUNNING",
        stage="queued",
        message="服务端已收到请求，正在进入问答流程。",
        progress_percent=2,
        input_summary={"question": request.question.strip(), "document_ids": request.document_ids or []},
        output_summary={"request_id": request_id},
    )
    return QAService(db).ask(request.question, request.document_ids, request_id)


@router.post("/demo", response_model=DemoAskResponse)
def run_demo_rag(request: DemoAskRequest, db: Session = Depends(get_db)) -> DemoAskResponse:
    return QAService(db).run_demo_experience(request.context_text, request.question)


@router.get("/progress/{request_id}", response_model=QAProgressResponse)
def get_qa_progress(request_id: str) -> QAProgressResponse:
    return QAService.get_progress(request_id)
