from fastapi import APIRouter, Depends, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.document import DocumentDetail, DocumentListItem, DocumentUploadResponse
from app.services.document_service import DocumentService


router = APIRouter()


@router.get("", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentListItem]:
    service = DocumentService(db)
    return service.list_documents()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(upload_file: UploadFile, db: Session = Depends(get_db)) -> DocumentUploadResponse:
    service = DocumentService(db)
    return service.save_upload(upload_file)


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentDetail:
    service = DocumentService(db)
    return service.get_document(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> Response:
    service = DocumentService(db)
    service.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{document_id}/reprocess", response_model=DocumentUploadResponse)
def reprocess_document(document_id: int, db: Session = Depends(get_db)) -> DocumentUploadResponse:
    service = DocumentService(db)
    return service.reprocess_document(document_id)
