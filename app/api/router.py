from fastapi import APIRouter

from app.api.routes import documents, health, qa


api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, prefix="/api/documents", tags=["documents"])
api_router.include_router(qa.router, prefix="/api/qa", tags=["qa"])
