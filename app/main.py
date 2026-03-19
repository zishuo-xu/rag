from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.router import api_router
from app.core.config import get_settings


settings = get_settings()
static_dir = Path(__file__).resolve().parent / "static"

app = FastAPI(title=settings.app_name)
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")
