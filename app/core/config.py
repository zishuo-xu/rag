from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="rag", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")

    upload_dir: Path = Field(default=Path("storage/uploads"), alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=20, alias="MAX_FILE_SIZE_MB")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")

    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="", alias="EMBEDDING_MODEL")
    embedding_base_url: str = Field(default="https://api.openai.com/v1", alias="EMBEDDING_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    rerank_provider: str = Field(default="local", alias="RERANK_PROVIDER")
    rerank_api_key: str = Field(default="", alias="RERANK_API_KEY")
    rerank_model: str = Field(default="", alias="RERANK_MODEL")
    rerank_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        alias="RERANK_BASE_URL",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
