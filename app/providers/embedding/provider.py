import hashlib
import math
import re

import httpx
from openai import OpenAI

from app.core.config import get_settings
from app.db.models.document_chunk import EMBEDDING_DIMENSION


LOCAL_EMBEDDING_DIM = EMBEDDING_DIMENSION


class EmbeddingProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.embedding_model
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.api_key = settings.embedding_api_key
        self.enabled = bool(settings.embedding_api_key and settings.embedding_model)
        self.client = None
        self.use_volc_multimodal = self.enabled and "ark.cn-beijing.volces.com/api/v3" in self.base_url
        if self.enabled and not self.use_volc_multimodal:
            self.client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.enabled and self.use_volc_multimodal:
            try:
                return [self._embed_with_volc(text) for text in texts]
            except Exception:
                pass
        if self.enabled and self.client:
            try:
                response = self.client.embeddings.create(model=self.model, input=texts)
                return [item.embedding for item in response.data]
            except Exception:
                pass
        return [_local_embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.enabled and self.use_volc_multimodal:
            try:
                return self._embed_with_volc(text)
            except Exception:
                pass
        if self.enabled and self.client:
            try:
                response = self.client.embeddings.create(model=self.model, input=text)
                return response.data[0].embedding
            except Exception:
                pass
        return _local_embed(text)

    @property
    def provider_name(self) -> str:
        return self.model if self.enabled and self.model else "local-hash-embedding"

    def _embed_with_volc(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/embeddings/multimodal",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        vector = _extract_embedding(payload)
        if not vector:
            raise ValueError("empty embedding response")
        return vector


def _local_embed(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % LOCAL_EMBEDDING_DIM
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    return ascii_tokens + cjk_chars


def _extract_embedding(payload: dict) -> list[float] | None:
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("embedding"), list):
        return data["embedding"]
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return first["embedding"]
    if isinstance(payload.get("embedding"), list):
        return payload["embedding"]
    return None
