import json
from dataclasses import dataclass
from urllib import error, request

from app.core.config import get_settings


@dataclass
class ExternalRerankItem:
    index: int
    score: float


class AliyunRerankProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider_name = settings.rerank_provider
        self.model = settings.rerank_model
        self.base_url = settings.rerank_base_url.rstrip("/")
        self.api_key = settings.rerank_api_key
        self.timeout_seconds = 20.0
        self.enabled = bool(
            self.provider_name == "aliyun"
            and self.api_key
            and self.model
            and self.base_url
        )

    def rerank(self, *, query: str, documents: list[str], top_n: int) -> list[ExternalRerankItem]:
        if not self.enabled:
            raise RuntimeError("aliyun rerank is not configured")
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(max(top_n, 1), len(documents)),
            "return_documents": False,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.base_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"aliyun rerank http {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"aliyun rerank network error: {exc}") from exc

        payload = json.loads(raw)
        output = payload.get("output") or {}
        results = payload.get("results") or output.get("results") or []
        reranked: list[ExternalRerankItem] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score")
            if index is None or score is None:
                continue
            reranked.append(ExternalRerankItem(index=int(index), score=float(score)))
        return reranked
