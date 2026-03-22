import re
from dataclasses import dataclass

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.providers.rerank.aliyun_provider import AliyunRerankProvider
from app.utils.semantic_tags import derive_semantic_tags


@dataclass
class RerankResult:
    chunk: DocumentChunk
    document: Document
    rerank_score: float
    keyword_score: int
    vector_score: float
    external_score: float = 0.0


class RerankService:
    def __init__(self) -> None:
        self.external_provider = AliyunRerankProvider()

    def rerank(
        self,
        question: str,
        candidates: list[tuple[DocumentChunk, Document, int, float]],
        *,
        limit: int = 4,
    ) -> list[RerankResult]:
        normalized_question = question.strip().lower()
        question_terms = _extract_terms(question)
        query_tags = set(derive_semantic_tags(question))
        reranked: list[RerankResult] = []

        for chunk, document, keyword_score, vector_score in candidates:
            chunk_text = chunk.chunk_text or ""
            lowered_chunk = chunk_text.lower()
            normalized_title = _normalize_section_title(chunk.section_title)
            exact_phrase_bonus = 12.0 if normalized_question and normalized_question in lowered_chunk else 0.0
            title_bonus = _title_bonus(question_terms, normalized_title)
            coverage_score = _coverage_score(question_terms, lowered_chunk)
            density_score = _density_score(question_terms, lowered_chunk)
            position_bonus = max(0.0, 2.0 - chunk.chunk_index * 0.02)
            semantic_tags = _chunk_semantic_tags(chunk)
            semantic_tag_bonus = _semantic_tag_bonus(query_tags, semantic_tags)
            meta_penalty = _meta_section_penalty(normalized_title, chunk_text)

            rerank_score = (
                vector_score * 8.0
                + keyword_score * 2.5
                + coverage_score * 6.0
                + density_score * 4.0
                + title_bonus
                + exact_phrase_bonus
                + position_bonus
                + semantic_tag_bonus
                - meta_penalty
            )

            reranked.append(
                RerankResult(
                    chunk=chunk,
                    document=document,
                    rerank_score=rerank_score,
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                    external_score=0.0,
                )
            )

        reranked = self._apply_external_rerank(question, reranked)
        reranked.sort(key=_result_sort_key, reverse=True)
        return reranked[:limit]

    def _apply_external_rerank(self, question: str, reranked: list[RerankResult]) -> list[RerankResult]:
        if not self.external_provider.enabled or not reranked:
            return reranked

        local_sorted = sorted(reranked, key=_result_sort_key, reverse=True)
        candidate_limit = min(len(local_sorted), 12)
        external_candidates = local_sorted[:candidate_limit]
        documents = [_rerank_document_text(item) for item in external_candidates]

        try:
            external_results = self.external_provider.rerank(
                query=question,
                documents=documents,
                top_n=candidate_limit,
            )
        except Exception:
            return reranked

        external_scores = {item.index: item.score for item in external_results}
        updated: list[RerankResult] = []
        for index, item in enumerate(external_candidates):
            score = external_scores.get(index, 0.0)
            updated.append(
                RerankResult(
                    chunk=item.chunk,
                    document=item.document,
                    rerank_score=item.rerank_score + score * 20.0,
                    keyword_score=item.keyword_score,
                    vector_score=item.vector_score,
                    external_score=score,
                )
            )

        return updated + local_sorted[candidate_limit:]


def _extract_terms(text: str) -> list[str]:
    lowered = text.strip().lower()
    ascii_terms = re.findall(r"[a-z0-9_]+", lowered)
    cjk_phrases = [phrase for phrase in re.findall(r"[\u4e00-\u9fff]{3,}", lowered) if len(phrase) <= 8]
    cjk_bigrams = [lowered[index : index + 2] for index in range(len(lowered) - 1) if _is_cjk_pair(lowered[index : index + 2])]
    terms = ascii_terms + cjk_phrases + cjk_bigrams
    return list(dict.fromkeys(term for term in terms if term.strip()))


def _is_cjk_pair(token: str) -> bool:
    return len(token) == 2 and all("\u4e00" <= char <= "\u9fff" for char in token)


def _coverage_score(question_terms: list[str], chunk_text: str) -> float:
    if not question_terms:
        return 0.0
    matched = sum(1 for term in question_terms if term in chunk_text)
    return matched / len(question_terms)


def _density_score(question_terms: list[str], chunk_text: str) -> float:
    if not question_terms or not chunk_text:
        return 0.0
    total_hits = sum(min(chunk_text.count(term), 4) for term in question_terms)
    normalized_length = max(len(chunk_text) / 120.0, 1.0)
    return total_hits / normalized_length


def _title_bonus(question_terms: list[str], section_title: str | None) -> float:
    if not section_title:
        return 0.0
    lowered_title = section_title.lower()
    matched = sum(1 for term in question_terms if term in lowered_title)
    return matched * 2.0


def _meta_section_penalty(section_title: str | None, chunk_text: str) -> float:
    if not section_title:
        return 0.0
    penalty = 0.0
    if any(marker in section_title for marker in ["适合评测", "问答点"]):
        penalty += 20.0
    if any(marker in section_title for marker in ["研究目的", "分析范围"]):
        penalty += 8.0
    if "这份文档适合测试以下问题类型" in chunk_text:
        penalty += 12.0
    return penalty


def _normalize_section_title(section_title: str | None) -> str | None:
    if not section_title:
        return None
    normalized = re.sub(r"\s+", " ", section_title).strip().lower()
    return normalized or None


def _chunk_semantic_tags(chunk: DocumentChunk) -> set[str]:
    if chunk.semantic_tags_json:
        return set(chunk.semantic_tags_json)
    metadata_tags = []
    if chunk.metadata_json and isinstance(chunk.metadata_json, dict):
        metadata_tags = chunk.metadata_json.get("semantic_tags") or []
    if metadata_tags:
        return set(metadata_tags)
    return set(derive_semantic_tags(chunk.chunk_text or "", chunk.section_title))


def _semantic_tag_bonus(query_tags: set[str], chunk_tags: set[str]) -> float:
    if not query_tags or not chunk_tags:
        return 0.0
    overlap = query_tags & chunk_tags
    if not overlap:
        return 0.0
    bonus = len(overlap) * 3.0
    if {"globalization", "us_market"} <= overlap:
        bonus += 2.0
    if {"globalization", "mexico_factory"} <= overlap:
        bonus += 2.0
    if {"globalization", "morocco_factory"} <= overlap:
        bonus += 2.0
    return bonus


def _rerank_document_text(item: RerankResult) -> str:
    title = item.chunk.section_title or "未标注"
    chunk_text = item.chunk.chunk_text or ""
    return f"章节: {title}\n内容: {chunk_text}"


def _result_sort_key(item: RerankResult) -> tuple[float, float, int, float, int]:
    return (
        item.external_score,
        item.rerank_score,
        item.keyword_score,
        item.vector_score,
        -item.chunk.chunk_index,
    )
