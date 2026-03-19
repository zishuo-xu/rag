import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.services.rerank_service import RerankResult


@dataclass
class ProcessedHit:
    document: Document
    chunk_index: int
    chunk_span: str
    section_title: str | None
    content: str
    score: float
    keyword_score: int
    vector_score: float


class RetrievalPostprocessor:
    def postprocess(self, reranked_hits: list[RerankResult], *, limit: int = 4) -> list[ProcessedHit]:
        if not reranked_hits:
            return []

        merged_hits = self._merge_adjacent_hits(reranked_hits)
        deduped_hits: list[ProcessedHit] = []

        for hit in merged_hits:
            if self._is_duplicate(hit, deduped_hits):
                continue
            deduped_hits.append(hit)
            if len(deduped_hits) >= limit:
                break

        return deduped_hits

    def _merge_adjacent_hits(self, reranked_hits: list[RerankResult]) -> list[ProcessedHit]:
        grouped: dict[int, list[RerankResult]] = {}
        for item in reranked_hits:
            grouped.setdefault(item.document.id, []).append(item)

        processed: list[ProcessedHit] = []
        for items in grouped.values():
            by_chunk_index: dict[int, RerankResult] = {}
            for item in items:
                existing = by_chunk_index.get(item.chunk.chunk_index)
                if existing is None or item.rerank_score > existing.rerank_score:
                    by_chunk_index[item.chunk.chunk_index] = item

            sorted_items = sorted(by_chunk_index.values(), key=lambda item: item.chunk.chunk_index)
            current_group: list[RerankResult] = []

            for item in sorted_items:
                if not current_group:
                    current_group.append(item)
                    continue

                previous = current_group[-1]
                if self._should_merge(previous.chunk, item.chunk):
                    current_group.append(item)
                    continue

                processed.append(self._build_processed_hit(current_group))
                current_group = [item]

            if current_group:
                processed.append(self._build_processed_hit(current_group))

        processed.sort(key=lambda item: item.score, reverse=True)
        return processed

    def _should_merge(self, left: DocumentChunk, right: DocumentChunk) -> bool:
        if right.chunk_index - left.chunk_index != 1:
            return False
        if left.document_id != right.document_id:
            return False
        if left.section_title and right.section_title and left.section_title != right.section_title:
            return False
        return True

    def _build_processed_hit(self, items: list[RerankResult]) -> ProcessedHit:
        first = items[0]
        chunk_indices = [item.chunk.chunk_index for item in items]
        merged_content = self._merge_contents([item.chunk.chunk_text for item in items])
        return ProcessedHit(
            document=first.document,
            chunk_index=chunk_indices[0],
            chunk_span=_build_chunk_span(chunk_indices),
            section_title=_normalize_section_title(first.chunk.section_title),
            content=merged_content[:900],
            score=max(item.rerank_score for item in items),
            keyword_score=max(item.keyword_score for item in items),
            vector_score=max(item.vector_score for item in items),
        )

    def _merge_contents(self, contents: list[str]) -> str:
        merged: list[str] = []
        for content in contents:
            if not merged:
                merged.append(content.strip())
                continue
            previous = merged[-1]
            overlap = _longest_suffix_prefix(previous, content)
            if overlap >= 30:
                merged[-1] = previous + content[overlap:]
            else:
                merged.append(content.strip())
        return "\n\n".join(part for part in merged if part)

    def _is_duplicate(self, candidate: ProcessedHit, accepted: list[ProcessedHit]) -> bool:
        normalized_candidate = _normalize_text(candidate.content)
        for item in accepted:
            if item.document.id != candidate.document.id:
                continue
            if _span_overlaps(item.chunk_span, candidate.chunk_span):
                return True
            normalized_existing = _normalize_text(item.content)
            if normalized_existing == normalized_candidate:
                return True
            similarity = SequenceMatcher(None, normalized_existing, normalized_candidate).ratio()
            if similarity >= 0.9:
                return True
        return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_section_title(section_title: str | None) -> str | None:
    if not section_title:
        return None
    normalized = re.sub(r"\s+", " ", section_title).strip()
    return normalized or None


def _build_chunk_span(indices: list[int]) -> str:
    if not indices:
        return "-"
    if len(indices) == 1:
        return str(indices[0])
    return f"{indices[0]}-{indices[-1]}"


def _span_overlaps(left: str, right: str) -> bool:
    left_start, left_end = _parse_span(left)
    right_start, right_end = _parse_span(right)
    return max(left_start, right_start) <= min(left_end, right_end)


def _parse_span(span: str) -> tuple[int, int]:
    if "-" not in span:
        value = int(span)
        return value, value
    start, end = span.split("-", 1)
    return int(start), int(end)


def _longest_suffix_prefix(left: str, right: str, max_window: int = 180) -> int:
    left_tail = left[-max_window:]
    right_head = right[:max_window]
    max_overlap = min(len(left_tail), len(right_head))
    for overlap in range(max_overlap, 0, -1):
        if left_tail[-overlap:] == right_head[:overlap]:
            return overlap
    return 0
