import re
import time
from collections import Counter
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk, EMBEDDING_DIMENSION
from app.db.repositories.qa_repository import QARepository
from app.providers.embedding.provider import EmbeddingProvider
from app.providers.llm.openai_provider import OpenAILLMProvider
from app.schemas.qa import AskResponse, CitationItem, QAHistoryDetailResponse, QAProgressResponse
from app.services.query_rewrite_service import QueryRewriteService
from app.services.redis_service import get_qa_progress, set_qa_progress
from app.services.retrieval_postprocessor import ProcessedHit, RetrievalPostprocessor
from app.services.rerank_service import RerankService
from app.utils.semantic_tags import derive_semantic_tags


class QAService:
    def __init__(self, db: Session) -> None:
        self.repo = QARepository(db)
        self.settings = get_settings()
        self.llm_provider = OpenAILLMProvider()
        self.embedding_provider = EmbeddingProvider()
        self.rerank_service = RerankService()
        self.retrieval_postprocessor = RetrievalPostprocessor()
        self.query_rewrite_service = QueryRewriteService()

    def list_history(self):
        return self.repo.list_history()

    def get_history_detail(self, request_id: str) -> QAHistoryDetailResponse:
        record = self.repo.get_history_detail(request_id)
        if not record:
            raise HTTPException(status_code=404, detail="未找到对应的问答记录")
        citations = [CitationItem.model_validate(item) for item in (record.citations_json or [])]
        return QAHistoryDetailResponse(
            request_id=record.request_id,
            question=record.question,
            answer=record.answer,
            citations=citations,
            llm_input_text=record.llm_input_text,
            llm_output_text=record.llm_output_text,
            llm_provider_status=record.llm_provider_status,
            llm_fallback_reason=record.llm_fallback_reason,
            model_name=record.model_name,
            response_time_ms=record.response_time_ms,
            status=record.status,
            created_time=record.created_time,
        )

    def ask(self, question: str, document_ids: list[int] | None = None, request_id: str | None = None) -> AskResponse:
        return self.ask_with_options(
            question,
            document_ids=document_ids,
            request_id=request_id,
            persist_record=True,
            track_progress=True,
        )

    def ask_with_options(
        self,
        question: str,
        *,
        document_ids: list[int] | None = None,
        request_id: str | None = None,
        persist_record: bool = True,
        track_progress: bool = True,
    ) -> AskResponse:
        started_at = time.perf_counter()
        request_id = request_id or f"req_{uuid4().hex[:12]}"
        normalized_question = question.strip()
        llm_input_text: str | None = None
        llm_output_text: str | None = None
        llm_provider_status: str | None = None
        llm_fallback_reason: str | None = None
        self._set_progress(
            request_id,
            status="RUNNING",
            stage="received",
            message="已接收问题，准备开始处理。",
            progress_percent=5,
            input_summary={"question": normalized_question, "document_ids": document_ids or []},
            output_summary={"request_id": request_id, "normalized_question": normalized_question},
            enabled=track_progress,
        )
        rewrite_result = self.query_rewrite_service.rewrite(normalized_question)
        rewritten_question = rewrite_result.rewritten_question
        self._set_progress(
            request_id,
            status="RUNNING",
            stage="received",
            message="问题已规范化，准备进入检索。",
            progress_percent=12,
            input_summary={"question": normalized_question, "document_ids": document_ids or []},
            output_summary={
                "request_id": request_id,
                "normalized_question": normalized_question,
                "rewritten_question": rewritten_question,
                "applied_rules": rewrite_result.applied_rules,
            },
            enabled=track_progress,
        )
        tokens = _extract_tokens(rewritten_question)
        try:
            self._set_progress(
                request_id,
                status="RUNNING",
                stage="embedding",
                message="正在生成问题向量。",
                progress_percent=20,
                input_summary={
                    "question": normalized_question,
                    "rewritten_question": rewritten_question,
                    "applied_rules": rewrite_result.applied_rules,
                    "token_preview": tokens[:10],
                    "token_count": len(tokens),
                },
                enabled=track_progress,
            )
            query_embedding = self.embedding_provider.embed_query(rewritten_question)
            self._set_progress(
                request_id,
                status="RUNNING",
                stage="embedding",
                message="问题向量已生成。",
                progress_percent=28,
                input_summary={
                    "question": normalized_question,
                    "rewritten_question": rewritten_question,
                    "applied_rules": rewrite_result.applied_rules,
                    "token_preview": tokens[:10],
                    "token_count": len(tokens),
                },
                output_summary={
                    "provider": self.embedding_provider.provider_name,
                    "dimension": len(query_embedding),
                    "vector_preview": _vector_preview(query_embedding),
                },
                enabled=track_progress,
            )

            self._set_progress(
                request_id,
                status="RUNNING",
                stage="retrieving",
                message="正在检索相关片段。",
                progress_percent=40,
                input_summary={
                    "vector_dimension": len(query_embedding),
                    "vector_preview": _vector_preview(query_embedding),
                    "document_filter": document_ids or [],
                },
                enabled=track_progress,
            )
            vector_rows = []
            vector_retrieval_mode = "pgvector"
            if len(query_embedding) == EMBEDDING_DIMENSION:
                vector_rows = self.repo.list_similar_chunks(query_embedding, top_k=30, document_ids=document_ids)
            else:
                vector_retrieval_mode = "keyword_only_due_to_embedding_dim_mismatch"
            candidates: list[tuple[DocumentChunk, Document, int, float]] = []
            for chunk, document, distance in vector_rows:
                keyword_score = _score_text(tokens, chunk.chunk_text)
                vector_score = max(0.0, 1.0 - float(distance))
                if keyword_score > 0 or vector_score > 0:
                    candidates.append((chunk, document, keyword_score, vector_score))

            if not candidates:
                chunk_rows = self.repo.list_searchable_chunks(document_ids)
                for chunk, document in chunk_rows:
                    keyword_score = _score_text(tokens, chunk.chunk_text)
                    if keyword_score > 0:
                        candidates.append((chunk, document, keyword_score, 0.0))

            focused_candidates, focus_summary = _focus_candidates_by_document(rewritten_question, candidates)
            if focused_candidates:
                candidates = focused_candidates

            self._set_progress(
                request_id,
                status="RUNNING",
                stage="retrieving",
                message="相关片段检索完成。",
                progress_percent=50,
                input_summary={
                    "vector_dimension": len(query_embedding),
                    "vector_preview": _vector_preview(query_embedding),
                    "document_filter": document_ids or [],
                },
                output_summary={
                    "vector_hit_count": len(vector_rows),
                    "candidate_count": len(candidates),
                    "candidate_preview": _candidate_preview(candidates),
                    "focus_documents": focus_summary,
                    "vector_retrieval_mode": vector_retrieval_mode,
                },
                enabled=track_progress,
            )

            self._set_progress(
                request_id,
                status="RUNNING",
                stage="reranking",
                message="正在对候选片段做二次重排。",
                progress_percent=60,
                input_summary={"candidate_count": len(candidates), "candidate_preview": _candidate_preview(candidates)},
                enabled=track_progress,
            )
            reranked_hits = self.rerank_service.rerank(rewritten_question, candidates, limit=12)
            processed_hits = self.retrieval_postprocessor.postprocess(reranked_hits, limit=16)
            answer_hit_limit = 4 if self.llm_provider.enabled else 3
            top_hits = _select_answer_hits(rewritten_question, processed_hits, limit=answer_hit_limit)
            self._set_progress(
                request_id,
                status="RUNNING",
                stage="reranking",
                message="候选片段重排完成。",
                progress_percent=68,
                input_summary={"candidate_count": len(candidates), "candidate_preview": _candidate_preview(candidates)},
                output_summary={
                    "reranked_hit_count": len(reranked_hits),
                    "top_hit_count": len(top_hits),
                    "top_hit_preview": _processed_hit_preview(top_hits),
                    "rewritten_question": rewritten_question,
                },
                enabled=track_progress,
            )

            if top_hits:
                citations = [
                    CitationItem(
                        citation_id=index + 1,
                        document_id=item.document.id,
                        file_name=item.document.file_name,
                        chunk_index=item.chunk_index,
                        chunk_span=item.chunk_span,
                        section_title=_normalize_section_title(item.section_title),
                        content=item.content[:220],
                        score=int(item.score * 100),
                    )
                    for index, item in enumerate(top_hits)
                ]
                citations = _order_citations_for_answer(rewritten_question, citations)
                self._set_progress(
                    request_id,
                    status="RUNNING",
                    stage="generating",
                    message="正在调用模型生成答案。",
                    progress_percent=80,
                    input_summary={
                        "question": normalized_question,
                        "rewritten_question": rewritten_question,
                        "applied_rules": rewrite_result.applied_rules,
                        "model": self.settings.llm_model or "keyword-retrieval",
                        "citation_count": len(citations),
                        "citation_preview": _citation_preview(citations),
                        "llm_input_preview": self._build_llm_input_preview(rewritten_question, citations),
                    },
                    enabled=track_progress,
                )
                answer, model_name, llm_input_text, llm_output_text, llm_provider_status, llm_fallback_reason = self._generate_answer(rewritten_question, citations)
                status = "SUCCESS"
                self._set_progress(
                    request_id,
                    status="RUNNING",
                    stage="generating",
                    message="答案生成完成。",
                    progress_percent=88,
                    input_summary={
                        "question": normalized_question,
                        "rewritten_question": rewritten_question,
                        "model": self.settings.llm_model or "keyword-retrieval",
                        "citation_count": len(citations),
                        "citation_preview": _citation_preview(citations),
                        "llm_input_preview": _preview_text(llm_input_text),
                    },
                    output_summary={
                        "model_name": model_name,
                        "answer_preview": answer[:220],
                        "llm_output_preview": _preview_text(llm_output_text),
                        "llm_provider_status": llm_provider_status,
                        "llm_fallback_reason": _preview_text(llm_fallback_reason),
                    },
                    enabled=track_progress,
                )
            else:
                citations = []
                answer = "知识库中暂未检索到足够相关的内容，请尝试换个问法或重新上传更相关的文档。"
                model_name = "keyword-retrieval"
                llm_provider_status = "no_retrieval_result"
                llm_fallback_reason = "未检索到足够相关的片段，未调用外部 LLM。"
                status = "FAILED"

            elapsed_time_ms = int((time.perf_counter() - started_at) * 1000)
            self._set_progress(
                request_id,
                status="RUNNING",
                stage="persisting",
                message="正在写入问答记录并整理返回结果。",
                progress_percent=95,
                input_summary={"status": status, "model_name": model_name, "citation_count": len(citations)},
                output_summary={"elapsed_time_ms": elapsed_time_ms},
                enabled=track_progress,
            )
            if persist_record:
                self.repo.create_record(
                    request_id=request_id,
                    question=normalized_question,
                    answer=answer,
                    citations_json=[item.model_dump() for item in citations] or None,
                    top_chunks_json=[item.model_dump() for item in citations] or None,
                    llm_input_text=llm_input_text,
                    llm_output_text=llm_output_text,
                    llm_provider_status=llm_provider_status,
                    llm_fallback_reason=llm_fallback_reason,
                    model_name=model_name,
                    response_time_ms=elapsed_time_ms,
                    status=status,
                )
            final_status = "COMPLETED" if status == "SUCCESS" else "FAILED"
            final_message = "问答处理完成。" if status == "SUCCESS" else "未检索到足够相关内容。"
            self._set_progress(
                request_id,
                status=final_status,
                stage="completed",
                message=final_message,
                progress_percent=100,
                input_summary={"status": status, "model_name": model_name},
                output_summary={
                    "elapsed_time_ms": elapsed_time_ms,
                    "citation_count": len(citations),
                    "answer_preview": answer[:220],
                    "llm_input_preview": _preview_text(llm_input_text),
                    "llm_output_preview": _preview_text(llm_output_text),
                    "llm_provider_status": llm_provider_status,
                    "llm_fallback_reason": _preview_text(llm_fallback_reason),
                },
                enabled=track_progress,
            )
            return AskResponse(
                request_id=request_id,
                question=normalized_question,
                answer=answer,
                citations=citations,
                elapsed_time_ms=elapsed_time_ms,
                model_name=model_name,
                generation_mode=_generation_mode(model_name, llm_provider_status),
                llm_provider_status=llm_provider_status,
                llm_fallback_reason=llm_fallback_reason,
            )
        except Exception as exc:
            self._set_progress(
                request_id,
                status="FAILED",
                stage="failed",
                message="问答处理失败。",
                progress_percent=100,
                error_message=str(exc),
                output_summary={"error_type": type(exc).__name__},
                enabled=track_progress,
            )
            raise

    @staticmethod
    def get_progress(request_id: str) -> QAProgressResponse:
        payload = get_qa_progress(request_id)
        if not payload:
            raise HTTPException(status_code=404, detail="未找到对应的问答进度")
        return QAProgressResponse.model_validate(payload)

    def _set_progress(
        self,
        request_id: str,
        *,
        status: str,
        stage: str,
        message: str,
        progress_percent: int,
        error_message: str | None = None,
        input_summary: dict | None = None,
        output_summary: dict | None = None,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            return
        set_qa_progress(
            request_id,
            status=status,
            stage=stage,
            message=message,
            progress_percent=progress_percent,
            error_message=error_message,
            input_summary=input_summary,
            output_summary=output_summary,
        )

    def _generate_answer(self, question: str, citations: list[CitationItem]) -> tuple[str, str, str | None, str | None, str, str | None]:
        context_blocks = [
            (
                f"[{citation.citation_id}] 文件: {citation.file_name}; "
                f"Chunk: {citation.chunk_index}; "
                f"章节: {citation.section_title or '未标注'}; "
                f"内容: {_compress_citation_content(citation.content, question)}"
            )
            for citation in citations
        ]
        if self.llm_provider.enabled:
            try:
                result = self.llm_provider.generate_answer(question=question, context_blocks=context_blocks)
                if result and result.answer:
                    return (
                        result.answer,
                        result.model_name,
                        result.input_text,
                        result.output_text,
                        result.provider_status,
                        result.fallback_reason,
                    )
            except Exception as exc:
                fallback_answer = _build_answer(question, citations)
                fallback_input = self._build_llm_input_preview(question, citations)
                return (
                    fallback_answer,
                    "keyword-retrieval",
                    fallback_input,
                    fallback_answer,
                    "local_fallback",
                    f"external_llm_failed: {type(exc).__name__}: {exc}",
                )
        fallback_answer = _build_answer(question, citations)
        fallback_input = self._build_llm_input_preview(question, citations)
        return fallback_answer, "keyword-retrieval", fallback_input, fallback_answer, "local_fallback", "llm not configured"

    def _build_llm_input_preview(self, question: str, citations: list[CitationItem]) -> str:
        context_blocks = [
            (
                f"[{citation.citation_id}] 文件: {citation.file_name}; "
                f"Chunk: {citation.chunk_index}; "
                f"章节: {citation.section_title or '未标注'}; "
                f"内容: {_compress_citation_content(citation.content, question)}"
            )
            for citation in citations
        ]
        if self.llm_provider.enabled:
            return self.llm_provider.build_prompt(question=question, context_blocks=context_blocks)
        return self._build_fallback_prompt(question, context_blocks)

    def _build_fallback_prompt(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n".join(context_blocks)
        return (
            f"用户问题：{question}\n\n"
            "当前未启用外部 LLM，系统将基于检索结果生成检索式摘要。\n\n"
            f"参考资料：\n{context}"
        )


def _extract_tokens(text: str) -> list[str]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk_bigrams = [text[index : index + 2] for index in range(len(text) - 1) if _is_cjk_pair(text[index : index + 2])]
    cjk_phrases = [phrase for phrase in re.findall(r"[\u4e00-\u9fff]{3,}", text) if len(phrase) <= 8]
    tokens = [token for token in ascii_tokens + cjk_phrases + cjk_bigrams if token.strip()]
    deduped = [token for token, count in Counter(tokens).items() if count >= 1]
    return deduped


def _is_cjk_pair(token: str) -> bool:
    return len(token) == 2 and all("\u4e00" <= char <= "\u9fff" for char in token)


def _score_text(tokens: list[str], text: str) -> int:
    lowered = text.lower()
    score = 0
    for token in tokens:
        if token in lowered:
            score += min(lowered.count(token), 3)
    return score


def _build_answer(question: str, citations: list[CitationItem]) -> str:
    if not citations:
        return "知识库暂无足够依据。"

    query_tags = set(derive_semantic_tags(question))
    lead = _build_lead_sentence(question, citations, query_tags)
    evidence_lines = []
    for citation in citations[:3]:
        summary = _summarize_citation(citation, question)
        evidence_lines.append(f"- [{citation.citation_id}] {summary}")
    return f"{lead}\n\n依据要点：\n" + "\n".join(evidence_lines)


def _order_citations_for_answer(question: str, citations: list[CitationItem]) -> list[CitationItem]:
    query_tags = set(derive_semantic_tags(question))
    ordered = sorted(
        citations,
        key=lambda citation: _citation_answer_priority(citation, query_tags),
        reverse=True,
    )
    return [
        citation.model_copy(update={"citation_id": index + 1})
        for index, citation in enumerate(ordered)
    ]


def _select_answer_hits(question: str, hits: list[ProcessedHit], limit: int = 3) -> list[ProcessedHit]:
    query_tags = set(derive_semantic_tags(question))
    ordered = sorted(
        hits,
        key=lambda hit: _answer_hit_priority(hit, query_tags),
        reverse=True,
    )
    selected: list[ProcessedHit] = []
    seen_sections: set[str] = set()
    for hit in ordered:
        section_key = (hit.section_title or "").strip().lower()
        if section_key and section_key in seen_sections:
            continue
        selected.append(hit)
        if section_key:
            seen_sections.add(section_key)
        if len(selected) >= limit:
            break

    if selected:
        return selected
    return ordered[:limit]


def _answer_hit_priority(hit: ProcessedHit, query_tags: set[str]) -> tuple[float, int]:
    hit_tags = set(derive_semantic_tags(hit.content, hit.section_title))
    overlap = len(query_tags & hit_tags)
    lexical_bonus = _question_match_bonus(hit.content, hit.section_title, query_tags)
    generic_penalty = 1 if hit.section_title and any(
        marker in hit.section_title for marker in ["结论", "展望", "建议", "风险提示", "股价走势"]
    ) else 0
    meta_penalty = 1 if hit.section_title and any(
        marker in hit.section_title for marker in ["适合评测", "问答点", "研究目的", "分析范围"]
    ) else 0
    specificity_bonus = 1 if hit.section_title and any(
        marker in hit.section_title for marker in ["客户", "合作", "财务", "估值", "基本情况", "技术路线", "关注重点"]
    ) else 0
    return (
        hit.score + overlap * 12 + lexical_bonus + specificity_bonus * 4 - generic_penalty * 8 - meta_penalty * 18,
        -hit.chunk_index,
    )


def _citation_answer_priority(citation: CitationItem, query_tags: set[str]) -> tuple[float, int, int]:
    chunk_tags = set(derive_semantic_tags(citation.content, citation.section_title))
    overlap_score = len(query_tags & chunk_tags)
    lexical_bonus = _question_match_bonus(citation.content, citation.section_title, query_tags)
    generic_penalty = 1 if citation.section_title and any(
        marker in citation.section_title for marker in ["结论", "展望", "建议", "风险提示", "股价走势"]
    ) else 0
    meta_penalty = 1 if citation.section_title and any(
        marker in citation.section_title for marker in ["适合评测", "问答点", "研究目的", "分析范围"]
    ) else 0
    title_score = 1 if citation.section_title else 0
    specificity_bonus = 1 if citation.section_title and any(
        marker in citation.section_title for marker in ["客户", "合作", "财务", "估值", "基本情况", "技术路线", "关注重点"]
    ) else 0
    return (
        overlap_score * 10
        + lexical_bonus
        + specificity_bonus * 4
        - generic_penalty * 6
        - meta_penalty * 18
        + title_score * 2
        + citation.score,
        title_score,
        -citation.chunk_index,
    )


def _normalize_section_title(section_title: str | None) -> str | None:
    if not section_title:
        return None
    normalized = re.sub(r"\s+", " ", section_title).strip()
    return normalized or None


def _question_match_bonus(content: str, section_title: str | None, query_tags: set[str]) -> float:
    source = f"{section_title or ''} {content}"
    bonus = 0.0
    if "company_profile" in query_tags and any(marker in source for marker in ["上市", "股票代码", "基本情况"]):
        bonus += 8.0
    if "globalization" in query_tags and any(marker in source for marker in ["美国", "墨西哥", "摩洛哥", "欧洲", "海外"]):
        bonus += 10.0
    if "customer" in query_tags and any(marker in source for marker in ["客户", "合作", "供应链"]):
        bonus += 8.0
    if "finance" in query_tags and any(marker in source for marker in ["营收", "净利润", "财务", "毛利率", "现金流"]):
        bonus += 8.0
    if "valuation" in query_tags and any(marker in source for marker in ["估值", "市盈率", "目标价"]):
        bonus += 8.0
    if "smart_driving" in query_tags and any(marker in source for marker in ["智驾", "智能驾驶", "HSD", "XNGP", "NAD", "ANP"]):
        bonus += 8.0
    return bonus


def _build_lead_sentence(question: str, citations: list[CitationItem], query_tags: set[str]) -> str:
    if "globalization" in query_tags:
        coverage = []
        joined = " ".join(citation.content for citation in citations)
        for label in ["美国", "墨西哥", "摩洛哥", "欧洲市场", "海外市场"]:
            if label in joined:
                coverage.append(label)
        if coverage:
            return f"基于当前命中的资料，关于“{question}”，相关信息主要集中在{'、'.join(coverage[:4])}等海外布局。"
    return f"基于当前命中的资料，关于“{question}”，可以先从最相关的 {min(len(citations), 3)} 条依据来看。"


def _summarize_citation(citation: CitationItem, question: str) -> str:
    text = re.sub(r"\s+", " ", citation.content).strip()
    sentences = re.split(r"(?<=[。！？；])", text)
    question_terms = _extract_tokens(question)
    selected: list[str] = []
    for sentence in sentences:
        compact = sentence.strip()
        if not compact:
            continue
        if any(term in compact for term in question_terms):
            selected.append(compact)
        if len(selected) >= 2:
            break
    if not selected:
        selected = [sentence.strip() for sentence in sentences if sentence.strip()][:2]
    summary = " ".join(selected).strip()
    summary = summary[:160].rstrip("，；、 ")
    if citation.section_title:
        return f"{citation.section_title}：{summary}"
    return summary


def _focus_candidates_by_document(
    question: str,
    candidates: list[tuple[DocumentChunk, Document, int, float]],
) -> tuple[list[tuple[DocumentChunk, Document, int, float]], list[dict]]:
    if not candidates:
        return candidates, []

    grouped: dict[int, list[tuple[DocumentChunk, Document, int, float]]] = {}
    for item in candidates:
        grouped.setdefault(item[1].id, []).append(item)

    if len(grouped) <= 1:
        return candidates, _document_focus_summary(grouped)

    question_terms = set(_extract_tokens(question))
    question_tags = set(derive_semantic_tags(question))
    document_scores: list[tuple[int, float]] = []

    for document_id, items in grouped.items():
        document = items[0][1]
        file_name = document.file_name.lower()
        file_terms = set(_extract_tokens(document.file_name))
        file_tags = set(derive_semantic_tags(document.file_name))
        top_keyword = max(item[2] for item in items)
        top_vector = max(item[3] for item in items)
        top_combined = max(item[2] * 1.5 + item[3] * 10.0 for item in items)
        filename_term_overlap = len(question_terms & file_terms)
        filename_tag_overlap = len(question_tags & file_tags)
        entity_bonus = _entity_bonus(question, file_name)
        score = top_combined + filename_term_overlap * 6.0 + filename_tag_overlap * 8.0 + entity_bonus
        document_scores.append((document_id, score))

    document_scores.sort(key=lambda item: item[1], reverse=True)
    top_score = document_scores[0][1]
    allowed_ids = {
        document_id
        for document_id, score in document_scores
        if score >= top_score * 0.72
    }
    if len(allowed_ids) == 1 and len(document_scores) > 1 and document_scores[1][1] >= top_score * 0.9:
        allowed_ids.add(document_scores[1][0])

    focused = [item for item in candidates if item[1].id in allowed_ids]
    summary = [
        {
            "document_id": document_id,
            "score": round(score, 2),
            "selected": document_id in allowed_ids,
            "file_name": grouped[document_id][0][1].file_name,
        }
        for document_id, score in document_scores
    ]
    return focused, summary


def _document_focus_summary(grouped: dict[int, list[tuple[DocumentChunk, Document, int, float]]]) -> list[dict]:
    summary = []
    for document_id, items in grouped.items():
        summary.append(
            {
                "document_id": document_id,
                "score": round(max(item[2] * 1.5 + item[3] * 10.0 for item in items), 2),
                "selected": True,
                "file_name": items[0][1].file_name,
            }
        )
    return summary


def _entity_bonus(question: str, file_name_lower: str) -> float:
    entity_map = {
        "伯特利": "伯特利",
        "地平线": "地平线",
        "hsd": "地平线",
        "准入": "试点",
        "上路通行": "试点",
        "试点": "试点",
        "智能网联汽车": "试点",
    }
    bonus = 0.0
    lowered_question = question.lower()
    for marker, expected in entity_map.items():
        if marker.lower() in lowered_question and expected.lower() in file_name_lower:
            bonus += 18.0
    return bonus


def _vector_preview(vector: list[float], size: int = 8) -> list[float]:
    return [round(value, 4) for value in vector[:size]]


def _candidate_preview(candidates: list[tuple[DocumentChunk, Document, int, float]], limit: int = 5) -> list[dict]:
    preview = []
    for chunk, document, keyword_score, vector_score in candidates[:limit]:
        preview.append(
            {
                "file_name": document.file_name,
                "chunk_index": chunk.chunk_index,
                "keyword_score": keyword_score,
                "vector_score": round(vector_score, 4),
            }
        )
    return preview


def _rerank_preview(items, limit: int = 4) -> list[dict]:
    preview = []
    for item in items[:limit]:
        preview.append(
            {
                "file_name": item.document.file_name,
                "chunk_index": item.chunk.chunk_index,
                "rerank_score": round(item.rerank_score, 4),
                "keyword_score": item.keyword_score,
                "vector_score": round(item.vector_score, 4),
            }
        )
    return preview


def _processed_hit_preview(items: list[ProcessedHit], limit: int = 4) -> list[dict]:
    preview = []
    for item in items[:limit]:
        preview.append(
            {
                "file_name": item.document.file_name,
                "chunk_span": item.chunk_span,
                "rerank_score": round(item.score, 4),
                "keyword_score": item.keyword_score,
                "vector_score": round(item.vector_score, 4),
            }
        )
    return preview


def _citation_preview(citations: list[CitationItem], limit: int = 4) -> list[dict]:
    return [
        {
            "citation_id": citation.citation_id,
            "file_name": citation.file_name,
            "chunk_index": citation.chunk_index,
            "chunk_span": citation.chunk_span,
        }
        for citation in citations[:limit]
    ]


def _preview_text(text: str | None, limit: int = 220) -> str | None:
    if not text:
        return None
    return text[:limit]


def _compress_citation_content(content: str, question: str, limit: int = 180) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    sentences = [item.strip() for item in re.split(r"(?<=[。！？；])", normalized) if item.strip()]
    question_terms = _extract_tokens(question)
    selected: list[str] = []

    for sentence in sentences:
        if any(term in sentence for term in question_terms):
            selected.append(sentence)
        if len(" ".join(selected)) >= limit:
            break

    if not selected:
        selected = sentences[:2]

    compressed = " ".join(selected).strip()
    return compressed[:limit].rstrip("，；、 ")


def _generation_mode(model_name: str, llm_provider_status: str | None) -> str:
    if llm_provider_status and llm_provider_status.startswith("external_"):
        return "external_llm"
    if model_name == "keyword-retrieval":
        return "fallback_summary"
    return "unknown"
