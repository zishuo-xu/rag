import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.db.models.document import Document
from app.services.qa_service import QAService


@dataclass
class EvalCase:
    case_id: str
    category: str
    difficulty: str
    question: str
    document_ids: list[int] | None
    document_names: list[str] | None
    expected_document_ids: list[int]
    expected_document_names: list[str]
    expected_absent_document_ids: list[int]
    expected_absent_document_names: list[str]
    max_unexpected_citations: int | None
    expected_keywords: list[str]
    must_hit_section_titles: list[str]
    golden_answer: str | None
    min_citations: int
    manual_score: int | None
    manual_notes: str | None
    notes: str | None


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                case_id=payload["case_id"],
                category=payload.get("category", "default"),
                difficulty=payload.get("difficulty", "medium"),
                question=payload["question"],
                document_ids=payload.get("document_ids"),
                document_names=payload.get("document_names"),
                expected_document_ids=payload.get("expected_document_ids", []),
                expected_document_names=payload.get("expected_document_names", []),
                expected_absent_document_ids=payload.get("expected_absent_document_ids", []),
                expected_absent_document_names=payload.get("expected_absent_document_names", []),
                max_unexpected_citations=payload.get("max_unexpected_citations"),
                expected_keywords=payload.get("expected_keywords", []),
                must_hit_section_titles=payload.get("must_hit_section_titles", []),
                golden_answer=payload.get("golden_answer"),
                min_citations=int(payload.get("min_citations", 1)),
                manual_score=payload.get("manual_score"),
                manual_notes=payload.get("manual_notes"),
                notes=payload.get("notes"),
            )
        )
    return cases


def keyword_recall(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    matched = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
    return matched / len(expected_keywords)


def document_hit(citation_document_ids: list[int], expected_document_ids: list[int]) -> bool | None:
    if not expected_document_ids:
        return None
    return any(document_id in expected_document_ids for document_id in citation_document_ids)


def unexpected_citation_count(citation_document_ids: list[int], expected_document_ids: list[int], expected_absent_document_ids: list[int]) -> int | None:
    if not expected_document_ids and not expected_absent_document_ids:
        return None
    expected_set = set(expected_document_ids)
    absent_set = set(expected_absent_document_ids)
    count = 0
    for document_id in citation_document_ids:
        if absent_set and document_id in absent_set:
            count += 1
            continue
        if expected_set and document_id not in expected_set:
            count += 1
    return count


def document_purity(citation_document_ids: list[int], expected_document_ids: list[int], expected_absent_document_ids: list[int]) -> float | None:
    if not citation_document_ids:
        return None
    unexpected = unexpected_citation_count(citation_document_ids, expected_document_ids, expected_absent_document_ids)
    if unexpected is None:
        return None
    return max(0.0, 1.0 - unexpected / len(citation_document_ids))


def section_hit(citation_sections: list[str], expected_sections: list[str]) -> bool | None:
    if not expected_sections:
        return None
    normalized_sections = [_normalize_section_title(section) for section in citation_sections if section]
    for expected in expected_sections:
        expected_lower = _normalize_section_title(expected)
        if any(expected_lower in section for section in normalized_sections):
            return True
    return False


def _normalize_section_title(section: str) -> str:
    return re.sub(r"\s+", " ", section).strip().lower()


def unique_preserving_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def binary_precision_at_k(ranked_items: list[str], relevant_items: list[str], k: int) -> float | None:
    if not relevant_items or k <= 0:
        return None
    top_k = ranked_items[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_items)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(top_k)


def binary_recall_at_k(ranked_items: list[str], relevant_items: list[str], k: int) -> float | None:
    if not relevant_items or k <= 0:
        return None
    top_k = ranked_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def binary_mrr_at_k(ranked_items: list[str], relevant_items: list[str], k: int) -> float | None:
    if not relevant_items or k <= 0:
        return None
    relevant_set = set(relevant_items)
    for index, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def binary_ndcg_at_k(ranked_items: list[str], relevant_items: list[str], k: int) -> float | None:
    if not relevant_items or k <= 0:
        return None
    relevant_set = set(relevant_items)
    dcg = 0.0
    for index, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / _log2(index + 1)
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return None
    idcg = sum(1.0 / _log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def answer_token_scores(answer: str, golden_answer: str | None) -> tuple[float | None, float | None, float | None]:
    if not golden_answer:
        return None, None, None
    answer_tokens = extract_eval_terms(answer)
    golden_tokens = extract_eval_terms(golden_answer)
    if not answer_tokens or not golden_tokens:
        return None, None, None
    answer_set = set(answer_tokens)
    golden_set = set(golden_tokens)
    overlap = len(answer_set & golden_set)
    precision = overlap / len(answer_set) if answer_set else None
    recall = overlap / len(golden_set) if golden_set else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if precision == 0.0 or recall == 0.0 else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_l_scores(answer: str, golden_answer: str | None) -> tuple[float | None, float | None, float | None]:
    if not golden_answer:
        return None, None, None
    answer_chars = [char for char in re.sub(r"\s+", "", answer) if char.strip()]
    golden_chars = [char for char in re.sub(r"\s+", "", golden_answer) if char.strip()]
    if not answer_chars or not golden_chars:
        return None, None, None
    lcs = _lcs_length(answer_chars, golden_chars)
    precision = lcs / len(answer_chars)
    recall = lcs / len(golden_chars)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for index, right_item in enumerate(right, start=1):
            cached = dp[index]
            if left_item == right_item:
                dp[index] = previous + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous = cached
    return dp[-1]


def _log2(value: int) -> float:
    return 0.0 if value <= 0 else math.log2(value)


def golden_answer_recall(answer: str, golden_answer: str | None) -> float | None:
    if not golden_answer:
        return None
    golden_tokens = extract_eval_terms(golden_answer)
    if not golden_tokens:
        return None
    answer_lower = answer.lower()
    matched = sum(1 for token in golden_tokens if token in answer_lower)
    return matched / len(golden_tokens)


def extract_eval_terms(text: str) -> list[str]:
    ascii_terms = [token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 2]
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    merged = ascii_terms + cjk_terms
    return list(dict.fromkeys(term.strip().lower() for term in merged if term.strip()))


def manual_score_bucket(score: int | None) -> str | None:
    if score is None:
        return None
    if score <= 0:
        return "bad"
    if score == 1:
        return "partial"
    return "good"


def resolve_document_ids_by_names(db, names: list[str] | None) -> list[int]:
    if not names:
        return []
    rows = (
        db.query(Document.id, Document.file_name, Document.updated_time, Document.created_time)
        .filter(Document.file_name.in_(names))
        .filter(Document.status == "SUCCESS")
        .order_by(Document.file_name.asc(), Document.updated_time.desc(), Document.created_time.desc(), Document.id.desc())
        .all()
    )
    latest_by_name: dict[str, int] = {}
    for document_id, file_name, _updated_time, _created_time in rows:
        if file_name not in latest_by_name:
            latest_by_name[file_name] = int(document_id)
    return sorted(latest_by_name.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight RAG evaluation suite.")
    parser.add_argument(
        "--cases",
        default="evals/qa_cases.jsonl",
        help="Path to JSONL evaluation cases.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable external LLM generation and only evaluate retrieval + fallback summary.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write a JSON report.",
    )
    parser.add_argument(
        "--review-output",
        default="",
        help="Optional path to write a CSV review sheet for manual scoring.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    if not cases:
        raise SystemExit("No evaluation cases found.")

    db = SessionLocal()
    try:
        service = QAService(db)
        if args.disable_llm:
            service.llm_provider.enabled = False
            service.llm_provider.client = None

        rows: list[dict] = []
        for case in cases:
            case_document_ids = sorted(set((case.document_ids or []) + resolve_document_ids_by_names(db, case.document_names)))
            expected_document_ids = sorted(set((case.expected_document_ids or []) + resolve_document_ids_by_names(db, case.expected_document_names)))
            expected_absent_document_ids = sorted(
                set((case.expected_absent_document_ids or []) + resolve_document_ids_by_names(db, case.expected_absent_document_names))
            )
            response = service.ask_with_options(
                case.question,
                document_ids=case_document_ids or None,
                request_id=f"eval_{case.case_id}_{uuid4().hex[:8]}",
                persist_record=False,
                track_progress=False,
            )
            citation_document_ids = [citation.document_id for citation in response.citations]
            citation_sections = [citation.section_title or "" for citation in response.citations]
            ranked_document_ids = unique_preserving_order([str(document_id) for document_id in citation_document_ids])
            relevant_document_ids = [str(document_id) for document_id in expected_document_ids]
            ranked_sections = unique_preserving_order([_normalize_section_title(section) for section in citation_sections if section])
            relevant_sections = [_normalize_section_title(section) for section in case.must_hit_section_titles]
            answer_keyword_score = keyword_recall(response.answer, case.expected_keywords)
            retrieval_hit = document_hit(citation_document_ids, expected_document_ids)
            unexpected_count = unexpected_citation_count(
                citation_document_ids,
                expected_document_ids,
                expected_absent_document_ids,
            )
            purity = document_purity(
                citation_document_ids,
                expected_document_ids,
                expected_absent_document_ids,
            )
            purity_ok = (
                True
                if case.max_unexpected_citations is None or unexpected_count is None
                else unexpected_count <= case.max_unexpected_citations
            )
            section_match = section_hit(citation_sections, case.must_hit_section_titles)
            golden_recall = golden_answer_recall(response.answer, case.golden_answer)
            doc_precision_at_4 = binary_precision_at_k(ranked_document_ids, relevant_document_ids, 4)
            doc_recall_at_4 = binary_recall_at_k(ranked_document_ids, relevant_document_ids, 4)
            doc_mrr_at_4 = binary_mrr_at_k(ranked_document_ids, relevant_document_ids, 4)
            doc_ndcg_at_4 = binary_ndcg_at_k(ranked_document_ids, relevant_document_ids, 4)
            section_precision_at_4 = binary_precision_at_k(ranked_sections, relevant_sections, 4)
            section_recall_at_4 = binary_recall_at_k(ranked_sections, relevant_sections, 4)
            section_mrr_at_4 = binary_mrr_at_k(ranked_sections, relevant_sections, 4)
            section_ndcg_at_4 = binary_ndcg_at_k(ranked_sections, relevant_sections, 4)
            answer_token_precision, answer_token_recall, answer_token_f1 = answer_token_scores(response.answer, case.golden_answer)
            rouge_l_precision, rouge_l_recall, rouge_l_f1 = rouge_l_scores(response.answer, case.golden_answer)
            citation_ok = len(response.citations) >= case.min_citations
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "question": case.question,
                    "document_ids": case_document_ids,
                    "document_names": case.document_names,
                    "status": response.model_name,
                    "success": response.answer.strip() != "",
                    "elapsed_time_ms": response.elapsed_time_ms,
                    "citation_count": len(response.citations),
                    "citation_ok": citation_ok,
                    "retrieval_hit": retrieval_hit,
                    "unexpected_citation_count": unexpected_count,
                    "document_purity": purity,
                    "document_purity_ok": purity_ok,
                    "section_hit": section_match,
                    "answer_keyword_recall": answer_keyword_score,
                    "golden_answer_recall": golden_recall,
                    "doc_precision_at_4": doc_precision_at_4,
                    "doc_recall_at_4": doc_recall_at_4,
                    "doc_mrr_at_4": doc_mrr_at_4,
                    "doc_ndcg_at_4": doc_ndcg_at_4,
                    "section_precision_at_4": section_precision_at_4,
                    "section_recall_at_4": section_recall_at_4,
                    "section_mrr_at_4": section_mrr_at_4,
                    "section_ndcg_at_4": section_ndcg_at_4,
                    "answer_token_precision": answer_token_precision,
                    "answer_token_recall": answer_token_recall,
                    "answer_token_f1": answer_token_f1,
                    "rouge_l_precision": rouge_l_precision,
                    "rouge_l_recall": rouge_l_recall,
                    "rouge_l_f1": rouge_l_f1,
                    "llm_provider_status": response.llm_provider_status,
                    "answer_preview": response.answer[:240],
                    "citation_sections": citation_sections,
                    "citation_file_names": [citation.file_name for citation in response.citations],
                    "expected_document_ids": expected_document_ids,
                    "expected_document_names": case.expected_document_names,
                    "must_hit_section_titles": case.must_hit_section_titles,
                    "golden_answer": case.golden_answer,
                    "manual_score": case.manual_score,
                    "manual_score_bucket": manual_score_bucket(case.manual_score),
                    "manual_notes": case.manual_notes,
                    "notes": case.notes,
                }
            )

        retrieval_scores = [row["retrieval_hit"] for row in rows if row["retrieval_hit"] is not None]
        purity_scores = [row["document_purity"] for row in rows if row["document_purity"] is not None]
        purity_ok_scores = [row["document_purity_ok"] for row in rows if row["document_purity_ok"] is not None]
        section_scores = [row["section_hit"] for row in rows if row["section_hit"] is not None]
        keyword_scores = [row["answer_keyword_recall"] for row in rows if row["answer_keyword_recall"] is not None]
        golden_scores = [row["golden_answer_recall"] for row in rows if row["golden_answer_recall"] is not None]
        doc_precision_scores = [row["doc_precision_at_4"] for row in rows if row["doc_precision_at_4"] is not None]
        doc_recall_scores = [row["doc_recall_at_4"] for row in rows if row["doc_recall_at_4"] is not None]
        doc_mrr_scores = [row["doc_mrr_at_4"] for row in rows if row["doc_mrr_at_4"] is not None]
        doc_ndcg_scores = [row["doc_ndcg_at_4"] for row in rows if row["doc_ndcg_at_4"] is not None]
        section_precision_scores = [row["section_precision_at_4"] for row in rows if row["section_precision_at_4"] is not None]
        section_recall_scores = [row["section_recall_at_4"] for row in rows if row["section_recall_at_4"] is not None]
        section_mrr_scores = [row["section_mrr_at_4"] for row in rows if row["section_mrr_at_4"] is not None]
        section_ndcg_scores = [row["section_ndcg_at_4"] for row in rows if row["section_ndcg_at_4"] is not None]
        token_precision_scores = [row["answer_token_precision"] for row in rows if row["answer_token_precision"] is not None]
        token_recall_scores = [row["answer_token_recall"] for row in rows if row["answer_token_recall"] is not None]
        token_f1_scores = [row["answer_token_f1"] for row in rows if row["answer_token_f1"] is not None]
        rouge_precision_scores = [row["rouge_l_precision"] for row in rows if row["rouge_l_precision"] is not None]
        rouge_recall_scores = [row["rouge_l_recall"] for row in rows if row["rouge_l_recall"] is not None]
        rouge_f1_scores = [row["rouge_l_f1"] for row in rows if row["rouge_l_f1"] is not None]
        citation_scores = [row["citation_ok"] for row in rows]
        success_scores = [row["success"] for row in rows]
        manual_scores = [row["manual_score"] for row in rows if row["manual_score"] is not None]
        avg_latency = mean(row["elapsed_time_ms"] for row in rows)

        summary = {
            "case_count": len(rows),
            "success_rate": sum(success_scores) / len(success_scores),
            "retrieval_hit_rate": (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None,
            "document_purity": mean(purity_scores) if purity_scores else None,
            "document_purity_pass_rate": (sum(purity_ok_scores) / len(purity_ok_scores)) if purity_ok_scores else None,
            "section_hit_rate": (sum(section_scores) / len(section_scores)) if section_scores else None,
            "citation_coverage": sum(citation_scores) / len(citation_scores),
            "answer_keyword_recall": mean(keyword_scores) if keyword_scores else None,
            "golden_answer_recall": mean(golden_scores) if golden_scores else None,
            "doc_precision_at_4": mean(doc_precision_scores) if doc_precision_scores else None,
            "doc_recall_at_4": mean(doc_recall_scores) if doc_recall_scores else None,
            "doc_mrr_at_4": mean(doc_mrr_scores) if doc_mrr_scores else None,
            "doc_ndcg_at_4": mean(doc_ndcg_scores) if doc_ndcg_scores else None,
            "section_precision_at_4": mean(section_precision_scores) if section_precision_scores else None,
            "section_recall_at_4": mean(section_recall_scores) if section_recall_scores else None,
            "section_mrr_at_4": mean(section_mrr_scores) if section_mrr_scores else None,
            "section_ndcg_at_4": mean(section_ndcg_scores) if section_ndcg_scores else None,
            "answer_token_precision": mean(token_precision_scores) if token_precision_scores else None,
            "answer_token_recall": mean(token_recall_scores) if token_recall_scores else None,
            "answer_token_f1": mean(token_f1_scores) if token_f1_scores else None,
            "rouge_l_precision": mean(rouge_precision_scores) if rouge_precision_scores else None,
            "rouge_l_recall": mean(rouge_recall_scores) if rouge_recall_scores else None,
            "rouge_l_f1": mean(rouge_f1_scores) if rouge_f1_scores else None,
            "manual_average_score": mean(manual_scores) if manual_scores else None,
            "average_latency_ms": avg_latency,
        }

        category_summary: dict[str, dict] = {}
        for category in sorted({row["category"] for row in rows}):
            category_rows = [row for row in rows if row["category"] == category]
            category_retrieval = [row["retrieval_hit"] for row in category_rows if row["retrieval_hit"] is not None]
            category_purity = [row["document_purity"] for row in category_rows if row["document_purity"] is not None]
            category_purity_ok = [row["document_purity_ok"] for row in category_rows if row["document_purity_ok"] is not None]
            category_sections = [row["section_hit"] for row in category_rows if row["section_hit"] is not None]
            category_keywords = [row["answer_keyword_recall"] for row in category_rows if row["answer_keyword_recall"] is not None]
            category_golden = [row["golden_answer_recall"] for row in category_rows if row["golden_answer_recall"] is not None]
            category_doc_precision = [row["doc_precision_at_4"] for row in category_rows if row["doc_precision_at_4"] is not None]
            category_doc_recall = [row["doc_recall_at_4"] for row in category_rows if row["doc_recall_at_4"] is not None]
            category_doc_mrr = [row["doc_mrr_at_4"] for row in category_rows if row["doc_mrr_at_4"] is not None]
            category_doc_ndcg = [row["doc_ndcg_at_4"] for row in category_rows if row["doc_ndcg_at_4"] is not None]
            category_section_precision = [row["section_precision_at_4"] for row in category_rows if row["section_precision_at_4"] is not None]
            category_section_recall = [row["section_recall_at_4"] for row in category_rows if row["section_recall_at_4"] is not None]
            category_section_mrr = [row["section_mrr_at_4"] for row in category_rows if row["section_mrr_at_4"] is not None]
            category_section_ndcg = [row["section_ndcg_at_4"] for row in category_rows if row["section_ndcg_at_4"] is not None]
            category_token_f1 = [row["answer_token_f1"] for row in category_rows if row["answer_token_f1"] is not None]
            category_rouge_l_f1 = [row["rouge_l_f1"] for row in category_rows if row["rouge_l_f1"] is not None]
            category_manual = [row["manual_score"] for row in category_rows if row["manual_score"] is not None]
            category_summary[category] = {
                "case_count": len(category_rows),
                "success_rate": sum(row["success"] for row in category_rows) / len(category_rows),
                "retrieval_hit_rate": (sum(category_retrieval) / len(category_retrieval)) if category_retrieval else None,
                "document_purity": mean(category_purity) if category_purity else None,
                "document_purity_pass_rate": (sum(category_purity_ok) / len(category_purity_ok)) if category_purity_ok else None,
                "section_hit_rate": (sum(category_sections) / len(category_sections)) if category_sections else None,
                "citation_coverage": sum(row["citation_ok"] for row in category_rows) / len(category_rows),
                "answer_keyword_recall": mean(category_keywords) if category_keywords else None,
                "golden_answer_recall": mean(category_golden) if category_golden else None,
                "doc_precision_at_4": mean(category_doc_precision) if category_doc_precision else None,
                "doc_recall_at_4": mean(category_doc_recall) if category_doc_recall else None,
                "doc_mrr_at_4": mean(category_doc_mrr) if category_doc_mrr else None,
                "doc_ndcg_at_4": mean(category_doc_ndcg) if category_doc_ndcg else None,
                "section_precision_at_4": mean(category_section_precision) if category_section_precision else None,
                "section_recall_at_4": mean(category_section_recall) if category_section_recall else None,
                "section_mrr_at_4": mean(category_section_mrr) if category_section_mrr else None,
                "section_ndcg_at_4": mean(category_section_ndcg) if category_section_ndcg else None,
                "answer_token_f1": mean(category_token_f1) if category_token_f1 else None,
                "rouge_l_f1": mean(category_rouge_l_f1) if category_rouge_l_f1 else None,
                "manual_average_score": mean(category_manual) if category_manual else None,
                "average_latency_ms": mean(row["elapsed_time_ms"] for row in category_rows),
            }

        print(f"Cases: {len(rows)}")
        print(f"Success rate: {sum(success_scores)}/{len(success_scores)} = {summary['success_rate']:.2%}")
        if retrieval_scores:
            print(f"Retrieval hit rate: {sum(retrieval_scores)}/{len(retrieval_scores)} = {summary['retrieval_hit_rate']:.2%}")
        if purity_scores:
            print(f"Document purity: {summary['document_purity']:.2%}")
        if purity_ok_scores:
            print(f"Document purity pass rate: {sum(purity_ok_scores)}/{len(purity_ok_scores)} = {summary['document_purity_pass_rate']:.2%}")
        if section_scores:
            print(f"Section hit rate: {sum(section_scores)}/{len(section_scores)} = {summary['section_hit_rate']:.2%}")
        if doc_precision_scores:
            print(f"Doc Precision@4: {summary['doc_precision_at_4']:.2%}")
        if doc_recall_scores:
            print(f"Doc Recall@4: {summary['doc_recall_at_4']:.2%}")
        if doc_mrr_scores:
            print(f"Doc MRR@4: {summary['doc_mrr_at_4']:.4f}")
        if doc_ndcg_scores:
            print(f"Doc nDCG@4: {summary['doc_ndcg_at_4']:.4f}")
        if section_precision_scores:
            print(f"Section Precision@4: {summary['section_precision_at_4']:.2%}")
        if section_recall_scores:
            print(f"Section Recall@4: {summary['section_recall_at_4']:.2%}")
        if section_mrr_scores:
            print(f"Section MRR@4: {summary['section_mrr_at_4']:.4f}")
        if section_ndcg_scores:
            print(f"Section nDCG@4: {summary['section_ndcg_at_4']:.4f}")
        if citation_scores:
            print(f"Citation coverage: {sum(citation_scores)}/{len(citation_scores)} = {summary['citation_coverage']:.2%}")
        if keyword_scores:
            print(f"Answer keyword recall: {summary['answer_keyword_recall']:.2%}")
        if golden_scores:
            print(f"Golden answer recall: {summary['golden_answer_recall']:.2%}")
        if token_precision_scores:
            print(f"Answer token precision: {summary['answer_token_precision']:.2%}")
        if token_recall_scores:
            print(f"Answer token recall: {summary['answer_token_recall']:.2%}")
        if token_f1_scores:
            print(f"Answer token F1: {summary['answer_token_f1']:.2%}")
        if rouge_f1_scores:
            print(f"ROUGE-L F1: {summary['rouge_l_f1']:.2%}")
        if manual_scores:
            print(f"Manual average score: {summary['manual_average_score']:.2f}")
        print(f"Average latency: {summary['average_latency_ms']:.1f} ms")
        print("")
        print("By category:")
        for category, metrics in category_summary.items():
            keyword_part = "-" if metrics["answer_keyword_recall"] is None else f"{metrics['answer_keyword_recall']:.2%}"
            golden_part = "-" if metrics["golden_answer_recall"] is None else f"{metrics['golden_answer_recall']:.2%}"
            doc_recall_part = "-" if metrics["doc_recall_at_4"] is None else f"{metrics['doc_recall_at_4']:.2%}"
            doc_mrr_part = "-" if metrics["doc_mrr_at_4"] is None else f"{metrics['doc_mrr_at_4']:.3f}"
            section_recall_part = "-" if metrics["section_recall_at_4"] is None else f"{metrics['section_recall_at_4']:.2%}"
            section_mrr_part = "-" if metrics["section_mrr_at_4"] is None else f"{metrics['section_mrr_at_4']:.3f}"
            token_f1_part = "-" if metrics["answer_token_f1"] is None else f"{metrics['answer_token_f1']:.2%}"
            rouge_f1_part = "-" if metrics["rouge_l_f1"] is None else f"{metrics['rouge_l_f1']:.2%}"
            retrieval_part = "-" if metrics["retrieval_hit_rate"] is None else f"{metrics['retrieval_hit_rate']:.2%}"
            purity_part = "-" if metrics["document_purity"] is None else f"{metrics['document_purity']:.2%}"
            purity_pass_part = "-" if metrics["document_purity_pass_rate"] is None else f"{metrics['document_purity_pass_rate']:.2%}"
            section_part = "-" if metrics["section_hit_rate"] is None else f"{metrics['section_hit_rate']:.2%}"
            manual_part = "-" if metrics["manual_average_score"] is None else f"{metrics['manual_average_score']:.2f}"
            print(
                f"- {category}: cases={metrics['case_count']}, success={metrics['success_rate']:.2%}, "
                f"retrieval={retrieval_part}, purity={purity_part}, purity_pass={purity_pass_part}, "
                f"section_hit={section_part}, doc_recall@4={doc_recall_part}, doc_mrr@4={doc_mrr_part}, "
                f"section_recall@4={section_recall_part}, section_mrr@4={section_mrr_part}, "
                f"citations={metrics['citation_coverage']:.2%}, keyword_recall={keyword_part}, "
                f"golden_recall={golden_part}, token_f1={token_f1_part}, rouge_l_f1={rouge_f1_part}, manual_score={manual_part}, "
                f"latency={metrics['average_latency_ms']:.1f}ms"
            )
        print("")
        print("Per case:")
        for row in rows:
            keyword_part = "-" if row["answer_keyword_recall"] is None else f"{row['answer_keyword_recall']:.2%}"
            golden_part = "-" if row["golden_answer_recall"] is None else f"{row['golden_answer_recall']:.2%}"
            doc_recall_part = "-" if row["doc_recall_at_4"] is None else f"{row['doc_recall_at_4']:.2%}"
            doc_mrr_part = "-" if row["doc_mrr_at_4"] is None else f"{row['doc_mrr_at_4']:.3f}"
            section_recall_part = "-" if row["section_recall_at_4"] is None else f"{row['section_recall_at_4']:.2%}"
            section_mrr_part = "-" if row["section_mrr_at_4"] is None else f"{row['section_mrr_at_4']:.3f}"
            token_f1_part = "-" if row["answer_token_f1"] is None else f"{row['answer_token_f1']:.2%}"
            rouge_f1_part = "-" if row["rouge_l_f1"] is None else f"{row['rouge_l_f1']:.2%}"
            retrieval_part = "-" if row["retrieval_hit"] is None else str(row["retrieval_hit"])
            purity_part = "-" if row["document_purity"] is None else f"{row['document_purity']:.2%}"
            purity_ok_part = "-" if row["document_purity_ok"] is None else str(row["document_purity_ok"])
            section_part = "-" if row["section_hit"] is None else str(row["section_hit"])
            manual_part = "-" if row["manual_score"] is None else str(row["manual_score"])
            print(
                f"- {row['case_id']} [{row['category']}/{row['difficulty']}]: latency={row['elapsed_time_ms']}ms, "
                f"citations={row['citation_count']}, citation_ok={row['citation_ok']}, "
                f"retrieval_hit={retrieval_part}, purity={purity_part}, purity_ok={purity_ok_part}, section_hit={section_part}, "
                f"doc_recall@4={doc_recall_part}, doc_mrr@4={doc_mrr_part}, "
                f"section_recall@4={section_recall_part}, section_mrr@4={section_mrr_part}, "
                f"answer_keyword_recall={keyword_part}, golden_recall={golden_part}, token_f1={token_f1_part}, rouge_l_f1={rouge_f1_part}, "
                f"manual_score={manual_part}, provider={row['llm_provider_status'] or row['status']}"
            )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "summary": summary,
                        "by_category": category_summary,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("")
            print(f"Report written to: {output_path}")

        if args.review_output:
            review_path = Path(args.review_output)
            review_path.parent.mkdir(parents=True, exist_ok=True)
            with review_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "case_id",
                        "category",
                        "difficulty",
                        "question",
                        "golden_answer",
                        "must_hit_section_titles",
                        "answer_preview",
                        "citation_sections",
                        "manual_score",
                        "manual_notes",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "case_id": row["case_id"],
                            "category": row["category"],
                            "difficulty": row["difficulty"],
                            "question": row["question"],
                            "golden_answer": row["golden_answer"] or "",
                            "must_hit_section_titles": " | ".join(row["must_hit_section_titles"] or []),
                            "answer_preview": row["answer_preview"],
                            "citation_sections": " | ".join(row["citation_sections"] or []),
                            "manual_score": row["manual_score"] if row["manual_score"] is not None else "",
                            "manual_notes": row["manual_notes"] or "",
                        }
                    )
            print(f"Review sheet written to: {review_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
