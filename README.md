# RAG Knowledge Base

## Local run

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill model keys if needed.
3. Initialize the database:

```bash
.venv/bin/python scripts/init_db.py
```

4. Start the app:

```bash
uvicorn app.main:app --reload
```

## Available endpoints

- `GET /health`
- `GET /health/deps`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{id}`
- `POST /api/documents/{id}/reprocess`
- `DELETE /api/documents/{id}`
- `POST /api/qa/ask`
- `GET /api/qa/history`

## Current capabilities

- Document upload with local file persistence
- Synchronous parsing for `txt`, `md`, `pdf`, `docx`
- Basic text cleaning and chunk splitting
- Chunk embedding generation with external provider or local fallback
- Hybrid retrieval with vector similarity and keyword matching
- Basic keyword retrieval QA over chunks
- Optional OpenAI Responses API answer generation when `LLM_API_KEY` is configured
- Document list, detail, and delete
- Simple browser UI at `/`

## Evaluation

Run the lightweight regression suite:

```bash
.venv/bin/python scripts/eval_rag.py
```

Run retrieval-only evaluation without external LLM:

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm
```

Write a JSON report:

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm --output evals/reports/latest.json
```

Write a manual review sheet:

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm --review-output evals/reports/review_sheet.csv
```

Default cases live in `evals/qa_cases.jsonl`.
Each case can carry `category`, `difficulty`, `must_hit_section_titles`, `golden_answer`, `manual_score`, and `manual_notes` fields.
The script prints category-level metrics and can export a CSV sheet for human review.

## Industry-style evaluation

Reset the current knowledge-base documents and seed the curated industry corpus:

```bash
.venv/bin/python scripts/reset_seed_eval_corpus.py --corpus-dir evals/industry_corpus
```

Run the larger industry-style benchmark over the full corpus:

```bash
.venv/bin/python scripts/eval_rag.py --cases evals/industry_cases.jsonl --disable-llm --output evals/reports/industry_eval_latest.json --review-output evals/reports/industry_review_sheet.csv
```

This benchmark currently uses:

- `10` seeded sample documents in `evals/industry_corpus`
- `36` evaluation cases in `evals/industry_cases.jsonl`
- retrieval metrics such as `Precision@4`, `Recall@4`, `MRR@4`, `nDCG@4`
- answer metrics such as `answer token F1` and `ROUGE-L F1`

The latest report is written to `evals/reports/industry_eval_latest.json`.
