# RAG Knowledge Base

一个可本地运行的 RAG 知识库问答系统，当前已经具备：

- 文档上传、解析、清洗、切分、向量化
- `pgvector` 检索、PostgreSQL 全文检索、关键词补充、rerank、去重合并
- 外部 LLM 生成答案与引用展示
- 问答历史、处理进度、LLM 输入输出回放
- 单文档、多文档、行业样本评测

## 当前架构

- 应用：`FastAPI + 静态页面`
- 数据库：`PostgreSQL + pgvector`
- 缓存：`Redis`
- Embedding：火山云 embedding
- LLM：当前默认已验证 `deepseek-chat`

详细架构见 [当前项目架构说明.md](/Users/xuzishuo/ai-work/rag/当前项目架构说明.md)。

## 当前默认主线

当前验证效果最好的生成链路是：

- `deepseek-chat`
- 压缩上下文
- 事实化提示词
- 题型定向约束
- 简洁句式收束
- 题型驱动的上下文组织

当前 `11` 题端到端 LLM 样本回归结果见：
- [industry_eval_with_deepseek_chat_sample_concise_format.json](/Users/xuzishuo/ai-work/rag/evals/reports/industry_eval_with_deepseek_chat_sample_concise_format.json)

核心指标：
- `golden_answer_recall = 61.75%`
- `answer_token_f1 = 40.62%`
- `ROUGE-L F1 = 73.98%`
- `average_latency = 3641.7 ms`

可选外部 rerank：
- 阿里百炼 `qwen3-rerank`
- 接入方式为“本地规则型粗排 + 阿里 rerank 精排前 12 个候选 + 失败自动回退”

## 当前能力

文档侧：
- 上传 `txt / md / pdf / docx`
- 文档列表、详情、删除、重新处理
- 原始文件保存在 `storage/uploads`
- 上传与重处理默认走后台异步处理，状态流转为 `QUEUED -> PROCESSING -> SUCCESS / FAILED`
- 文档列表可查看当前处理阶段与进度说明，失败后可直接重新处理
- 文档列表可查看总耗时与各阶段耗时，便于定位大文件慢在解析、切分还是向量化
- 文档任务通过 `Redis` 队列投递，由独立 worker 消费，避免 Web 服务重启导致任务中断
- `document_task` 表已独立记录每次上传/重处理任务，保留任务来源、状态、阶段、耗时和失败原因
- 健康检查会展示 document worker 在线状态，worker 启动时会自动恢复长时间卡住的陈旧任务
- PDF 解析会保留页级标记，并对扫描版 / 无文字层 PDF 给出更明确的错误提示
- `document_chunk` 已结构化沉淀 `page_start / page_end / section_title / semantic_tags_json`，便于后续检索过滤和排序

检索侧：
- Query Rewrite
- 向量检索 + PostgreSQL 全文检索 + 关键词检索
- 本地规则型 rerank
- 可选阿里百炼 `qwen3-rerank` 外部 rerank
- 相邻 chunk 合并与重复证据去重
- 支持只在选中文档内提问
- rerank 已优先读取显式结构化语义标签，而不是只依赖 `metadata_json`

问答侧：
- 外部 LLM 生成答案
- fallback 检索式摘要
- 引用编号与来源片段联动
- LLM 输入/输出、回退原因持久化

前端侧：
- 首页直接可操作
- 首页提供可用于分享的 RAG 过程可视化流程区
- 首页提供“RAG 分享体验区”，可直接输入短资料和问题演示一次迷你 RAG 过程
- 实时进度展示
- 调试视图
- 问答历史与单次回放
- 文档任务历史面板

## 本地运行

1. 安装依赖

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

如需启用阿里百炼 rerank，可补充：

```env
RERANK_PROVIDER=aliyun
RERANK_API_KEY=your_dashscope_key
RERANK_MODEL=qwen3-rerank
RERANK_BASE_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
```

可选的大文件向量化批次配置：

```env
EMBEDDING_BATCH_SIZE=100
PERSIST_BATCH_SIZE=200
```

3. 初始化数据库

```bash
.venv/bin/python scripts/init_db.py
```

4. 启动文档 Worker

```bash
.venv/bin/python scripts/run_document_worker.py
```

5. 启动服务

```bash
.venv/bin/uvicorn app.main:app --reload --port 8011
```

6. 打开页面

[http://127.0.0.1:8011](http://127.0.0.1:8011)

## 核心接口

- `GET /health`
- `GET /health/deps`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{id}`
- `POST /api/documents/{id}/reprocess`
- `DELETE /api/documents/{id}`
- `POST /api/qa/ask`
- `POST /api/qa/demo`
- `GET /api/qa/history`
- `GET /api/qa/history/{request_id}`
- `GET /api/qa/progress/{request_id}`

## 评测

基础回归：

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm --output evals/reports/latest.json
```

导出人工复核表：

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm --review-output evals/reports/review_sheet.csv
```

当前人工复核建议：
- 单模型复核表会新增 `relevance_score / hallucination_score / conciseness_score / citation_support_score`
- 主线与阿里 rerank 的人工对比表见 [industry_manual_compare_main_vs_aliyun.csv](/Users/xuzishuo/ai-work/rag/evals/reports/industry_manual_compare_main_vs_aliyun.csv)

## 行业样本评测

先重置当前知识库并写入行业样本：

```bash
.venv/bin/python scripts/reset_seed_eval_corpus.py --corpus-dir evals/industry_corpus
```

再跑行业评测：

```bash
.venv/bin/python scripts/eval_rag.py --cases evals/industry_cases.jsonl --disable-llm --output evals/reports/industry_eval_latest.json --review-output evals/reports/industry_review_sheet.csv
```

当前行业基线：
- `10` 份样本文档
- `36` 条评测题
- 支持 `Precision@4 / Recall@4 / MRR@4 / nDCG@4`
- 支持 `answer token F1 / ROUGE-L F1`

最新端到端抽样外部 LLM 结果：
- [industry_eval_with_deepseek_chat_tiny_compressed.json](/Users/xuzishuo/ai-work/rag/evals/reports/industry_eval_with_deepseek_chat_tiny_compressed.json)

更贴近真实用户问法的 LLM 样本集：
- [industry_cases_llm_realistic.jsonl](/Users/xuzishuo/ai-work/rag/evals/industry_cases_llm_realistic.jsonl)

运行方式：

```bash
.venv/bin/python scripts/eval_rag.py --cases evals/industry_cases_llm_realistic.jsonl --output evals/reports/industry_eval_with_deepseek_chat_realistic.json --review-output evals/reports/industry_review_with_deepseek_chat_realistic.csv
```

当前这套默认主线在 `19` 题真实问法集上的结果：
- `retrieval_hit_rate = 100%`
- `section_hit_rate = 100%`
- `golden_answer_recall = 52.64%`
- `answer_token_f1 = 35.26%`
- `ROUGE-L F1 = 68.02%`
- `average_latency = 2876.2 ms`

最新 `11` 题生成质量小样本实验表明，“题型驱动的上下文组织”对生成质量有净增益：
- [industry_eval_with_deepseek_chat_realistic_profile_context_sample11.json](/Users/xuzishuo/ai-work/rag/evals/reports/industry_eval_with_deepseek_chat_realistic_profile_context_sample11.json)
- `answer_keyword_recall = 92.42%`
- `golden_answer_recall = 63.57%`
- `answer_token_f1 = 42.06%`
- `ROUGE-L F1 = 71.66%`
- `average_latency = 4575.6 ms`

阿里百炼 rerank 接入后的 `19` 题真实问法回归结果：
- [industry_eval_with_aliyun_rerank_realistic.json](/Users/xuzishuo/ai-work/rag/evals/reports/industry_eval_with_aliyun_rerank_realistic.json)
- `document_purity = 94.74%`
- `doc_precision@4 = 89.47%`
- `golden_answer_recall = 54.90%`
- `answer_token_f1 = 33.70%`
- `ROUGE-L F1 = 68.59%`
- `average_latency = 3117.6 ms`

## 目录

- [app](/Users/xuzishuo/ai-work/rag/app)：应用代码
- [scripts](/Users/xuzishuo/ai-work/rag/scripts)：初始化、评测、重置语料脚本
- [evals](/Users/xuzishuo/ai-work/rag/evals)：评测题集、语料、报告
- [当前项目架构说明.md](/Users/xuzishuo/ai-work/rag/当前项目架构说明.md)：当前真实架构
- [ROADMAP.md](/Users/xuzishuo/ai-work/rag/ROADMAP.md)：后续开发路线

## 仓库协作

- [CONTRIBUTING.md](/Users/xuzishuo/ai-work/rag/CONTRIBUTING.md)
- [LICENSE](/Users/xuzishuo/ai-work/rag/LICENSE)
- `.github/ISSUE_TEMPLATE`
- `.github/PULL_REQUEST_TEMPLATE.md`

## 下一步

当前最值得继续做的是：

- 固化当前最佳主线，持续用更真实问法做回归
- 引用数量 / 顺序自适应
- 更高质量的 rerank 与引用组织
- 混合检索继续优化
- 人工评分与趋势对比
