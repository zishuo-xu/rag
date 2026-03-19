# RAG Knowledge Base

一个可本地运行的 RAG 知识库问答系统，当前已经具备：

- 文档上传、解析、清洗、切分、向量化
- `pgvector` 检索、关键词补充、rerank、去重合并
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

## 当前能力

文档侧：
- 上传 `txt / md / pdf / docx`
- 文档列表、详情、删除、重新处理
- 原始文件保存在 `storage/uploads`

检索侧：
- Query Rewrite
- 向量检索 + 关键词检索
- 本地规则型 rerank
- 相邻 chunk 合并与重复证据去重
- 支持只在选中文档内提问

问答侧：
- 外部 LLM 生成答案
- fallback 检索式摘要
- 引用编号与来源片段联动
- LLM 输入/输出、回退原因持久化

前端侧：
- 首页直接可操作
- 实时进度展示
- 调试视图
- 问答历史与单次回放

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

3. 初始化数据库

```bash
.venv/bin/python scripts/init_db.py
```

4. 启动服务

```bash
.venv/bin/uvicorn app.main:app --reload --port 8011
```

5. 打开页面

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

## 目录

- [app](/Users/xuzishuo/ai-work/rag/app)：应用代码
- [scripts](/Users/xuzishuo/ai-work/rag/scripts)：初始化、评测、重置语料脚本
- [evals](/Users/xuzishuo/ai-work/rag/evals)：评测题集、语料、报告
- [当前项目架构说明.md](/Users/xuzishuo/ai-work/rag/当前项目架构说明.md)：当前真实架构
- [ROADMAP.md](/Users/xuzishuo/ai-work/rag/ROADMAP.md)：后续开发路线

## 下一步

当前最值得继续做的是：

- 简单题 / 复杂题路由，减少不必要的 LLM 调用
- 上下文进一步压缩与答案模板优化
- 外部 LLM 抽样评测自动化
- 更高质量的 rerank 与引用组织
