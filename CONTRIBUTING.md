# Contributing

感谢你对这个仓库感兴趣。

## 开始之前

建议先阅读这些文件：

- [README.md](/Users/xuzishuo/ai-work/rag/README.md)
- [当前项目架构说明.md](/Users/xuzishuo/ai-work/rag/当前项目架构说明.md)
- [ROADMAP.md](/Users/xuzishuo/ai-work/rag/ROADMAP.md)

## 开发建议

1. 保持改动尽量聚焦，不要在一个提交里混入多个无关主题。
2. 新增功能后，同步更新架构文档。
3. 影响问答质量的改动，尽量先跑评测脚本。
4. 如果变更会影响模型、检索或评测结果，请在 PR 描述里写清楚前后差异。

## 推荐本地流程

1. 安装依赖

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. 初始化数据库

```bash
.venv/bin/python scripts/init_db.py
```

3. 启动服务

```bash
.venv/bin/uvicorn app.main:app --reload --port 8011
```

4. 跑评测

```bash
.venv/bin/python scripts/eval_rag.py --disable-llm --output evals/reports/latest.json
```

## 提交建议

- 提交信息尽量清楚描述目的，例如：
  - `Improve retrieval reranking for policy questions`
  - `Add industry evaluation corpus`
  - `Optimize deepseek-chat prompt and context compression`

## Issue 与 PR

- Bug 请尽量附上复现步骤、日志和期望结果。
- RAG 质量问题请尽量附上问题、命中引用和评测结果。
- PR 请说明：
  - 改动内容
  - 为什么这样改
  - 如何验证
  - 是否更新文档
