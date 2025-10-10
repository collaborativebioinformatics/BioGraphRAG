# v0.1.0 — Initial public release

Highlights
- Graph-native RAG for biomedical QA: question → vector seeds → Neptune expansion → PCST-like pruning → GNN+LLM answering with citations.
- Vector store options: OpenSearch (managed/serverless) or local FAISS backend for dev.
- Config via YAML with environment variable overrides (BIO_KG_*).
- Minimal tests for Neptune expansion; reproducible Make targets.

Notable changes
- Add FAISS backend to `src/retrieval/vector_store.py` (create/upsert/query).
- Add CI (ruff + pytest), pre-commit hooks, packaging skeleton (`pyproject.toml`).
- Documentation polish with quickstart and references (GraphRAG blog, G‑Retriever, STaRK, Scientific Data 2023).

Getting started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make embed
make qa
```

Local dev without AWS
```bash
export BIO_KG_vector_store__backend=faiss
make embed
make qa
```
