# dual bio graph rag (PrimeKG + PubMedKG)

This project implements a dual-knowledge-graph GraphRAG system for biomedical Q&A. It ingests PrimeKG (precision medicine KG) and PubMedKG side-by-side, preserves each graph’s native schema via namespaced labels (`PRIME_*`, `PKG_*`), and performs query-time joins using vector similarity and graph expansion. PrimeKG background: [PrimeKG](https://github.com/mims-harvard/PrimeKG?tab=readme-ov-file).

Key idea: vector KNN → graph expansion (1–2 hops) → pruning → PyG (GNN + text) fact fusion → LLM answer with citations.

Reference inspiration (PyG example): [txt2kg_rag.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/llm/txt2kg_rag.py)

---

## What you can do
- Load two KGs (PrimeKG + PKG) into Amazon Neptune via the bulk loader.
- Retrieve a compact subgraph relevant to a question and get a grounded, cited answer.
- Use a PyTorch Geometric fusion step to rank facts before prompting the LLM (mandatory in this repo).

---

- `data/`
  - `local/primekg/exports/`, `local/pubmedkg/exports/`: place your raw CSV/TSV exports here.
  - `graph/prime/`, `graph/pkg/`: generated Neptune openCypher CSVs (`nodes.csv`, `edges.csv`).
- `src/`
  - `ingest/`: converters and loaders
    - `prime_to_neptune.py`, `pkg_to_neptune.py`: raw → `data/graph/*` CSVs (1:1 fields, namespaced labels).
    - `neptune_loader.py`: emits Neptune bulk-load payload JSON (submit via Neptune bulk loader).
  - `embeddings/`: `text_embed.py`, `node_embed.py` to build vector inputs for OpenSearch.
  - `retrieval/`: `vector_store.py` (OpenSearch), `expand.py` (Neo4j/Neptune expansion), `prune.py`, `g_retriever.py`.
  - `rag/pipeline.py`: modular GraphRAG pipeline (encode → seed → expand → prune → PyG fusion → prompt).
  - `qa/`: `answer.py` (end-to-end QA runner), `prompt.py`.
  - `gnn/`: `pyg_rag.py` (PyG utilities for graph+text fusion and fact ranking).
  - `api/`, `ui/`: optional API/UI.
  - `utils/`: configuration, IO, logging, seeding helpers.
- `Makefile`: one-liners for setup, ingest, load, embeddings, QA, API/UI.

---

## Prerequisites
- Python 3.11+
- Amazon Neptune cluster with openCypher enabled and reachable from the application environment
- OpenSearch (Serverless or self-managed)
- LLM endpoint (OpenAI-compatible HTTP API)

---

## End-to-end (Neptune path, PyG mandatory)

1) Install
```bash
make setup
```

2) Stage raw exports
- Put PrimeKG exports under `data/local/primekg/exports/` (filenames in `configs/ingest_prime.yaml`).
- Put PubMedKG exports under `data/local/pubmedkg/exports/` (filenames in `configs/ingest_pkg.yaml`).

3) Convert raw → graph CSVs (preserve fields; add namespaced labels)
```bash
make prime_to_neptune
make pkg_to_neptune
```

4) Generate Neptune loader payloads and submit via bulk loader
```bash
make load_prime
make load_pkg
# then use the Neptune console or CLI to start the bulk load jobs
```

5) Initialize vectors (OpenSearch index + embeddings + upsert)
```bash
make embed
```

6) Ask questions (GraphRAG with PyG fusion)
```bash
make qa
# or run the module directly
python -m src.qa.answer --config configs/default.yaml --question-file configs/demo_questions.yaml
```

7) Optional dev servers
```bash
make api
make ui
```

---

## Notes on Amazon Neptune deployment
- Graph expansion uses Neptune’s openCypher endpoint (`/opencypher`) over HTTPS.
- The application signs requests with IAM SigV4 (`neptune-db` service); ensure the runtime role or profile has permissions such as `neptune-db:connect`, `neptune-db:ExecuteOpenCypherQuery`, and `sts:GetCallerIdentity`.
- Neptune must be reachable from the application host (same VPC/subnet or connected via VPN/peering) with security groups allowing inbound port 8182 from the app.
- Enable CloudWatch logs/metrics for Neptune and consider AWS Budgets or alarms for cost monitoring.

---

## How the pipeline works
1) Encode question → OpenSearch KNN (seeds across both namespaces)
2) Graph expansion in Neptune (1–2 hops, namespace + degree limits)
3) Prune the subgraph to a readable size
4) PyG fusion (mandatory):
   - Build PyG graph from the subgraph
   - Compute graph embedding (GNN)
   - Encode combined text of facts
   - Fuse graph + text vectors and rank facts
   - Keep top facts for the prompt
5) Prompt the LLM and return a concise answer with PMIDs/experiment IDs

All of this is implemented in `src/rag/pipeline.py` and called by `src/qa/answer.py`.

---

## Configuration knobs (common)
- Retrieval in `configs/*yaml` under `retrieval.*`:
  - `top_k`: vector KNN hits
  - `expansion_hops`: 1–2 hops
  - `prune_max_nodes`: subgraph size after pruning
  - `prune_max_degree`: degree cap during expansion
  - `namespaces.include`: e.g., `[PRIME_, PKG_]`
- PyG fusion under `pyg_rag.*`:
  - `top_facts`: number of facts kept for the prompt (e.g., 40)
- OpenSearch under `open_search.*`: endpoint/index/dimension
- LLM under `llm.*`: base URL, key, model, temperature

---

## Troubleshooting
- Neptune connectivity: confirm VPC routing and that the app security group can reach port 8182 on the Neptune cluster.
- IAM permissions: verify the caller identity resolves (`aws sts get-caller-identity`) and that the Neptune actions mentioned above are attached.
- OpenSearch index: run `python -m src.retrieval.vector_store --config configs/default.yaml --create` once if needed.
- Empty subgraph: increase `top_k`, `expansion_hops`, or relax `prune_max_degree`.
- PyG errors: ensure `gnn/pyg_rag.py` dependencies are installed; check CUDA vs CPU settings.

---

## License and references
- GraphRAG idea inspired by NVIDIA and PyG’s example: [txt2kg_rag.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/llm/txt2kg_rag.py)
- PrimeKG: `https://github.com/mims-harvard/PrimeKG?tab=readme-ov-file`
- PubMedKG: `https://pubmedkg.github.io/`
