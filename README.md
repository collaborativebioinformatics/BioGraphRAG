<img width="474" height="376" alt="Screenshot 2025-10-03 at 12 44 02 AM" src="https://github.com/user-attachments/assets/ddc9f200-05bd-41a2-87bf-04030ddda915" />

## Overview

Natural language queries using GenAI systems are known to generate hallucinations of facts and may not be able to provide evidence for explanations. For use in biomedical research, GenAI must provide valid and well-grounded results. Therefore, a biomedical system that generates results based on factual biomedical knowledge and publications are needed.

BioGraphRAG bridges biomedical data and publication knowledge graphs with GraphRAG, based upon the G-retriever architecture, to enhance and ensure that the natural language responses are generated from only information from the knowledge sources.

We tackle biomedical question answering by pairing graph-native retrieval with neural reasoning. For every question, we retrieve a targeted slice of our knowledge graphs and pass that evidence to a graph-aware neural reader.

Given a question $Q_i$, we seek an answer set $A_i \subseteq V$ from a large graph $G=(V,E)$. Our pipeline formalizes subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) optimization, then conditions a GNN+LLM reader on the retrieved subgraph. This design reduces hallucinations by grounding in the graph, while scaling beyond a single-context window for dense biomedical domains via structured retrieval, aligning with the GraphRAG patterns described in the NVIDIA technical blog and the G-Retriever paper.

**Goal:** Implement a system to use GNNs + LLM to integrate knowledge graphs made from large public datasets with literature to turn a free-text biomedical question into a grounded answer with biomedical information and PMIDs / experiment IDs.

## ✨ TL;DR

- **Question → Embedding → Retrieval**: Encode the user’s question and fetch top-k seeds from Clinical + Public KGs.
- **Graph Expansion**: Expand seeds in Neptune with openCypher (1–2 hops, label filters, degree caps).
- **Pruning**: Apply a PCST-like algorithm to create a compact, evidence-rich subgraph.
- **Answering**: Feed the subgraph + snippets into an LLM to generate a grounded answer with biomedical information and PMIDs/experiment IDs.

<img width="5102" height="2808" alt="image" src="https://github.com/user-attachments/assets/aadc47f9-9653-4b34-b337-51375c00b754" />

## Methods

### Knowledge Graph and Indexing

Our knowledge graph stores heterogeneous biomedical entities such as drugs, diseases, genes, and proteins, alongside richly typed relationships. Ingestion enforces schema-level guarantees (label-specific uniqueness, referential integrity) and attaches text embeddings to every node. We maintain both semantic indexes (vector similarity) and structural indexes (label/property) so that questions can be seeded semantically while traversals stay performant. Production graphs live in Amazon Neptune with supporting artifacts in Amazon S3, and configuration defaults are kept in `configs/default.yaml` with loaders under `src/ingest/`.

### Retrieval and Base Subgraph Construction

Incoming questions are embedded with our text encoder. We identify the top-matching nodes via cosine similarity, then expand their one-hop neighborhoods inside Neptune using IAM-signed openCypher queries that enforce namespace prefixes and degree caps. The result is a base subgraph that balances recall against the combinatorial growth typical in dense biomedical graphs.

### Prize Assignment and PCST Pruning

Within the base subgraph we assign a "prize" score to nodes and edges based on three cues: semantic similarity to the question, membership in the original seed set, and curated edge semantics. Traversal costs discourage overly large subgraphs. We then run a prize-collecting Steiner tree procedure that returns a compact, connected subgraph with high total prize. This pruning stage preserves multi-hop evidence while keeping the context manageable for downstream models.

### Neural Reader: GNN + LLM Integration

The pruned subgraph passes through a PyTorch Geometric GATv1 encoder to produce node representations that capture multi-hop structure and textual attributes. We serialize the subgraph into an ordered, human-readable description (node names, descriptions, relation types) and feed it—alongside the question and a soft prompt derived from the GNN outputs—into an instruction-tuned large language model. The LLM remains frozen so we benefit from its language fluency while the GNN-derived prompt focuses attention on graph evidence.

### Training Objectives

Supervision uses tuples of question, answer nodes, and source subgraphs. Training optimizes two losses jointly: a node-level loss that encourages the model to rank true answers above distractors, and a generation loss that compels the LLM to produce grounded natural-language answers conditioned on the serialized subgraph and soft prompt.

### Inference

At inference we return the generated answer together with the top-ranked answer nodes extracted from the subgraph. When recall is critical, we append additional high-prize nodes from the pruned subgraph, effectively ensembling the retrieval and reasoning stages.

### System Implementation

Retrieval and pruning logic resides in `src/retrieval/g_retriever.py` and `src/retrieval/expand.py`, while the graph-aware reader and reranking pipeline are implemented in `src/gnn/pyg_rag.py` and orchestrated via `src/rag/pipeline.py`. Neptune remains the source of truth for production graphs, with strict schema validation and uniqueness checks during ingest. Neptune loaders live in `src/ingest/`. Configuration, seeds, and embedding utilities are centralized in `configs/default.yaml`, `src/utils/seed.py`, and `src/embeddings/` respectively.

### Reproducibility and Evaluation

We run with fixed random seeds, log every relevant hyperparameter, and evaluate on multi-hop biomedical QA benchmarks using metrics such as hits at K, recall, and mean reciprocal rank. Sensitivity analyses cover the number of retrieved seeds, expansion depth, prize schedules, and pruning strength, echoing the coupled hyperparameter behavior observed in GraphRAG literature.

### References

- NVIDIA Technical Blog: Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases — see [https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/](https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/)
- G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (arXiv:2402.07630) — see [https://arxiv.org/pdf/2402.07630](https://arxiv.org/pdf/2402.07630)
- STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases — see [https://arxiv.org/abs/2404.13207](https://arxiv.org/abs/2404.13207)
- Building a knowledge graph to enable precision medicine — Scientific Data (2023) — see [https://www.nature.com/articles/s41597-023-02094-0](https://www.nature.com/articles/s41597-023-02094-0)

## Repository Layout

- `configs/` — YAML configs for ingestion, retrieval, and demo questions.
- `docs/` — Architecture notes (`methods.md`) and dual-KG design primer (`dual_kg.md`).
- `infra/docker/` — Container images for API and Streamlit UI deployments.
- `src/` — Application code; see `src/rag/pipeline.py` for the end-to-end flow.
- `tests/` — Neptune expansion regression tests.
- `OPTIONAL_ADDITIONS.txt` — Hardening and observability backlog.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate embeddings and seed vector store
make embed

# Run an end-to-end demo
make qa
```

### Run locally (no AWS) using FAISS vector store

```bash
export BIO_KG_vector_store__backend=faiss
make embed
make qa
```

## Data Preparation

1. Download PrimeKG extracts into `data/local/primekg/exports/` using the filenames referenced in `configs/ingest_prime.yaml`.
2. Place PubMedKG exports under `data/local/pubmedkg/exports/` or point the config to an S3 prefix.
3. Convert raw files into Neptune-friendly CSVs:

   ```bash
   make prime_to_neptune
   make pkg_to_neptune
   ```
4. Upload the generated CSVs to S3 and trigger Neptune bulk loads via `make load_prime` / `make load_pkg`.

## Make Targets

- `make setup` — create virtualenv and install dependencies.
- `make embed` — build embeddings, create the OpenSearch index, and upsert vectors.
- `make qa` — answer questions from `configs/demo_questions.yaml`.
- `make api` / `make ui` — launch FastAPI and Streamlit front-ends for interactive use.

## Running Retrieval & QA

1. Ensure Neptune is reachable (`configs/default.yaml` → `neptune.endpoint`).
2. Seed the vector store with embeddings (`make embed`).
3. Execute `python -m src.qa.answer --config configs/default.yaml --question-file configs/demo_questions.yaml` to retrieve subgraphs and generate grounded answers.
4. Inspect logs in `logs/` for expansion/pruning diagnostics.

## Evaluation & Testing

- `pytest -q` runs unit tests, including Neptune expansion coverage in `tests/test_expand_neptune.py`.
- Extend evaluation by comparing factual grounding metrics or integrating STaRK benchmark datasets for cross-validation of retrieval quality [STaRK](https://github.com/snap-stanford/STaRK).

## Docker Images

- `infra/docker/Dockerfile.api` — FastAPI service packaging the QA endpoint.
- `infra/docker/Dockerfile.ui` — Streamlit UI for browsing retrieved subgraphs.
  Build with:

```bash
docker build -t biographrag-api -f infra/docker/Dockerfile.api.
docker build -t biographrag-ui -f infra/docker/Dockerfile.ui.
```

## Production deployment (AWS)

The system is designed to run fully on AWS with Amazon Neptune (openCypher), Amazon OpenSearch for vector retrieval, and S3 for graph CSVs. For production, supply endpoints and credentials via environment variables (preferred) or update `configs/default.yaml` and override with env.

### Required environment variables (provide at run time)

- `AWS_DEFAULT_REGION` — e.g., `us-east-1`
- `BIO_KG_neptune__endpoint` — Writer endpoint with port 8182, e.g., `https://<cluster>.neptune.amazonaws.com:8182`
- `BIO_KG_neptune__iam_role_arn` — IAM role ARN used by Neptune bulk loader (S3 read)
- `BIO_KG_s3__bucket` — S3 bucket name for graph CSVs (prefixes `prime/`, `pkg/`)
- `BIO_KG_open_search__endpoint` — Managed OpenSearch domain endpoint (IAM SigV4 auth)
- `BIO_KG_open_search__use_iam_auth` — `true` (recommended)
- `BIO_KG_open_search__service` — `es` for Managed OpenSearch or `aoss` for Serverless
- `BIO_KG_llm__api_base` — OpenAI‑compatible base URL (exposes `/v1/chat/completions`)
- `BIO_KG_llm__api_key` — API key for the LLM endpoint

### API endpoints (FastAPI)

- `GET /health` → `{ "status": "ok" }`
- `POST /qa` → run retrieval + PyG fusion + LLM; body:

  ```json
  { "question": "Which PrimeKG findings highlight EGFR involvement in colon cancer?" }
  ```

  Response includes the grounded answer, prompt, nodes, edges, and evidence hits.

### What the user must supply (summary)

- Neptune writer endpoint (port 8182) and bulk‑loader role ARN
- S3 bucket for graph CSVs
- OpenSearch endpoint (Managed or Serverless) with IAM access for the app role
- OpenAI‑compatible LLM base URL and API key

These are not bundled with the repository and must be provided at deploy time.

## Configuration Highlights

- `graph.backend`: switch between Neptune (default) and future adapters.
- `retrieval.top_k` / `expansion_hops`: shape the subgraph frontier.
- `pyg_rag.top_facts`: control evidence window fed to the LLM.
- `open_search.*`: set endpoint, index name, and IAM auth toggle.
- `llm.*`: configure API gateway for the generation model.

## Deployment Playbook

1. Provision AWS resources (Neptune, OpenSearch, S3) via IaC or console.
2. Populate S3 with PrimeKG / PubMedKG graph CSVs.
3. Trigger Neptune bulk loads using `src/ingest/neptune_loader.py` helpers.
4. Deploy API/UI containers to ECS or EKS; mount config overrides via environment variables.
5. Configure CI (GitHub Actions) to lint, test, and build Docker artifacts on pull requests.
6. Monitor query latency and evidence coverage through structured logs and optional CloudWatch dashboards.

## Optional Additions to implement

- Infrastructure (CI/CD and IaC)
  - GitHub Actions for lint/tests on pull requests
- pixi and UV for env Setup
- checkpointed incremental loads
- Structured JSON logs in src/utils/logging.py
- Testing
  - Unit tests for utils, retrieval, and ETL
  - Tiny synthetic datasets in tests/
- - Lightweight explorer in src/ui to visualize subgraphs and evidence

## Contributing

- Fork, create a feature branch, and ensure `pytest` + `ruff` pass locally.
- Document new configs or endpoints in `docs/` and update demo questions if QA surfaces new entities.
- Open a PR with clear reproducibility notes and, where possible, add tests.

## License

MIT License. See `LICENSE` for the full text.

## Citation

If you use BioGraphRAG in published work, please cite the NVIDIA GraphRAG blog and G-Retriever paper listed above.

## Acknowledgements

- BioGraphRAG builds upon GraphRAG patterns, PyTorch Geometric utilities, and the collective work of the hackathon team.
-
