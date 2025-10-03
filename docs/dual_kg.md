# Dual Knowledge Graph Architecture

The dual-KG deployment loads PrimeKG (precision medicine knowledge graph) and PubMedKG side-by-side into Amazon Neptune. Each KG retains its native schema and semantics. Downstream retrieval performs query-time joins by exact identifiers or vector similarity, avoiding any harmonization or synthetic augmentation.

## Key concepts

- **Namespaces**: PrimeKG nodes and edges are prefixed with `PRIME_`, while PubMedKG nodes and edges use `PKG_`. This allows graph queries and OpenSearch filters to stay scoped to their original sources.
- **Separation of concerns**: Each KG is ingested from raw exports to Neptune-ready CSVs independently. The GraphRAG pipeline combines them only during retrieval.
- **Query-time joining**: Answer generation relies on two strategies:
  - Exact matches on shared identifiers (e.g., proteins, diseases) retrieved from both graphs.
  - Vector similarity between publications, findings, and entities across namespaces via OpenSearch.

## Data flow

1. **Raw exports**
   - Place PrimeKG exports (e.g., `kg.csv`) under `data/local/primekg/exports/` (filenames configurable in `configs/ingest_prime.yaml`).
   - Place PubMedKG CSV/TSV extracts under `data/local/pubmedkg/exports/` (filenames configurable in `configs/ingest_pkg.yaml`).

2. **PrimeKG mapping** (`src/ingest/prime_to_neptune.py`)
   - Reads the PrimeKG edge list (`kg.csv`) or pre-split node/edge CSVs and preserves all properties.
   - Writes `data/graph/prime/nodes.csv` and `data/graph/prime/edges.csv` with namespaced labels like `PRIME_Entity`, `PRIME_RELATION`.

3. **PubMedKG mapping** (`src/ingest/pkg_to_neptune.py`)
   - Reads publication, mention, and citation files.
   - Emits `data/graph/pkg/nodes.csv` and `data/graph/pkg/edges.csv` using labels `PKG_Publication`, `PKG_MENTIONS`, `PKG_CITES`.

4. **Graph database loading**
   - **Neptune** (`src/ingest/neptune_loader.py`): generates loader payload JSON targeting S3 prefixes `graph/prime/` and `graph/pkg/`. Submit each JSON payload to Neptune’s bulk loader API to populate the graph.

5. **Embeddings & Vector Store**
   - `src/embeddings/text_embed.py` builds document vectors from PKG publications and PrimeKG nodes.
   - `src/embeddings/node_embed.py` produces embeddings for all nodes across namespaces.
   - `src/retrieval/vector_store.py` stores vectors in OpenSearch with namespace metadata for filtering.

6. **GraphRAG Retrieval**
   - Vector KNN (OpenSearch) selects seed nodes across both namespaces.
   - `src/retrieval/expand_neptune.py` expands within Neptune’s openCypher endpoint up to the configured hop count while respecting namespace filters and degree limits.
   - `src/retrieval/prune.py` trims the subgraph before LLM answer synthesis.

## Operational summary

- Run `make prime_to_neptune` and `make pkg_to_neptune` after staging input files.
- Upload `data/graph/prime/*.csv` and `data/graph/pkg/*.csv` to S3 prefixes referenced by `configs/default.yaml`.
- Execute `make load_prime` and `make load_pkg` to produce loader payloads, then invoke the Neptune bulk loader separately for each prefix.
- After loading Neptune, run `make embed` to populate OpenSearch and `make qa` to exercise the GraphRAG pipeline.

## Query-time joining strategies

- **Exact key linking**: If both graphs contain entities with matching identifiers (e.g., `EGFR`), the expanded subgraph will contain nodes from both namespaces, enabling downstream reasoning.
- **Semantic proximity**: Vector similarity across embeddings connects, for example, PKG publications mentioning a gene with PrimeKG findings about related pathways.

## Residual considerations

- Ensure that CSV exports include all necessary properties; the mappers do not infer missing fields.
- Monitor namespace coverage when adding new data sources—extend the `retrieval.namespaces.include` list in `configs/default.yaml` as needed.
- For large datasets, consider chunking the Neptune load into multiple batches per namespace to stay within service limits.
