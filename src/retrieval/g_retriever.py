"""High-level GraphRAG orchestration combining retrieval, expansion, pruning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from src.embeddings.text_embed import build_document_embeddings
from src.embeddings.node_embed import build_node_embeddings
from src.retrieval.vector_store import create_index, query_vectors, upsert_vectors
from src.retrieval.prune import prune_subgraph
from src.utils.config import load_config
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def initialize_vector_store(config_path: str) -> None:
    """Create the OpenSearch index and populate it with graph embeddings."""

    seed_everything()
    cfg = load_config(config_path)

    create_index(config_path)

    doc_path = build_document_embeddings(config_path)
    node_path = build_node_embeddings(config_path)

    doc_embeddings = _load_embeddings(doc_path)
    node_embeddings = _load_embeddings(node_path)

    documents: List[Dict] = []
    for _, row in doc_embeddings.iterrows():
        payload: Dict = {
            "id": row["id"],
            "type": row.get("type"),
            "text": row.get("text"),
            "namespace": _namespace_from_type(row.get("type")),
            "vector": row.drop(labels=["id", "type", "text"]).tolist(),
        }
        documents.append(payload)
    upsert_vectors(config_path, documents)

    node_documents: List[Dict] = []
    for _, row in node_embeddings.iterrows():
        payload = {
            "id": row["id"],
            "type": row.get("type"),
            "text": row.get("text"),
            "namespace": _namespace_from_type(row.get("type")),
            "vector": row.drop(labels=["id", "type", "text"]).tolist(),
        }
        node_documents.append(payload)
    upsert_vectors(config_path, node_documents)


def retrieve_graph_context(config_path: str, question_embedding: List[float]) -> Dict[str, List[Dict]]:
    """Expand and prune the neighbourhood around vector-seed candidates."""

    cfg = load_config(config_path)
    top_k = cfg.get("retrieval.top_k", 8)
    hops = cfg.get("retrieval.expansion_hops", 2)
    max_degree = cfg.get("retrieval.prune_max_degree", 10)
    max_nodes = cfg.get("retrieval.prune_max_nodes", 40)

    hits = query_vectors(config_path, question_embedding, top_k=top_k)
    seed_ids = [hit["id"].split("::")[-1] for hit in hits]
    LOGGER.info("Vector hits: %s", seed_ids)

    expand_fn = _get_expander(config_path)
    expansion = expand_fn(config_path, seed_ids, hops=hops, max_degree=max_degree)
    nodes = [entry for result in expansion for entry in result.get("nodes", [])]
    edges = [entry for result in expansion for entry in result.get("rels", [])]

    pruned_nodes, pruned_edges = prune_subgraph(nodes, edges, max_nodes=max_nodes)
    if not pruned_nodes:
        LOGGER.info("Retrieve graph context produced zero nodes after pruning")
    return {
        "nodes": pruned_nodes,
        "edges": pruned_edges,
        "evidence": hits,
    }


def _load_embeddings(path: str | Path):
    """Load the embeddings CSV produced by the embedding scripts."""

    return pd.read_csv(Path(path))


def _namespace_from_type(label: Optional[str]) -> str:
    """Infer the namespace prefix from the document/node type."""

    if not label:
        return ""
    if label.startswith("PRIME_"):
        return "PRIME_"
    if label.startswith("PKG_"):
        return "PKG_"
    return label.split("_")[0]


def _get_expander(config_path: str):
    cfg = load_config(config_path)
    backend = (cfg.get("graph.backend") or "neptune").lower()
    if backend == "neptune":
        from src.retrieval.expand_neptune import expand_subgraph

        return expand_subgraph
    raise ValueError("Only 'neptune' backend is supported")

