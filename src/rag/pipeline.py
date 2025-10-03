"""GraphRAG pipeline inspired by NVIDIA GraphRAG, tailored for dual PrimeKG + PKG.

The goal is to expose a set of *small* functions that can be stitched together to
build retrieval-and-generation flows. Keeping each step separate makes it easier
to unit test, to swap out implementations, or to reuse the pipeline in other
scripts.

Stages covered here (in order):

``encode_question``
    Embed the question text using a SentenceTransformer model.

``vector_seed``
    Retrieve the top-k candidate nodes from OpenSearch.

``expand_graph``
    Fan out from those candidates in the graph database (Neptune) while respecting
    namespace + hop & degree limits.

``prune_graph``
    Reduce the frontier to a manageable number of nodes/edges.

``pyg_fusion``
    Run a PyTorch Geometric based fusion to score facts that matter most.

``assemble_prompt``
    Produce the final prompt string handed to the LLM.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sentence_transformers import SentenceTransformer

from src.retrieval.vector_store import query_vectors
from src.retrieval.prune import prune_subgraph
from src.qa.prompt import build_prompt
from src.gnn.pyg_rag import (
    build_pyg_from_subgraph,
    encode_texts,
    fuse_vectors,
    gnn_graph_embedding,
    rank_facts,
    set_structural_vectors,
    structural_fact_vectors,
    textify_subgraph,
)

from src.utils.config import load_config


def _get_expander(config_path: str):
    cfg = load_config(config_path)
    backend = (cfg.get("graph.backend") or "neptune").lower()
    if backend == "neptune":
        from src.retrieval.expand_neptune import expand_subgraph

        return expand_subgraph
    raise ValueError("Only 'neptune' backend is supported")


def encode_question(model_name: str, question: str) -> List[float]:
    """Return a dense embedding for the input question text.

    Parameters
    ----------
    model_name:
        Sentence-Transformer model id to load.
    question:
        Natural-language question to vectorise.

    Notes
    -----
    Using a local helper keeps the calling code light and consistent with the
    modular pipeline layout from the NVIDIA GraphRAG reference.
    """

    model = SentenceTransformer(model_name)
    return model.encode([question], show_progress_bar=False)[0].tolist()



def vector_seed(config_path: str, question_vec: List[float], top_k: int) -> List[Dict]:
    """Fetch the top-k candidate nodes from OpenSearch."""

    return query_vectors(config_path, question_vec, top_k=top_k)


def expand_graph(config_path: str, seed_ids: List[str], hops: int, max_degree: int) -> Tuple[List[Dict], List[Dict]]:
    """Expand from the seed ids using the configured graph backend."""

    expand_fn = _get_expander(config_path)
    expansion = expand_fn(config_path, seed_ids, hops=hops, max_degree=max_degree)
    nodes = [entry for result in expansion for entry in result.get("nodes", [])]
    edges = [entry for result in expansion for entry in result.get("rels", [])]
    return nodes, edges


def prune_graph(nodes: List[Dict], edges: List[Dict], max_nodes: int) -> Tuple[List[Dict], List[Dict]]:
    """Trim the expanded subgraph to satisfy the size budget."""

    return prune_subgraph(nodes, edges, max_nodes=max_nodes)


def pyg_fusion(cfg, nodes: List[Dict], edges: List[Dict], top_facts: int) -> Tuple[List[Dict], List[Dict]]:
    """Fuse PrimeKG + PubMedKG signals with PyG and keep the top facts."""

    text_facts = textify_subgraph(nodes, edges)
    if not text_facts:
        return nodes, edges

    data, node_index = build_pyg_from_subgraph(nodes, edges)

    gnn_vec = np.asarray(gnn_graph_embedding(data, cfg), dtype=np.float32)
    llm_matrix = encode_texts([" ".join(text_facts)], cfg)
    llm_vec = llm_matrix[0] if llm_matrix.shape[0] else np.zeros_like(gnn_vec)

    node_embeddings = getattr(data, "node_embeddings", None)
    if node_embeddings is not None:
        node_embeddings = node_embeddings.detach().cpu().numpy()

    struct_vectors = structural_fact_vectors(nodes, edges, node_embeddings, node_index)
    set_structural_vectors(struct_vectors)

    try:
        fused = fuse_vectors(gnn_vec, llm_vec)
        ranked_indices = rank_facts(fused, text_facts, cfg)
        top_indices = ranked_indices[:top_facts]
        return _filter_facts_by_indices(top_indices, nodes, edges)
    finally:
        set_structural_vectors(None)


def assemble_prompt(question: str, nodes: List[Dict], edges: List[Dict], evidence: List[Dict]) -> str:
    """Turn the selected facts into the final LLM prompt."""

    return build_prompt(question, nodes, edges, evidence)


def _filter_facts_by_indices(indices: List[int], nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    from src.gnn.pyg_rag import get_fact_lookup

    lookup = get_fact_lookup()
    selected_nodes: List[Dict] = []
    selected_edges: List[Dict] = []

    for idx in indices:
        if idx >= len(lookup):
            continue
        fact_type, fact_idx = lookup[idx]
        if fact_type == "node" and fact_idx < len(nodes):
            selected_nodes.append(nodes[fact_idx])
        elif fact_type == "edge" and fact_idx < len(edges):
            selected_edges.append(edges[fact_idx])

    if not selected_nodes and not selected_edges:
        return nodes, edges

    selected_node_ids = {node.get("~id") for node in selected_nodes}
    selected_edges = [edge for edge in selected_edges if edge.get("~from") in selected_node_ids and edge.get("~to") in selected_node_ids]
    return selected_nodes, selected_edges
