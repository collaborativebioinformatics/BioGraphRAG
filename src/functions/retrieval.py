from __future__ import annotations

from typing import Dict, List

from src.retrieval.g_retriever import initialize_vector_store, retrieve_graph_context


def initialize_vectors(config_path: str) -> None:
    """Create and populate the vector store (documents + nodes)."""
    initialize_vector_store(config_path)


def retrieve_context(config_path: str, question_embedding: List[float]) -> Dict[str, List[Dict]]:
    """Return nodes, edges, and evidence for a given question embedding."""
    return retrieve_graph_context(config_path, question_embedding)


