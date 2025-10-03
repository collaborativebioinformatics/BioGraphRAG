"""Feature extraction utilities for optional GNN reranker."""

from __future__ import annotations

from typing import Dict, List


def build_feature_matrix(nodes: List[Dict]) -> List[List[float]]:
    """Return a placeholder feature matrix."""

    return [[1.0] for _ in nodes]

