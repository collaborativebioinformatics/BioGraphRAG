"""PCST-like pruning heuristic for compact subgraphs."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import networkx as nx


LOGGER = logging.getLogger(__name__)


def prune_subgraph(nodes: List[Dict], rels: List[Dict], max_nodes: int = 40) -> Tuple[List[Dict], List[Dict]]:
    graph = nx.Graph()

    for node in nodes:
        node_id = node.get("~id") or node.get("id")
        graph.add_node(node_id, data=node)

    for rel in rels:
        start = rel.get("~from") or rel.get("from")
        end = rel.get("~to") or rel.get("to")
        weight = _edge_weight(rel)
        graph.add_edge(start, end, data=rel, weight=weight)

    prizes = {node: _node_prize(graph.nodes[node]["data"]) for node in graph.nodes}
    sorted_nodes = sorted(prizes.items(), key=lambda item: item[1], reverse=True)

    selected_nodes = set()
    for node, _ in sorted_nodes:
        selected_nodes.add(node)
        if len(selected_nodes) >= max_nodes:
            break

    induced = graph.subgraph(selected_nodes).copy()

    pruned_nodes = [graph.nodes[n]["data"] for n in induced.nodes]
    pruned_edges = [graph.edges[u, v]["data"] for u, v in induced.edges]
    return pruned_nodes, pruned_edges


def _node_prize(node: Dict) -> float:
    label = node.get("~label") or node.get("label")
    if label == "Finding":
        return 3.0
    if label == "Protein":
        return 2.5
    if label == "Experiment":
        return 2.0
    if label == "Publication":
        return 1.5
    return 1.0


def _edge_weight(edge: Dict) -> float:
    label = edge.get("~label") or edge.get("label")
    if label in {"MEASURES", "DERIVED_FROM"}:
        return 0.5
    if label in {"IMPLICATES", "MENTIONS"}:
        return 0.8
    return 1.0

