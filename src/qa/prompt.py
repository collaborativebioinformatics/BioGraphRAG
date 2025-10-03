"""Prompt builder for GraphRAG responses."""

from __future__ import annotations

from typing import Dict, List


def build_prompt(question: str, nodes: List[Dict], edges: List[Dict], evidence: List[Dict]) -> str:
    lines = ["You are an expert biomedical assistant. Use the graph context to answer succinctly and cite PMIDs or experiment IDs."]
    lines.append(f"Question: {question}")
    lines.append("\nGraph Context:")

    for node in nodes:
        node_id = node.get("~id") or node.get("id")
        label = node.get("~label") or node.get("label")
        desc = node.get("description") or node.get("name") or ""
        lines.append(f"- Node[{label}] {node_id}: {desc}")

    for edge in edges:
        start = edge.get("~from") or edge.get("from")
        end = edge.get("~to") or edge.get("to")
        label = edge.get("~label") or edge.get("label")
        lines.append(f"- Edge[{label}] {start} -> {end}")

    if evidence:
        lines.append("\nVector Retrieval Evidence:")
        for item in evidence:
            lines.append(f"- {item['id']} (score={item['score']:.3f})")

    lines.append("\nAnswer (include citations and brief rationale):")
    return "\n".join(lines)

