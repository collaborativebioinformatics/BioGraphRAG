"""LLM interface for producing answers from GraphRAG context with optional PyG fusion."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.rag.pipeline import (
    encode_question,
    vector_seed,
    expand_graph,
    prune_graph,
    pyg_fusion,
    assemble_prompt,
)
from src.qa.prompt import build_prompt
from src.retrieval.g_retriever import initialize_vector_store, retrieve_graph_context
from src.utils.config import load_config
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def answer_question(config_path: str, question: str) -> Dict:
    """Run the GraphRAG pipeline end-to-end and return answer + evidence.

    Parameters
    ----------
    config_path:
        Path to a YAML config file containing retrieval, PyG, LLM, and
        connection parameters.
    question:
        Natural-language question supplied by the user.

    Returns
    -------
    dict
        A dictionary containing the original question, the LLM answer text,
        the prompt that was sent, and the nodes/edges/evidence gathered during
        retrieval.

    Notes
    -----
    The function performs the following steps:
    1. Load configuration & seed random number generators.
    2. Encode the question into a dense vector.
    3. Retrieve vector seeds via OpenSearch.
    4. Expand and prune the graph neighbourhood around the seeds.
    5. Apply PyTorch Geometric fusion to rank the most relevant facts.
    6. Assemble the final prompt and call the target LLM.
    """

    seed_everything()
    cfg = load_config(config_path)

    # Encode question -----------------------------------------------------
    model_name = cfg.get("embedding_model.document_model", "sentence-transformers/all-MiniLM-L12-v2")
    question_vector = encode_question(model_name, question)

    # Vector seeds → expand → prune ---------------------------------------
    top_k = cfg.get("retrieval.top_k", 8)
    hops = cfg.get("retrieval.expansion_hops", 2)
    max_degree = cfg.get("retrieval.prune_max_degree", 10)
    max_nodes = cfg.get("retrieval.prune_max_nodes", 40)

    hits = vector_seed(config_path, question_vector, top_k)
    if not hits:
        LOGGER.warning("Vector search returned no candidates for question: %s", question)
        return {
            "question": question,
            "answer": "No relevant context found.",
            "prompt": "",
            "nodes": [],
            "edges": [],
            "evidence": [],
        }

    seed_ids = [hit["id"].split("::")[-1] for hit in hits if "id" in hit]
    nodes, edges = expand_graph(config_path, seed_ids, hops, max_degree)
    nodes, edges = prune_graph(nodes, edges, max_nodes)

    if not nodes:
        LOGGER.warning("Graph expansion produced no nodes for question: %s", question)
        return {
            "question": question,
            "answer": "No relevant context found after expansion.",
            "prompt": "",
            "nodes": [],
            "edges": [],
            "evidence": hits,
        }

    # Mandatory PyG fusion -----------------------------------------------
    top_facts = cfg.get("pyg_rag", {}).get("top_facts", 40)
    nodes, edges = pyg_fusion(cfg, nodes, edges, top_facts)

    # Prompt & LLM completion --------------------------------------------
    prompt = assemble_prompt(question, nodes, edges, hits)

    api_base = cfg.get("llm.api_base")
    if not api_base:
        raise ValueError("llm.api_base must be set in the configuration")

    api_key = cfg.get("llm.api_key")
    temperature = cfg.get("llm.temperature", 0.2)
    max_tokens = cfg.get("llm.max_new_tokens", 512)
    llm_model = cfg.get("llm.model_name")

    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "changeme":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a biomedical assistant. Provide concise answers with"
                    " experiment IDs and PMIDs as citations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("LLM request failed: %s", exc)
        raise

    data = response.json()
    answer_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "question": question,
        "answer": answer_text,
        "prompt": prompt,
        "nodes": nodes,
        "edges": edges,
        "evidence": hits,
    }


def _filter_facts_by_indices(indices: List[int], nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return only the nodes/edges corresponding to selected fact indices."""
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


def main() -> None:
    """Entry point for CLI usage."""

    parser = argparse.ArgumentParser(description="Run GraphRAG QA pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question-file", default="configs/demo_questions.yaml")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.initialize:
        initialize_vector_store(args.config)

    if args.question_file.endswith(".json"):
        questions = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    else:
        import yaml  # type: ignore

        questions = yaml.safe_load(Path(args.question_file).read_text(encoding="utf-8"))

    for item in questions:
        result = answer_question(args.config, item["question"])
        LOGGER.info("Answer: %s", result["answer"])


if __name__ == "__main__":  # pragma: no cover
    main()

