"""Convert PrimeKG exports into Neptune openCypher CSVs (namespaced PRIME_*)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map PrimeKG exports to Neptune openCypher CSVs")
    parser.add_argument("--config", default="configs/ingest_prime.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--edges-file")
    parser.add_argument("--nodes-file")
    parser.add_argument("--nodes-glob")
    parser.add_argument("--relationships-glob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = Path(args.input_dir or cfg.get("prime_ingest.input_dir"))
    output_dir = ensure_dir(args.output_dir or cfg.get("prime_ingest.output_dir", "data/graph/prime"))

    edges_file = args.edges_file or cfg.get("prime_ingest.edges_file")
    nodes_file = args.nodes_file or cfg.get("prime_ingest.nodes_file")
    nodes_glob = args.nodes_glob or cfg.get("prime_ingest.nodes_glob", "prime_nodes_*.csv")
    rels_glob = args.relationships_glob or cfg.get("prime_ingest.relationships_glob", "prime_edges_*.csv")

    if edges_file and (input_dir / edges_file).exists():
        edges = pd.read_csv(input_dir / edges_file, low_memory=False)
        nodes = _derive_nodes_from_edges(edges)
        mapped_nodes = _map_prime_nodes(nodes)
        mapped_edges = _map_prime_edges(edges)
    else:
        node_files = sorted(input_dir.glob(nodes_glob))
        rel_files = sorted(input_dir.glob(rels_glob))
        if not node_files and not nodes_file:
            raise FileNotFoundError("PrimeKG nodes not found; provide nodes_file or nodes_glob matches")
        if not rel_files and not edges_file:
            raise FileNotFoundError("PrimeKG relationships not found; provide edges_file or relationships_glob matches")

        nodes = pd.concat([pd.read_csv(p) for p in node_files], ignore_index=True, sort=False) if node_files else pd.read_csv(input_dir / nodes_file)
        rels = pd.concat([pd.read_csv(p) for p in rel_files], ignore_index=True, sort=False) if rel_files else pd.read_csv(input_dir / edges_file)
        mapped_nodes = _map_prime_nodes(nodes)
        mapped_edges = _map_prime_edges(rels)

    write_dataframe(mapped_nodes, output_dir / "nodes.csv", index=False)
    write_dataframe(mapped_edges, output_dir / "edges.csv", index=False)


def _derive_nodes_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    # PrimeKG edge list columns: x_id, y_id, x_type, y_type, relation (plus optional text fields)
    left = edges[["x_id", "x_type"]].rename(columns={"x_id": "~id", "x_type": "~label"})
    right = edges[["y_id", "y_type"]].rename(columns={"y_id": "~id", "y_type": "~label"})
    nodes = pd.concat([left, right], ignore_index=True)
    nodes.drop_duplicates(subset=["~id"], inplace=True)
    return nodes


def _map_prime_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    df = nodes.copy()
    if "~id" not in df.columns:
        if "id" in df.columns:
            df.rename(columns={"id": "~id"}, inplace=True)
        else:
            raise KeyError("PrimeKG nodes must contain '~id' or 'id'")
    if "~label" not in df.columns:
        if "type" in df.columns:
            df.rename(columns={"type": "~label"}, inplace=True)
        else:
            df["~label"] = "PRIME_Entity"
    df["~label"] = df["~label"].apply(_prime_label)
    return df


def _map_prime_edges(rels: pd.DataFrame) -> pd.DataFrame:
    df = rels.copy()
    if {"x_id", "y_id"}.issubset(df.columns):
        df.rename(columns={"x_id": "~from", "y_id": "~to"}, inplace=True)
    elif {"start_id", "end_id"}.issubset(df.columns):
        df.rename(columns={"start_id": "~from", "end_id": "~to"}, inplace=True)
    else:
        raise KeyError("Edges must contain 'x_id,y_id' or 'start_id,end_id'")

    if "relation" in df.columns:
        df.rename(columns={"relation": "~label"}, inplace=True)
    elif "type" in df.columns:
        df.rename(columns={"type": "~label"}, inplace=True)
    else:
        df["~label"] = "PRIME_REL"

    df["~label"] = df["~label"].apply(_prime_label)
    return df


def _prime_label(label: str) -> str:
    if not isinstance(label, str):
        return "PRIME_Entity"
    if label.startswith("PRIME_"):
        return label
    # Map plain types to PRIME_* namespace
    clean = label.strip().replace(" ", "_")
    return f"PRIME_{clean}"


if __name__ == "__main__":  # pragma: no cover
    main()
