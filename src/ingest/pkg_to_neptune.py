"""Convert PubMedKG extracts into Neptune openCypher CSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map PubMedKG extracts to Neptune openCypher CSVs")
    parser.add_argument("--config", default="configs/ingest_pkg.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--publications-file")
    parser.add_argument("--mentions-file")
    parser.add_argument("--citations-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = Path(args.input_dir or cfg.get("pkg_ingest.input_dir"))
    output_dir = ensure_dir(args.output_dir or cfg.get("pkg_ingest.output_dir", "data/graph/pkg"))

    publications_file = args.publications_file or cfg.get("pkg_ingest.publications_file", "pubmed_publications.csv")
    mentions_file = args.mentions_file or cfg.get("pkg_ingest.mentions_file", "pubmed_mentions.csv")
    citations_file = args.citations_file or cfg.get("pkg_ingest.citations_file", "pubmed_citations.csv")

    publications_path = input_dir / publications_file
    mentions_path = input_dir / mentions_file
    citations_path = input_dir / citations_file

    LOGGER.info("Reading PubMedKG files from %s", input_dir)

    pubs = _read_required(publications_path)
    mentions = _read_required(mentions_path)
    citations = _read_required(citations_path)

    LOGGER.info("Loaded %s publications, %s mentions, %s citations", len(pubs), len(mentions), len(citations))

    nodes = _map_nodes(pubs, mentions)
    edges_mentions = _map_mentions(mentions)
    edges_citations = _map_citations(citations)

    node_path = output_dir / "nodes.csv"
    edge_path = output_dir / "edges.csv"

    combined_edges = pd.concat([edges_mentions, edges_citations], ignore_index=True, sort=False)

    write_dataframe(nodes, node_path, index=False)
    write_dataframe(combined_edges, edge_path, index=False)

    LOGGER.info("Wrote Neptune nodes to %s", node_path)
    LOGGER.info("Wrote Neptune edges to %s", edge_path)


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required PubMedKG file missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"PubMedKG file is empty: {path}")
    return df


def _map_nodes(publications: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    pub_nodes = publications.copy()
    if "pmid" not in pub_nodes.columns:
        raise KeyError("Publications file must contain 'pmid'")
    pub_nodes.rename(columns={"pmid": "~id"}, inplace=True)
    pub_nodes["~label"] = "PKG_Publication"

    mention_entities = mentions[["entity_id", "entity_type"]].dropna().drop_duplicates()
    mention_entities.rename(columns={"entity_id": "~id"}, inplace=True)
    mention_entities["~label"] = mention_entities["entity_type"].apply(_normalize_entity_label)
    mention_entities.drop(columns=["entity_type"], inplace=True)

    combined = pd.concat([pub_nodes, mention_entities], ignore_index=True, sort=False)
    combined.drop_duplicates(subset="~id", inplace=True)
    return combined


def _map_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    required = {"pmid", "entity_id"}
    missing = required - set(mentions.columns)
    if missing:
        raise KeyError(f"Mentions file missing columns: {', '.join(sorted(missing))}")

    df = mentions.copy()
    df.rename(columns={"pmid": "~from", "entity_id": "~to"}, inplace=True)
    df["~label"] = "PKG_MENTIONS"
    return df


def _map_citations(citations: pd.DataFrame) -> pd.DataFrame:
    required = {"source_pmid", "target_pmid"}
    missing = required - set(citations.columns)
    if missing:
        raise KeyError(f"Citations file missing columns: {', '.join(sorted(missing))}")

    df = citations.copy()
    df.rename(columns={"source_pmid": "~from", "target_pmid": "~to"}, inplace=True)
    df["~label"] = "PKG_CITES"
    return df


def _normalize_entity_label(entity_type: str) -> str:
    base = entity_type.strip().replace(" ", "_") or "Entity"
    if base.startswith("PKG_"):
        return base
    return f"PKG_{base}"


if __name__ == "__main__":  # pragma: no cover
    main()
