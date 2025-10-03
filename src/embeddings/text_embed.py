"""Utilities to compute text/document embeddings from dual PrimeKG + PubMedKG graphs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def build_document_embeddings(config_path: str) -> Path:
    """Generate text embeddings for PrimeKG + PubMedKG documents."""

    cfg = load_config(config_path)
    seed_everything()

    model_name = cfg.get("embedding_model.document_model", "sentence-transformers/all-MiniLM-L12-v2")
    model = SentenceTransformer(model_name)

    docs = _collect_documents(cfg)
    if not docs:
        raise ValueError("No documents available for text embedding. Ensure PrimeKG and PubMedKG CSVs exist.")

    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=False)

    df_emb = pd.DataFrame(embeddings)
    df_emb.insert(0, "id", [doc["id"] for doc in docs])
    df_emb.insert(1, "type", [doc["type"] for doc in docs])
    df_emb.insert(2, "text", texts)

    output_dir = ensure_dir(cfg.get("paths.embeddings_dir", "data/embeddings"))
    output_path = write_dataframe(df_emb, Path(output_dir) / "document_embeddings.csv", index=False)
    LOGGER.info("Wrote document embeddings to %s", output_path)
    return output_path


def _collect_documents(cfg) -> List[dict]:
    """Collect PrimeKG nodes and PubMedKG publications to encode."""
    docs: List[dict] = []

    pkg_nodes = Path(cfg.get("pkg_ingest.output_dir", "data/graph/pkg")) / "nodes.csv"
    if not pkg_nodes.exists():
        raise FileNotFoundError(f"PubMedKG nodes CSV missing: {pkg_nodes}")

    publications = pd.read_csv(pkg_nodes)
    publications = publications[publications["~label"] == "PKG_Publication"]
    for _, row in publications.iterrows():
        pmid = row["~id"]
        text = _row_to_text("PKG_Publication", row)
        docs.append({"id": f"PKG::{pmid}", "text": text, "type": "PKG_Publication"})

    prime_nodes = Path(cfg.get("prime_ingest.output_dir", "data/graph/prime")) / "nodes.csv"
    if not prime_nodes.exists():
        raise FileNotFoundError(f"PrimeKG nodes CSV missing: {prime_nodes}")

    prime_df = pd.read_csv(prime_nodes)
    prime_df["~label"] = prime_df["~label"].astype(str)
    prime_df = prime_df[prime_df["~label"].str.startswith("PRIME_")]
    for _, row in prime_df.iterrows():
        node_id = row["~id"]
        label = row.get("~label", "PRIME_Entity")
        text = _row_to_text(label, row)
        docs.append({"id": f"PRIME::{node_id}", "text": text, "type": label})

    return docs


def _row_to_text(label: str, row: pd.Series) -> str:
    """Serialise properties into a pipe-delimited string for embedding."""
    parts = [label]
    for column, value in row.items():
        if column in {"~id", "~label"}:
            continue
        if pd.isna(value):
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def main() -> None:
    """CLI entry point for building document embeddings."""
    parser = argparse.ArgumentParser(description="Build document embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    build_document_embeddings(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()
