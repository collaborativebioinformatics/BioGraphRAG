"""Node embedding utilities for dual PrimeKG + PubMedKG graphs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def build_node_embeddings(config_path: str) -> Path:
    """Build sentence embeddings for graph nodes across both namespaces."""

    cfg = load_config(config_path)
    seed_everything()

    model_name = cfg.get("embedding_model.node_model", "sentence-transformers/all-MiniLM-L12-v2")
    model = SentenceTransformer(model_name)

    nodes = _load_all_nodes(cfg)
    rows = _to_embedding_rows(nodes)
    if not rows:
        raise ValueError("No graph nodes available for embeddings. Run the PrimeKG/PubMedKG loaders first.")

    texts = [row["text"] for row in rows]
    embeddings = model.encode(texts, show_progress_bar=False)

    df_emb = pd.DataFrame(embeddings)
    df_emb.insert(0, "id", [row["id"] for row in rows])
    df_emb.insert(1, "type", [row["type"] for row in rows])
    df_emb.insert(2, "text", texts)

    output_dir = ensure_dir(cfg.get("paths.embeddings_dir", "data/embeddings"))
    output_path = write_dataframe(df_emb, Path(output_dir) / "node_embeddings.csv", index=False)
    LOGGER.info("Wrote node embeddings to %s", output_path)
    return output_path


def _load_all_nodes(cfg) -> pd.DataFrame:
    """Load node CSVs from both namespaces into a single DataFrame."""
    paths = [
        Path(cfg.get("prime_ingest.output_dir", "data/graph/prime")) / "nodes.csv",
        Path(cfg.get("pkg_ingest.output_dir", "data/graph/pkg")) / "nodes.csv",
    ]
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Nodes CSV not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Nodes CSV is empty: {path}")
        if "~id" not in df.columns or "~label" not in df.columns:
            raise KeyError(f"Nodes CSV missing '~id' or '~label': {path}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _to_embedding_rows(df: pd.DataFrame) -> List[dict]:
    """Return unique node records with id/type/text fields for encoding."""
    rows: List[dict] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        node_id = str(row["~id"])
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        label = str(row["~label"])
        text = _row_to_text(row)
        rows.append({"id": node_id, "type": label, "text": text})
    return rows


def _row_to_text(row: pd.Series) -> str:
    """Serialise node properties into a single text string."""
    parts: List[str] = [str(row["~label"])]
    for column, value in row.items():
        if column in {"~id", "~label"}:
            continue
        if pd.isna(value):
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def main() -> None:
    """CLI entry point for building node embeddings."""
    parser = argparse.ArgumentParser(description="Build node embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    build_node_embeddings(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()

