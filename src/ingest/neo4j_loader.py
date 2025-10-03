"""Utilities to bulk load PrimeKG + PubMedKG CSV exports into Neo4j.

The loader expects Neptune-style CSVs (``nodes.csv`` and ``edges.csv``) emitted by
``prime_to_neptune.py`` and ``pkg_to_neptune.py``. It can be invoked via the
command line or imported directly from other scripts/tests.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable

from neo4j import GraphDatabase

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Neo4j loader."""
    parser = argparse.ArgumentParser(description="Load PrimeKG + PubMedKG CSVs into Neo4j")
    parser.add_argument("--config", default="configs/neo4j.yaml", help="Path to configuration file")
    parser.add_argument("--prime-dir", default="data/graph/prime", help="Directory containing PrimeKG nodes/edges CSVs")
    parser.add_argument("--pkg-dir", default="data/graph/pkg", help="Directory containing PubMedKG nodes/edges CSVs")
    return parser.parse_args()


def main() -> None:
    """CLI entry point that reads args and executes the load."""
    args = parse_args()
    cfg = load_config(args.config)

    load_neo4j_from_dirs(cfg, Path(args.prime_dir), Path(args.pkg_dir))


def load_neo4j_from_dirs(cfg, prime_dir: Path, pkg_dir: Path) -> None:
    """Programmatically load PrimeKG + PubMedKG CSVs into Neo4j.

    Parameters
    ----------
    cfg:
        :class:`~src.utils.config.AppConfig` instance with Neo4j settings under
        ``neo4j``.
    prime_dir:
        Directory containing PrimeKG ``nodes.csv`` and ``edges.csv`` files.
    pkg_dir:
        Directory containing PubMedKG ``nodes.csv`` and ``edges.csv`` files.
    """
    neo4j_cfg = _read_neo4j_config(cfg.raw.get("neo4j", {}))
    prime_dir = Path(prime_dir).resolve()
    pkg_dir = Path(pkg_dir).resolve()

    if not prime_dir.exists():
        raise FileNotFoundError(f"PrimeKG graph directory not found: {prime_dir}")
    if not pkg_dir.exists():
        raise FileNotFoundError(f"PKG graph directory not found: {pkg_dir}")

    LOGGER.info("Loading Neo4j data from %s (PrimeKG) and %s (PKG)", prime_dir, pkg_dir)

    driver = GraphDatabase.driver(neo4j_cfg["uri"], auth=(neo4j_cfg["username"], neo4j_cfg["password"]))
    try:
        with driver.session(database=neo4j_cfg.get("database", "neo4j")) as session:
            _create_constraints(session)
            _load_dataset(session, prime_dir / "nodes.csv", prime_dir / "edges.csv", namespace="PRIME")
            _load_dataset(session, pkg_dir / "nodes.csv", pkg_dir / "edges.csv", namespace="PKG")
    finally:
        driver.close()


def _read_neo4j_config(raw: Dict[str, str]) -> Dict[str, str]:
    """Validate required Neo4j keys and return the config mapping."""
    missing = [key for key in ("uri", "username", "password") if not raw.get(key)]
    if missing:
        raise ValueError(f"Neo4j configuration missing keys: {', '.join(missing)}")
    return raw


def _create_constraints(session) -> None:
    """Ensure uniqueness constraints exist for the PrimeKG/PKG namespaces."""
    constraints = [
        "CREATE CONSTRAINT prime_id IF NOT EXISTS FOR (n:PRIME) REQUIRE n.`~id` IS UNIQUE",
        "CREATE CONSTRAINT pkg_id IF NOT EXISTS FOR (n:PKG) REQUIRE n.`~id` IS UNIQUE",
    ]
    for statement in constraints:
        session.run(statement)
    LOGGER.info("Ensured Neo4j uniqueness constraints for PRIME and PKG namespaces")


def _load_dataset(session, nodes_path: Path, edges_path: Path, *, namespace: str) -> None:
    """Load a single namespace's node/edge CSVs into Neo4j."""
    if not nodes_path.exists():
        LOGGER.warning("Nodes CSV not found for namespace %s: %s", namespace, nodes_path)
        return
    if not edges_path.exists():
        LOGGER.warning("Edges CSV not found for namespace %s: %s", namespace, edges_path)
        return

    node_rows = list(_read_csv(nodes_path))
    edge_rows = list(_read_csv(edges_path))

    LOGGER.info("Loading %s nodes and %s edges for namespace %s", len(node_rows), len(edge_rows), namespace)

    for row in node_rows:
        _merge_node(session, row)

    for row in edge_rows:
        _create_relationship(session, row)

    LOGGER.info("Finished loading namespace %s", namespace)


def _merge_node(session, row: Dict[str, str]) -> None:
    """MERGE a node row into the target database."""
    node_id = row.get("~id")
    if not node_id:
        LOGGER.warning("Skipping node without ~id: %s", row)
        return

    dynamic_label = row.get("~label")
    namespace_label = _namespace_from_label(dynamic_label)
    labels = [label for label in {namespace_label, dynamic_label} if label]

    properties = {k: _coerce_value(v) for k, v in row.items() if v is not None}

    query = (
        "MERGE (n {`~id`: $id}) "
        "SET n += $props "
        "WITH n "
        "CALL apoc.create.addLabels(n, $labels) YIELD node "
        "RETURN node"
    )
    session.run(query, id=node_id, props=properties, labels=labels)


def _create_relationship(session, row: Dict[str, str]) -> None:
    """Create a relationship between two nodes (idempotent)."""
    start_id = row.get("~from")
    end_id = row.get("~to")
    rel_type = row.get("~label")
    if not start_id or not end_id or not rel_type:
        LOGGER.warning("Skipping relationship with missing identifiers: %s", row)
        return

    properties = {k: _coerce_value(v) for k, v in row.items() if k not in {"~from", "~to"} and v is not None}

    query = (
        "MATCH (a {`~id`: $start_id}), (b {`~id`: $end_id}) "
        "CALL apoc.create.relationship(a, $type, $props, b) YIELD rel "
        "RETURN rel"
    )
    session.run(query, start_id=start_id, end_id=end_id, type=rel_type, props=properties)


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    """Yield rows from a CSV, normalising empty strings to ``None``."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {key: (value if value != "" else None) for key, value in row.items()}
            yield cleaned


def _namespace_from_label(label: str | None) -> str | None:
    """Return the namespace to add based on the dynamic label."""
    if not label:
        return None
    if label.startswith("PRIME_"):
        return "PRIME"
    if label.startswith("PKG_"):
        return "PKG"
    return label.split("_")[0]


def _coerce_value(value: str | None) -> object:
    """Convert a CSV value into the appropriate Python primitive."""
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value



if __name__ == "__main__":  # pragma: no cover
    main()
