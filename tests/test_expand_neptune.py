from __future__ import annotations

from pathlib import Path
import boto3
import pytest

from src.retrieval.expand_neptune import expand_subgraph


def _write_config(tmp_path: Path, use_iam: bool = False) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "project:",
                "  region: us-east-1",
                "graph:",
                "  backend: neptune",
                "  id_property: \"~id\"",
                "  label_prefixes:",
                "    - PRIME_",
                "    - PKG_",
                "neptune:",
                "  endpoint: https://example-neptune:8182",
                f"  use_iam_auth: {'true' if use_iam else 'false'}",
                "  opencypher_path: /opencypher",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def test_expand_subgraph_no_seeds_returns_empty(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    assert expand_subgraph(str(config_path), [], hops=1, max_degree=1) == []


@pytest.mark.integration
def test_sts_identity_available() -> None:
    client = boto3.client("sts")
    identity = client.get_caller_identity()
    assert "Arn" in identity and identity["Arn"], "STS identity must resolve for IAM signing"

