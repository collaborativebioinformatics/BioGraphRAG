"""Generate Neptune loader payload JSON for specific graph prefixes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import ensure_dir


LOGGER = logging.getLogger(__name__)


def build_loader_payload(config_path: str, prefix: str | None = None) -> Path:
    cfg = load_config(config_path)
    bucket = cfg.get("s3.bucket")
    base_prefix = cfg.get("s3.prefixes.graph", "graph/")
    iam_role = cfg.get("neptune.iam_role_arn")
    endpoint = cfg.get("neptune.endpoint")
    region = cfg.get("project.region", "us-east-1")
    graph_prefix = prefix or base_prefix

    payload = {
        "source": {
            "s3BucketArn": f"arn:aws:s3:::{bucket}",
            "s3Key": graph_prefix,
        },
        "iamRoleArn": iam_role,
        "format": cfg.get("neptune.graph_format", "openCypher"),
        "region": region,
        "endpoint": endpoint,
        "mode": "NEW",
        "failOnError": True,
        "parallelism": cfg.get("neptune.max_concurrency", 2),
    }

    loader_cfg = cfg.get("neptune_loader", {})
    output_dir = ensure_dir(loader_cfg.get("output_dir", "configs"))
    filename_prefix = loader_cfg.get("file_name_prefix", "neptune_loader")
    safe_prefix = graph_prefix.strip("/").replace("/", "_") or "root"
    output_path = Path(output_dir) / f"{filename_prefix}_{safe_prefix}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    LOGGER.info("Wrote Neptune loader payload to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Neptune loader payload")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prefix")
    args = parser.parse_args()

    build_loader_payload(args.config, prefix=args.prefix)


if __name__ == "__main__":  # pragma: no cover
    main()




