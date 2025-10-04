"""Configuration utilities for the dual bio graph rag project."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class AppConfig:
    raw: Dict[str, Any]

    def get(self, dotted_path: str, default: Optional[Any] = None) -> Any:
        node: Any = self.raw
        for part in dotted_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle) or {}

    env_prefix = "BIO_KG_"
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        dotted = key.removeprefix(env_prefix).lower().replace("__", ".")
        _assign(data, dotted, value)

    if overrides:
        for dotted, value in overrides.items():
            _assign(data, dotted, value)

    return AppConfig(raw=data)


def _assign(tree: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
