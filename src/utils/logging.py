"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path / "app.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path / "app.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)



