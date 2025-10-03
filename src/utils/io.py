"""I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    return file_path


def write_lines(lines: Iterable[str], path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
    return file_path
"""I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    return file_path


def write_lines(lines: Iterable[str], path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
    return file_path



