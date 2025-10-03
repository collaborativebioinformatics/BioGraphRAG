"""Base classes for dataset ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import pandas as pd

from src.utils.config import AppConfig
from src.utils.io import ensure_dir, write_dataframe


LOGGER = logging.getLogger(__name__)


@dataclass
class BaseIngestor:
    name: str
    config: AppConfig
    sample_size: int = 200
    output_format: str = "csv"
    local_root: Path = field(init=False)
    raw_dir: Path = field(init=False)
    staging_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        local_root = Path(self.config.get("paths.local_data_dir", "data/local"))
        self.local_root = ensure_dir(local_root / self.name)
        self.raw_dir = ensure_dir(self.local_root / "raw")
        self.staging_dir = ensure_dir(self.local_root / "processed")

    def run(self) -> Dict[str, Path]:
        outputs = self.ingest()
        files: Dict[str, Path] = {}
        for label, df in outputs.items():
            destination = self.staging_dir / f"{label}.{self.output_format}"
            LOGGER.info("Writing %s (%s rows)", destination, len(df))
            write_dataframe(df, destination, index=False)
            files[label] = destination
            self._maybe_upload_to_s3(destination)
        return files

    def ingest(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def _maybe_upload_to_s3(self, path: Path) -> None:
        bucket = self.config.get("s3.bucket")
        if not bucket or bucket.startswith("your-"):
            LOGGER.debug("Skipping S3 upload for %s; bucket not configured", path)
            return

        prefix = self.config.get("s3.prefixes.staging", "staging/")
        key = f"{prefix}{self.name}/{path.name}"

        try:
            boto3.client("s3").upload_file(str(path), bucket, key)
            LOGGER.info("Uploaded %s to s3://%s/%s", path, bucket, key)
        except (NoCredentialsError, BotoCoreError) as exc:
            LOGGER.warning("S3 upload skipped for %s (%s)", path, exc)
"""Base classes for dataset ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import pandas as pd

from src.utils.config import AppConfig
from src.utils.io import ensure_dir, write_dataframe


LOGGER = logging.getLogger(__name__)


@dataclass
class BaseIngestor:
    name: str
    config: AppConfig
    sample_size: int = 500
    output_format: str = "csv"
    local_root: Path = field(init=False)
    raw_dir: Path = field(init=False)
    staging_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        local_data_root = Path(self.config.get("paths.local_data_dir", "data/local"))
        self.local_root = ensure_dir(local_data_root / self.name)
        self.raw_dir = ensure_dir(self.local_root / "raw")
        self.staging_dir = ensure_dir(self.local_root / "processed")

    def run(self) -> Dict[str, Path]:
        outputs = self.ingest()
        written: Dict[str, Path] = {}
        for label, df in outputs.items():
            filename = f"{label}.{self.output_format}"
            destination = self.staging_dir / filename
            LOGGER.info("Writing %s (%s rows)", destination, len(df))
            write_dataframe(df, destination, index=False)
            written[label] = destination
            self._maybe_upload_to_s3(destination)
        return written

    def ingest(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def _maybe_upload_to_s3(self, path: Path) -> None:
        bucket = self.config.get("s3.bucket")
        if not bucket or bucket.startswith("your-"):
            LOGGER.debug("Skipping S3 upload for %s; bucket not configured", path)
            return

        prefix = self.config.get("s3.prefixes.staging", "staging/")
        key = f"{prefix}{self.name}/{path.name}"

        try:
            boto3.client("s3").upload_file(str(path), bucket, key)
            LOGGER.info("Uploaded %s to s3://%s/%s", path, bucket, key)
        except (NoCredentialsError, BotoCoreError) as exc:
            LOGGER.warning("S3 upload skipped for %s (%s)", path, exc)




