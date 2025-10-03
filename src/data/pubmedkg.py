"""PubMedKG ingest utilities for raw dataset extracts."""

from __future__ import annotations

import argparse
import logging
from typing import Dict

import pandas as pd

from .base import BaseIngestor
from src.utils.config import load_config
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)

README = """\
Usage: place PubMedKG TSV/CSV extracts under
`paths.local_data_dir/pubmedkg/processed`. Files must include
`pubmed_publications.csv`, `pubmed_mentions.csv`, and `pubmed_citations.csv`.
The ingestor will fail fast if any file is missing or empty.
"""


class PubMedKgIngestor(BaseIngestor):
    REQUIRED_FILES = {
        "pubmed_publications": "pubmed_publications.csv",
        "pubmed_mentions": "pubmed_mentions.csv",
        "pubmed_citations": "pubmed_citations.csv",
    }

    def ingest(self) -> Dict[str, pd.DataFrame]:
        seed_everything()

        data = {}
        for label, filename in self.REQUIRED_FILES.items():
            path = self.staging_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"PubMedKG ingest missing required file: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"PubMedKG ingest found empty file: {path}")
            LOGGER.info("Loaded %s rows from %s", len(df), path)
            data[label] = df

        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PubMedKG extracts")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ingestor = PubMedKgIngestor(name="pubmedkg", config=cfg)
    ingestor.run()


if __name__ == "__main__":  # pragma: no cover
    print(README)
    main()




