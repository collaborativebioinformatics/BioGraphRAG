"""AWS helper utilities."""

from __future__ import annotations

import argparse
import logging

import boto3

from .config import load_config


LOGGER = logging.getLogger(__name__)


def ensure_budget(config_path: str, profile: str | None = None, limit: float = 100.0) -> None:
    cfg = load_config(config_path)
    region = cfg.get("project.region", "us-east-1")
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("budgets")
    account_id = session.client("sts").get_caller_identity()["Account"]

    try:
        client.create_budget(
            AccountId=account_id,
            Budget={
                "BudgetName": "PrimeKGGraphRAG",
                "BudgetLimit": {"Amount": str(limit), "Unit": "USD"},
                "TimeUnit": "MONTHLY",
                "BudgetType": "COST",
            },
        )
        LOGGER.info("Created AWS Budget PrimeKGGraphRAG")
    except client.exceptions.DuplicateRecordException:
        LOGGER.info("AWS Budget PrimeKGGraphRAG already exists")


def teardown_resources(config_path: str, profile: str | None = None) -> None:
    LOGGER.info("Teardown placeholder: stop Neptune cluster, delete indexes, clean S3 manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS helpers for PrimeKG GraphRAG")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ensure-budget", action="store_true")
    parser.add_argument("--teardown", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()

    if args.ensure_budget:
        ensure_budget(args.config, args.profile)
    if args.teardown:
        teardown_resources(args.config, args.profile)


if __name__ == "__main__":  # pragma: no cover
    main()
"""AWS helper utilities."""

from __future__ import annotations

import argparse
import logging

import boto3

from .config import load_config


LOGGER = logging.getLogger(__name__)


def ensure_budget(config_path: str, profile: str | None = None, limit: float = 100.0) -> None:
    cfg = load_config(config_path)
    region = cfg.get("project.region", "us-east-1")
    session = boto3.Session(profile_name=profile, region_name=region)
    budgets = session.client("budgets")
    account_id = session.client("sts").get_caller_identity()["Account"]

    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget={
                "BudgetName": "GraphRAGHackathon",
                "BudgetLimit": {"Amount": str(limit), "Unit": "USD"},
                "TimeUnit": "MONTHLY",
                "BudgetType": "COST",
            },
        )
        LOGGER.info("Created AWS budget GraphRAGHackathon")
    except budgets.exceptions.DuplicateRecordException:
        LOGGER.info("AWS budget GraphRAGHackathon already exists")


def teardown_resources(config_path: str, profile: str | None = None) -> None:
    LOGGER.info("Teardown placeholder: ensure Neptune/EC2/OpenSearch resources are stopped manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS helper CLI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ensure-budget", action="store_true")
    parser.add_argument("--teardown", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()

    if args.ensure_budget:
        ensure_budget(args.config, args.profile)
    if args.teardown:
        teardown_resources(args.config, args.profile)


if __name__ == "__main__":  # pragma: no cover
    main()



