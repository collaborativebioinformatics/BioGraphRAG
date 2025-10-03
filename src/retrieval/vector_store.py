"""OpenSearch vector index helpers for dual PrimeKG + PubMedKG graphs."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Iterable, List

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def _client_with_auth(endpoint: str, use_iam_auth: bool, region: str, service: str) -> OpenSearch:
    http_auth = None
    if use_iam_auth:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials not found for OpenSearch SigV4 auth")
        frozen = credentials.get_frozen_credentials()
        http_auth = AWS4Auth(frozen.access_key, frozen.secret_key, region, service, session_token=frozen.token)

    return OpenSearch(
        hosts=[endpoint],
        http_auth=http_auth,
        use_ssl=endpoint.startswith("https"),
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def create_index(config_path: str) -> None:
    cfg = load_config(config_path)
    endpoint = cfg.get("open_search.endpoint")
    index_name = cfg.get("open_search.index_name")
    dimension = cfg.get("open_search.embedding_dimension", 768)
    use_iam = bool(cfg.get("open_search.use_iam_auth", True))
    region = cfg.get("project.region", "us-east-1")
    service = cfg.get("open_search.service", "aoss")  # 'aoss' for Serverless, 'es' for managed

    client = _client_with_auth(endpoint, use_iam, region, service)
    if client.indices.exists(index=index_name):
        LOGGER.info("Index %s already exists", index_name)
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "namespace": {"type": "keyword"},
                "text": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    LOGGER.info("Created OpenSearch index %s", index_name)


def upsert_vectors(config_path: str, documents: Iterable[Dict]) -> None:
    cfg = load_config(config_path)
    endpoint = cfg.get("open_search.endpoint")
    index_name = cfg.get("open_search.index_name")
    use_iam = bool(cfg.get("open_search.use_iam_auth", True))
    region = cfg.get("project.region", "us-east-1")
    service = cfg.get("open_search.service", "aoss")
    client = _client_with_auth(endpoint, use_iam, region, service)

    actions = []
    count = 0
    for doc in documents:
        doc_type = doc.get("type") or ""
        namespace = doc.get("namespace") or doc_type.split("_")[0]
        document = {
            "id": doc["id"],
            "type": doc_type,
            "namespace": namespace,
            "text": doc.get("text"),
            "vector": doc["vector"],
        }
        actions.append({"index": {"_index": index_name, "_id": document["id"]}})
        actions.append(document)
        count += 1

    if not actions:
        LOGGER.warning("No documents provided for upsert")
        return

    bulk_body = "\n".join(json.dumps(action) for action in actions) + "\n"
    client.bulk(bulk_body)
    LOGGER.info("Upserted %s vectors", count)


def query_vectors(config_path: str, query_vector: List[float], top_k: int = 8) -> List[Dict]:
    cfg = load_config(config_path)
    endpoint = cfg.get("open_search.endpoint")
    index_name = cfg.get("open_search.index_name")
    namespaces = cfg.get("retrieval.namespaces.include", [])
    use_iam = bool(cfg.get("open_search.use_iam_auth", True))
    region = cfg.get("project.region", "us-east-1")
    service = cfg.get("open_search.service", "aoss")
    client = _client_with_auth(endpoint, use_iam, region, service)

    query: Dict[str, Dict] = {
        "size": top_k,
        "query": {
            "bool": {
                "must": {
                    "knn": {"vector": {"vector": query_vector, "k": top_k}}
                },
                "filter": [{"terms": {"namespace": namespaces}}] if namespaces else [],
            }
        },
    }

    response = client.search(index=index_name, body=query)
    hits = response.get("hits", {}).get("hits", [])
    return [
        {
            "id": hit["_source"]["id"],
            "type": hit["_source"].get("type"),
            "namespace": hit["_source"].get("namespace"),
            "score": hit.get("_score"),
            "text": hit["_source"].get("text"),
        }
        for hit in hits
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSearch vector store helper")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()

    if args.create:
        create_index(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()

