"""Vector store helpers for dual PrimeKG + PubMedKG graphs.

Supports two backends selected via `vector_store.backend` in config:
- `opensearch` (default): uses OpenSearch KNN index via opensearch-py
- `faiss`: local FAISS index persisted to `paths.vector_index_path`
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Iterable, List, Tuple

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import faiss
import numpy as np
from pathlib import Path

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def _backend(cfg) -> str:
    backend = (cfg.get("vector_store.backend") or "opensearch").lower()
    if backend not in {"opensearch", "faiss"}:
        raise ValueError("vector_store.backend must be 'opensearch' or 'faiss'")
    return backend


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


def _faiss_paths(cfg) -> Tuple[Path, Path, Path]:
    index_path = Path(cfg.get("paths.vector_index_path", "data/embeddings/vector.index")).resolve()
    meta_ids_path = index_path.with_suffix(".ids.npy")
    meta_info_path = index_path.with_suffix(".meta.npy")
    return index_path, meta_ids_path, meta_info_path


def _faiss_load_or_init(cfg, dim: int) -> Tuple[faiss.IndexFlatIP, List[str], np.ndarray]:
    index_path, ids_path, meta_path = _faiss_paths(cfg)
    if index_path.exists() and ids_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        ids: List[str] = np.load(ids_path, allow_pickle=True).tolist()
        meta = np.load(meta_path, allow_pickle=True)
        return index, ids, meta
    index = faiss.IndexFlatIP(dim)
    return index, [], np.empty((0, 2), dtype=object)


def _faiss_save(cfg, index: faiss.IndexFlatIP, ids: List[str], meta: np.ndarray) -> None:
    index_path, ids_path, meta_path = _faiss_paths(cfg)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.array(ids, dtype=object), allow_pickle=True)
    np.save(meta_path, meta, allow_pickle=True)


def create_index(config_path: str) -> None:
    cfg = load_config(config_path)
    backend = _backend(cfg)
    dimension = cfg.get("open_search.embedding_dimension", 768)

    if backend == "opensearch":
        endpoint = cfg.get("open_search.endpoint")
        index_name = cfg.get("open_search.index_name")
        use_iam = bool(cfg.get("open_search.use_iam_auth", True))
        region = cfg.get("project.region", "us-east-1")
        service = cfg.get("open_search.service", "aoss")
        client = _client_with_auth(endpoint, use_iam, region, service)
        if client.indices.exists(index=index_name):
            LOGGER.info("Index %s already exists", index_name)
            return
        body = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "namespace": {"type": "keyword"},
                    "text": {"type": "text"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "nmslib"},
                    },
                }
            },
        }
        client.indices.create(index=index_name, body=body)
        LOGGER.info("Created OpenSearch index %s", index_name)
        return

    # faiss backend: nothing to create yet; file will be created on first upsert
    index, ids, meta = _faiss_load_or_init(cfg, dimension)
    _faiss_save(cfg, index, ids, meta)
    LOGGER.info("Initialized FAISS index at %s", _faiss_paths(cfg)[0])


def upsert_vectors(config_path: str, documents: Iterable[Dict]) -> None:
    cfg = load_config(config_path)
    backend = _backend(cfg)
    if backend == "opensearch":
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
        LOGGER.info("Upserted %s vectors (OpenSearch)", count)
        return

    # faiss
    dim = cfg.get("open_search.embedding_dimension", 768)
    index, ids, meta = _faiss_load_or_init(cfg, dim)
    vectors = []
    new_ids: List[str] = []
    new_meta_rows: List[Tuple[str, str]] = []
    for doc in documents:
        vec = np.asarray(doc["vector"], dtype="float32")
        # Normalize to cosine similarity via inner product
        norm = np.linalg.norm(vec) + 1e-12
        vec = vec / norm
        vectors.append(vec)
        new_ids.append(doc["id"])
        new_meta_rows.append((doc.get("type"), (doc.get("namespace") or "")))
    if not vectors:
        LOGGER.warning("No documents provided for upsert (FAISS)")
        return
    arr = np.vstack(vectors).astype("float32")
    index.add(arr)
    ids.extend(new_ids)
    if meta.size == 0:
        meta = np.array(new_meta_rows, dtype=object)
    else:
        meta = np.vstack([meta, np.array(new_meta_rows, dtype=object)])
    _faiss_save(cfg, index, ids, meta)
    LOGGER.info("Upserted %s vectors (FAISS)", len(new_ids))


def query_vectors(config_path: str, query_vector: List[float], top_k: int = 8) -> List[Dict]:
    cfg = load_config(config_path)
    backend = _backend(cfg)
    namespaces = cfg.get("retrieval.namespaces.include", [])
    if backend == "opensearch":
        endpoint = cfg.get("open_search.endpoint")
        index_name = cfg.get("open_search.index_name")
        use_iam = bool(cfg.get("open_search.use_iam_auth", True))
        region = cfg.get("project.region", "us-east-1")
        service = cfg.get("open_search.service", "aoss")
        client = _client_with_auth(endpoint, use_iam, region, service)
        query: Dict[str, Dict] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {"knn": {"vector": {"vector": query_vector, "k": top_k}}},
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

    # faiss
    dim = cfg.get("open_search.embedding_dimension", 768)
    index, ids, meta = _faiss_load_or_init(cfg, dim)
    if index.ntotal == 0:
        return []
    q = np.asarray(query_vector, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-12)
    q = q.reshape(1, -1)
    scores, ind = index.search(q, top_k)
    results: List[Dict] = []
    for score, idx in zip(scores[0], ind[0]):
        if idx == -1:
            continue
        entry = {
            "id": ids[idx],
            "type": str(meta[idx, 0]) if meta.size else None,
            "namespace": str(meta[idx, 1]) if meta.size else None,
            "score": float(score),
            "text": None,
        }
        if namespaces and entry["namespace"] not in namespaces:
            continue
        results.append(entry)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSearch vector store helper")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()

    if args.create:
        create_index(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()

