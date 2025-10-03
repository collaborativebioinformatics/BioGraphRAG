from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urljoin

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
import requests

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def expand_subgraph(
    config_path: str,
    seed_nodes: List[str],
    hops: int = 2,
    max_degree: int = 10,
) -> List[Dict[str, Any]]:
    if not seed_nodes:
        LOGGER.debug("No seed nodes; skipping expansion")
        return []

    cfg = load_config(config_path)
    region = cfg.get("project.region", "us-east-1")
    neptune_cfg = cfg.get("neptune") or {}
    endpoint = neptune_cfg.get("endpoint")
    use_iam = bool(neptune_cfg.get("use_iam_auth", True))
    oc_path = neptune_cfg.get("opencypher_path", "/opencypher")
    id_prop = cfg.get("graph.id_property", "~id")
    prefixes = _coerce_prefixes(cfg.get("graph.label_prefixes", ["PRIME_", "PKG_"]))

    if not endpoint:
        raise ValueError("neptune.endpoint must be set in config")
    if not prefixes:
        raise ValueError("graph.label_prefixes must include at least one prefix")

    url = urljoin(endpoint.rstrip("/") + "/", oc_path.lstrip("/"))
    cypher = _build_query(prefixes, id_prop)
    payload = {
        "query": cypher,
        "parameters": {
            "ids": list(seed_nodes),
            "hops": int(hops),
            "maxDegree": int(max_degree),
        },
    }
    data = json.dumps(payload)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    LOGGER.debug("Posting openCypher query to Neptune", extra={"endpoint": url, "hops": hops})
    response = _execute_request(url, data, headers, use_iam, region)

    response.raise_for_status()
    return _parse_response(response.json(), id_prop, prefixes)


def _execute_request(
    url: str,
    data: str,
    headers: Dict[str, str],
    use_iam: bool,
    region: str,
) -> requests.Response:
    if not use_iam:
        return requests.post(url, data=data, headers=headers, timeout=30)

    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials not found for SigV4 signing")

    frozen = credentials.get_frozen_credentials()
    readonly = ReadOnlyCredentials(frozen.access_key, frozen.secret_key, frozen.token)
    request = AWSRequest(method="POST", url=url, data=data, headers=headers)
    SigV4Auth(readonly, "neptune-db", region).add_auth(request)
    prepared = requests.Request(
        "POST",
        url,
        data=data,
        headers=dict(request.headers),
    ).prepare()
    with requests.Session() as client:
        return client.send(prepared, timeout=30)


def _build_query(prefixes: List[str], id_prop: str) -> str:
    label_predicates = " OR ".join(
        f"ANY(label IN labels(n) WHERE label STARTS WITH '{p}')" for p in prefixes
    )
    rel_predicates = " OR ".join(
        f"type(r) STARTS WITH '{p}'" for p in prefixes
    )
    return "\n".join(
        [
            "MATCH (n)",
            f"WHERE n.`{id_prop}` IN $ids AND (" + label_predicates + ")",
            "MATCH p=(n)-[r*..$hops]-(m)",
            "WHERE ALL(r IN relationships(p) WHERE (" + rel_predicates + "))",
            "  AND ALL(r IN relationships(p) WHERE size((startNode(r))--()) <= $maxDegree)",
            "RETURN nodes(p) AS nodes, relationships(p) AS rels",
            "LIMIT 2000",
        ]
    )


def _parse_response(payload: Dict[str, Any], id_prop: str, prefixes: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError(f"Neptune returned errors: {errors}")

    results = payload.get("results") or []
    records: List[Dict[str, Any]] = []
    for row in results:
        nodes_raw = list(_extract_column(row, "nodes"))
        rels_raw = list(_extract_column(row, "rels"))
        if not nodes_raw and not rels_raw:
            continue

        nodes, node_lookup = _transform_nodes(nodes_raw, id_prop, prefixes)
        rels = [_to_rel_dict(rel_raw, node_lookup, id_prop) for rel_raw in rels_raw]
        records.append({"nodes": nodes, "rels": rels})
    return records


def _extract_column(row: Dict[str, Any], key: str) -> Iterable[Dict[str, Any]]:
    if not isinstance(row, dict):
        return []
    if key in row and isinstance(row[key], list):
        return row[key]

    bindings = row.get("bindings")
    if isinstance(bindings, dict):
        value = bindings.get(key)
        if isinstance(value, list):
            return value
    if isinstance(bindings, list):
        for binding in bindings:
            if isinstance(binding, dict) and isinstance(binding.get(key), list):
                return binding[key]

    row_values = row.get("row")
    if isinstance(row_values, list):
        index_map = {"nodes": 0, "rels": 1}
        idx = index_map.get(key)
        if idx is not None and idx < len(row_values):
            value = row_values[idx]
            if isinstance(value, list):
                return value
    return []


def _transform_nodes(
    nodes_raw: Iterable[Dict[str, Any]],
    id_prop: str,
    prefixes: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[Any, str]]:
    nodes: List[Dict[str, Any]] = []
    lookup: Dict[Any, str] = {}

    for raw in nodes_raw:
        converted, internal_id = _to_node_dict(raw, id_prop, prefixes)
        nodes.append(converted)
        if internal_id is None:
            continue
        key_raw = internal_id
        lookup[key_raw] = converted.get("~id")
        lookup[str(key_raw)] = converted.get("~id")
    return nodes, lookup


def _coerce_prefixes(prefixes: Any) -> List[str]:
    if isinstance(prefixes, (list, tuple)):
        return [str(p) for p in prefixes if p]
    if isinstance(prefixes, str) and prefixes:
        return [prefixes]
    return ["PRIME_", "PKG_"]


def _to_node_dict(
    node: Dict[str, Any],
    id_prop: str,
    prefixes: List[str],
) -> Tuple[Dict[str, Any], Any]:
    node = dict(node or {})
    properties = node.get("properties") or {}
    labels = node.get("labels") or node.get("label") or []
    if isinstance(labels, str):
        labels = [labels]

    out: Dict[str, Any] = dict(properties)
    logical_id = _select_logical_id(out, node, id_prop)
    out["~id"] = logical_id
    out["~label"] = _select_label(out.get("~label"), labels, prefixes)
    return out, node.get("id")


def _to_rel_dict(
    rel: Dict[str, Any],
    node_lookup: Dict[Any, str],
    id_prop: str,
) -> Dict[str, Any]:
    rel = dict(rel or {})
    properties = rel.get("properties") or {}

    out: Dict[str, Any] = dict(properties)
    out["~label"] = rel.get("type") or properties.get("~label")
    out["~from"] = _resolve_endpoint(rel.get("start") or rel.get("startNode"), node_lookup, id_prop)
    out["~to"] = _resolve_endpoint(rel.get("end") or rel.get("endNode"), node_lookup, id_prop)
    return out


def _resolve_endpoint(  # type: ignore[return-value]
    endpoint: Any,
    node_lookup: Dict[Any, str],
    id_prop: str,
) -> Any:
    if endpoint is None:
        return None
    if isinstance(endpoint, dict):
        properties = endpoint.get("properties") or {}
        candidate = properties.get(id_prop) or properties.get("~id")
        if candidate is not None:
            return str(candidate)
        candidate = endpoint.get(id_prop) or endpoint.get("~id")
        if candidate is not None:
            return str(candidate)
        endpoint = endpoint.get("id")

    if endpoint in node_lookup:
        return node_lookup[endpoint]
    if str(endpoint) in node_lookup:
        return node_lookup[str(endpoint)]
    return None


def _select_logical_id(
    properties: Dict[str, Any],
    node: Dict[str, Any],
    id_prop: str,
) -> Any:
    for candidate in (id_prop, "~id", ":ID", "id"):
        value = properties.get(candidate)
        if value is not None:
            return str(value)
        value = node.get(candidate)
        if value is not None:
            return str(value)
    return None


def _select_label(
    existing: Any,
    labels: Iterable[str],
    prefixes: List[str],
) -> Any:
    if isinstance(existing, str) and any(existing.startswith(p) for p in prefixes):
        return existing
    for label in labels:
        if isinstance(label, str) and any(label.startswith(p) for p in prefixes):
            return label
    return None


