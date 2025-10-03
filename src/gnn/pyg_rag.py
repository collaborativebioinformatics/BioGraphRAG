"""PyTorch Geometric utilities for GraphRAG fact ranking.

Inspired by NVIDIA GraphRAG and the PyG ``txt2kg_rag`` example, this module
provides helper functions to build PyTorch Geometric graphs from the retrieved
subgraph, compute structural/textual embeddings, and rank facts for LLM prompts.
The code is kept modular so higher-level pipelines can call only the pieces they
need (e.g., structural embeddings, text fusion, ranking).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, DefaultDict

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Node2Vec, SAGEConv
from torch_geometric.utils import to_undirected


_FACT_LOOKUP: List[Tuple[str, int]] = []
_STRUCTURAL_VECS: Optional[np.ndarray] = None


def build_pyg_from_subgraph(nodes: List[Dict], edges: List[Dict]) -> Tuple[Data, Dict[str, int]]:
    """Create a homogeneous PyG ``Data`` graph from dual-KG nodes/edges."""
    if not nodes:
        empty = Data()
        empty.x = torch.empty((0, 3), dtype=torch.float)
        empty.edge_index = torch.empty((2, 0), dtype=torch.long)
        return empty, {}

    node_index: Dict[str, int] = {}
    features: List[List[float]] = []

    sorted_nodes = sorted(nodes, key=lambda item: item.get("~id", ""))
    for idx, node in enumerate(sorted_nodes):
        node_id = node.get("~id")
        if node_id is None:
            continue
        node_index[node_id] = idx

    in_degree = [0] * len(node_index)
    out_degree = [0] * len(node_index)
    edge_sources: List[int] = []
    edge_targets: List[int] = []

    for rel in edges:
        start = rel.get("~from")
        end = rel.get("~to")
        start_idx = node_index.get(start)
        end_idx = node_index.get(end)
        if start_idx is None or end_idx is None or start_idx == end_idx:
            continue
        out_degree[start_idx] += 1
        in_degree[end_idx] += 1
        edge_sources.extend([start_idx, end_idx])
        edge_targets.extend([end_idx, start_idx])

    for node in sorted_nodes:
        node_id = node.get("~id")
        if node_id is None:
            continue
        idx = node_index[node_id]
        label = node.get("~label", "") or ""
        namespace_id = _namespace_id(label)
        features.append([float(in_degree[idx]), float(out_degree[idx]), float(namespace_id)])

    data = Data()
    data.x = torch.tensor(features, dtype=torch.float)
    if edge_sources:
        raw_edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        data.edge_index = to_undirected(raw_edge_index)
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
    return data, node_index


def build_pyg_from_subgraph_rich(nodes: List[Dict], edges: List[Dict]) -> Tuple[Data, Dict[str, int]]:
    """Create a homogeneous PyG ``Data`` graph with richer node/edge features.

    Node features include: in_degree, out_degree, namespace_id, one-hot node type,
    and common biomedical scalar attributes if present (confidence/publication_count).
    Edge features include: one-hot edge type plus optional weights/confidence.
    """
    if not nodes:
        empty = Data()
        empty.x = torch.empty((0, 3), dtype=torch.float)
        empty.edge_index = torch.empty((2, 0), dtype=torch.long)
        return empty, {}

    # Build label vocabularies local to this subgraph
    node_labels: List[str] = []
    edge_labels: List[str] = []
    for n in nodes:
        node_labels.append((n.get("~label") or "unknown"))
    for e in edges:
        edge_labels.append((e.get("~label") or "unknown"))
    node_label_to_idx = {lbl: i for i, lbl in enumerate(sorted(set(node_labels)))}
    edge_label_to_idx = {lbl: i for i, lbl in enumerate(sorted(set(edge_labels)))}

    # Index nodes and accumulators
    node_index: Dict[str, int] = {}
    sorted_nodes = sorted(nodes, key=lambda item: item.get("~id", ""))
    for idx, node in enumerate(sorted_nodes):
        node_id = node.get("~id")
        if node_id is None:
            continue
        node_index[node_id] = idx

    in_degree = [0] * len(node_index)
    out_degree = [0] * len(node_index)
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    edge_attr_rows: List[List[float]] = []

    # Build edge index and edge attributes
    for rel in edges:
        start = rel.get("~from")
        end = rel.get("~to")
        start_idx = node_index.get(start)
        end_idx = node_index.get(end)
        if start_idx is None or end_idx is None or start_idx == end_idx:
            continue
        out_degree[start_idx] += 1
        in_degree[end_idx] += 1
        # Undirected twin edges
        edge_sources.extend([start_idx, end_idx])
        edge_targets.extend([end_idx, start_idx])

        # Edge features (one row per directed edge we add)
        etype = rel.get("~label", "unknown")
        edge_type_vec = _one_hot(edge_label_to_idx.get(etype, 0), len(edge_label_to_idx))
        weight = float(rel.get("weight", rel.get("confidence", 1.0)) or 1.0)
        base_feat = edge_type_vec + [weight]
        # Push for both directions to match twin edges
        edge_attr_rows.append(base_feat)
        edge_attr_rows.append(base_feat)

    # Node features
    features: List[List[float]] = []
    for node in sorted_nodes:
        node_id = node.get("~id")
        if node_id is None:
            continue
        idx = node_index[node_id]
        label = node.get("~label", "") or "unknown"
        namespace_id = _namespace_id(label)
        node_type_vec = _one_hot(node_label_to_idx.get(label, 0), len(node_label_to_idx))
        # Optional biomedical scalars
        confidence = float(node.get("confidence_score", 0.0) or 0.0)
        pub_count = float(node.get("publication_count", 0.0) or 0.0)
        # Assemble feature vector
        base = [float(in_degree[idx]), float(out_degree[idx]), float(namespace_id)]
        features.append(base + node_type_vec + [confidence, pub_count])

    data = Data()
    data.x = torch.tensor(features, dtype=torch.float) if features else torch.empty((0, 3), dtype=torch.float)
    if edge_sources:
        raw_edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        data.edge_index = to_undirected(raw_edge_index)
        if edge_attr_rows:
            data.edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 0), dtype=torch.float)
    return data, node_index


def build_hetero_pyg_from_subgraph(nodes: List[Dict], edges: List[Dict]) -> HeteroData:
    """Create a ``HeteroData`` graph preserving node/edge types.

    This is optional and useful when downstream models exploit heterogeneity.
    The function constructs per-type node feature matrices using degree,
    namespace, and a per-type one-hot of the local label vocabulary.
    Edge stores contain ``edge_index`` and per-edge one-hot type plus weight.
    """
    data = HeteroData()
    if not nodes:
        return data

    # Group nodes by label
    by_label: DefaultDict[str, List[Dict]] = DefaultDict(list)
    for n in nodes:
        by_label[(n.get("~label") or "unknown")].append(n)

    # Create local indices across all nodes to compute degrees
    # and to map ids back during edge creation
    _, global_index = build_pyg_from_subgraph(nodes, edges)

    # Build per-type node features
    all_node_labels = sorted(by_label.keys())
    for lbl in all_node_labels:
        type_nodes = sorted(by_label[lbl], key=lambda item: item.get("~id", ""))
        x_rows: List[List[float]] = []
        node_ids: List[str] = []
        for node in type_nodes:
            node_id = node.get("~id")
            if node_id is None:
                continue
            node_ids.append(node_id)
            idx = global_index.get(node_id, -1)
            indeg = 0.0
            outdeg = 0.0
            if idx >= 0:
                # Degrees will be recomputed per-type as zeros; use namespace + bias
                # for simple, robust features across types
                pass
            namespace_id = _namespace_id(lbl)
            # Per-type one-hot among all node labels
            type_vec = _one_hot(all_node_labels.index(lbl), len(all_node_labels))
            x_rows.append([indeg, outdeg, float(namespace_id)] + type_vec)

        if x_rows:
            data[lbl].x = torch.tensor(x_rows, dtype=torch.float)
            data[lbl].node_ids = node_ids

    # Build edges grouped by (src_type, rel, dst_type)
    def find_type_and_idx(node_id: str) -> Tuple[str, int]:
        for t in all_node_labels:
            ids: List[str] = getattr(data[t], "node_ids", [])
            try:
                return t, ids.index(node_id)
            except ValueError:
                continue
        return "", -1

    rel_groups: DefaultDict[Tuple[str, str, str], List[Dict]] = DefaultDict(list)
    for e in edges:
        s_id = e.get("~from")
        t_id = e.get("~to")
        rel = (e.get("~label") or "rel")
        s_type, _ = find_type_and_idx(s_id)
        t_type, _ = find_type_and_idx(t_id)
        if s_type and t_type:
            rel_groups[(s_type, rel, t_type)].append(e)

    for (s_type, rel, t_type), rel_edges in rel_groups.items():
        edge_index_rows: List[List[int]] = []
        edge_attr_rows: List[List[float]] = []
        # One-hot over the relations present in this bipartite store
        # (usually length 1, but keep general)
        rel_vocab = {rel: 0}
        for e in rel_edges:
            s_id = e.get("~from")
            t_id = e.get("~to")
            _, s_idx = find_type_and_idx(s_id)
            _, t_idx = find_type_and_idx(t_id)
            if s_idx < 0 or t_idx < 0:
                continue
            edge_index_rows.append([s_idx, t_idx])
            weight = float(e.get("weight", e.get("confidence", 1.0)) or 1.0)
            edge_attr_rows.append(_one_hot(rel_vocab[rel], len(rel_vocab)) + [weight])
        if edge_index_rows:
            data[(s_type, rel, t_type)].edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
            data[(s_type, rel, t_type)].edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)
    return data


def gnn_graph_embedding(data: Data, cfg) -> List[float]:
    """Train a lightweight structural encoder and return the pooled graph vector."""
    dim = int(cfg.get("pyg_rag.dim", 64))
    epochs = max(1, int(cfg.get("pyg_rag.epochs", 1)))
    learning_rate = float(cfg.get("pyg_rag.learning_rate", 0.01))
    walk_length = int(cfg.get("pyg_rag.walk_length", 20))
    context_size = int(cfg.get("pyg_rag.context_size", 10))
    walks_per_node = int(cfg.get("pyg_rag.walks_per_node", 5))

    if data.x is None or data.x.numel() == 0:
        num_nodes = data.num_nodes if data.num_nodes is not None else 0
        data.x = torch.ones((num_nodes, 3), dtype=torch.float)
    if data.num_nodes == 0:
        return [0.0] * dim

    model_choice = (cfg.get("pyg_rag.model", "node2vec") or "node2vec").lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_choice == "graphsage":
        return _graphsage_embedding(data, device, dim, epochs, learning_rate)

    return _node2vec_embedding(
        data,
        device,
        dim,
        epochs,
        learning_rate,
        walk_length,
        context_size,
        walks_per_node,
    )


def textify_subgraph(nodes: List[Dict], edges: List[Dict]) -> List[str]:
    """Convert graph data into human-readable fact strings with lookup bookkeeping."""
    global _FACT_LOOKUP
    texts: List[str] = []
    _FACT_LOOKUP = []

    sorted_nodes = sorted(
        list(enumerate(nodes)),
        key=lambda item: item[1].get("~id", ""),
    )
    for idx, node in sorted_nodes:
        node_id = node.get("~id", "unknown")
        label = node.get("~label", "node")
        details = _format_properties(node, skip_keys={"~id", "~label"})
        text = f"{label}({node_id}) {details}".strip()
        texts.append(text)
        _FACT_LOOKUP.append(("node", idx))

    sorted_edges = sorted(
        list(enumerate(edges)),
        key=lambda item: (
            item[1].get("~label", ""),
            item[1].get("~from", ""),
            item[1].get("~to", ""),
        ),
    )
    for idx, rel in sorted_edges:
        rel_label = rel.get("~label", "rel")
        start = rel.get("~from", "")
        end = rel.get("~to", "")
        details = _format_properties(rel, skip_keys={"~label", "~from", "~to"})
        text = f"{rel_label}({start}->{end}) {details}".strip()
        texts.append(text)
        _FACT_LOOKUP.append(("edge", idx))

    return texts


def encode_texts(texts: List[str], cfg) -> np.ndarray:
    """Return SentenceTransformer embeddings for the provided texts."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model_name = cfg.get("embedding_model.document_model", "sentence-transformers/all-MiniLM-L12-v2")
    model = _load_sentence_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(embeddings, dtype=np.float32)


def fuse_vectors(gnn_vec: np.ndarray, llm_vec: np.ndarray) -> np.ndarray:
    """L2-normalise graph and text vectors before concatenation."""
    gnn = _safe_normalize_vector(gnn_vec)
    llm = _safe_normalize_vector(llm_vec)
    return np.concatenate([llm, gnn], axis=0)


def rank_facts(fused_vec: np.ndarray, fact_texts: List[str], cfg) -> List[int]:
    """Order facts by cosine similarity against the fused query vector."""
    if fused_vec.size == 0 or not fact_texts:
        return list(range(len(fact_texts)))

    text_matrix = encode_texts(fact_texts, cfg)
    if text_matrix.size == 0:
        return list(range(len(fact_texts)))

    llm_dim = text_matrix.shape[1]
    fused_unit = _safe_normalize_vector(fused_vec)
    text_part = fused_unit[:llm_dim]

    struct_vectors = _STRUCTURAL_VECS
    if struct_vectors is not None and struct_vectors.shape[0] == len(fact_texts):
        struct_vectors = _normalize_rows(struct_vectors)
        gnn_part = fused_unit[llm_dim:]
        if gnn_part.size:
            gnn_part = _safe_normalize_vector(gnn_part)
            fact_matrix = np.concatenate([_normalize_rows(text_matrix), struct_vectors], axis=1)
            fused_final = np.concatenate([text_part, gnn_part], axis=0)
        else:
            fact_matrix = _normalize_rows(text_matrix)
            fused_final = text_part
    else:
        fact_matrix = _normalize_rows(text_matrix)
        fused_final = text_part

    scores = fact_matrix @ fused_final
    return np.argsort(scores)[::-1].tolist()


def structural_fact_vectors(
    nodes: List[Dict],
    edges: List[Dict],
    node_embeddings: Optional[np.ndarray],
    node_index: Dict[str, int],
) -> Optional[np.ndarray]:
    """Return structural vectors aligned with ``fact_texts`` order."""
    if node_embeddings is None:
        return None

    embeddings = np.asarray(node_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        return None

    lookup = _FACT_LOOKUP
    if not lookup:
        return None

    fact_vectors = np.zeros((len(lookup), embeddings.shape[1]), dtype=np.float32)

    for row, (kind, ref) in enumerate(lookup):
        if kind == "node":
            if 0 <= ref < len(nodes):
                node = nodes[ref]
                node_id = node.get("~id")
                if node_id is not None:
                    emb_idx = node_index.get(node_id)
                    if emb_idx is not None and emb_idx < embeddings.shape[0]:
                        fact_vectors[row] = embeddings[emb_idx]
        elif kind == "edge":
            if 0 <= ref < len(edges):
                rel = edges[ref]
                start_idx = node_index.get(rel.get("~from"))
                end_idx = node_index.get(rel.get("~to"))
                vectors: List[np.ndarray] = []
                if start_idx is not None and start_idx < embeddings.shape[0]:
                    vectors.append(embeddings[start_idx])
                if end_idx is not None and end_idx < embeddings.shape[0]:
                    vectors.append(embeddings[end_idx])
                if vectors:
                    fact_vectors[row] = np.mean(vectors, axis=0)

    return fact_vectors


def get_fact_lookup() -> List[Tuple[str, int]]:
    """Expose the mapping used to tie facts back to their nodes/edges."""
    return list(_FACT_LOOKUP)


def set_structural_vectors(vectors: Optional[np.ndarray]) -> None:
    """Store the latest structural vectors for access during ranking."""
    global _STRUCTURAL_VECS
    _STRUCTURAL_VECS = None if vectors is None else np.asarray(vectors, dtype=np.float32)


def _node2vec_embedding(
    data: Data,
    device: torch.device,
    dim: int,
    epochs: int,
    learning_rate: float,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
) -> List[float]:
    """Learn Node2Vec embeddings and return their mean vector."""
    if data.num_edges == 0 or data.edge_index.numel() == 0:
        node_count = data.num_nodes if data.num_nodes is not None else data.x.size(0)
        zeros = np.zeros((node_count, dim), dtype=np.float32)
        data.node_embeddings = torch.from_numpy(zeros)
        return zeros.mean(axis=0).tolist() if node_count else [0.0] * dim

    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        sparse=True,
    ).to(device)

    batch_size = min(256, max(1, data.num_nodes))
    loader = model.loader(batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        embeddings = model.embedding.weight.detach().cpu()
    data.node_embeddings = embeddings
    graph_vec = embeddings.mean(dim=0)
    return graph_vec.numpy().tolist()


def _graphsage_embedding(data: Data, device: torch.device, dim: int, epochs: int, learning_rate: float) -> List[float]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    if edge_index.numel() == 0:
        node_count = data.num_nodes if data.num_nodes is not None else x.size(0)
        zeros = np.zeros((node_count, dim), dtype=np.float32)
        data.node_embeddings = torch.from_numpy(zeros)
        return zeros.mean(axis=0).tolist() if node_count else [0.0] * dim

    model = _GraphSAGEEncoder(x.size(1), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(out, x)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        embeddings = model(x, edge_index).detach().cpu()
    data.node_embeddings = embeddings
    graph_vec = embeddings.mean(dim=0)
    return graph_vec.numpy().tolist()


class _GraphSAGEEncoder(torch.nn.Module):
    """Two-layer mean GraphSAGE encoder with ReLU between layers."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        mid_channels = max(hidden_channels, in_channels)
        self.conv1 = SAGEConv(in_channels, mid_channels)
        self.conv2 = SAGEConv(mid_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def _safe_normalize_vector(vector: Iterable[float]) -> np.ndarray:
    """Safely L2-normalise a vector, guarding against zero norms."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def _one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * max(1, size)
    if 0 <= index < max(1, size):
        vec[index] = 1.0
    return vec


def _namespace_id(label: str) -> int:
    """Map namespace prefixes to integer ids for structural features."""
    if label.startswith("PRIME_"):
        return 0
    if label.startswith("PKG_"):
        return 1
    return 2


def _format_properties(item: Dict, skip_keys: set[str]) -> str:
    """Format node/edge properties as ``key: value`` pairs."""
    parts: List[str] = []
    for key in sorted(item.keys()):
        if key in skip_keys:
            continue
        value = item[key]
        if value in (None, "", []):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str) -> SentenceTransformer:
    """Cache and return a SentenceTransformer model."""
    return SentenceTransformer(model_name)
