### Methods

#### Overview
We tackle biomedical question answering by pairing graph-native retrieval with neural reasoning. For every question, we retrieve a targeted slice of our knowledge graph and pass that evidence to a graph-aware neural reader. This design cuts hallucinations and keeps the system scalable for dense biomedical domains, aligning with the GraphRAG patterns described in the NVIDIA technical blog and the G-Retriever paper.

#### Knowledge Graph and Indexing
Our knowledge graph stores heterogeneous biomedical entities such as drugs, diseases, genes, and proteins, alongside richly typed relationships. Ingestion enforces schema-level guarantees (label-specific uniqueness, referential integrity) and attaches text embeddings to every node. We maintain both semantic indexes (vector similarity) and structural indexes (label/property) so that questions can be seeded semantically while traversals stay performant. Production graphs live in Amazon Neptune with supporting artifacts in Amazon S3, and configuration defaults are kept in `configs/default.yaml` with loaders under `src/ingest/`.

#### Retrieval and Base Subgraph Construction
Incoming questions are embedded with our text encoder. We identify the top-matching nodes via cosine similarity, then expand their one-hop neighborhoods inside Neptune using IAM-signed openCypher queries that enforce namespace prefixes and degree caps. The result is a base subgraph that balances recall against the combinatorial growth typical in dense biomedical graphs.

#### Prize Assignment and PCST Pruning
Within the base subgraph we assign a "prize" score to nodes and edges based on three cues: semantic similarity to the question, membership in the original seed set, and curated edge semantics. Traversal costs discourage overly large subgraphs. We then run a prize-collecting Steiner tree procedure that returns a compact, connected subgraph with high total prize. This pruning stage preserves multi-hop evidence while keeping the context manageable for downstream models.

#### Neural Reader: GNN + LLM Integration
The pruned subgraph passes through a PyTorch Geometric GATv1 encoder to produce node representations that capture multi-hop structure and textual attributes. We serialize the subgraph into an ordered, human-readable description (node names, descriptions, relation types) and feed it—alongside the question and a soft prompt derived from the GNN outputs—into an instruction-tuned large language model. The LLM remains frozen so we benefit from its language fluency while the GNN-derived prompt focuses attention on graph evidence.

#### Training Objectives
Supervision uses tuples of question, answer nodes, and source subgraphs. Training optimizes two losses jointly: a node-level loss that encourages the model to rank true answers above distractors, and a generation loss that compels the LLM to produce grounded natural-language answers conditioned on the serialized subgraph and soft prompt.

#### Inference
At inference we return the generated answer together with the top-ranked answer nodes extracted from the subgraph. When recall is critical, we append additional high-prize nodes from the pruned subgraph, effectively ensembling the retrieval and reasoning stages.

#### System Implementation
Retrieval and pruning logic resides in `src/retrieval/g_retriever.py` and `src/retrieval/expand.py`, while the graph-aware reader and reranking pipeline are implemented in `src/gnn/pyg_rag.py` and orchestrated via `src/rag/pipeline.py`. Neptune remains the source of truth for production graphs, with strict schema validation and uniqueness checks during ingest. Neptune loaders live in `src/ingest/`. Configuration, seeds, and embedding utilities are centralized in `configs/default.yaml`, `src/utils/seed.py`, and `src/embeddings/` respectively.

#### Reproducibility and Evaluation
We run with fixed random seeds, log every relevant hyperparameter, and evaluate on multi-hop biomedical QA benchmarks using metrics such as hits at K, recall, and mean reciprocal rank. Sensitivity analyses cover the number of retrieved seeds, expansion depth, prize schedules, and pruning strength, echoing the coupled hyperparameter behavior observed in GraphRAG literature.

#### References
- NVIDIA Technical Blog: Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases — `https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/`
- G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (arXiv:2402.07630) — `https://arxiv.org/pdf/2402.07630`



