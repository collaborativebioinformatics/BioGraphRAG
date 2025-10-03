"""Legacy Neo4j expander placeholder."""


def expand_subgraph(*args, **kwargs):
    raise RuntimeError("Neo4j backend removed. Set graph.backend: neptune in configs/default.yaml.")

