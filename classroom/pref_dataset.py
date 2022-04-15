from .pref_dag import PrefDAG
from .pref_graph import PrefGraph
from typing import Any


class PrefDataset:
    """`PrefDataset` is a wrapper for a `PrefGraph` or `PrefDAG` which allows for
    easily sampling batches of preferences for use in training a reward model."""
    def __init__(self, graph: PrefGraph, closure: bool = True):
        """Create a new `PrefDataset` from a `PrefGraph` or `PrefDAG`. By default,
        the dataset will sample from the transitive closure of the graph if it is
        a DAG. If you want to sample from the original graph, pass `closure=False`."""
        if isinstance(graph, PrefDAG) and closure:
            graph = graph.transitive_closure()
        
        # Sort the edges by the first node value to ensure a deterministic order
        edge_iter: Any = graph.edges(data=True)
        self._edges: list[tuple[str, str, dict]] = sorted(edge_iter, key=lambda e: e[0])
    
    def __getitem__(self, idx: int):
        a, b, data = self._edges[idx]
        return a, b, data['weight'] or 0.5
    
    def __iter__(self):
        """Iterate over the dataset, yielding a batch of preferences."""
        return iter(self._edges)
    
    def __len__(self):
        return len(self._edges)