from .pref_dag import PrefDAG
from .pref_graph import PrefGraph
from pathlib import Path
from typing import Any
import pickle


class PrefDataset:
    """`PrefDataset` is a wrapper for a `PrefGraph` or `PrefDAG` which allows for
    easily sampling batches of preferences for use in training a reward model."""
    def __init__(self, graph: PrefGraph, clip_dir: Path | str, closure: bool = True):
        """Create a new `PrefDataset` from a `PrefGraph` or `PrefDAG`. By default,
        the dataset will sample from the transitive closure of the graph if it is
        a DAG. If you want to sample from the original graph, pass `closure=False`."""
        if isinstance(graph, PrefDAG) and closure:
            graph = graph.transitive_closure()
        
        # Sort the edges by the first node value to ensure a deterministic order
        edge_iter: Any = graph.edges()
        self._clip_dir = Path(clip_dir)
        self._edges: list[tuple[str, str]] = sorted(edge_iter, key=lambda e: e[0])
        self._graph = graph
    
    def __getitem__(self, idx: int) -> tuple[Any, Any, float]:
        a, b = self._edges[idx]
        
        with open(self._clip_dir / f'{a}.pkl', 'rb') as f:
            a_clip = pickle.load(f)
        
        with open(self._clip_dir / f'{b}.pkl', 'rb') as f:
            b_clip = pickle.load(f)
        
        return a_clip, b_clip, self._graph.pref_prob(a, b)
    
    def __iter__(self):
        """Iterate over the dataset, yielding a batch of preferences."""
        for i in range(len(self)):
            yield self[i]
    
    def __len__(self):
        return len(self._edges)