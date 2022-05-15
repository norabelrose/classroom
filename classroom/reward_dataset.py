from .pref_graph import PrefGraph
from pathlib import Path
from typing import Any
import pickle


class RewardDataset:
    """
    `RewardDataset` is a wrapper for a `PrefGraph` or `PrefDAG` which allows for
    easily sampling batches of clips and their associated MLE rewards for use in
    training a reward model using a regression loss.
    """
    def __init__(self, path: Path | str):
        self.clip_path = Path(path).parent / 'clips'

        with open(path, 'rb') as f:
            graph = pickle.load(f)
            assert isinstance(graph, PrefGraph)

        node_iter: Any = graph.nodes(data='reward') # type: ignore
        self._nodes: list[tuple[str, float]] = sorted(node_iter, key=lambda e: e[0])

    def __getitem__(self, idx: int):
        node_id, reward = self._nodes[idx]
        
        with open(self.clip_path / f'{node_id}.pkl', 'rb') as f:
            clip = pickle.load(f)
        
        return clip, reward

    def __iter__(self):
        """Iterate over the dataset, yielding a batch of (clip, reward) pairs."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of clips with estimated rewards in the dataset."""
        return len(self._nodes)
