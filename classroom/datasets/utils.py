from ..tree_util import pytree_stack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar
import numpy as np
import pickle


T = TypeVar("T")
@dataclass
class BatchedDataset(Sequence[T]):
    inner: Sequence[T]
    batch_size: int

    def __len__(self) -> int:
        return len(self.inner) // self.batch_size
    
    def __getitem__(self, index: int) -> T:
        if index >= len(self):
            raise IndexError
        
        end = min(self.batch_size * (index + 1), len(self.inner))
        batch = pytree_stack([self.inner[i] for i in range(self.batch_size * index, end)])
        assert batch
        return batch    # type: ignore


@dataclass
class SubsetDataset(Sequence[T]):
    inner: Sequence[T]
    indices: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, index: int) -> T:
        return self.inner[self.indices[index]]


class ClipDataset(Sequence):
    """`ClipDataset` lazily loads clips from the disk."""
    def __init__(self, path: Path | str, transform: Callable = lambda x: x):
        clip_path = Path(path) / 'clips'
        self._clips = sorted(clip_path.expanduser().glob('*.pkl'))
        self._transform = transform

    def __getitem__(self, idx: int):
        with open(self._clips[idx], 'rb') as f:
            return self._transform(pickle.load(f))

    def __iter__(self):
        """Iterate over the dataset, yielding a batch of (clip, reward) pairs."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of clips with estimated rewards in the dataset."""
        return len(self._clips)
