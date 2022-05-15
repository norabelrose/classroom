from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from scipy.stats import rankdata
from typing import Any, Callable, Sequence, TypeVar
import numpy as np
import pickle


T = TypeVar("T")
@dataclass
class BatchedDataset(Sequence[T]):
    inner: Sequence[T]
    batch_size: int

    def __len__(self) -> int:
        return len(self.inner) // self.batch_size
    
    def __getitem__(self, index: int) -> list[T]:
        if index >= len(self):
            raise IndexError
        
        end = min(self.batch_size * (index + 1), len(self.inner))
        batch = [self.inner[i] for i in range(self.batch_size * index, end)]
        assert batch, f"Batch is empty: {len(self)=} {index=} {end=} {len(self.inner)=} {self.batch_size=}"
        return batch


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


class InMemoryClipDataset(Sequence):
    def __init__(
            self,
            path: Path | str,
            filter_fn: Callable = lambda _: True,
            transform: Callable = lambda x: x
        ):
        clip_path = Path(path) / 'clips'
        paths = sorted(clip_path.expanduser().glob('*.pkl'))

        self._clips = list(filter(filter_fn, (pickle.load(path.open('rb')) for path in paths)))
        # self._labels = rankdata([clip.state.reward.sum() for clip in self._clips]) / len(self)
        self._labels = [clip.state.reward.sum() for clip in self._clips]
        self._transform = transform

    def __getitem__(self, item: int | slice) -> tuple[Any, float]:
        raw_clip = self._clips[item]
        return self._transform(raw_clip), raw_clip.state.reward.sum()   # type: ignore

    def __iter__(self):
        """Iterate over the dataset, yielding a batch of (clip, reward) pairs."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of clips with estimated rewards in the dataset."""
        return len(self._clips)


class InMemoryPrefDataset(Sequence):
    def __init__(
            self,
            path: Path | str,
            filter_fn: Callable = lambda _: True,
            transform: Callable = lambda x: x
        ):
        clip_path = Path(path) / 'clips'
        paths = sorted(clip_path.expanduser().glob('*.pkl'))

        file_iter = (pickle.load(path.open('rb')) for path in paths)
        clip_iter = map(transform, filter(filter_fn, file_iter))
        self._pairs = list(combinations(clip_iter, 2))

    def __getitem__(self, item: int | slice) -> tuple[Any, float]:
        clip1, clip2 = self._pairs[item]
        rew1 = clip1.state.reward.sum() # type: ignore
        rew2 = clip2.state.reward.sum() # type: ignore
        return (clip1, clip2), float(rew1 > rew2)

    def __iter__(self):
        """Iterate over the dataset, yielding a batch of (clip, reward) pairs."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of distinct pairs in the dataset."""
        return len(self._pairs)
