from ..tree_util import pytree_stack
from typing import Sequence
import numpy as np


class EnsembleDataset(Sequence):
    def __init__(self, inner: Sequence, num_copies: int, *, seed: int = 0):
        self.inner = inner
        self.num_copies = num_copies

        prng = np.random.default_rng(seed)
        self.indices = np.stack([
            prng.permutation(len(inner)) for _ in range(num_copies)
        ])
    
    def __getitem__(self, idx: int):
        return pytree_stack([self.inner[i] for i in self.indices[:, idx]])

    def __len__(self):
        return len(self.inner)
