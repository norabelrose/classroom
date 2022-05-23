from .synthetic_prefs import SyntheticPrefs
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
from typing import Callable
import numpy as np
import pickle


class SyntheticListwisePrefs(SyntheticPrefs):
    """
    Creates a dataset of synthetic listwise preferences between clips generated
    stochastically from the environment reward. The env rewards must be stored
    in a CSV named `rewards.csv` in the dataset directory. Notably, unlike
    `SyntheticPairwisePrefs`, this dataset is automatically batched and requires
    a `batch_size` parameter at initialization.
    """
    def __init__(
            self, path: Path | str, batch_size: int, *,
            beta: float = 5.0, mistake_prob: float = 0.01, normalize: bool = True, seed: int = 0, transform: Callable = lambda x: x
        ):
        self.batch_size = batch_size
        super().__init__(path, beta=beta, mistake_prob=mistake_prob, normalize=normalize, seed=seed, transform=transform)
    
    def max_kendalltau(self) -> float:
        """
        Returns a theoretical upper bound on the rank correlation achievable by a preference model trained on this dataset,
        based on the level of aleatoric uncertainty in the dataset.
        """
        return np.mean([
            kendalltau(noiseless, noisy).correlation
            for noiseless, noisy in zip(self._noiseless_ranks.T, self._noisy_ranks.T)
        ])
    
    def max_spearmanr(self) -> float:
        """
        Returns a theoretical upper bound on the rank correlation achievable by a preference model trained on this dataset,
        based on the level of aleatoric uncertainty in the dataset.
        """
        return np.mean([
            spearmanr(noiseless, noisy).correlation
            for noiseless, noisy in zip(self._noiseless_ranks.T, self._noisy_ranks.T)
        ])
    
    def resample_prefs(self):
        """Sample a new set of batches as well as new rankings for each batch."""
        # Truncate the dataset so that it's evenly divisible by the batch size
        perm = self._prng.permutation(len(self.clip_ids))
        num_clips = (len(perm) // self.batch_size) * self.batch_size

        # Shuffle & batch the dataset
        clip_batches = self.clip_ids[perm][:num_clips].reshape(-1, self.batch_size)
        score_batches = self.beta * self.rewards[perm][:num_clips].reshape(-1, self.batch_size)
        self._noiseless_batches = np.take_along_axis(clip_batches, np.argsort(-score_batches), axis=1)

        self._noiseless_ranks = np.zeros_like(self._noiseless_batches)
        np.put_along_axis(self._noiseless_ranks, np.argsort(-score_batches), np.arange(self.batch_size), axis=1)

        # Simulate the Plackett-Luce random utility model
        unnormalized = np.exp(score_batches)
        prob_batches = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        prob_batches = self.mistake_prob / self.batch_size + (1 - self.mistake_prob) * prob_batches

        # np.random.Generator.choice() only supports 1D prob vectors so we need this for loop
        self._noisy_batches = np.zeros_like(self._noiseless_batches)
        self._noisy_ranks = np.zeros_like(self._noiseless_batches)
        for i, (clips, probs) in enumerate(zip(clip_batches, prob_batches)):
            indices = self._prng.choice(self.batch_size, p=probs, replace=False, size=self.batch_size)
            self._noisy_batches[i] = clips[indices]
            self._noisy_ranks[i, indices] = np.arange(self.batch_size)

    def __getitem__(self, item: int) -> np.ndarray:
        return np.stack([
            self.transform(
                pickle.load(
                    open(self.path / 'clips' / f'{clip_id}.pkl', 'rb')
                )
            )
            for clip_id in self._noisy_batches[item]
        ])

    def __len__(self):
        """Number of distinct batches in the dataset."""
        return len(self._noisy_batches)
