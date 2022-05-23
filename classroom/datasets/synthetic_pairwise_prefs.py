from .synthetic_prefs import SyntheticPrefs
from scipy.special import expit
from typing import Any
import numpy as np


class SyntheticPairwisePrefs(SyntheticPrefs):
    """
    Creates a dataset of synthetic pairwise preferences between clips generated
    stochastically from the environment reward. The env rewards must be stored
    in a CSV named `rewards.csv` in the dataset directory.
    """    
    def max_accuracy(self) -> float:
        """
        Returns a theoretical upper bound on the accuracy achievable by a preference model trained on this dataset,
        based on the level of aleatoric uncertainty in the dataset.
        """
        return np.mean(self._noiseless_prefs == self._noisy_prefs)
    
    def resample_prefs(self):
        """Sample a new set of clip pairs as well as new preferences for each pair."""
        # Sample a random set of N pairs, where clips are sampled without replacement
        perm = self._prng.permutation(len(self.clip_ids))
        self._pairs = self.clip_ids[perm].reshape(-1, 2)
        
        # Compute preference probabilities for each pair
        reward_pairs = self.rewards[perm].reshape(-1, 2)
        scores = self.beta * (reward_pairs[:, 0] - reward_pairs[:, 1])

        # There's a `mistake_prob` chance of picking a clip uniformly at random
        pref_probs = self.mistake_prob * 0.5 + (1 - self.mistake_prob) * expit(scores)
        self._noiseless_prefs = reward_pairs[:, 0] > reward_pairs[:, 1]
        self._noisy_prefs = self._prng.uniform(size=len(self._pairs)) < pref_probs

    def __getitem__(self, item: int) -> tuple[Any, Any, np.ndarray]:
        id_a, id_b = self._pairs[item]
        clip_a = self.clip_with_id(id_a)
        clip_b = self.clip_with_id(id_b)

        return clip_a, clip_b, self._noisy_prefs[item]

    def __len__(self):
        """Number of distinct pairs in the dataset."""
        return len(self._noisy_prefs)
