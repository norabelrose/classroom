from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence
import csv
import numpy as np


class SyntheticPrefs(Sequence, ABC):
    """Base class for listwise and pairwise synthetic preference datasets."""
    def __init__(
            self,
            path: Path | str,
            *,
            beta: float = 5.0,
            mistake_prob: float = 0.1,
            normalize: bool = True,
            seed: int = 0,
            transform: Callable = lambda x: x
        ):
        assert beta > 0, f"Rationality parameter `beta` must be positive: {beta=}"
        assert 0 <= mistake_prob < 1, f"Mistake probability must be in [0, 1): {mistake_prob=}"

        self.beta = beta
        self.mistake_prob = mistake_prob
        self.path = Path(path)
        self.transform = transform
        self._prng = np.random.default_rng(seed)

        # Load the rewards from the CSV
        path = Path(path)
        with path.joinpath('rewards.csv').open() as f:
            reader = csv.reader(f)
            assert next(reader) == ['clip_id', 'reward'], "Rewards CSV must have header with 'clip_id' and 'reward'"

            rows = sorted(
                (int(id_str), float(reward_str))
                for id_str, reward_str in reader
            )
            clip_ids, rewards = zip(*rows)
            self.clip_ids = np.array(clip_ids)
        
        # Normalize the rewards to zero mean and unit variance
        if normalize:
            self.rewards: np.ndarray = (rewards - np.mean(rewards)) / np.std(rewards)
        
        # Now sample the preferences
        self.resample_prefs()
    
    def __repr__(self):
        return f"{type(self).__name__}(path='{self.path}', beta={self.beta}, mistake_prob={self.mistake_prob})"
    
    @abstractmethod
    def resample_prefs(self):
        pass
