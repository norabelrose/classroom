from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Sequence
import csv
import numpy as np
import pickle


class SyntheticPrefs(Sequence, ABC):
    """Base class for listwise and pairwise synthetic preference datasets."""
    def __init__(
            self,
            path: Path | str,
            *,
            beta: float = 5.0,
            in_memory: bool = False,
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

        # Load the rewards from the CSV(s)
        all_rewards = []
        self.clip_paths: dict[int, Path] = {}

        for csv_path in self.path.rglob('rewards.csv'):
            clip_dir = csv_path.parent / 'clips'
            print(f"Loading rewards from {csv_path}")

            with csv_path.open() as f:
                reader = csv.reader(f)
                if next(reader) != ['clip_id', 'reward']:
                    raise ValueError("Rewards CSV must have header with 'clip_id' and 'reward'")

                rows = sorted(
                    (int(id_str), float(reward_str))
                    for id_str, reward_str in reader
                )
                clip_ids, rewards = zip(*rows)
                self.clip_paths.update(
                    (int(id_), clip_dir / f'{id_}.pkl')
                    for id_ in clip_ids
                )
                all_rewards.extend(rewards)
        
        self.clip_ids = np.array(list(self.clip_paths))
        
        # Normalize the rewards to zero mean and unit variance
        if normalize:
            self.rewards: np.ndarray = (all_rewards - np.mean(all_rewards)) / np.std(all_rewards)
        else:
            self.rewards = np.array(all_rewards)

        # Load all clips into memory if requested
        if in_memory:
            with Pool() as pool:
                self._clip_cache = dict(
                    pool.map(
                        partial(_load_clip, transform=self.transform),
                        self.clip_paths.items()
                    )
                )
        else:
            self._clip_cache = {}
        
        # Now sample the preferences
        self.resample_prefs()
    
    def clip_with_id(self, id_: int) -> Any:
        """Returns the clip with the given id."""
        clip = self._clip_cache.get(id_)
        if clip is not None:    # Cache hit
            return clip
        
        with open(self.path / 'clips' / f'{id_}.pkl', 'rb') as f:   # Cache miss
            return self.transform(pickle.load(f))
    
    def __repr__(self):
        return f"{type(self).__name__}(path='{self.path}', beta={self.beta}, mistake_prob={self.mistake_prob})"
    
    @abstractmethod
    def resample_prefs(self):
        pass


def _load_clip(item: tuple[int, Path], transform: Callable) -> tuple[int, Any]:
    clip_id, clip_path = item
    with clip_path.open('rb') as f:
        return clip_id, transform(pickle.load(f))
