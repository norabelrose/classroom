from dataclasses import dataclass
from numpy import dtype, bool_, floating, integer
from typing import Any, TypeVar
import numpy as np


# Prefs are integer NumPy arrays, rewards are floating
PrefTensor = np.ndarray[Any, dtype[bool_]]
RewardTensor = np.ndarray[Any, dtype[floating]]

@dataclass
class SimTeacher:
    """
    Generates simulated human feedback by stochastically sampling preferences
    from a specified parametric reward function.
    """
    mistake_prob: float = 0.1
    temperature: float = 1.0

    def sample(self, rewards: RewardTensor) -> PrefTensor:
        """Given a tensor of rewards/returns"""
        diffs = rewards - rewards[..., None]
        probs = 1 / (1 + np.exp(diffs / self.temperature))

        return np.random.uniform(size=rewards.shape) < probs
