from numpy.typing import ArrayLike
from scipy.stats import kendalltau, spearmanr
from typing import Literal
import numpy as np


class RankCorrelation:
    """Convenience class for computing the rank correlation of a model's predictions."""
    def __init__(self, corr_type: Literal['kendalltau', 'spearmanr'] = 'kendalltau'):
        self._corr_fn = kendalltau if corr_type == 'kendalltau' else spearmanr
        self.reset()
    
    def reset(self):
        self.scores = []
        self.labels = []
    
    def update(self, scores: ArrayLike, labels: ArrayLike):
        self.scores.append(scores)
        self.labels.append(labels)
    
    def compute(self) -> float:
        scores = np.concatenate(self.scores, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        return self._corr_fn(scores, labels).correlation
