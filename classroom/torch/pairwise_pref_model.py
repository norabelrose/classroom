from abc import ABC
from torch import nn, Tensor
from typing import Iterable, SupportsFloat
import torch
import torch.nn.functional as F


def mean_update(running_mean, new_value, n: int):
    """Online update of a running mean."""
    return (n * running_mean + new_value) / (n + 1)


class PairwisePrefModel(nn.Module, ABC):
    """Abstract base class for pairwise preference models."""
    def pref_logit(self, a: Tensor, b: Tensor) -> Tensor:
        batch = torch.cat([a, b])
        a_pred, b_pred = self(batch).chunk(2)
        return a_pred - b_pred
    
    @torch.no_grad()
    def test(self, data: Iterable[tuple[Tensor, Tensor, Tensor]]) -> dict[str, float]:
        was_training = self.training
        self.eval()

        metrics: dict[str, SupportsFloat] = {'accuracy': 0, 'loss': 0}
        for i, (a, b, pref) in enumerate(data):
            pref = pref.cuda()
            pref_logit = self.pref_logit(a.cuda(), b.cuda())
            hits = pref == (pref_logit > 0)

            metrics['accuracy'] = mean_update(
                metrics['accuracy'], hits.float().mean(), i
            )
            metrics['loss'] = mean_update(
                metrics['loss'],
                F.binary_cross_entropy_with_logits(pref_logit, pref.float()), i
            )
        
        self.training = was_training
        return {k: float(v) for k, v in metrics.items()}
