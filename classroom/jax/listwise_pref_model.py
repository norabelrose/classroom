from ..metrics import RankCorrelation
from .pref_model import PrefModel
from flax.training.train_state import TrainState
from typing import Iterable
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np


# The clips are guaranteed to be in their ground truth order in the batch
ListwisePref = jnp.ndarray


def listwise_pref_loss(scores: jnp.ndarray) -> jnp.ndarray:
    """
    Negative log likelihood of the ground truth preference ranking under a Plackett-Luce random
    utility model with the given predicted `scores`. Under Placket-Luce, we imagine that the GT
    ranking was generated sequentially in descending order, by sampling without replacement from
    the available options in proportion to `exp(scores)`.
    """
    scores = jnp.flip(scores)

    # The option set shrinks by one every time a choice is made; e.g. the highest ranked option was
    # selected from among the entire set of options, the second highest from the whole set minus the
    # one just picked, etc. This allows us to efficiently compute the log likelihood of the GT ranking
    # using a log-cumsum-exp trick.
    stabilizer: jnp.ndarray = scores.max()
    log_normalizer = jnp.log(jnp.cumsum(jnp.exp(scores - stabilizer), 0) + 1e-7)
    return -jnp.mean(scores - log_normalizer - stabilizer)


class ListwisePrefModel(PrefModel):    
    def train_step(self, state: TrainState, batch: ListwisePref) -> tuple[float, TrainState]:
        def compute_loss(model: 'ListwisePrefModel') -> jnp.ndarray:
            return listwise_pref_loss(model(batch))
        
        grad_fn = jax.value_and_grad(nn.apply(compute_loss, self))
        (loss, grads) = grad_fn({'params': state.params})

        state = state.apply_gradients(grads=grads['params'])
        return loss, state

    def test(self, data: Iterable[ListwisePref]) -> dict[str, float]:
        losses = []
        rank_corr = RankCorrelation('kendalltau')

        for clips in data:
            scores = self(clips)
            losses.append(listwise_pref_loss(scores))
            rank_corr.update(scores, np.arange(len(scores), 0, -1))
        
        return {'kendalltau': rank_corr.compute(), 'loss': np.mean(losses)}
