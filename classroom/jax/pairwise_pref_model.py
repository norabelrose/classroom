from .pref_model import PrefModel
from flax.training.train_state import TrainState
from typing import Iterable, SupportsFloat
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax


PairwisePref = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

def mean_update(running_mean, new_value, n: int):
    """Online update of a running mean."""
    return (n * running_mean + new_value) / (n + 1)


class PairwisePrefModel(PrefModel):
    def pref_logit(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        batch = jnp.concatenate([a, b])
        a_pred, b_pred = jnp.split(self(batch).mean(axis=-1), 2)
        return a_pred - b_pred
    
    def train_step(self, state: TrainState, batch: PairwisePref) -> tuple[float, TrainState]:
        a, b, prefs = batch
        def compute_loss(model: 'PairwisePrefModel') -> jnp.ndarray:
            logits = model.pref_logit(a, b)
            return optax.sigmoid_binary_cross_entropy(logits, prefs.astype(jnp.float32)).mean()
        
        grad_fn = jax.value_and_grad(nn.apply(compute_loss, self))
        (loss, grads) = grad_fn({'params': state.params})

        state = state.apply_gradients(grads=grads['params'])
        return loss, state

    def test(self, data: Iterable[PairwisePref]) -> dict[str, jnp.ndarray]:
        metrics: dict[str, SupportsFloat] = {'acc': jnp.zeros(()), 'loss': jnp.zeros(())}
        for i, (a, b, pref) in enumerate(data):
            pref_logit = self.pref_logit(a, b)
            hits = pref == (pref_logit > 0)

            metrics['acc'] = mean_update(metrics['acc'], hits.mean(), i)
            metrics['loss'] = mean_update(
                metrics['loss'],
                optax.sigmoid_binary_cross_entropy(pref_logit, pref.astype(jnp.float32)).mean(), i
            )
        
        return {k: jnp.asarray(v) for k, v in metrics.items()}
