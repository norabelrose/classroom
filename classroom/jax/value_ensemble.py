from dataclasses import dataclass, field
from flax.training.train_state import TrainState
from functools import partial
from typing import Iterable
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import random


@dataclass
class ValueEnsemble:
    """Ensemble of value models."""
    model: nn.Module
    num_models: int = 5
    seed: int = field(default_factory=lambda: random.randrange(2 ** 32))

    def __post_init__(self):
        @jax.jit
        def compute_loss(variables: VariableDict, data: jnp.ndarray, labels: jnp.ndarray) -> float:
            logits = jnp.sum(self.model.apply(variables, data), axis=-1)
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None, None))
        def train_step(state: TrainState, data, labels) -> tuple[float, TrainState]:
            loss, grads = jax.value_and_grad(compute_loss)(state.params, data, labels)
            state = state.apply_gradients(grads=grads)
            return loss, state

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None))
        def predict(state: TrainState, data: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(self.model.apply(state.params, data), axis=-1)
    
    def fit(self, data: jnp.ndarray, labels: jnp.ndarray):
        @jax.vmap
        def init_state(key) -> TrainState:
            opt = optax.adam(learning_rate=1e-3)
            params = self.model.init(key, jnp.zeros(95))
            return TrainState.create(apply_fn=self.model.apply, params=params, tx=opt)
        
        master = jax.random.PRNGKey(self.seed)
        states = init_state()