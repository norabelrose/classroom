from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from functools import partial
from typing import Generic, Mapping, Sequence, TypeVar
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import optax


PrefBatch = TypeVar('PrefBatch', bound=Sequence)
class PrefModel(nn.Module, Generic[PrefBatch]):
    """Abstract base class for JAX-based preference models."""
    def init_training(self, seed, input_shape: tuple[int, ...]) -> TrainState:
        """Initialize parameters and optimizer state for training."""
        opt = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
        variables = self.init(jax.random.PRNGKey(seed), jnp.zeros(input_shape))
        return TrainState.create(apply_fn=self.apply, params=variables['params'], tx=opt)

    def fit(
            self,
            train: Sequence[PrefBatch],
            val: Sequence[PrefBatch],
            *,
            jit: bool = True,
            max_epochs: int = 100,
            patience: int = 0,
            seed: int | jnp.ndarray = 0
        ) -> TrainState:
        # Take a peek at the dataset to get the input shape.
        first_batch = train[0]
        clip = first_batch if isinstance(first_batch, np.ndarray) else first_batch[0]

        init_fn = partial(self.init_training, input_shape=clip.shape)
        ensemble = isinstance(seed, jnp.ndarray) and seed.size > 1
        if ensemble:
            init_fn = jax.vmap(init_fn)

        state = init_fn(seed)
        early_stop = EarlyStopping(patience=patience)
        step_fn = jax.jit(
            lambda state, batch: self.apply(state.params, state, batch, method=self.train_step)
        ) if jit else self.train_step

        if ensemble:
            step_fn = jax.vmap(step_fn)#, in_axes=(0, None))  # type: ignore
        
        eval_fn = lambda state: self.apply({'params': state.params}, val, method=self.test)
        if ensemble:
            eval_fn = jax.vmap(eval_fn)

        for epoch in range(max_epochs):
            losses = []
            for batch in train:
                loss, state = step_fn(state, batch)
                losses.append(loss)

            val_metrics = eval_fn(state)
            assert isinstance(val_metrics, Mapping)

            msg = f"Epoch {epoch + 1} train loss: {np.mean(losses):.3f}, "
            msg += ", ".join(
                f"val {k}: {v.mean():.3f} (sd {v.std():.3f})" if ensemble else f"val {k}: {v.mean():.3f}"
                for k, v in val_metrics.items()
            )
            
            print(msg)
            _, early_stop = early_stop.update(val_metrics['loss'].mean())
            if early_stop.should_stop:
                print(f"Early stopping; val loss plateaued at {early_stop.best_metric:.3f}")
                return state
        
        print(f"Training hit max epochs ({max_epochs}) without val loss plateauing.")
        return state
    
    def train_step(self, state: TrainState, batch: PrefBatch) -> tuple[float, TrainState]:
        raise NotImplementedError
