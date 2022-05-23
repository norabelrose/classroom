from brax.envs.env import Env, State, Wrapper
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map
from pathlib import Path
from .brax_renderer import BraxRenderer
from ..jax import tree_stack
from .utils import BraxClip
import csv
import jax.numpy as jnp
import numpy as np
import pickle
import time


class BraxRecorder(Wrapper):
    def __init__(
            self,
            env: Env,
            db_path: Path | str,
            clip_length: int = 120,
            clips_per_batch: int | None = None,  # Defaults to next_power_of_2(env.batch_size // 100)
        ):
        super().__init__(env)

        batch_size = get_env_batch_size(env)
        assert batch_size, "BraxRecorder requires vectorized environments."

        self._clip_length = clip_length
        self._clips_per_batch = clips_per_batch or max(1, next_power_of_2(batch_size // 100))
        self._db_path = Path(db_path)

        self._clip_dir = self._db_path / 'clips'
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        self._csv_writer = csv.writer(open(self._db_path / 'rewards.csv', 'w'))
        self._csv_writer.writerow(['clip_id', 'reward'])

        # Save a renderer that can be used to render the clips later
        with open(self._db_path / 'renderer.pkl', 'wb') as f:
            pickle.dump(BraxRenderer(env.sys), f)

        self._host_reset()
    
    def reset(self, rng: jnp.ndarray) -> State:
        # Let the host know that we're about to reset
        id_tap(lambda _, __: self._host_reset(), ())
        return super().reset(rng)
    
    def step(self, state: State, action: jnp.ndarray) -> State:
        # The info dictionary is the only field in the State dataclass that has scalar leaves;
        # all other leaves are arrays with the same leading batch dimension. To make the code
        # simple we just drop it from the state.
        clip_state = state.replace(info={})  # type: ignore[attr-defined]

        # For simplicity we record the states and actions from the env in the first
        # `clips_per_batch` environments in the batch
        id_tap(
            self._process_timestep,
            tree_map(lambda field: field[:self._clips_per_batch], BraxClip(clip_state, action)),
        )
        return super().step(state, action)
    
    def _host_reset(self):
        # We'll store the timesteps in a list, and then write them to the queue when we've
        # collected enough to make a full clip.
        self._transition_buffers = [[] for _ in range(self._clips_per_batch)]
    
    def _process_timestep(self, transition: BraxClip, _):
        """Host-side function called by `step()` to process a single timestep to be recorded."""
        for i in range(self._clips_per_batch):
            buffer = self._transition_buffers[i]
            sample: BraxClip = tree_map(lambda field: field[i], transition)
            buffer.append(sample)

            # We hit a terminal state but we haven't collected enough transitions to make a clip yet
            if sample.state.done and len(buffer) < self._clip_length:
                buffer.clear()
            
            # We've collected enough timesteps to make a clip.
            elif len(buffer) >= self._clip_length:
                # We use Unix timestamps, measured in nanoseconds, to generate ~unique filenames that
                # can be easily sorted by time. I decided to not use `monotonic_ns()` because it uses
                # an undefined reference time. I'm assuming leap seconds are not a serious problem here.
                timestamp = time.time_ns()
                clip_name = f"{timestamp}.pkl"
                with open(self._clip_dir / clip_name, 'wb') as f:
                    # Transpose our list of QPs into a QP where each field has a timestep dimension.
                    # This saves space in the queue and on disk.
                    clip = tree_map(np.asarray, tree_stack(buffer))
                    pickle.dump(clip, f)
                
                buffer.clear()
                self._csv_writer.writerow([timestamp, clip.state.reward.sum()])


def get_env_batch_size(env: Env) -> int:
    """Returns the batch size of the given env or wrapper."""
    while inner := getattr(env, 'env'):
        if batch_size := getattr(inner, 'batch_size'):
            return batch_size
    
    return 0


def next_power_of_2(x: int) -> int:
    # .bit_length() trick suggested by GitHub Copilot
    return 2 ** (x - 1).bit_length()
