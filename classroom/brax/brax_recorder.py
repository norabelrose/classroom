from brax import QP
from brax.envs.env import Env, State, Wrapper
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map
from pathlib import Path
from ..mmap import MmapQueueWriter
from .brax_renderer import BraxRenderer
from .utils import BraxClip, tree_stack
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
            min_clip_length: int | None = None,  # Defaults to clip_length // 2

            clips_per_batch: int | None = None,  # Defaults to next_power_of_2(env.batch_size // 100)
            flush_every: int = 100,
        ):
        super().__init__(env)

        batch_size = get_env_batch_size(env)
        assert batch_size, "BraxRecorder requires vectorized environments."

        self._clip_length = clip_length
        self._clips_per_batch = clips_per_batch or max(1, next_power_of_2(batch_size // 100))
        self._db_path = Path(db_path)
        self._flush_every = flush_every
        self._min_clip_length = min_clip_length or clip_length // 2

        self._clip_dir = self._db_path / 'clips'
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        # self._writer = MmapQueueWriter(clip_dir)

        # Save a renderer that can be used to render the clips later
        with open(self._db_path / 'renderer.pkl', 'wb') as f:
            pickle.dump(BraxRenderer(env.sys), f)

        self._host_reset()
    
    def reset(self, rng: jnp.ndarray) -> State:
        # Let the host know that we're about to reset
        id_tap(lambda _, __: self._host_reset(), ())
        return super().reset(rng)
    
    def step(self, state: State, action: jnp.ndarray) -> State:
        # For simplicity we record the states and actions from the env in the first
        # `clips_per_batch` environments in the batch
        stop = self._clips_per_batch
        id_tap(
            # It seems to be important for performance to convert to NumPy up front to ensure
            # that we don't cause any needless transfers from the host back to the device
            lambda x, _: self._process_timestep(*tree_map(lambda field: np.asarray(field), x)),
            (tree_map(lambda field: field[:stop], state.qp), action[:stop], state.done[:stop]),
        )
        return super().step(state, action)
    
    def _host_reset(self):
        # We'll store the timesteps in a list, and then write them to the queue when we've
        # collected enough to make a full clip.
        self._action_buffers = [[] for _ in range(self._clips_per_batch)]
        self._qp_buffers = [[] for _ in range(self._clips_per_batch)]
    
    def _process_timestep(self, qps: QP, actions: np.ndarray, dones: np.ndarray):
        """Host-side function called by `step()` to process a single timestep to be recorded."""
        states = [tree_map(lambda field: field[i], qps) for i in range(self._clips_per_batch)]
        
        for s, a, s_buf, a_buf, terminal in zip(states, actions, self._qp_buffers, self._action_buffers, dones):
            s_buf.append(s)
            a_buf.append(a)

            if terminal or len(s_buf) >= self._clip_length:
                # We've collected enough timesteps to make a clip.
                if len(s_buf) >= self._min_clip_length:
                    # Transpose our list of QPs into a QP where each field has a timestep dimension.
                    # This saves space in the queue and on disk.
                    clip_name = f"clip_{time.monotonic_ns()}.pkl"
                    with open(self._clip_dir / clip_name, 'wb') as f:
                        pickle.dump(BraxClip(tree_stack(s_buf), np.stack(a_buf)), f)
                
                s_buf.clear()
                a_buf.clear()


def get_env_batch_size(env: Env) -> int:
    """Returns the batch size of the given env or wrapper."""
    while inner := getattr(env, 'env'):
        if batch_size := getattr(inner, 'batch_size'):
            return batch_size
    
    return 0


def next_power_of_2(x: int) -> int:
    # .bit_length() trick suggested by GitHub Copilot
    return 2 ** (x - 1).bit_length()
