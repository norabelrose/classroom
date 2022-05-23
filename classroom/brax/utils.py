from brax import jumpy as jp
from brax.envs import State
from flax import struct


@struct.dataclass
class BraxClip:
    """One or many (state, action) pair(s)"""
    state: State
    action: jp.ndarray

    def __len__(self):
        """Returns the number of timesteps in the clip."""
        return self.action.shape[0]


def flatten_clip(clip: BraxClip) -> jp.ndarray:
    """Converts a clip to a flattened array."""
    return jp.concatenate([clip.state.obs, clip.action], axis=-1)
