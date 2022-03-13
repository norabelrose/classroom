from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Clip:
    """In `classroom`, 'clips' are data structures that store the (minimal) information
    necessary to reconstruct a segment of an agent's trajectory so that it can be
    evaluated by a human. In a simulated environment which is deterministic conditioned
    on its random seed, only the seed and the sequence of actions taken by the agent
    should be sufficient to reconstruct the trajectory segment. In other cases, it may
    be desirable or necessary to store environment states or observations as well.
    """
    seed: int
    timestamp: float
    actions: np.ndarray

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Clip':
        """Create a `Clip` from a structured NumPy array."""
        return cls(
            seed=int(array['seed']),
            timestamp=float(array['timestamp']),
            actions=array['actions'],
        )
