from ..renderer import Renderer
from .utils import BraxClip
from brax.io.html import render
from brax.io.image import render_array
from dataclasses import dataclass
from jax.tree_util import tree_map
import brax
import numpy as np


@dataclass(frozen=True)
class BraxRenderer(Renderer[BraxClip]):
    sys: brax.System
    
    def thumbnail(self, clip: BraxClip, frame: int) -> np.ndarray:
        # Thumbnails don't depend on the action
        states, _ = clip

        frame_state = tree_map(lambda field: field[frame], states)
        return render_array(self.sys, frame_state, 480, 480)
    
    def viewer_html(self, clip: BraxClip) -> str:
        states, _ = clip
        frames = [tree_map(lambda field: field[i], states) for i in range(states.pos.shape[0])]
        return render(self.sys, frames)
