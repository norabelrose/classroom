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

    def has_env_rewards(self) -> bool:
        return True

    def env_reward(self, clip: BraxClip) -> float | None:
        return float(clip.state.reward.sum())
    
    def thumbnail(self, clip: BraxClip, frame: int) -> np.ndarray:
        frame_state = tree_map(lambda field: field[frame], clip.state.qp)
        return render_array(self.sys, frame_state, 480, 480)
    
    def viewer_html(self, clip: BraxClip) -> str:
        qps = clip.state.qp
        frames = [tree_map(lambda field: field[i], qps) for i in range(qps.pos.shape[0])]
        return render(self.sys, frames)
