# from flask import make_response, Response
from gym.wrappers import RecordVideo
from .renderer import Renderer
import gym
import numpy as np
import tempfile


class GymRenderer(Renderer):
    def __init__(self, env: gym.Env):
        # Read environment metadata
        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            raise ValueError(f"{env} does not support the 'rgb_array' render mode.")
        
        self.tempdir = tempfile.TemporaryDirectory()
        self.env = RecordVideo(env, './videos/')
    
    def render(self, clip: np.ndarray) -> str:
        return self.env.render(mode='rgb_array')
    
    def thumbnail(self, clip: np.ndarray):
        return super().thumbnail(clip)
    
    def viewer_html(self) -> str:
        return '<video controls autoplay></video>'