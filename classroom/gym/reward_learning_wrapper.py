from pathlib import Path
import gym
import numpy as np
import pickle
import time


class RewardLearningWrapper(gym.Wrapper):
    """Wrapper class for a Gym environment for gathering human feedback."""
    env: gym.Env

    def __init__(self, env: gym.Env, db_path: Path | str):
        super().__init__(env)
        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            raise ValueError(f"{env} does not support the 'rgb_array' render mode.")

        self._db_path = Path(db_path)
        self._training = False

        self._state_buffer = []
        self._action_buffer = []
    
    @property
    def training(self) -> bool:
        return self._training
    
    @training.setter
    def training(self, value: bool):
        self._training = value
        
        # Make sure the current episode gets saved
        if self._action_buffer:
            self._save_clip()

    def _save_clip(self):
        with open(self._db_path / 'clips' / f'{time.time_ns()}.pkl', 'wb') as f:
            clip = dict(
                states=np.stack(self._state_buffer),
                actions=np.stack(self._action_buffer)
            )
            pickle.dump(clip, f)
        
        self._action_buffer.clear()
        self._state_buffer.clear()
    
    def step(self, action):
        # Record the action, adding the clip to the buffer if we've hit the end of a clip
        if self._training:
            self._state_buffer.append(self.env.render(mode='rgb_array'))
            self._action_buffer.append(action)

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if kwargs.get('return_info'):
            obs, _ = out
        else:
            obs = out
        
        self._state_buffer.append(obs)
        return out
