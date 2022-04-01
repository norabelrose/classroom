import gym
import numpy as np
import time


class GymRecorder(gym.Wrapper):
    """Wrapper class for a Gym environment for gathering human feedback."""
    env: gym.Env

    def __init__(self, env: gym.Env, clip_manager: ClipManager):
        super().__init__(env)

        assert not clip_manager.read_only, "Cannot use a read-only clip manager with a GymWrapper"
        self.clip_manager = clip_manager
        self._action_buffer = []
        self._obs_buffer = []

    def step(self, action):
        # Record the action, adding the clip to the buffer if we've hit the end of a clip
        self._action_buffer.append(action)
        if len(self._action_buffer) >= self.clip_manager.clip_length:
            self.clip_manager.add_clip(
                Clip(time.time(), np.stack(self._action_buffer))
            )
            self._action_buffer.clear()

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if kwargs.get('return_info'):
            obs, _ = out
        else:
            obs = out
        
        self._action_buffer.clear()
        self._obs_buffer.clear()
        self._obs_buffer.append(obs)
        return out
