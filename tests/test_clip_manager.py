from pathlib import Path
from whisper import ClipManager
import gym
import numpy as np
import tempfile
import time


def test_database_creation():
    env = gym.make('CartPole-v1')
    
    # Create a new database and write some clips to it
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)
        clip_length = 25
        capacity = 10

        clip_manager = ClipManager(db_path, env, clip_length, capacity)
        assert clip_manager._capacity.dtype == np.uint64
        assert clip_manager._write_cursor.dtype == np.uint64
        assert clip_manager.db_path == db_path
        assert clip_manager.clip_length == clip_length
        assert clip_manager.capacity == capacity

        # Add a clip
        clip_manager.add_clip(42, time.time(), [env.action_space.sample()] * clip_length)
        assert len(clip_manager) == 1
        assert clip_manager[0]['seed'] == 42

        # Now add 10 more clips, and check to see if the database has been extended
        for i in range(10):
            clip_manager.add_clip(i, time.time(), [env.action_space.sample()] * clip_length)
        
        assert len(clip_manager) == 11
        assert clip_manager.capacity > 10
