from pathlib import Path
from whisper import Clip, ClipManager
import gym
import numpy as np
import pytest
import tempfile
import time


# Arbitrary testing parameters
CLIP_LENGTH = 25


def test_database_creation():
    env = gym.make('CartPole-v1')
    
    # Create a new database and write some clips to it
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)

        manager = ClipManager(db_path, env, CLIP_LENGTH)
        manager.add_clip(
            Clip(42, time.time(), np.array([env.action_space.sample()] * CLIP_LENGTH))
        )
        assert len(manager) == 1
        assert manager[0].seed == 42

        # Make the manager exceed its current capacity
        old_capacity = manager.capacity
        for i in range(old_capacity):
            manager.add_clip(
                Clip(i, time.time(), np.array([env.action_space.sample()] * CLIP_LENGTH))
            )
        
        assert len(manager) == old_capacity + 1
        assert manager.capacity > old_capacity


def test_concurrent_managers():
    env = gym.make('CartPole-v1')

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir)

        producer = ClipManager(db_path, env, CLIP_LENGTH)
        consumer = ClipManager(db_path, read_only=True)

        # Add a clip
        producer.add_clip(
            Clip(42, time.time(), np.array([env.action_space.sample()] * CLIP_LENGTH))
        )
        assert len(consumer) == len(producer) == 1

        with pytest.raises(ValueError):
            # Attempt to add a clip to the read-only database
            consumer.add_clip(producer[0])
