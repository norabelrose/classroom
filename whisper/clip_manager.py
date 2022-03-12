from .clip import Clip
from dataclasses import astuple
from datetime import date
from pathlib import Path
from typing import Optional
import gym
import json
import numpy as np
import pickle


class ClipManager:
    """
    `ClipManager` objects manage large memory-mapped buffers of clips, along with metadata about the
    environment used to generate them. They can run either in read-only mode or read-write mode.
    No locking is used to ensure consistency, so many read-only managers can run in parallel on the
    same buffer, but only one read-write manager should be running at a time.
    """
    DEFAULT_CAPACITY = 1024

    def __init__(
            self,
            db_path: Path,
            env: Optional[gym.Env] = None,
            clip_length: Optional[int] = None,
            read_only: bool = False,
        ):
        self.db_path = db_path

        # Database directories are expected to contain the following files:
        # - clips.npy: the raw clip buffer
        # - env.pickle: the pickled Gym environment
        # - info.json: miscellaneous metadata about the experiment
        self.clip_file = db_path / 'clips.npy'
        env_file = db_path / 'env.pickle'
        info_file = db_path / 'info.json'

        # Memory map the clip buffer if it already exists
        if self.clip_file.exists():
            assert env_file.exists(), f"Database path {db_path} must contain an env.pickle file"
            assert info_file.exists(), f"Database path {db_path} must contain an info.json file"
            assert clip_length is None, f"Cannot specify a clip_length for an existing database"
            assert env is None, "Cannot specify an environment for an existing database"

            # First load the environment
            with env_file.open('rb') as f:
                env = pickle.load(f)
            
            # Now load the info.json file
            with info_file.open('r') as f:
                self.info = json.load(f)
                clip_length = self.info['clip_length']
            
            # Just read the first 16 bytes of the buffer to get the capacity and write cursor;
            # the rest of the buffer will be mapped on demand
            mode = 'r' if read_only else 'r+'

        # Create a new database directory
        else:
            assert env is not None, "Must provide a Gym environment to create a new database"
            assert not read_only, "Cannot create a new database in read-only mode"
            mode = 'w+'

            # Create the database directory
            db_path.mkdir(parents=True, exist_ok=True)

            # Save the environment
            with env_file.open('wb') as f:
                pickle.dump(env, f)
            
            # Create the info.json file
            self.info = {
                'clip_length': clip_length,
                'created': date.today().isoformat(),
            }
            with info_file.open('w') as f:
                json.dump(self.info, f, indent=2)
        
        # Construct the NumPy structured data type for the clips
        assert isinstance(env, gym.Env), "Environment must be a Gym environment"
        self.clip_dtype = np.dtype([
            ('seed', np.uint64),
            ('timestamp', np.float64),
            ('actions', env.action_space.dtype, (clip_length, *env.action_space.shape)),  # type: ignore[attr-defined]
        ])

        # The first 16 bytes of the buffer are reserved for the capacity and write cursor
        self._capacity = np.memmap(
            self.clip_file, dtype=np.uint64, mode=mode, offset=0, shape=(1,)
        )
        self._write_cursor = np.memmap(
            self.clip_file, dtype=np.uint64, mode=mode, offset=8, shape=(1,)
        )

        capacity = max(self.DEFAULT_CAPACITY, self.capacity)
        self._buffer = np.memmap(
            self.clip_file, dtype=self.clip_dtype, mode=mode, offset=16,
            shape=(capacity,)
        )
        # Initialize the capacity variable if needed
        if mode != 'r':
            self._capacity[:] = capacity
    
    def __getitem__(self, index: int) -> Clip:
        """Retrieve a clip from the database."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        
        return Clip.from_numpy(self._buffer[index])
    
    def __len__(self) -> int:
        """Number of clips available to be read in the database."""
        cursor = int(self._write_cursor)

        # Check if we need to re-map the buffer to a new size
        if cursor > len(self._buffer):
            assert self.capacity > cursor, "Internal error: write cursor is beyond capacity"
            self.reserve_capacity(self.capacity)
        
        return cursor
    
    def __repr__(self) -> str:
        return f"ClipServer(db_path={str(self.db_path)}, read_only={self.read_only})"
    
    def add_clip(self, clip: Clip):
        """
        Append a clip to the end of the buffer. This will fail if the database object is read-only.
        """
        # Check if we need to re-map the buffer to a new size
        if len(self) == self.capacity:
            self.reserve_capacity(self.capacity * 2)
        
        array = np.array(astuple(clip), dtype=self.clip_dtype)
        self._buffer[self._write_cursor] = array
        self._write_cursor[:] = int(self._write_cursor) + 1
    
    @property
    def capacity(self) -> int:
        return int(self._capacity)
    
    def flush(self):
        self._buffer.flush()
        self._capacity.flush()
        self._write_cursor.flush()

    @property
    def read_only(self) -> bool:
        """Indicates whether the ClipServer object can be used to write to the database."""
        return not self._buffer.flags.writeable
    
    def reserve_capacity(self, new_capacity: int):
        """Re-map the buffer to a new capacity."""        
        self._buffer = np.memmap(
            self.clip_file, dtype=self.clip_dtype, mode='r' if self.read_only else 'r+',
            offset=16, shape=(new_capacity,)
        )
        self._capacity[:] = new_capacity
    
    def snapshot(self) -> np.ndarray:
        """
        Return a read-only view into the database that is guaranteed not to change.
        This can be useful when you want to be sure that the length of the database isn't
        going to go up while you're processing it.
        """
        end = len(self)
        view = self._buffer[:end]
        view.flags.writeable = False
        return view
