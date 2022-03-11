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
    def __init__(
            self,
            db_path: Path,
            env: Optional[gym.Env] = None,
            clip_length: Optional[int] = None,
            capacity: int = 100_000,
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
                'capacity': capacity,
                'created': date.today().isoformat(),
            }
            with info_file.open('w') as f:
                json.dump(self.info, f, indent=2)
        
        # Construct the NumPy structured data type for the clips
        self.clip_dtype = np.dtype([
            ('seed', np.uint64),
            ('timestamp', np.float64),
            ('actions', env.action_space.dtype, (clip_length, *env.action_space.shape)),
        ])
        self._map_to_capacity(capacity, mode)
        self.clip_length = clip_length
    
    def _map_to_capacity(self, capacity: int, mode: str):
        """Private method for (re-)mapping the buffer to a (new) capacity."""
        # Memory map the clip buffer. Note that the first 16 bytes of the buffer are reserved for
        # two 64-bit unsigned integers: the buffer capacity, and the write cursor.
        self._buffer = np.memmap(
            self.clip_file, dtype=self.clip_dtype, mode=mode, offset=16, shape=(capacity,)
        )
        self._capacity = np.memmap(
            self.clip_file, dtype=np.uint64, mode=mode, offset=0, shape=()
        )
        self._write_cursor = np.memmap(
            self.clip_file, dtype=np.uint64, mode=mode, offset=8, shape=()
        )
        # Let read-only clients know that the buffer has been extended
        if not self.read_only:
            self._capacity.fill(capacity)
    
    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve a clip from the database."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        
        return self._buffer[index]
    
    def __len__(self) -> int:
        """Number of clips available to be read in the database."""
        cursor = int(self._write_cursor)

        # Check if we need to re-map the buffer to a new size
        if cursor > len(self._buffer):
            assert self.capacity > cursor, "Internal error: write cursor is beyond capacity"
            self._map_to_capacity(self.capacity, 'r' if self.read_only else 'r+')
        
        return cursor
    
    def __repr__(self) -> str:
        return f"ClipServer(db_path={str(self.db_path)}, read_only={self.read_only})"
    
    def add_clip(self, seed: int, timestamp: float, actions: np.ndarray):
        """
        Append a clip to the end of the buffer. This will fail if the database object is read-only.
        """
        # Check if we need to re-map the buffer to a new size
        if len(self) == self.capacity:
            self._map_to_capacity(self.capacity * 2, 'r' if self.read_only else 'r+')
        
        clip = np.array(
            (seed, timestamp, actions),
            dtype=self.clip_dtype, order='C'
        )
        self._buffer[self._write_cursor] = clip

        # Hackish way to increment the write cursor- the "normal" way doesn't work for np.memmap
        self._write_cursor.fill(int(self._write_cursor) + 1)
    
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
