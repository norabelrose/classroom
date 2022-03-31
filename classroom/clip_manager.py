from numpy.typing import ArrayLike
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pickle


class ClipManager:
    """
    `ClipManager` objects are handles for memory-mapped databases of clips, along with metadata about
    the environment used to generate them. They can be used either in read-only mode or read-write mode.
    No locking is used to ensure consistency, so many read-only managers can run in parallel on the
    same buffer, but only one read-write manager should be running at a time.

    Args:
        db_path: The path to the database directory.
        clip_dtype: The NumPy dtype of the clips in the database.
        clip_shape: The shape of the clips in the database. This defaults to an empty tuple, because
            we assume the clips are 'scalars' with structured dtypes by default.
        metadata: Any picklable object representing the environment used to generate the clips.
            It will be saved to the database directory as a metadata.pickle file.
        read_only: Whether this database handle is read-only.
    """
    DEFAULT_CAPACITY = 1024

    def __init__(
            self,
            db_path: Union[Path, str],
            clip_dtype: Optional[np.dtype] = None,
            clip_shape: tuple[int, ...] = (),
            metadata: Optional[Any] = None,
            read_only: bool = False,
        ):
        db_path = Path(db_path)
        self.db_path = db_path

        # Database directories are expected to contain the following files:
        # - clips.npy: the raw clip buffer
        # - metadata.pickle: a pickled representation of the environment
        self.clip_file = db_path / 'clips.npy'
        metadata_file = db_path / 'metadata.pickle'

        # Memory map the clip buffer if it already exists
        if self.clip_file.exists():
            assert metadata_file.exists(), f"Database path {db_path} must contain an metadata.pickle file"
            assert metadata is None, "Cannot specify an environment for an existing database"

            # First load the environment
            with metadata_file.open('rb') as f:
                metadata = pickle.load(f)
            
            # Just read the first 16 bytes of the buffer to get the capacity and write cursor;
            # the rest of the buffer will be mapped on demand
            mode = 'r' if read_only else 'r+'

        # Create a new database directory
        else:
            assert metadata is not None, "Must provide a Gym environment to create a new database"
            assert not read_only, "Cannot create a new database in read-only mode"
            mode = 'w+'

            # Create the database directory
            db_path.mkdir(parents=True, exist_ok=True)

            # Save the environment
            with metadata_file.open('wb') as f:
                pickle.dump(metadata, f)
        
        self.clip_dtype = clip_dtype
        self.env = metadata

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
            shape=(capacity, *clip_shape)
        )
        # Initialize the capacity variable if needed
        if mode != 'r':
            self._capacity[:] = capacity
    
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
            self.reserve_capacity(self.capacity)
        
        return cursor
    
    def __repr__(self) -> str:
        return f"ClipManager(db_path={str(self.db_path)}, read_only={self.read_only})"
    
    def add_clip(self, clip: ArrayLike):
        """Append a clip to the end of the buffer. This will fail if the database object is read-only."""
        # Check if we need to re-map the buffer to a new size
        if len(self) == self.capacity:
            self.reserve_capacity(self.capacity * 2)
        
        self._buffer[self._write_cursor] = np.asarray(clip)
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
        """Indicates whether the ClipManager object can be used to write to the database."""
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
