from abc import ABC
from numpy.typing import ArrayLike, DTypeLike
from pathlib import Path
from typing import Any, Callable, Literal
import numpy as np
import pickle


class _MmapHandle(ABC):
    """Abstract base class for `MmapReader` and `MmapWriter`"""
    _buffer: np.memmap

    def __init__(self, path: Path | str, mode: Literal['r', 'r+', 'w+']):
        self.path = Path(path)

        # The first 16 bytes of the buffer are reserved for the capacity and write cursor
        self._capacity = np.memmap(path, dtype=np.uint64, mode=mode, offset=0, shape=(1,))
        self._write_cursor = np.memmap(path, dtype=np.uint64, mode=mode, offset=8, shape=(1,))
    
    @property
    def capacity(self):
        return int(self._capacity)
    
    @property
    def dtype(self) -> np.dtype:
        return self._buffer.dtype
    
    def __len__(self):
        return int(self._write_cursor)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.path}, {self.dtype}, capacity={self.capacity}, len={len(self)})"


class MmapReader(_MmapHandle):
    """Read handle for a memory-mapped file which grows automatically to fit new data."""
    def __init__(self, path: Path | str, dtype: DTypeLike):
        super().__init__(path, 'r')
        self._buffer = np.memmap(path, dtype=dtype, mode='r', offset=16, shape=(self.capacity,))
    
    def __getitem__(self, item: int | slice) -> np.ndarray:
        """Retrieve an element from the file."""
        cur_length = len(self)  # __len__ is volatile, so only read it once

        if isinstance(item, slice):
            assert item.step is None, "Slices with a step are not yet supported"
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else cur_length
        else:
            start = item
            stop = item

        # Check if a MmapWriter grew the file and we need to re-map the buffer to the new size
        if len(self._buffer) <= stop < self.capacity:
            self._buffer = np.memmap(
                self.path, dtype=self.dtype, mode='r', offset=16, shape=(self.capacity,)
            )
        
        # Normal bounds checking
        if not (0 <= start < cur_length):
            raise IndexError(f"Start index {start} out of bounds")
        if not (0 <= stop <= cur_length):
            raise IndexError(f"Index {stop} out of bounds")
        
        return self._buffer[item]


class MmapWriter(_MmapHandle):
    """Write handle for a memory-mapped file that grows as needed. Only appending to the end of the
    file is supported in order to guarantee consistency for the read side without locking."""
    def __init__(
            self,
            path: Path | str,
            capacity: int = 2 ** 20,
            dtype: DTypeLike = np.uint8,
            *,
            exist_ok: bool = True
        ):
        path = Path(path)
        mode = 'r+' if exist_ok and path.exists() else 'w+'
        super().__init__(path, mode)

        # When we first create the file, the capacity field will be initialized to zero
        buffer_size = max(self.capacity, capacity)
        self._buffer = np.memmap(path, dtype=dtype, mode=mode, offset=16, shape=(buffer_size,))
        self._capacity[:] = buffer_size
    
    def extend(self, data: ArrayLike):
        """Add a new chunk of data to the end of the buffer"""
        data = np.atleast_1d(data).astype(self.dtype)
        assert data.shape[1:] == self._buffer.shape[1:], f"Data shape {data.shape} does not match buffer shape {self._buffer.shape}"
        
        # Resize the buffer if necessary
        old_length = len(self)
        new_length = old_length + len(data)
        if new_length > self.capacity:
            self.reserve_capacity(new_length * 2)
        
        # Write the data
        self._buffer[old_length:new_length] = data
        self._write_cursor[:] = new_length      # Update the write cursor
    
    def flush(self):
        """Flush the buffer to disk"""
        self._buffer.flush()
    
    def reserve_capacity(self, new_capacity: int):
        """Re-map the buffer to a new capacity."""        
        self._buffer = np.memmap(
            self.path, dtype=self.dtype, mode='r+',
            offset=16, shape=(new_capacity,)
        )
        self._capacity[:] = new_capacity


class MmapQueueReader:
    """Read handle for a memory-mapped list of arbitrary Python objects which grows to fit new data.
    Writers can append new elements but cannot shrink the list or modify existing elements."""
    def __init__(self, path: Path | str, *, deserialize_fn: Callable[[bytes], Any] = pickle.loads):
        self.deserialize_fn = deserialize_fn
        self.path = Path(path)
        
        self._data = MmapReader(self.path / 'data.npy', dtype=np.uint8)
        self._offsets = MmapReader(self.path / 'offsets.npy', dtype=np.uint64)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._offsets)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.path}, len={len(self)}, capacity={self._data.capacity} bytes, size={len(self._data)} bytes)"
    
    def __getitem__(self, i: int) -> Any:
        start = self._offsets[i - 1] if i > 0 else 0
        stop = self._offsets[i]
        data = self._data[start:stop]
        return self.deserialize_fn(data.tobytes())


class MmapQueueWriter:
    """Write handle for a memory-mapped list of arbitrary Python objects which grows to fit new data."""
    def __init__(
            self,
            path: Path | str,
            capacity: int = 2 ** 20,
            *,
            serialize_fn: Callable[[Any], bytes] = pickle.dumps
        ):
        self.serialize_fn = serialize_fn
        self.path = Path(path)
        
        self._data = MmapWriter(self.path / 'data.npy', capacity=capacity, dtype=np.uint8)
        self._offsets = MmapWriter(self.path / 'offsets.npy', capacity=2 ** 20, dtype=np.uint64)
    
    def __len__(self):
        return len(self._offsets)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.path}, len={len(self)}, capacity={self._data.capacity} bytes, size={len(self._data)} bytes)"
    
    def append(self, item: Any):
        data = np.frombuffer(self.serialize_fn(item), dtype=np.uint8)
        self._data.extend(data)
        self._offsets.extend(len(self._data))
    
    def flush(self):
        self._data.flush()
        self._offsets.flush()
