from abc import ABC, abstractmethod
from typing import Generic, Sized, TypeVar
import numpy as np


# Note that clips have to be Sized. The `__len__` method must return the
# number of frames or timesteps in the clip.
ClipType = TypeVar('ClipType', bound=Sized)

class Renderer(ABC, Generic[ClipType]):
    """A Renderer is responsible for converting clips into a human-viewable representation which
    can be displayed in an HTML page."""    
    @abstractmethod
    def thumbnail(self, clip: ClipType, frame: int) -> np.ndarray:
        """Returns a thumbnail of the clip at the given frame as a NumPy array. If `frame` is None,
        the thumbnail will be selected from the frame at index `len(clip) // 2`."""
        pass
    
    @abstractmethod
    def viewer_html(self, clip: ClipType) -> str:
        pass
