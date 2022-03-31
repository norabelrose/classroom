from abc import ABC, abstractmethod
from typing import Generic, TypeVar


RENDERER_REGISTRY: dict[str, type] = {}

ClipType = TypeVar('ClipType')

class Renderer(ABC, Generic[ClipType]):
    """A Renderer is responsible for converting clips into a human-viewable representation which
    can be displayed in an HTML page."""

    # Automatically register subclasses of Renderer
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RENDERER_REGISTRY[cls.__name__] = cls
    
    @abstractmethod
    def render(self, clip: ClipType) -> str:
        """Renders a clip, returning a Flask `Response` containing the information necessary
        for the client-side viewer to display the rendered clip."""
        raise NotImplementedError
    
    @abstractmethod
    def thumbnail(self, clip: ClipType):
        pass
    
    @abstractmethod
    def viewer_onload(self) -> str:
        """JavaScript snippet to be run on the onload event of the viewer's HTML element."""
        raise NotImplementedError
