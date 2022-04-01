# from .pref_graph import PrefGraph
from .renderer import Renderer
from pathlib import Path
from typing import Sized
import numpy as np
import pickle


class DatabaseEvalHandle:
    """This class is meant to be used by the process(es) serving the Classroom GUI.
    It can be thought of as a 'file handle' for the database which is read-only for
    clips of agent behavior, but read-write for preferences.
    
    Database directories are expected to contain:
        - `clips/`, a directory containing pickled clips. Clip filenames should be
            of the format `<timestamp>.pkl`, where `timestamp` is a
            Unix timestamp and `actor_id` is the unique ID of the actor process that
            generated the clip.
        - `prefs/`, a directory containing pickled `PrefGraph` objects, one for each
            human evaluator
        - `renderer.pkl`, a pickled `Renderer` object
    """
    def __init__(self, path: Path | str):
        self.path = Path(path)
        assert self.path.is_dir(), f"Database path {self.path} must be a directory"

        # Load the renderer
        with open(self.path / 'renderer.pkl', 'rb') as f:
            self.renderer: Renderer = pickle.load(f)
            assert isinstance(self.renderer, Renderer)
        
        clip_dir = self.path / 'clips'
        self.clip_paths = {
            path.stem: path
            for path in clip_dir.glob('*.pkl')
        }
    
    def __getitem__(self, clip_id: str):
        path = self.clip_paths[clip_id]
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def thumbnail(self, clip_id: str) -> np.ndarray:
        clip = self[clip_id]
        assert isinstance(clip, Sized)

        return self.renderer.thumbnail(clip, frame=len(clip) // 2)
    
    def viewer_html(self, clip_id: str) -> str:
        return self.renderer.viewer_html(self[clip_id])