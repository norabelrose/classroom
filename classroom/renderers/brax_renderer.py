from ..lazy_import import lazy_import
from typing import TYPE_CHECKING


# Avoid loading Brax until we really need it
if TYPE_CHECKING:
    from brax.envs import Env
    import brax
else:
    brax = lazy_import('brax')


class BraxRenderer(ClipReader):
    def render(self, clip_id: int):
        """Renders a clip, returning a Flask `Response` containing the information necessary
        for the client-side viewer to display the rendered clip."""
        return make_response(jsonify(self[clip_id]), 200)
    
    def viewer_onload(self) -> str:
        return """
        import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.0.10/js/viewer.js';
        var viewer = new Viewer(this, system);

        fetch()
        """
