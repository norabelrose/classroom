from .mmap import MmapQueueReader
from .pref_graph import PrefGraph
from .server_utils import expose
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, SupportsInt
from werkzeug import Response, Request, run_simple
from werkzeug.exceptions import Unauthorized
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.utils import send_file
import json
import secrets

# TODO: Refactor this
import io
from brax.envs.ant import Ant
from jax.tree_util import tree_map
from brax.io.image import render_array
from PIL import Image


class GuiServer:
    """Wrapper class for a Werkzeug app for gathering human feedback. To make things simple,
    we currently only support one run at a time. To gather feedback for multiple runs in parallel,
    you should use multiple `GuiServer` instances."""
    def __init__(
            self,
            allowed_ips: tuple[str, ...] = (),  # localhost is always implicitly allowed
            debug: bool = False,
            port: int = 5000,
            use_tls: bool = False,
            experiment_dir: Optional[Path] = None,
            num_tokens: int = 0,    # Number of user tokens to generate; 0 means no token authentication
        ):
        # TODO: Make this configurable
        self.reader = MmapQueueReader('/tmp/classroom/ant.classroom')
        self.allow_remote = bool(allowed_ips)
        self.debug = debug
        self.port = port
        self.root_dir = experiment_dir or Path.cwd()
        self.selected_run = None
        self.static_dir = Path(__file__).parent / 'client' / 'public'
        self.use_tls = use_tls

        self.prefs = PrefGraph(allow_cycles=True)
        for i, j in zip(range(10), range(1, 11)):
            self.prefs.add_pref(i, j)

        # Generate URL tokens for users to use to authenticate themselves
        self.tokens = {secrets.token_urlsafe(16) for _ in range(num_tokens)}
        self._endpoints = {
            name: func
            for name, func in self.__class__.__dict__.items() if getattr(func, 'exposed', False)
        }
        self._wsgi_app = IPWhitelistMiddleware(
            SharedDataMiddleware(
                Request.application(self.dispatch_request),
                {'/': str(self.static_dir)}
            ),
            allowed_ips
        )
    
    def dispatch_request(self, request: Request) -> Response:
        """HTTP requests for things other than static files, i.e. `fetch()` requests from JavaScript,
        are routed through this method."""
        path = request.path.lstrip('/')
        if not path:
            return send_file(self.static_dir / 'index.html', request.environ)
        
        # Direct one-to-one mapping from URL path to endpoint
        elif method := self._endpoints.get(path):
            print(request.headers)
            try:
                output = method(self, **request.args)
            except TypeError as e:
                print(f"Got bad request: {e}")
                return Response(status=400) # Bad request
            
            if isinstance(output, Response):
                return output
            else:
                return Response(json.dumps(output), mimetype='application/json')

        return Response(status=404)

    @expose
    def add_edge(self, source: SupportsInt, target: SupportsInt):
        """Adds an edge to the preference graph."""
        source, target = int(source), int(target)
        self.prefs.add_pref(source, target)
        print(f"Added edge: {source} -> {target}")
    
    @expose
    def available_runs(self):
        """Returns a list of available runs."""
        if not self.root_dir.exists():
            return []
        if self.root_dir.suffix == '.classroom':
            return []
        
        return [
            folder for folder in self.root_dir.iterdir()
            if folder.is_dir() and folder.suffix == '.classroom'
        ]
    
    @expose
    def graph(self):
        """Serve the preference graph in Cytoscape.js format."""
        return self.prefs.to_cytoscape()
    
    @expose
    def thumbnail(self, node: SupportsInt):
        """Serve the thumbnail image."""
        node = int(node)
        
        s, _ = self.reader[node]
        mid_step = tree_map(lambda x: x[len(x) // 2], s)
        pixels = render_array(Ant().sys, mid_step, 480, 480)
        img = Image.fromarray(pixels)

        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        return Response(output, mimetype='image/png')

    @expose
    def viewer_html(self, node: SupportsInt):
        """Serve the viewer HTML."""
        from brax.io.html import render
        node = int(node)
        s, _ = self.reader[node]
        html = render(Ant().sys, [tree_map(lambda field: field[i], s) for i in range(len(s.pos))])
        return Response(html, mimetype='text/html')
        
    @property
    def database_path(self) -> Optional[Path]:
        """Returns the path to the currently selected database directory."""
        return self.root_dir / self.selected_run if self.selected_run else None
    
    def serve_forever(self):
        """Run the Werkzeug app."""
        run_simple(
            '0.0.0.0' if self.allow_remote else 'localhost',
            self.port, self._wsgi_app,
            use_debugger=self.debug, use_reloader=self.debug,
            ssl_context='adhoc' if self.use_tls else None
        )


@dataclass
class IPWhitelistMiddleware:
    """Middleware that restricts access to certain IPs."""
    app: Callable
    allowed_ips: tuple[str, ...]
    
    def __call__(self, environ: dict[str, Any], start_response: Callable) -> Iterable[bytes]:
        request_ip = environ.get('REMOTE_ADDR')
        if request_ip != '127.0.0.1' and request_ip not in self.allowed_ips:
            return Unauthorized(description='Invalid IP address')(environ, start_response)
        
        return self.app(environ, start_response)
