from .database_eval_handle import DatabaseEvalHandle
from .pref_graph import PrefGraph

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from sanic import Request, Sanic
from sanic.exceptions import Unauthorized
from sanic.response import html, json, raw
import cv2
import warnings

ROOT_DIR = Path(__file__).parent


app = Sanic("Classroom")
app.static('/', ROOT_DIR / 'client/public')
app.static('/', ROOT_DIR / "client/public/index.html")


prefs = PrefGraph(allow_cycles=True)
for i, j in zip(range(10), range(1, 11)):
    prefs.add_pref(i, j)


@lru_cache()
def database_handle(root_dir: Path) -> DatabaseEvalHandle:
    return DatabaseEvalHandle(root_dir)


@app.on_request
def filter_ips(request: Request):
    whitelist = request.app.config['allowed_ips']

    if request.ip != '127.0.0.1' and request.ip not in whitelist:
        raise Unauthorized("Invalid IP address.")


@app.websocket('/feedback')
async def feedback_socket(request, ws):
    """Receive feedback from the client and send it to the database."""
    while True:
        comparison = await ws.recv()
        # comparison = json.loads(comparison)
        print(comparison)


@app.route("/graph")
async def graph(request):
    """Serve the preference graph in Cytoscape.js format."""
    return json(prefs.to_cytoscape())


@app.route("/thumbnail/<node:int>")
async def thumbnail(request, node: int):
    """Serve the thumbnail image."""
    pixels = database_handle(app.config['database']).get_thumbnail(int(node))
    _, img = cv2.imencode('.png', pixels[:, :, ::-1])
    return raw(img.data, content_type='image/png')


@app.route("/viewer_html/<node:int>")
async def viewer_html(request, node: int):
    """Serve the viewer HTML."""
    markup = database_handle(app.config['database']).get_viewer_html(int(node))
    return html(markup)


if __name__ == '__main__':
    parser = ArgumentParser(description="Run the Classroom GUI server.")
    parser.add_argument('--allowed-ips', nargs='*', type=str, help="List of allowed remote IPs.")
    parser.add_argument('--database', type=Path, help="Path to the database directory.")
    parser.add_argument('--usernames', nargs='*', type=str, default=(), help="Usernames for which to generate tokens.")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on.")
    args = parser.parse_args()

    # Prevent annoying DeprecationWarnings from inside Sanic
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='the imp module')

    app.config.update(**vars(args))
    app.run(
        host='localhost' if not args.allowed_ips else '0.0.0.0',
        port=args.port,
        fast=True   # Use all available cores when needed
    )
