from .database_eval_handle import DatabaseEvalHandle
from .pref_graph import PrefGraph
from .renderer import Renderer

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
    """Opens a WebSocket-based Remote Procedure Call connection with the client."""
    import json
    import random

    async def reply(msg_id: int, result):
        """Boilerplate for sending a reply to the client."""
        await ws.send(
            json.dumps({
                'id': msg_id,
                'result': result
            })
        )

    handle = DatabaseEvalHandle(request.app.config['database'])
    pref_graph = PrefGraph()
    clip_ids = list(handle.clip_paths)
    random.shuffle(clip_ids)

    while clip_ids:
        call = json.loads(await ws.recv())

        match call:
            case {'method': 'clips', 'params': _, 'id': msg_id}:
                await reply(msg_id, {
                    "clipA": clip_ids.pop(),
                    "clipB": clip_ids.pop()
                })
            case {'method': 'commit', 'params': {'better': better, 'worse': worse}, 'id': msg_id}:
                pref_graph.add_pref(better, worse)
                await reply(msg_id, {
                    "clipA": clip_ids.pop(),
                    "clipB": clip_ids.pop()
                })
            case {'method': 'getGraph', 'params': _, 'id': msg_id}:
                """Serve the preference graph in Cytoscape.js format."""
                await reply(msg_id, pref_graph.to_cytoscape())
            case _:
                warnings.warn(f"Malformed RPC message: {call}")


@app.route("/thumbnail/<node>")
async def thumbnail(request, node: str):
    """Serve the thumbnail image."""
    pixels = database_handle(app.config['database']).thumbnail(node)
    _, img = cv2.imencode('.png', pixels[:, :, ::-1])
    return raw(img.data, content_type='image/png')


@app.route("/viewer_html/<node>")
async def viewer_html(request, node: str):
    """Serve the viewer HTML."""
    markup = database_handle(app.config['database']).viewer_html(node)
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
