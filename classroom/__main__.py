from .database_eval_handle import DatabaseEvalHandle
from .pref_graph import PrefGraph
from .renderer import Renderer

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from sanic import Request, Sanic
from sanic.exceptions import Unauthorized
from sanic.response import html, raw
import cv2
import networkx as nx
import warnings

ROOT_DIR = Path(__file__).parent


app = Sanic("Classroom")
app.static('/', ROOT_DIR / 'client/public')
app.static('/', ROOT_DIR / "client/public/index.html")


@app.on_request
def maybe_open_handle(request: Request):
    ctx = request.app.ctx
    if not hasattr(ctx, 'handle'):
        ctx.handle = DatabaseEvalHandle(request.app.config['database'])


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

    handle: DatabaseEvalHandle = request.app.ctx.handle
    clip_ids = list(handle.clip_paths)
    random.shuffle(clip_ids)

    with PrefGraph.open(request.app.config['database'] / 'prefs.pkl') as pref_graph:
        while clip_ids:
            call = json.loads(await ws.recv())

            async def reply(result):
                """Boilerplate for sending a reply to the client."""
                await ws.send(json.dumps({'id': call.get('id'), 'result': result}))

            match call.get('method'), call.get('params'):
                # Add a strict preference to the graph, returning the next pair of clips to compare.
                case 'addPref', {'nodes': [str(clipA), str(clipB)], 'strict': bool(strict)}:
                    if strict:
                        pref_graph.add_pref(clipA, clipB)
                    else:
                        pref_graph.add_indifference(clipA, clipB)

                    await reply({
                        'clipA': pref_graph.median(),
                        'clipB': clip_ids.pop()
                    })
                
                # Return the first pair of clips to compare.
                case 'clips', None:
                    await reply({ 'clipA': clip_ids.pop(), 'clipB': clip_ids.pop() })
                
                # Serve the preference graph in Cytoscape.js format.
                case 'getGraph', None:
                    await reply({
                        'nodes': [
                            {'data': {'id': str(node), 'value': node, 'name': node_id_to_human_timestamp(node)}}
                            for node in pref_graph.strict_prefs.nodes
                        ],
                        'strictPrefs': [
                            {'data': {'source': a, 'target': b, 'strict': True}}
                            for (a, b) in pref_graph.strict_prefs.edges
                        ],
                        'indifferences': [
                            {'data': {'source': a, 'target': b, 'indiff': True}}
                            for (a, b) in pref_graph.indifferences.edges
                        ],
                        'connectedNodes': pref_graph.strict_prefs.number_of_nodes(),
                        'totalNodes': len(clip_ids),

                        'longestPath': nx.dag_longest_path_length(pref_graph.strict_prefs),
                        'numPrefs': pref_graph.strict_prefs.number_of_edges(),
                        'numIndifferences': pref_graph.indifferences.number_of_edges(),
                    })
                
                # Compute a planar layout for the graph
                case 'getPlanarLayout', None:
                    await reply({
                        node: dict(x=x, y=y)
                        for node, (x, y) in nx.planar_layout(pref_graph.strict_prefs).items()
                    })
                case _:
                    warnings.warn(f"Malformed RPC message: {call}")
                    await ws.send(json.dumps({'id': call.get('id'), 'error': 'Malformed RPC message.'}))


def node_id_to_human_timestamp(node_id: str) -> str:
    try:
        timestamp = int(node_id)
    except ValueError:
        return node_id
    
    return datetime.fromtimestamp(timestamp // 1_000_000_000).strftime('%b-%d %H:%M:%S')


@app.route("/thumbnail/<node>/<frame:int>")
async def thumbnail(request, node: str, frame: int = 60):
    """Serve the thumbnail image."""
    pixels = request.app.ctx.handle.thumbnail(node, frame)
    _, img = cv2.imencode('.png', pixels[:, :, ::-1])
    return raw(img.data, content_type='image/png')


@app.route("/viewer_html/<node>")
async def viewer_html(request, node: str):
    """Serve the viewer HTML."""
    markup = request.app.ctx.handle.viewer_html(node)
    return html(markup)


if __name__ == '__main__':
    parser = ArgumentParser(description="Run the Classroom GUI server.")
    parser.add_argument('--allowed-ips', nargs='*', type=str, help="List of allowed remote IPs.")
    parser.add_argument('--database', type=Path, help="Path to the database directory.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('--usernames', nargs='*', type=str, default=(), help="Usernames for which to generate tokens.")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on.")
    args = parser.parse_args()

    # Prevent annoying DeprecationWarnings from inside Sanic
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='the imp module')

    app.config.update(**vars(args))
    app.run(
        auto_reload=args.debug,
        debug=args.debug,
        host='localhost' if not args.allowed_ips else '0.0.0.0',
        port=args.port,
        fast=True   # Use all available cores when needed
    )
