from .bayes import estimate_rewards
from .pref_dag import PrefDAG
from .renderer import Renderer

from pathlib import Path
from sanic import Request, Sanic
from sanic.exceptions import Unauthorized
from sanic.response import html, raw
import cv2
import networkx as nx
import pickle
import warnings

ROOT_DIR = Path(__file__).parent


app = Sanic("Classroom")
app.static('/', ROOT_DIR / 'client/public')
app.static('/', ROOT_DIR / "client/public/index.html")


@app.on_request
def setup(request: Request):
    ctx = app.ctx

    if not hasattr(ctx, 'renderer'):
        # Load the renderer
        with open(app.config['database'] / 'renderer.pkl', 'rb') as f:
            ctx.renderer = pickle.load(f)
            assert isinstance(ctx.renderer, Renderer)


@app.on_request
def filter_ips(request: Request):
    whitelist = app.config['allowed_ips']

    if request.ip != '127.0.0.1' and request.ip not in whitelist:
        raise Unauthorized("Invalid IP address.")


@app.websocket('/feedback')
async def feedback_socket(request, ws):
    """Opens a WebSocket-based Remote Procedure Call connection with the client."""
    import json
    import random

    db_path: Path = app.config['database']
    clip_ids = [path.stem for path in db_path.glob('clips/*.pkl')]
    random.shuffle(clip_ids)

    with PrefDAG.open(db_path / 'prefs.pkl') as pref_graph:
        while clip_ids:
            call = json.loads(await ws.recv())

            async def reply(result):
                """Boilerplate for sending a reply to the client."""
                await ws.send(json.dumps({'id': call.get('id'), 'result': result}))

            match call.get('method'), call.get('params'):
                # Add a strict preference to the graph, returning the next pair of clips to compare.
                case 'addPref', {'nodes': [str(clipA), str(clipB)], 'strict': bool(strict)}:
                    if strict:
                        pref_graph.add_greater(clipA, clipB)
                    else:
                        pref_graph.add_equals(clipA, clipB)

                    await reply({
                        'clipA': pref_graph.median(),
                        'clipB': clip_ids.pop()
                    })
                
                # Return the first pair of clips to compare.
                case 'clips', None:
                    await reply({ 'clipA': clip_ids.pop(), 'clipB': clip_ids.pop() })
                
                # Serve the preference graph in Cytoscape.js format.
                case 'getGraph', None:
                    rewards = estimate_rewards(pref_graph, 'bradley-terry')
                    await reply({
                        'nodes': [
                            {'data': {
                                'id': str(node),
                                'value': node,
                                'name': f"{reward:.3f}"
                            }}
                            for node, reward in zip(pref_graph.nodes, rewards)
                        ],
                        'strictPrefs': [
                            {'data': {'source': a, 'target': b, 'strict': True}}
                            for (a, b) in pref_graph.strict_prefs.edges
                        ],
                        'indifferences': [
                            {'data': {'source': a, 'target': b, 'indiff': True}}
                            for (a, b) in pref_graph.indifferences.edges
                        ],
                        'connectedNodes': pref_graph.number_of_nodes(),
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


@app.route("/thumbnail/<node>/<frame:int>")
async def thumbnail(request, node: str, frame: int = 60):
    """Serve the thumbnail image."""
    db_path: Path = app.config['database']
    clip_path = db_path / 'clips' / f'{node}.pkl'
    with open(clip_path, 'rb') as f:
        clip = pickle.load(f)
        
    pixels = app.ctx.renderer.thumbnail(clip, frame)
    _, img = cv2.imencode('.png', pixels[:, :, ::-1])
    return raw(img.data, content_type='image/png')


@app.route("/viewer_html/<node>")
async def viewer_html(request, node: str):
    """Serve the viewer HTML."""
    db_path: Path = app.config['database']
    clip_path = db_path / 'clips' / f'{node}.pkl'
    with open(clip_path, 'rb') as f:
        clip = pickle.load(f)
    
    renderer: Renderer = app.ctx.renderer
    return html(renderer.viewer_html(clip))
