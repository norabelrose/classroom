from .bayes import update_rewards
from .graph_manager import GraphManager
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

    db_path: Path = app.config['database']
    strat = app.config.get('query_strategy', 'binsearch')

    with GraphManager.open(db_path / 'prefs.pkl', strat) as manager:
        G = manager.graph

        while not manager.done:
            call = json.loads(await ws.recv())

            async def reply(result):
                """Boilerplate for sending a reply to the client."""
                await ws.send(json.dumps({'id': call.get('id'), 'result': result}))

            match call.get('method'), call.get('params'):
                case 'add_pref', {'source': str(src), 'target': str(tgt), **attr}:
                    weight = attr.get('weight', 1)

                    # Report a human comparison from the Compare tab. This adds an
                    # edge to the graph, returning the next pair of clips to compare.
                    if manager.current_query == (src, tgt):
                        pref = '>'
                    elif manager.current_query == (tgt, src):
                        pref = '<'
                    
                    # Add or remove an arbitrary edge to the graph, outside of the Compare tab
                    else:
                        manager.add_pref(src, tgt, weight=weight)
                        await reply({ 'status': 'ok' })
                        continue

                    # Return the next pair of clips to compare
                    manager.commit_feedback(pref if weight > 0 else '=')

                    left, right = manager.current_query
                    await reply({ 'left': left, 'right': right })
                
                # Return the first pair of clips to compare.
                case 'clips', None:
                    left, right = manager.current_query
                    await reply({ 'left': left, 'right': right })
                
                # Serve the preference graph in Cytoscape.js format.
                case 'getGraph', None:
                    node_view = G.nonisolated.nodes
                    update_rewards(G)

                    await reply({
                        'nodes': [
                            {
                                'classes': ['clip'],
                                'data': {
                                    'id': str(node),
                                    'reward': node_view[node].get('reward', None),
                                }
                            }
                            for node in node_view
                        ],
                        'strictPrefs': [
                            {
                                'classes': ['pref'],
                                'data': {'source': a, 'target': b, 'strict': True}
                            }
                            for (a, b) in G.strict_prefs.edges
                        ],
                        'indifferences': [
                            {
                                'classes': ['pref'],
                                'data': {'source': a, 'target': b, 'strict': False}
                            }
                            for (a, b) in G.indifferences.edges
                        ]
                    })
                
                case 'getStats', None:
                    await reply({
                        'connectedNodes': len(G.nonisolated),
                        'totalNodes': len(G),

                        'longestPath': nx.dag_longest_path_length(G.strict_prefs),
                        'numPrefs': G.strict_prefs.number_of_edges(),
                        'numIndifferences': G.indifferences.number_of_edges(),
                    })
                
                case 'remove_pref', {'source': str(src), 'target': str(tgt)}:
                    manager.unlink(src, tgt)
                case _:
                    warnings.warn(f"Malformed RPC message: {call}")
                    await ws.send(json.dumps({'id': call.get('id'), 'error': "Malformed RPC message."}))


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
