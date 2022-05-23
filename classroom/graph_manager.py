from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from .pref_dag import PrefDAG
from .pref_graph import PrefGraph
from .query_strategies import QueryStrategy
import networkx as nx
import pickle
import time


class GraphManager:
    """
    Wrapper for a `PrefGraph` or `PrefDAG` object. Once a graph is wrapped
    you must route all edits to the graph through the relevant methods of this class.
    """
    @classmethod
    @contextmanager
    def open(cls, path: Path | str, query_strategy: str = 'binsearch'):
        """
        Create a GraphManager object wrapping a pickled `PrefGraph`, and ensure changes are saved
        to disk when the context is exited.
        """
        path = Path(path)
        if path.exists():
            with open(path, 'rb') as f:
                manager = pickle.load(f)
        else:
            clip_dir = path.parent / 'clips'
            dag = PrefDAG()

            last_populated = dag.graph.get('last_populated', 0)
            if last_populated > clip_dir.stat().st_mtime:
                return
            
            dag.add_nodes_from(path.stem for path in clip_dir.glob(f'*.pkl'))

            # The clips' filenames are Unix timestamps, but the human-readable ID
            # for each clip is its 1-based index in a sorted list of all the timestamps;
            # i.e. "Clip #42" is the 42nd clip to have been saved during learning.
            for i, node in enumerate(sorted(dag.nodes), start=1):
                dag.nodes[node]['id'] = i
            
            # Update the last populated time so we don't run this method again unnecessarily
            dag.graph['last_populated'] = time.time()
            manager = cls(dag, query_strategy, copy=False)
        
        try:
            yield manager
        finally:        
            with open(path, 'wb') as f:
                pickle.dump(manager, f)
    
    def __init__(
            self,
            graph: PrefGraph,
            query_strategy: str = 'binsearch',
            *,
            copy: bool = True
        ):
        # Copy to make sure we don't modify the original graph
        self._graph = type(graph)(graph) if copy else graph
        self._strategy = QueryStrategy.from_name(query_strategy, self._graph)
    
    def commit_feedback(self, feedback: Literal['>', '<', '=']):
        """
        Commit feedback to the graph, updating the query strategy's state and adding
        an entry to the undo stack.
        """
        a, b = self._strategy.current_query
        match feedback:
            case '>':
                self._graph.add_pref(a, b)
            case '<':
                self._graph.add_pref(b, a)
            case '=':
                self._graph.add_indiff(a, b)
        
        self._strategy.register_feedback(feedback)
    
    @property
    def current_query(self) -> tuple[str, str]:
        return self._strategy.current_query
    
    @property
    def done(self) -> bool:
        return self._strategy.done
    
    @property
    def graph(self) -> PrefGraph:
        """Read-only view onto the underlying graph."""
        return nx.graphviews.generic_graph_view(self._graph)
    
    def add_pref(self, a: str, b: str, **attr):
        """Add a preference to the graph."""
        self._graph.add_pref(a, b, **attr)
    
    def add_indiff(self, a: str, b: str):
        """Add an indifference to the graph."""
        self._graph.add_indiff(a, b)
    
    def unlink(self, a: str, b: str):
        """Remove a preference or indifference from the graph."""
        self._graph.unlink(a, b)
    