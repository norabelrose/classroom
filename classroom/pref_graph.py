from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator
import networkx as nx
import pickle
from .fas import eades_fas
if TYPE_CHECKING:   # Prevent circular import
    from .pref_dag import PrefDAG


class PrefGraph(nx.DiGraph):
    """
    `PrefGraph` represents a possibly cyclic set of preferences over clips as a weighted directed graph.
    Edge weights represent the strength of the preference of A over B, and indifferences are represented
    as edges with zero weights. Clips are represented as string IDs. If you want to prevent cycles from
    being added to the graph in an online fashion, you should probably use `PrefDAG` instead.
    """
    @classmethod
    @contextmanager
    def open(cls, path: Path | str):
        """Open a pickled `PrefGraph` from a file, and ensure it is saved when the context is exited."""
        path = Path(path)
        if path.exists():
            with open(path, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = cls()
        
        try:
            yield graph
        finally:        
            with open(path, 'wb') as f:
                pickle.dump(graph, f)
    
    @property
    def indifferences(self) -> nx.Graph:
        """Return a read-only, undirected view of the subgraph containing only indifferences."""
        return nx.graphviews.subgraph_view(
            self,
            filter_edge=lambda a, b: self.edges[a, b].get('weight', 1.0) == 0.0
        ).to_undirected(as_view=True)
    
    @property
    def strict_prefs(self) -> nx.DiGraph:
        """Return a read-only view of the subgraph containing only strict preferences."""
        return nx.graphviews.subgraph_view(
            self,
            filter_edge=lambda a, b: self.edges[a, b].get('weight', 1.0) > 0
        )
    
    def __repr__(self) -> str:
        num_indiff = self.indifferences.number_of_edges()
        return f'{type(self).__name__}({len(self.strict_prefs)} strict prefs, {num_indiff} indifferences)'
    
    def add_edge(self, a: str, b: str, weight: float = 1.0, **attr):
        """Add an edge to the graph, and check for coherence violations. Usually you
        should use the `add_greater` or `add_equals` wrapper methods instead of this method."""
        if weight < 0:
            raise CoherenceViolation("Preferences must have non-negative weight")
        
        super().add_edge(a, b, weight=weight, **attr)
    
    def add_greater(self, a: str, b: str, weight: float = 1.0, **attr):
        """Try to add the preference `a > b`, and throw an error if the expected coherence
        properties of the graph would be violated."""
        if weight <= 0.0:
            raise CoherenceViolation("Strict preferences must have positive weight")
        
        attr.update(weight=weight)
        self.add_edge(a, b, **attr)
    
    def add_equals(self, a: str, b: str, **attr):
        """Try to add the indifference relation `a ~ b`, and throw an error if the expected
        coherence properties of the graph would be violated."""
        if attr.setdefault('weight', 0.0) != 0.0:
            raise CoherenceViolation("Indifferences cannot have nonzero weight")

        self.add_edge(a, b, **attr)
    
    # Convenience aliases
    add_pref = add_greater
    add_indiff = add_equals
    
    def draw(self):
        """Displays a visualization of the graph using `matplotlib`. Strict preferences
        are shown as solid arrows, and indifferences are dashed lines."""
        strict_subgraph = self.strict_prefs

        pos = nx.drawing.spring_layout(strict_subgraph)
        nx.draw_networkx_nodes(strict_subgraph, pos)
        nx.draw_networkx_edges(strict_subgraph, pos)
        nx.draw_networkx_edges(self.indifferences, pos, arrowstyle='-', style='dashed')
        nx.draw_networkx_labels(strict_subgraph, pos)
    
    def equivalence_classes(self) -> Generator[set[str], None, None]:
        """Yields sets of nodes that are equivalent under the indifference relation."""
        return nx.connected_components(self.indifferences)
    
    def find_acyclic_subgraph(self) -> 'PrefDAG':
        """Return an acyclic subgraph of this graph as a `PrefDAG`. The algorithm will try
        to remove as few preferences as possible, but it is not guaranteed to be optimal.
        If the graph is already acyclic, the returned `PrefDAG` will be isomorphic to this graph."""
        fas = set(eades_fas(self.strict_prefs))
        return PrefDAG((
            (u, v, d) for u, v, d in self.edges(data=True)  # type: ignore
            if (u, v) not in fas
        ))
    
    def is_quasi_transitive(self) -> bool:
        """Return whether the strict preferences are acyclic."""
        return nx.is_directed_acyclic_graph(self.strict_prefs)
    
    def pref_prob(self, a: str, b: str, eps: float = 5e-324) -> float:
        """Return the probability that `a` is preferred to `b`."""
        a_weight = self.pref_weight(a, b)
        denom = a_weight + self.pref_weight(b, a)

        # If there's no strict preference between a and b, then the
        # probability that A is preferred to B is 1/2.
        return (a_weight + eps) / (denom + 2 * eps)
    
    def pref_weight(self, a: str, b: str, default: float = 0.0) -> float:
        """
        Return the weight of the preference `a > b`, or 0.0 if there is no such
        preference. Preferences with no explicit weight are assumed to have weight 1.
        """
        attrs = self.edges.get((a, b), None)
        return attrs.get('weight', 1.0) if attrs is not None else default
    
    def unlink(self, a: str, b: str):
        """Remove the preference relation between `a` and `b`."""
        try:
            self.remove_edge(a, b)
        except nx.NetworkXError:
            # Try removing the edge in the other direction.
            try:
                self.remove_edge(b, a)
            except nx.NetworkXError:
                raise KeyError(f"No preference relation between {a} and {b}")


class CoherenceViolation(Exception):
    """Raised when an operation would violate the coherence of the graph."""
    pass
