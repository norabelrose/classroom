from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import networkx as nx
import pickle


class PrefGraph(nx.DiGraph):
    """`PrefGraph` represents a partial weak preference ordering over clips as a weighted directed graph.
    Edge weights represent the strength of the preference of A over B, and indifferences are represented
    as edges with zero weight. Clips are represented with string IDs.

    By default, `PrefGraph` enforces certain coherence properties expected of preference orderings:
    - Quasi-transitivity: Strict preferences must be transitive, and therefore the subgraph representing
    them must be acyclic. Violating this property will result in a `TransitivityViolation` exception, which
    has a `cycle` attribute that can be used to display the offending cycle to the user. We do not assume
    indifferences are transitive due to the Sorites paradox.
    See <https://en.wikipedia.org/wiki/Sorites_paradox#Resolutions_in_utility_theory> for discussion.
    - Non-negativity: Edge weights must be non-negative.

    If you need to bypass these checks, you can use `add_edge_unsafe` to do so.
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
        return f'PrefGraph({num_indiff} indifferences, {len(self.strict_prefs)} strict preferences)'
    
    def add_edge(self, a: str, b: str, weight: float = 1.0, **attr):
        """Add an edge to the graph, and check for coherence violations. Usually you
        should use the `add_greater` or `add_equals` wrapper methods instead of this method."""
        if weight < 0:
            raise CoherenceViolation("Preferences must have non-negative weight")
        
        self.add_edge_unsafe(a, b, **attr)
        if weight > 0:
            # This is a strict preference, so we should check for cycles
            try:
                cycle = nx.find_cycle(self.strict_prefs, source=a)
            except nx.NetworkXNoCycle:
                pass
            else:
                # Remove the edge we just added.
                self.remove_edge(a, b)

                ex = TransitivityViolation(f"Adding {a} > {b} would create a cycle: {cycle}")
                ex.cycle = cycle
                raise ex
    
    def add_edge_unsafe(self, a: str, b: str, weight: float = 1.0, **attr):
        """Add a preference without checking for coherence violations. Should probably only be used in
        cases where you explicitly want to model incoherent preferences."""
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
    
    def searchsorted(self) -> Generator[str, bool, int]:
        """Coroutine for asynchronously performing a binary search on the strict preference relation."""
        ordering = list(nx.topological_sort(self.strict_prefs))
        lo, hi = 0, len(ordering)

        while lo < hi:
            pivot = (lo + hi) // 2
            greater = yield ordering[pivot]
            if greater:
                lo = pivot + 1
            else:
                hi = pivot
        
        return lo
    
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
    
    def is_quasi_transitive(self) -> bool:
        """Return whether the strict preferences are acyclic. This should return `True` unless
        you've used a method like `add_greater_unsafe` to avoid coherence checks."""
        return nx.is_directed_acyclic_graph(self.strict_prefs)
    
    def median(self) -> str:
        """Return the node at index n // 2 of a topological ordering of the strict preference relation."""
        middle_idx = len(self.strict_prefs) // 2

        for i, node in enumerate(nx.topological_sort(self.strict_prefs)):
            if i == middle_idx:
                return node
        
        raise RuntimeError("Could not find median")
    
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

class TransitivityViolation(CoherenceViolation):
    """Raised when a mutation of a `PrefGraph` would cause transitivity to be violated"""
    cycle: list[int]
