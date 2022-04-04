from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import networkx as nx
import pickle


class PrefGraph:
    """`PrefGraph` represents a partial preference ordering over clips as a graph with two types of edges:
    weighted directed edges that represent strict preferences, and undirected edges that represent indifferences.
    Clips are represented by hashable IDs.

    Importantly, unlike the strict preference
    relation, the indifference relation is *not* required to be transitive. This is because indifference
    is assumed to be approximate; A ~ B means something like |A - B| < ε, where ε is the smallest
    discernible difference in desirability. If the latent ordering is A < B < C, it might be the
    case that |A - B| < ε and |B - C| < ε, yet |A - C| > ε, yielding a non-transitive indifference relation.
    
    By default, `PrefGraph` only stores preferences that are explicitly added with `add_pref`
    or `add_indifference`, and does not add implicit preferences to enforce transitivity. On the other hand,
    transitivity can be falsified by finding a cycle in the graph; if A > B, B > C, and C > A,
    then transitivity would imply A > C, which contradicts antisymmetry (A > C and C > A cannot
    both hold). This means that a `PrefGraph` can be interpreted as transitive iff its strict
    preferences are acyclic.
    """
    @classmethod
    @contextmanager
    def open(cls, path: Path | str):
        """Open a `PrefGraph` from a file, and ensure it is saved when the context is exited."""
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
    
    def __init__(self):
        self.indifferences = nx.Graph()
        self.strict_prefs = nx.DiGraph()
    
    def __contains__(self, pair: tuple[str, str]) -> bool:
        """Return whether there is an edge from `a` to `b`."""
        return pair in self.strict_prefs or pair in self.indifferences
    
    def __getitem__(self, edge: tuple[str, str]) -> Any:
        """Return the attributes of the edge from `a` to `b`."""
        return self.indifferences[edge] if edge in self.indifferences else self.strict_prefs[edge]
    
    def __repr__(self) -> str:
        num_indiff = self.indifferences.number_of_edges()
        return f'PrefGraph({num_indiff} indifferences, {len(self.strict_prefs)} strict preferences)'
    
    def add_pref(self, a: str, b: str, weight: float = 1.0, **attr):
        """Add the preference `a > b`, with optional keyword attributes."""
        assert a != b, "Strict preference relations are irreflexive"

        # Preserve non-negativity of weights; edges with negative weight are normalized to edges with
        # positive weight in the opposite direction.
        if weight < 0:
            weight = -weight
            a, b = b, a
        
        attr.update(weight=weight)
        self.strict_prefs.add_edge(a, b, **attr)

        # Check to see if adding this preference created a cycle. If so, we want to include the
        # new cycle in the exception so that it can potentially be displayed to the user.
        try:
            cycle = nx.find_cycle(self.strict_prefs, source=a)
        except nx.NetworkXNoCycle:
            pass
        else:
            # Remove the edge we just added.
            self.strict_prefs.remove_edge(a, b)

            ex = TransitivityViolation(f"Adding {a} > {b} would create a cycle: {cycle}")
            ex.cycle = cycle
            raise ex
    
    def add_indifference(self, a: str, b: str, **attr):
        """Add the indifference relation `a ~ b`."""
        self.indifferences.add_edge(a, b, **attr)
        self.strict_prefs.add_node(a)
        self.strict_prefs.add_node(b)
    
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
    
    def cycles(self) -> list[list[str]]:
        """Return a list of cycles in the graph."""
        return list(nx.simple_cycles(self.strict_prefs))

    def draw(self):
        """Displays a visualization of the graph using `matplotlib`. Strict preferences
        are shown as solid arrows, and indifferences are dashed lines."""
        pos = nx.drawing.spring_layout(self.strict_prefs)
        nx.draw_networkx_nodes(self.strict_prefs, pos)
        nx.draw_networkx_edges(self.strict_prefs, pos)
        nx.draw_networkx_edges(self.indifferences, pos, arrowstyle='-', style='dashed')
        nx.draw_networkx_labels(self.strict_prefs, pos)
    
    def is_transitive(self) -> bool:
        """Return whether the strict preferences can be interpreted as transitive, that is, whether
        they are acyclic."""
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
        if (a, b) in self.indifferences:
            self.indifferences.remove_edge(a, b)
        elif (a, b) in self.strict_prefs:
            self.strict_prefs.remove_edge(a, b)
        else:
            raise KeyError(f"No preference relation between {a} and {b}")


class TransitivityViolation(Exception):
    """Raised when a mutation of a `PrefGraph` would cause transitivity to be violated"""
    cycle: list[int]
