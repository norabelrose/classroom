from .pref_dag import PrefDAG
from .pref_graph import PrefGraph
import networkx as nx
import random


class GraphSorter:
    """
    Implements a simple binary search algorithm for inserting a new node into
    a preference graph in roughly O(log n) time. The basic idea is to first find
    the topological generations of the graph- essentially, a topological ordering
    where "equal" nodes are grouped together- then do binary search on this grouped
    ordering.
    """
    def __init__(self, graph: PrefDAG, seed: int | None = None):
        self.graph = graph

        # We use reverse topological ordering so that least preferred nodes come first;
        # the other way around is too confusing.
        self.generations = [
            sorted(gen)
            for gen in nx.topological_generations(graph.nonisolated.strict_prefs)
        ]
        self.generations.reverse()

        self.isolated = [n for n in graph.nodes if graph.degree(n) == 0]
        random.seed(seed)
        random.shuffle(self.isolated)
        self.reset()
    
    def current_pair(self):
        return self.query, self.pivot
    
    def reset(self):
        # Bootstrap the pivot generation if needed
        if not self.generations and self.isolated:
            self.generations = [[self.isolated.pop()]]
        
        self.lo = 0
        self.hi = len(self.generations)

        # We've run out of isolated nodes to query
        if not len(self.isolated):
            self.query = None
            self.pivot = None
            return
        
        self.query = self.isolated.pop()
        self.sample_pivot()
    
    def sample_pivot(self):
        midpoint = (self.lo + self.hi) // 2
        pivot_gen = self.generations[midpoint]
        
        idx = random.randrange(0, len(pivot_gen))
        self.pivot = pivot_gen[idx]

    @property
    def done(self):
        return self.lo >= self.hi
    
    def lesser(self):
        """Mark the query as being less than the pivot."""
        self.graph.add_greater(self.pivot, self.query)

        pivot_gen = (self.lo + self.hi) // 2
        self.hi = pivot_gen

        # Create new generation on the left if needed
        if pivot_gen == 0 or self.done:
            self.generations.insert(self.lo, [self.query])
            self.reset()        
        else:
            self.sample_pivot()
        
        return self.current_pair()
    
    def equals(self):
        """Mark the query as being equal to the pivot."""
        self.graph.add_equals(self.query, self.pivot)

        pivot_gen = (self.lo + self.hi) // 2
        self.generations[pivot_gen].append(self.query)
        self.reset()

        return self.current_pair()
    
    def greater(self):
        """Mark the query as being greater than the pivot."""
        self.graph.add_greater(self.query, self.pivot)

        pivot_gen = (self.lo + self.hi) // 2
        self.lo = pivot_gen + 1

        # Create new generation on the right if needed
        if pivot_gen == len(self.generations) - 1 or self.done:
            self.generations.insert(self.hi, [self.query])
            self.reset()
        else:
            self.sample_pivot()
        
        return self.current_pair()
