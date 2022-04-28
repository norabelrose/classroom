from typing import Literal
from ..pref_dag import PrefDAG
from .query_strategy import QueryStrategy
import networkx as nx
import random


class BinarySearch(QueryStrategy):
    """
    Implements a simple binary search algorithm for inserting a new node into
    a `PrefDAG` with O(log n) human comparisons. The basic idea is to first find
    the topological generations of the graph- essentially, a topological ordering
    where "equal" nodes are grouped together- then do binary search on this grouped
    ordering.
    """
    short_name = 'binsearch'
    
    def __init__(self, graph: PrefDAG, seed: int | None = None):
        assert isinstance(graph, PrefDAG), "BinarySearch requires a PrefDAG"
        assert len(graph) >= 2, "BinarySearch requires at least 2 nodes"
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
    
    @property
    def current_query(self):
        return self.query, self.pivot

    @property
    def done(self):
        return not self.isolated and self.lo >= self.hi
        
    def reset(self):
        # We've run out of isolated nodes to query
        if not len(self.isolated):
            return
        
        # Bootstrap the pivot generation if needed
        if not self.generations and self.isolated:
            self.generations = [[self.isolated.pop()]]
        
        self.lo = 0
        self.hi = len(self.generations)
        
        self.query = self.isolated.pop()
        self.sample_pivot()
    
    def sample_pivot(self):
        midpoint = (self.lo + self.hi) // 2
        pivot_gen = self.generations[midpoint]
        
        idx = random.randrange(0, len(pivot_gen))
        self.pivot = pivot_gen[idx]
    
    def register_feedback(self, feedback: Literal['>', '<', '=']):
        pivot_gen = (self.lo + self.hi) // 2

        match feedback:
            case '>':
                self.lo = pivot_gen + 1

                # Create new generation on the right if needed
                if pivot_gen == len(self.generations) - 1 or self.lo >= self.hi:
                    self.generations.insert(self.hi, [self.query])
                    self.reset()
                else:
                    self.sample_pivot()
            case '<':
                self.hi = pivot_gen

                # Create new generation on the left if needed
                if pivot_gen == 0 or self.lo >= self.hi:
                    self.generations.insert(self.lo, [self.query])
                    self.reset()        
                else:
                    self.sample_pivot()
            case '=':
                self.generations[pivot_gen].append(self.query)
                self.reset()
