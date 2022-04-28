from itertools import combinations
from typing import Literal
from ..pref_graph import PrefGraph
from .query_strategy import QueryStrategy
import random


class RandomSampling(QueryStrategy):
    """
    Samples pairs of clips uniformly at random without replacement; that is, if there
    is already an edge between clip A and clip B in either direction, the pairs (A, B)
    and (B, A) will never be sampled.
    """
    short_name = 'random'
    
    def __init__(self, G: PrefGraph, seed: int | None = None):
        E = G.edges
        self._graph = G

        self.pairs = [
            (a, b) for a, b in combinations(G.nodes, 2)
            if (a, b) not in E and (b, a) not in E
        ]
        random.seed(seed)
        random.shuffle(self.pairs)
    
    @property
    def current_query(self) -> tuple[str, str]:
        return self.pairs[-1]
    
    @property
    def done(self) -> bool:
        return not self.pairs
    
    def register_feedback(self, _: Literal['>', '<', '=']):
        del self.pairs[-1]
