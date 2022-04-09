"""
Ranked choice voting algorithms to aggregate multiple individuals' preference graphs
into a single collective preference ordering.
"""
from .pref_graph import PrefGraph, TransitivityViolation
from collections import Counter
from typing import Iterable
import networkx as nx


def ranked_pairs(ballots: Iterable[PrefGraph]) -> PrefGraph:
    """
    Nicolaus Tidemann's ranked pairs voting algorithm. Returns a weighted DAG where
    the weights correspond to the margin of victory for the victor in each pair.
    O(n^2) time and space complexity.
    See <https://en.wikipedia.org/wiki/Ranked_pairs>
    """
    tally = Counter((
        (candidate, loser)
        for ballot in ballots
        for candidate in ballot
        for loser in nx.dag.descendants(ballot.strict_prefs, candidate)  # Indifferences are ignored
    ))

    # Add pairs with the highest vote margin to the output graph until
    # doing so would create a cycle
    results = PrefGraph()
    for (winner, runner_up), count in tally.most_common():
        try:
            results.add_greater(winner, runner_up, weight=count)
        except TransitivityViolation:
            break
    
    return results
