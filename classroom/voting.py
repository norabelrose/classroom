"""
Ranked choice voting algorithms to aggregate multiple individuals' preference graphs
into a single collective preference ordering.
"""
from .pref_dag import PrefDAG, TransitivityViolation
from collections import Counter
from typing import Iterable


def ranked_pairs(ballots: Iterable[PrefDAG]) -> PrefDAG:
    """
    Nicolaus Tidemann's ranked pairs voting algorithm. Returns a weighted DAG where
    the weights correspond to the margin of victory for the victor in each pair.
    See <https://en.wikipedia.org/wiki/Ranked_pairs>
    """
    tally = Counter((
        (winner, loser)
        for ballot in ballots
        for winner, loser in ballot.transitive_closure().strict_prefs.edges
    ))

    # Iterate over pairs in descending order of vote count, adding each one to
    # the graph if it doesn't violate transitivity.
    results = PrefDAG()
    for (winner, runner_up), count in tally.most_common():
        try:
            results.add_pref(winner, runner_up, weight=count)
        except TransitivityViolation:
            continue
    
    return results
