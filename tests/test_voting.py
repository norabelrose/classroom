from classroom import PrefDAG
from classroom.voting import ranked_pairs
from hypothesis_networkx import graph_builder
from hypothesis import given, strategies as st
from itertools import pairwise
import networkx as nx


# Generate random DAGs by orienting undirected edges by the order of the node values
dags = graph_builder().map(lambda g: PrefDAG(map(sorted, g.edges)))

@given(ballots=st.lists(dags, min_size=1))
def test_ranked_pairs(ballots: list[PrefDAG]):
    results = ranked_pairs(ballots)

    # Coherence
    assert nx.is_directed_acyclic_graph(results)

    # TODO: Test Condorcet property etc.


# The example given in the ranked pairs Wikipedia article
def test_ranked_pairs_example():
    ballots = (
        [PrefDAG(pairwise('wxzy'))] * 7 +
        [PrefDAG(pairwise('wyxz'))] * 2 +
        [PrefDAG(pairwise('xyzw'))] * 4 +
        [PrefDAG(pairwise('xzwy'))] * 5 +
        [PrefDAG(pairwise('ywxz'))] * 1 +
        [PrefDAG(pairwise('yzwx'))] * 8
    )
    results = ranked_pairs(ballots)
    ordering = list(nx.topological_sort(results))

    assert ordering == ['w', 'x', 'y', 'z']