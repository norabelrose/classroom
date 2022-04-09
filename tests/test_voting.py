from classroom import PrefGraph
from classroom.voting import ranked_pairs
from hypothesis_networkx import graph_builder
from hypothesis import given, strategies as st
from itertools import pairwise
import networkx as nx


# Generate random DAGs by orienting undirected edges by the order of the node values
dags = graph_builder().map(lambda g: PrefGraph(map(sorted, g.edges)))

@given(ballots=st.lists(dags, min_size=1))
def test_ranked_pairs(ballots: list[PrefGraph]):
    results = ranked_pairs(ballots)

    # Coherence
    assert nx.is_directed_acyclic_graph(results)

    # TODO: Test Condorcet property etc.


# The example given in the ranked pairs Wikipedia article
def test_ranked_pairs_example():
    ballots = (
        [PrefGraph(pairwise('wxzy'))] * 7 +
        [PrefGraph(pairwise('wyxz'))] * 2 +
        [PrefGraph(pairwise('xyzw'))] * 4 +
        [PrefGraph(pairwise('xzwy'))] * 5 +
        [PrefGraph(pairwise('ywxz'))] * 1 +
        [PrefGraph(pairwise('yzwx'))] * 8
    )
    results = ranked_pairs(ballots)
    ordering = list(nx.topological_sort(results))

    assert ordering == ['w', 'x', 'z']