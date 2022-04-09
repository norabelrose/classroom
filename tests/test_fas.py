from classroom.fas import max_acyclic_subgraph
from hypothesis_networkx import graph_builder
from hypothesis import given, strategies as st
import networkx as nx


@given(graph=graph_builder(graph_type=nx.DiGraph), method=st.sampled_from(['berger_shor', 'eades']))
def test_max_acyclic_subgraph(graph: nx.DiGraph, method):
    subgraph = max_acyclic_subgraph(graph, method)

    # All methods have theoretical worst case FAS size <= m // 2 or less
    assert subgraph.number_of_edges() >= graph.number_of_edges() // 2
    assert nx.is_directed_acyclic_graph(subgraph)

    # If the input graph was already acyclic, the output should be isomorphic
    # TODO: Figure out why this fails for Berger-Shor
    if method == 'eades' and nx.is_directed_acyclic_graph(graph):
        assert nx.is_isomorphic(subgraph, graph)
    
    # matcher = nx.algorithms.isomorphism.DiGraphMatcher(graph, subgraph)
    # assert matcher.subgraph_is_isomorphic()
