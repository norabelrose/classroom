from classroom.fas import berger_shor_fas, eades_fas, fas_to_max_acyclic_subgraph
from hypothesis_networkx import graph_builder
from hypothesis import given, strategies as st
import networkx as nx


@given(graph=graph_builder(graph_type=nx.DiGraph))
def test_berger_shor(graph: nx.DiGraph):
    subgraph = fas_to_max_acyclic_subgraph(graph, berger_shor_fas(graph, seed=7295))

    # Theoretical worst case FAS size is m / 2 - c1 / delta ** 0.5
    assert subgraph.number_of_edges() >= graph.number_of_edges() // 2
    assert nx.is_directed_acyclic_graph(subgraph)


@given(graph=graph_builder(graph_type=nx.DiGraph))
def test_eades(graph: nx.DiGraph):
    subgraph = fas_to_max_acyclic_subgraph(graph, eades_fas(graph))

    assert subgraph.number_of_edges() >= graph.number_of_edges() // 2
    assert nx.is_directed_acyclic_graph(subgraph)
