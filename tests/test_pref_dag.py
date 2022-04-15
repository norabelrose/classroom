from classroom import PrefDAG
from contextlib import nullcontext
from hypothesis_networkx import graph_builder
from hypothesis import given
import networkx as nx
import pytest


@given(graph=graph_builder(graph_type=nx.DiGraph))
def test_init(graph: nx.DiGraph):
    # Raise an error on initialization iff the graph is not a DAG. Annoyingly NetworkX actually
    # catches the TransitivityViolation exception thrown by PrefDAG.add_edges_from and rethrows
    # it with a generic error message.
    with pytest.raises(Exception) if not nx.is_directed_acyclic_graph(graph) else nullcontext():
        PrefDAG(graph)

# Sample random DAGs by orienting the edges of random undirected graphs by the order of the node values
@given(graph=graph_builder().map(lambda g: PrefDAG(map(sorted, g.edges))))
def test_transitive_closure(graph: PrefDAG):
    tc = graph.transitive_closure()
    assert nx.is_isomorphic(tc.strict_prefs, nx.transitive_closure_dag(graph.strict_prefs))
    assert nx.is_isomorphic(tc.indifferences, graph.indifferences)