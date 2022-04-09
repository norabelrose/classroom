from classroom import PrefDAG
from contextlib import nullcontext
from hypothesis_networkx import graph_builder
from hypothesis import given
import networkx as nx
import pytest


@given(graph=graph_builder(graph_type=nx.DiGraph))
def test_pref_dag_init(graph: nx.DiGraph):
    # Raise an error on initialization iff the graph is not a DAG. Annoyingly NetworkX actually
    # catches the TransitivityViolation exception thrown by PrefDAG.add_edges_from and rethrows
    # it with a generic error message.
    with pytest.raises(Exception) if not nx.is_directed_acyclic_graph(graph) else nullcontext():
        PrefDAG(graph)
