from classroom import GraphSorter, PrefDAG
from hypothesis import given, strategies as st
import networkx as nx


# Test that GraphSorter implements a valid sorting algorithm
@given(st.lists(st.integers(), unique=True))
def test_graph_sorter(nodes: list[int]):
    graph = PrefDAG()
    graph.add_nodes_from(nodes)

    sorter = GraphSorter(graph)
    while True:
        query, pivot = sorter.current_pair()
        if query is None:
            break

        if query > pivot:
            sorter.greater()
        elif query < pivot:
            sorter.lesser()
        else:
            sorter.equals()
    
    assert list(nx.topological_sort(graph)) == sorted(nodes, reverse=True)