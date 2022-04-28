from classroom import GraphManager, PrefDAG, QueryStrategy
from hypothesis import given, strategies as st
import networkx as nx


# Test that all the built-in query strategies implement valid sorting algorithms
# for nodes with a known total order- in this case, integers.
@given(
    # We have to use unique=True here since NetworkX doesn't allow duplicate nodes.
    st.lists(st.integers(), min_size=2, unique=True),
    st.sampled_from(QueryStrategy.available_strategies)
)
def test_correctness(nodes: list[int], strategy: str):
    graph = PrefDAG()
    graph.add_nodes_from(nodes)
    manager = GraphManager(graph, strategy)

    while not manager.done:
        a, b = manager.current_query

        if a > b:
            manager.commit_feedback('>')
        elif a < b:
            manager.commit_feedback('<')
        else:
            manager.commit_feedback('=')
    
    assert list(nx.topological_sort(manager.graph)) == sorted(nodes, reverse=True)