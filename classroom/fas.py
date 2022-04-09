"""
Algorithms for finding the maximal acyclic subgraph of a directed graph. We use
these algorithms to remove intransitivities from preference graphs while minimizing
the number of preferences we have to discard. This problem is NP-hard in the general
case, but we can use heuristics to find decent approximations.
<https://en.wikipedia.org/wiki/Feedback_arc_set>
"""
from dataclasses import dataclass, field
from typing import Any, cast, Iterable, Literal
import networkx as nx
import numpy as np
import random


@dataclass
class LinkedListEntry:
    value: Any
    next: 'LinkedListEntry | None' = None
    prev: 'LinkedListEntry | None' = None

    def unlink(self):
        if self.prev is not None:
            self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev

@dataclass
class LinkedList:
    """Doubly linked list which unlinks entries from any other list they might be in when they are added.
    See https://github.com/dagrejs/dagre/blob/933822b991ea9b077521e5b797b8b4927c94f0a4/lib/data/list.js"""
    sentinel: LinkedListEntry = field(default_factory=lambda: LinkedListEntry(None))
    
    def enqueue(self, entry: LinkedListEntry):
        entry.unlink()
        entry.next = self.sentinel.next
        if self.sentinel.next is not None:
            self.sentinel.next.prev = entry
        
        self.sentinel.next = entry
        entry.prev = self.sentinel
    
    def dequeue(self) -> LinkedListEntry | None:
        entry = self.sentinel.next
        if entry is not None and entry is not self.sentinel:
            entry.unlink()
            return entry


def berger_shor_fas(graph: nx.DiGraph, seed: int | None = None) -> list:
    """See https://github.com/vereena42/Feedback-Arc-Set-Heuristics/blob/master/berger_shor.py"""
    nodes = list(graph.nodes)
    visited = np.zeros(len(nodes))
    edge_list = []

    # Create random linear ordering on vertices
    random.seed(seed)
    random.shuffle(nodes)

    # Main loop of Berger-Shor algorithm
    for node in nodes:
        left = []
        right = []
        visited[node] = True
        for neigh in graph.predecessors(node):
            if visited[neigh] == False:
                left.append((neigh, node))
        
        for neigh in graph.successors(node):
            if visited[neigh] == False:
                right.append((node, neigh))
        
        if len(right) < len(left):
            edge_list.extend(right)
        else:
            edge_list.extend(left)
    
    return edge_list


def eades_fas(graph: nx.DiGraph, weight: str = 'weight'):
    """
    Linear time algorithm for computing a "small" feedback arc set for a digraph.
    Variation of 'A fast and effective heuristic for the feedback arc set problem'
    by Eades, Lin, and Smyth (1993) for weighted digraphs. Adapted and ported from
    https://github.com/dagrejs/dagre/blob/933822b991ea9b077521e5b797b8b4927c94f0a4/lib/greedy-fas.js
    """    
    # Pylance and networkx *really* don't get along so just erase the type
    G: Any = graph.copy()

    # Pre-compute the weighted in and out-degree of each node, while
    # also updating the max in and out degrees in the graph.
    max_in, max_out = 0, 0
    for node in G.nodes:
        in_deg, out_deg = G.in_degree(node, weight=weight), G.out_degree(node, weight=weight)
        G.nodes[node].update(_in=in_deg, _out=out_deg)
        max_in = max(max_in, in_deg)
        max_out = max(max_out, out_deg)
    
    # Now we know how many buckets we need
    buckets = [LinkedList() for _ in range(max_in + max_out + 3)]
    zero_idx = max_in + 1

    def assign_bucket(node):
        attr = G.nodes[node]
        entry = attr.get('_entry')
        if entry is None:
            entry = attr['_entry'] = LinkedListEntry(node)
        
        if not attr['_in']:
            buckets[0].enqueue(entry)
        elif not attr['_out']:
            buckets[-1].enqueue(entry)
        else:
            delta = attr['_out'] - attr['_in']
            buckets[delta + zero_idx].enqueue(entry)
    
    def remove_node(node, return_parents = False) -> list:
        parents = []
        
        G.nodes[node]['_entry'].unlink()

        for a, b, w in G.in_edges(node, data=weight, default=1):
            G.nodes[a]['_out'] -= w
            assign_bucket(a)
            if return_parents:
                parents.append((a, b))
        
        for a, b, w in G.out_edges(node, data=weight, default=1):
            G.nodes[b]['_in'] -= w
            assign_bucket(b)
        
        G.remove_node(node)
        return parents
    
    for node in G.nodes:
        assign_bucket(node)

    while G:
        # Remove sinks & sources
        while sink := buckets[0].dequeue():
            remove_node(sink.value)
        while source := buckets[-1].dequeue():
            remove_node(source.value)
        
        # Remove the node with largest delta
        for bucket in reversed(buckets[:-1]):
            if entry := bucket.dequeue():
                yield from remove_node(entry.value, return_parents=True)
                break


def fas_to_max_acyclic_subgraph(graph: nx.DiGraph, fas: Iterable):
    """Given a feedback arc set, return the maximum acyclic subgraph."""
    subgraph = cast(nx.DiGraph, graph.copy())
    for a, b in fas:
        subgraph.remove_edge(a, b)
    
    return subgraph


def max_acyclic_subgraph(
        graph: nx.DiGraph,
        method: Literal['berger_shor', 'eades'],
        seed: int | None = 42,  # None for non-deterministic output; ignored by Eades
    ) -> nx.DiGraph:
    """Given a potentially cyclic digraph, return (an approximation to) the maximum acyclic subgraph."""
    match method:
        case 'berger_shor':
            return fas_to_max_acyclic_subgraph(graph, berger_shor_fas(graph, seed=seed))
        case 'eades':
            return fas_to_max_acyclic_subgraph(graph, eades_fas(graph))
        case _:
            raise ValueError(f'Unknown method: {method}')