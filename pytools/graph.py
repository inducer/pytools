from __future__ import annotations


__copyright__ = """
Copyright (C) 2009-2013 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 James Stevens
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


__doc__ = """
Graph Algorithms
================

.. note::

    These functions are mostly geared towards directed graphs (digraphs).

.. autofunction:: reverse_graph
.. autofunction:: a_star
.. autofunction:: compute_sccs
.. autoexception:: CycleError
.. autofunction:: compute_topological_order
.. autofunction:: compute_transitive_closure
.. autofunction:: contains_cycle
.. autofunction:: compute_induced_subgraph
.. autofunction:: as_graphviz_dot
.. autofunction:: validate_graph
.. autofunction:: is_connected

Type Variables Used
-------------------

.. class:: NodeT

    Type of a graph node, can be any hashable type.

.. class:: GraphT

    A :class:`collections.abc.Mapping` representing a directed
    graph. The mapping contains one key representing each node in the
    graph, and this key maps to a :class:`collections.abc.Collection` of its
    successor nodes. Note that most functions expect that every graph node
    is included as a key in the graph.
"""

from typing import (
    Any, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, MutableSet,
    Optional, Set, Tuple, TypeVar)


try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


NodeT = TypeVar("NodeT", bound=Hashable)


GraphT: TypeAlias[NodeT] = Mapping[NodeT, Collection[NodeT]]


# {{{ reverse_graph

def reverse_graph(graph: GraphT[NodeT]) -> GraphT[NodeT]:
    """
    Reverses a graph *graph*.

    :returns: A :class:`dict` representing *graph* with edges reversed.
    """
    result: Dict[NodeT, Set[NodeT]] = {}

    for node_key, successor_nodes in graph.items():
        # Make sure every node is in the result even if it has no successors
        result.setdefault(node_key, set())

        for successor in successor_nodes:
            result.setdefault(successor, set()).add(node_key)

    return {k: frozenset(v) for k, v in result.items()}

# }}}


# {{{ a_star

def a_star(  # pylint: disable=too-many-locals
        initial_state: NodeT, goal_state: NodeT, neighbor_map: GraphT[NodeT],
        estimate_remaining_cost: Optional[Callable[[NodeT], float]] = None,
        get_step_cost: Callable[[Any, NodeT], float] = lambda x, y: 1
        ) -> List[NodeT]:
    """
    With the default cost and heuristic, this amounts to Dijkstra's algorithm.
    """

    from heapq import heappop, heappush

    if estimate_remaining_cost is None:
        # pylint: disable=function-redefined
        def estimate_remaining_cost(x: NodeT) -> float:
            if x != goal_state:
                return 1
            else:
                return 0

    class AStarNode:
        __slots__ = ["state", "parent", "path_cost"]

        def __init__(self, state: NodeT, parent: Any, path_cost: float) -> None:
            self.state = state
            self.parent = parent
            self.path_cost = path_cost

    inf = float("inf")
    init_remcost = estimate_remaining_cost(initial_state)
    assert init_remcost != inf

    queue = [(init_remcost, AStarNode(initial_state, parent=None, path_cost=0))]
    visited_states = set()

    while queue:
        _, top = heappop(queue)
        visited_states.add(top.state)

        if top.state == goal_state:
            result = []
            it: Optional[AStarNode] = top
            while it is not None:
                result.append(it.state)
                it = it.parent
            return result[::-1]

        for state in neighbor_map[top.state]:
            if state in visited_states:
                continue

            remaining_cost = estimate_remaining_cost(state)
            if remaining_cost == inf:
                continue
            step_cost = get_step_cost(top, state)

            estimated_path_cost = top.path_cost+step_cost+remaining_cost
            heappush(queue,
                (estimated_path_cost,
                    AStarNode(state, top, path_cost=top.path_cost + step_cost)))

    raise RuntimeError("no solution")

# }}}


# {{{ compute SCCs with Tarjan's algorithm

def compute_sccs(graph: GraphT[NodeT]) -> List[List[NodeT]]:
    to_search = set(graph.keys())
    visit_order: Dict[NodeT, int] = {}
    scc_root = {}
    sccs = []

    while to_search:
        top = next(iter(to_search))
        call_stack: List[Tuple[NodeT, Iterator[NodeT], Optional[NodeT]]] = [(top,
                                                                 iter(graph[top]),
                                                                 None)]
        visit_stack = []
        visiting = set()

        scc: List[NodeT] = []

        while call_stack:
            top, children, last_popped_child = call_stack.pop()

            if top not in visiting:
                # Unvisited: mark as visited, initialize SCC root.
                count = len(visit_order)
                visit_stack.append(top)
                visit_order[top] = count
                scc_root[top] = count
                visiting.add(top)
                to_search.discard(top)

            # Returned from a recursion, update SCC.
            if last_popped_child is not None:
                scc_root[top] = min(
                    scc_root[top],
                    scc_root[last_popped_child])

            for child in children:
                if child not in visit_order:
                    # Recurse.
                    call_stack.append((top, children, child))
                    call_stack.append((child, iter(graph[child]), None))
                    break
                if child in visiting:
                    scc_root[top] = min(
                        scc_root[top],
                        visit_order[child])
            else:
                if scc_root[top] == visit_order[top]:
                    scc = []
                    while visit_stack[-1] != top:
                        scc.append(visit_stack.pop())
                    scc.append(visit_stack.pop())
                    for item in scc:
                        visiting.remove(item)
                    sccs.append(scc)

    return sccs

# }}}


# {{{ compute topological order

class CycleError(Exception):
    """
    Raised when a topological ordering cannot be computed due to a cycle.

    :attr node: Node in a directed graph that is part of a cycle.
    """
    def __init__(self, node: NodeT) -> None:
        self.node = node


class HeapEntry:
    """
    Helper class to compare associated keys while comparing the elements in
    heap operations.

    Only needs to define :func:`pytools.graph.__lt__` according to
    <https://github.com/python/cpython/blob/8d21aa21f2cbc6d50aab3f420bb23be1d081dac4/Lib/heapq.py#L135-L138>.
    """
    def __init__(self, node: NodeT, key: Any) -> None:
        self.node = node
        self.key = key

    def __lt__(self, other: "HeapEntry") -> bool:
        return self.key < other.key


def compute_topological_order(graph: GraphT[NodeT],
                              key: Optional[Callable[[NodeT], Any]] = None) \
                              -> List[NodeT]:
    """Compute a topological order of nodes in a directed graph.

    :arg key: A custom key function may be supplied to determine the order in
        break-even cases. Expects a function of one argument that is used to
        extract a comparison key from each node of the *graph*.

    :returns: A :class:`list` representing a valid topological ordering of the
        nodes in the directed graph.

    .. note::

        * Requires the keys of the mapping *graph* to be hashable.
        * Implements `Kahn's algorithm <https://w.wiki/YDy>`__.

    .. versionadded:: 2020.2
    """
    # all nodes have the same keys when not provided
    keyfunc = key if key is not None else (lambda x: 0)

    from heapq import heapify, heappop, heappush

    order = []

    # {{{ compute nodes_to_num_predecessors

    nodes_to_num_predecessors = {node: 0 for node in graph}

    for node in graph:
        for child in graph[node]:
            nodes_to_num_predecessors[child] = (
                    nodes_to_num_predecessors.get(child, 0) + 1)

    # }}}

    total_num_nodes = len(nodes_to_num_predecessors)

    # heap: list of instances of HeapEntry(n) where 'n' is a node in
    # 'graph' with no predecessor. Nodes with no predecessors are the
    # schedulable candidates.
    heap = [HeapEntry(n, keyfunc(n))
            for n, num_preds in nodes_to_num_predecessors.items()
            if num_preds == 0]
    heapify(heap)

    while heap:
        # pick the node with least key
        node_to_be_scheduled = heappop(heap).node
        order.append(node_to_be_scheduled)

        # discard 'node_to_be_scheduled' from the predecessors of its
        # successors since it's been scheduled
        for child in graph.get(node_to_be_scheduled, ()):
            nodes_to_num_predecessors[child] -= 1
            if nodes_to_num_predecessors[child] == 0:
                heappush(heap, HeapEntry(child, keyfunc(child)))

    if len(order) != total_num_nodes:
        # any node which has a predecessor left is a part of a cycle
        raise CycleError(next(iter(n for n, num_preds in
            nodes_to_num_predecessors.items() if num_preds != 0)))

    return order

# }}}


# {{{ compute transitive closure

def compute_transitive_closure(
        graph: Mapping[NodeT, MutableSet[NodeT]]) -> GraphT[NodeT]:
    """Compute the transitive closure of a directed graph using Warshall's
    algorithm.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The mapping contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.MutableSet` of
        nodes that are connected to the node by outgoing edges. This graph may
        contain cycles. This object must be picklable. Every graph node must
        be included as a key in the graph.

    :returns: The transitive closure of the graph, represented using the same
        data type.

    .. versionadded:: 2020.2
    """
    # Warshall's algorithm

    from copy import deepcopy
    closure = deepcopy(graph)

    # (assumes all graph nodes are included in keys)
    for k in graph.keys():
        for n1 in graph.keys():
            for n2 in graph.keys():
                if k in closure[n1] and n2 in closure[k]:
                    closure[n1].add(n2)

    return closure

# }}}


# {{{ check for cycle

def contains_cycle(graph: GraphT[NodeT]) -> bool:
    """Determine whether a graph contains a cycle.

    :returns: A :class:`bool` indicating whether the graph contains a cycle.

    .. versionadded:: 2020.2
    """

    try:
        compute_topological_order(graph)
        return False
    except CycleError:
        return True

# }}}


# {{{ compute induced subgraph

def compute_induced_subgraph(graph: Mapping[NodeT, Set[NodeT]],
                             subgraph_nodes: Set[NodeT]) -> GraphT[NodeT]:
    """Compute the induced subgraph formed by a subset of the vertices in a
    graph.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The mapping contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.Set` of nodes
        that are connected to the node by outgoing edges.

    :arg subgraph_nodes: A :class:`collections.abc.Set` containing a subset of
        the graph nodes in the graph.

    :returns: A :class:`dict` representing the induced subgraph formed by
        the subset of the vertices included in `subgraph_nodes`.

    .. versionadded:: 2020.2
    """

    new_graph = {}
    for node, children in graph.items():
        if node in subgraph_nodes:
            new_graph[node] = children & subgraph_nodes
    return new_graph

# }}}


# {{{ as_graphviz_dot

def as_graphviz_dot(graph: GraphT[NodeT],
                    node_labels: Optional[Callable[[NodeT], str]] = None,
                    edge_labels: Optional[Callable[[NodeT, NodeT], str]] = None) \
                    -> str:
    """
    Create a visualization of the graph *graph* in the
    `dot <http://graphviz.org/>`__ language.

    :arg node_labels: An optional function that returns node labels
        for each node.

    :arg edge_labels: An optional function that returns edge labels
        for each pair of nodes.

    :returns: A string in the `dot <http://graphviz.org/>`__ language.
    """
    from pytools import UniqueNameGenerator
    id_gen = UniqueNameGenerator(forced_prefix="mynode")

    from pytools.graphviz import dot_escape

    if node_labels is None:
        node_labels = lambda x: str(x)

    if edge_labels is None:
        edge_labels = lambda x, y: ""

    node_to_id = {}

    for node, targets in graph.items():
        if node not in node_to_id:
            node_to_id[node] = id_gen()
        for t in targets:
            if t not in node_to_id:
                node_to_id[t] = id_gen()

    # Add nodes
    content = "\n".join(
        [f'{node_to_id[node]} [label="{dot_escape(node_labels(node))}"];'
         for node in node_to_id.keys()])

    content += "\n"

    # Add edges
    content += "\n".join(
        [f"{node_to_id[node]} -> {node_to_id[t]} "
         f'[label="{dot_escape(edge_labels(node, t))}"];'
         for (node, targets) in graph.items()
         for t in targets])

    return f"digraph mygraph {{\n{ content }\n}}\n"

# }}}


# {{{ validate graph

def validate_graph(graph: GraphT[NodeT]) -> None:
    """
    Validates that all successor nodes of each node in *graph* are keys in
    *graph* itself. Raises a :class:`ValueError` if not.
    """
    seen_nodes: Set[NodeT] = set()

    for children in graph.values():
        seen_nodes.update(children)

    if not seen_nodes <= graph.keys():
        raise ValueError(
            f"invalid graph, missing keys: {seen_nodes-graph.keys()}")

# }}}


# {{{

def is_connected(graph: GraphT[NodeT]) -> bool:
    """
    Returns whether all nodes in *graph* are connected, ignoring
    the edge direction.

    :returns: A :class:`bool` indicating whether the graph is connected.
    """
    if not graph:
        # https://cs.stackexchange.com/questions/52815/is-a-graph-of-zero-nodes-vertices-connected
        return True

    visited = set()

    undirected_graph = {node: set(children) for node, children in graph.items()}

    for node, children in graph.items():
        for child in children:
            undirected_graph[child].add(node)

    def dfs(node: NodeT) -> None:
        visited.add(node)
        for child in undirected_graph[node]:
            if child not in visited:
                dfs(child)

    dfs(next(iter(graph.keys())))

    return visited == graph.keys()


# vim: foldmethod=marker
