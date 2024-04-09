import sys

import pytest


def test_compute_sccs():
    import random

    from pytools.graph import compute_sccs

    rng = random.Random(0)

    def generate_random_graph(nnodes):
        graph = {i: set() for i in range(nnodes)}
        for i in range(nnodes):
            for j in range(nnodes):
                # Edge probability 2/n: Generates decently interesting inputs.
                if rng.randint(0, nnodes - 1) <= 1:
                    graph[i].add(j)
        return graph

    def verify_sccs(graph, sccs):
        visited = set()

        def visit(node):
            if node in visited:
                return []
            else:
                visited.add(node)
                result = []
                for child in graph[node]:
                    result = result + visit(child)
                return [*result, node]

        for scc in sccs:
            scc = set(scc)
            assert not scc & visited
            # Check that starting from each element of the SCC results
            # in the same set of reachable nodes.
            for scc_root in scc:
                visited.difference_update(scc)
                result = visit(scc_root)
                assert set(result) == scc, (set(result), scc)

    for nnodes in range(10, 20):
        for _ in range(40):
            graph = generate_random_graph(nnodes)
            verify_sccs(graph, compute_sccs(graph))


def test_compute_topological_order():
    from pytools.graph import CycleError, compute_topological_order

    empty = {}
    assert compute_topological_order(empty) == []

    disconnected = {1: [], 2: [], 3: []}
    assert len(compute_topological_order(disconnected)) == 3

    line = list(zip(range(10), ([i] for i in range(1, 11))))
    import random
    random.seed(0)
    random.shuffle(line)
    expected = list(range(11))
    assert compute_topological_order(dict(line)) == expected

    claw = {1: [2, 3], 0: [1]}
    assert compute_topological_order(claw)[:2] == [0, 1]

    repeated_edges = {1: [2, 2], 2: [0]}
    assert compute_topological_order(repeated_edges) == [1, 2, 0]

    self_cycle = {1: [1]}
    with pytest.raises(CycleError):
        compute_topological_order(self_cycle)

    cycle = {0: [2], 1: [2], 2: [3], 3: [4, 1]}
    with pytest.raises(CycleError):
        compute_topological_order(cycle)


def test_transitive_closure():
    from pytools.graph import compute_transitive_closure

    # simple test
    graph = {
        1: {2},
        2: {3},
        3: {4},
        4: set(),
        }

    expected_closure = {
        1: {2, 3, 4},
        2: {3, 4},
        3: {4},
        4: set(),
        }

    closure = compute_transitive_closure(graph)

    assert closure == expected_closure

    # test with branches that reconnect
    graph = {
        1: {2},
        2: set(),
        3: {1},
        4: {1},
        5: {6, 7},
        6: {7},
        7: {1},
        8: {3, 4},
        }

    expected_closure = {
        1: {2},
        2: set(),
        3: {1, 2},
        4: {1, 2},
        5: {1, 2, 6, 7},
        6: {1, 2, 7},
        7: {1, 2},
        8: {1, 2, 3, 4},
        }

    closure = compute_transitive_closure(graph)

    assert closure == expected_closure

    # test with cycles
    graph = {
        1: {2},
        2: {3},
        3: {4},
        4: {1},
        }

    expected_closure = {
        1: {1, 2, 3, 4},
        2: {1, 2, 3, 4},
        3: {1, 2, 3, 4},
        4: {1, 2, 3, 4},
        }

    closure = compute_transitive_closure(graph)

    assert closure == expected_closure


def test_graph_cycle_finder():

    from pytools.graph import contains_cycle

    graph = {
        "a": {"b", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": set(),
        }

    assert not contains_cycle(graph)

    graph = {
        "a": {"b", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": {"a"},
        }

    assert contains_cycle(graph)

    graph = {
        "a": {"a", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": set(),
        }

    assert contains_cycle(graph)

    graph = {
        "a": {"a"},
        }

    assert contains_cycle(graph)


def test_induced_subgraph():

    from pytools.graph import compute_induced_subgraph

    graph = {
        "a": {"b", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": {"h", "i", "j"},
        }

    node_subset = {"b", "c", "e", "f", "g"}

    expected_subgraph = {
        "b": {"e"},
        "c": {"f"},
        "e": set(),
        "f": {"g"},
        "g": set(),
        }

    subgraph = compute_induced_subgraph(graph, node_subset)

    assert subgraph == expected_subgraph


def test_prioritized_topological_sort_examples():

    from pytools.graph import compute_topological_order

    keys = {"a": 4, "b": 3, "c": 2, "e": 1, "d": 4}
    dag = {
            "a": ["b", "c"],
            "b": [],
            "c": ["d", "e"],
            "d": [],
            "e": []}

    assert compute_topological_order(dag, key=keys.get) == [
            "a", "c", "e", "b", "d"]

    keys = {"a": 7, "b": 2, "c": 1, "d": 0}
    dag = {
            "d": set("c"),
            "b": set("a"),
            "a": set(),
            "c": set("a"),
            }

    assert compute_topological_order(dag, key=keys.get) == ["d", "c", "b", "a"]


def test_prioritized_topological_sort():

    import random

    from pytools.graph import compute_topological_order
    rng = random.Random(0)

    def generate_random_graph(nnodes):
        graph = {i: set() for i in range(nnodes)}
        for i in range(nnodes):
            # to avoid cycles only consider edges node_i->node_j where j > i.
            for j in range(i+1, nnodes):
                # Edge probability 4/n: Generates decently interesting inputs.
                if rng.randint(0, nnodes - 1) <= 2:
                    graph[i].add(j)
        return graph

    nnodes = rng.randint(40, 100)
    rev_dep_graph = generate_random_graph(nnodes)
    dep_graph = {i: set() for i in range(nnodes)}

    for i in range(nnodes):
        for rev_dep in rev_dep_graph[i]:
            dep_graph[rev_dep].add(i)

    keys = [rng.random() for _ in range(nnodes)]
    topo_order = compute_topological_order(rev_dep_graph, key=keys.__getitem__)

    for scheduled_node in topo_order:
        nodes_with_no_deps = {node for node, deps in dep_graph.items()
                    if len(deps) == 0}

        # check whether the order is a valid topological order
        assert scheduled_node in nodes_with_no_deps
        # check whether priorities are upheld
        assert keys[scheduled_node] == min(
                keys[node] for node in nodes_with_no_deps)

        # 'scheduled_node' is scheduled => no longer a dependency
        dep_graph.pop(scheduled_node)

        for deps in dep_graph.values():
            deps.discard(scheduled_node)

    assert len(dep_graph) == 0


def test_as_graphviz_dot():
    graph = {"A": ["B", "C"],
             "B": [],
             "C": ["A"]}

    from pytools.graph import NodeT, as_graphviz_dot

    def edge_labels(n1: NodeT, n2: NodeT) -> str:
        if n1 == "A" and n2 == "B":
            return "foo"

        return ""

    def node_labels(node: NodeT) -> str:
        if node == "A":
            return "foonode"

        return str(node)

    res = as_graphviz_dot(graph, node_labels=node_labels, edge_labels=edge_labels)

    assert res == \
"""digraph mygraph {
mynodeid [label="foonode"];
mynodeid_0 [label="B"];
mynodeid_1 [label="C"];
mynodeid -> mynodeid_0 [label="foo"];
mynodeid -> mynodeid_1 [label=""];
mynodeid_1 -> mynodeid [label=""];
}
"""


def test_reverse_graph():
    graph = {
        "a": frozenset(("b", "c")),
        "b": frozenset(("d", "e")),
        "c": frozenset(("d", "f")),
        "d": frozenset(),
        "e": frozenset(),
        "f": frozenset(("g",)),
        "g": frozenset(("h", "i", "j")),
        "h": frozenset(),
        "i": frozenset(),
        "j": frozenset(),
        }

    from pytools.graph import reverse_graph
    assert graph == reverse_graph(reverse_graph(graph))


def test_validate_graph():
    from pytools.graph import validate_graph
    graph1 = {
            "d": set("c"),
            "b": set("a"),
            "a": set(),
            "c": set("a"),
            }

    validate_graph(graph1)

    graph2 = {
            "d": set("d"),
            "b": set("c"),
            "a": set("b"),
            "c": set("a"),
            }

    validate_graph(graph2)

    graph3 = {
        "a": {"b", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": {"h", "i", "j"},  # h, i, j missing from keys
        }

    with pytest.raises(ValueError):
        validate_graph(graph3)

    validate_graph({})


def test_is_connected():
    from pytools.graph import is_connected
    graph1 = {
            "d": set("c"),
            "b": set("a"),
            "a": set(),
            "c": set("a"),
            }

    assert is_connected(graph1)

    graph2 = {
            "d": set("d"),
            "b": set("c"),
            "a": set("b"),
            "c": set("a"),
            }

    assert not is_connected(graph2)

    graph3 = {
        "a": {"b", "c"},
        "b": {"d", "e"},
        "c": {"d", "f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": {},
        }

    assert is_connected(graph3)

    graph4 = {
        "a": {"c"},
        "b": {"d", "e"},
        "c": {"f"},
        "d": set(),
        "e": set(),
        "f": {"g"},
        "g": {},
        }

    assert not is_connected(graph4)

    assert is_connected({})


def test_propagation_graph_tools():
    from pytools.graph import (
        get_reachable_nodes,
        undirected_graph_from_edges,
    )

    vars = {"a", "b", "c", "d", "e", "f", "g"}

    constraints = [
        ("a", "b"),
        ("a", "d"),
        ("c", "d"),
        ("e", "f"),
        ("f", "g")
    ]

    all_reachable_nodes = {
        "a": frozenset({"a", "b", "c", "d"}),
        "b": frozenset({"a", "b", "c", "d"}),
        "c": frozenset({"a", "b", "c", "d"}),
        "e": frozenset({"e", "f", "g"}),
        "f": frozenset({"e", "f", "g"}),
        "g": frozenset({"e", "f", "g"})
    }

    propagation_graph = undirected_graph_from_edges(constraints)
    assert (
        all_reachable_nodes[var] == get_reachable_nodes(propagation_graph, var)
        for var in vars
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
