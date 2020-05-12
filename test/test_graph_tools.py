import sys
import pytest


def test_compute_topological_order():
    from loopy.tools import compute_topological_order, CycleError

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
