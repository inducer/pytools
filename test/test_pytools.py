from __future__ import absolute_import, division, with_statement

import sys
import pytest

import logging
logger = logging.getLogger(__name__)


@pytest.mark.skipif("sys.version_info < (2, 5)")
def test_memoize_method_clear():
    from pytools import memoize_method

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        @memoize_method
        def f(self):
            self.run_count += 1
            return 17

    sc = SomeClass()
    sc.f()
    sc.f()
    assert sc.run_count == 1

    sc.f.clear_cache(sc)  # pylint: disable=no-member


def test_memoize_method_with_uncached():
    from pytools import memoize_method_with_uncached

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        @memoize_method_with_uncached(uncached_args=[1], uncached_kwargs=["z"])
        def f(self, x, y, z):
            del x, y, z
            self.run_count += 1
            return 17

    sc = SomeClass()
    sc.f(17, 18, z=19)
    sc.f(17, 19, z=20)
    assert sc.run_count == 1
    sc.f(18, 19, z=20)
    assert sc.run_count == 2

    sc.f.clear_cache(sc)  # pylint: disable=no-member


def test_memoize_method_nested():
    from pytools import memoize_method_nested

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        def f(self):

            @memoize_method_nested
            def inner(x):
                self.run_count += 1
                return 2*x

            inner(5)
            inner(5)

    sc = SomeClass()
    sc.f()
    assert sc.run_count == 1


def test_p_convergence_verifier():
    pytest.importorskip("numpy")

    from pytools.convergence import PConvergenceVerifier

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 0.1**order)
    pconv_verifier()

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 0.5**order)
    pconv_verifier()

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 2)
    with pytest.raises(AssertionError):
        pconv_verifier()


def test_memoize():
    from pytools import memoize
    count = [0]

    @memoize(use_kwargs=True)
    def f(i, j=1):
        count[0] += 1
        return i + j

    assert f(1) == 2
    assert f(1, 2) == 3
    assert f(2, j=3) == 5
    assert count[0] == 3
    assert f(1) == 2
    assert f(1, 2) == 3
    assert f(2, j=3) == 5
    assert count[0] == 3


def test_memoize_keyfunc():
    from pytools import memoize
    count = [0]

    @memoize(key=lambda i, j=(1,): (i, len(j)))
    def f(i, j=(1,)):
        count[0] += 1
        return i + len(j)

    assert f(1) == 2
    assert f(1, [2]) == 2
    assert f(2, j=[2, 3]) == 4
    assert count[0] == 2
    assert f(1) == 2
    assert f(1, (2,)) == 2
    assert f(2, j=(2, 3)) == 4
    assert count[0] == 2


@pytest.mark.parametrize("dims", [2, 3])
def test_spatial_btree(dims, do_plot=False):
    pytest.importorskip("numpy")
    import numpy as np
    nparticles = 2000
    x = -1 + 2*np.random.rand(dims, nparticles)
    x = np.sign(x)*np.abs(x)**1.9
    x = (1.4 + x) % 2 - 1

    bl = np.min(x, axis=-1)
    tr = np.max(x, axis=-1)
    print(bl, tr)

    from pytools.spatial_btree import SpatialBinaryTreeBucket
    tree = SpatialBinaryTreeBucket(bl, tr, max_elements_per_box=10)
    for i in range(nparticles):
        tree.insert(i, (x[:, i], x[:, i]))

    if do_plot:
        import matplotlib.pyplot as pt
        pt.gca().set_aspect("equal")
        pt.plot(x[0], x[1], "x")
        tree.plot(fill=None)
        pt.show()


def test_generate_numbered_unique_names():
    from pytools import generate_numbered_unique_names

    gen = generate_numbered_unique_names("a")
    assert next(gen) == (0, "a")
    assert next(gen) == (1, "a_0")

    gen = generate_numbered_unique_names("b", 6)
    assert next(gen) == (7, "b_6")


def test_find_module_git_revision():
    import pytools
    print(pytools.find_module_git_revision(pytools.__file__, n_levels_up=1))


def test_reshaped_view():
    import pytools
    import numpy as np

    a = np.zeros((10, 2))
    b = a.T
    c = pytools.reshaped_view(a, -1)
    assert c.shape == (20,)
    with pytest.raises(AttributeError):
        pytools.reshaped_view(b, -1)


def test_processlogger():
    logging.basicConfig(level=logging.INFO)

    from pytools import ProcessLogger
    plog = ProcessLogger(logger, "testing the process logger",
            long_threshold_seconds=0.01)

    from time import sleep
    with plog:
        sleep(0.3)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
