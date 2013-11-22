from __future__ import division, with_statement

import pytest
import sys  # noqa


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

    sc.f.clear_cache(sc)


def test_memoize_method_with_uncached():
    from pytools import memoize_method_with_uncached

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        @memoize_method_with_uncached(uncached_args=[1], uncached_kwargs=["z"])
        def f(self, x, y, z):
            self.run_count += 1
            return 17

    sc = SomeClass()
    sc.f(17, 18, z=19)
    sc.f(17, 19, z=20)
    assert sc.run_count == 1
    sc.f(18, 19, z=20)
    assert sc.run_count == 2

    sc.f.clear_cache(sc)


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
