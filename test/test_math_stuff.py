from __future__ import division
from __future__ import absolute_import


def test_variance():
    data = [4, 7, 13, 16]

    def naive_var(data):
        n = len(data)
        return ((
            sum(di**2 for di in data)
            - sum(data)**2/n)
            / (n-1))

    from pytools import variance
    orig_variance = variance(data, entire_pop=False)

    assert abs(naive_var(data) - orig_variance) < 1e-15

    data = [1e9 + x for x in data]
    assert abs(variance(data, entire_pop=False) - orig_variance) < 1e-15
