from __future__ import division, with_statement
from __future__ import absolute_import

import pytest  # noqa
import sys  # noqa
from six.moves import range
from six.moves import zip


def test_persistent_dict():
    from pytools.persistent_dict import PersistentDict
    pdict = PersistentDict("pytools-test")
    pdict.clear()

    from random import randrange

    def rand_str(n=20):
        return "".join(
                chr(65+randrange(26))
                for i in range(n))

    keys = [(randrange(2000), rand_str(), None) for i in range(20)]
    values = [randrange(2000) for i in range(20)]

    d = dict(list(zip(keys, values)))

    for k, v in zip(keys, values):
        pdict[k] = v
        pdict.store(k, v, info_files={"hey": str(v)})

    for k, v in list(d.items()):
        assert d[k] == pdict[k]

    for k, v in zip(keys, values):
        pdict.store(k, v+1, info_files={"hey": str(v)})

    for k, v in list(d.items()):
        assert d[k] + 1 == pdict[k]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
