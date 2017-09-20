from __future__ import division, with_statement, absolute_import

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


class PDictTestValue(object):

    def __init__(self, val):
        self.val = val

    def __getstate__(self):
        return {"val": self.val}

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.val)


def test_persistent_dict_in_memory_cache():
    from pytools.persistent_dict import PersistentDict
    pdict = PersistentDict("pytools-in-memory-cache-test", in_mem_cache_size=3)
    pdict.clear()

    pdict[1] = PDictTestValue(1)
    pdict[2] = PDictTestValue(2)
    pdict[3] = PDictTestValue(3)
    pdict[4] = PDictTestValue(4)

    # {{{ test LRU policy

    val1 = pdict[1]

    assert pdict[1] is val1
    pdict[2]
    assert pdict[1] is val1
    pdict[3]
    assert pdict[1] is val1
    pdict[2]
    assert pdict[1] is val1
    pdict[4]
    assert pdict[1] is not val1

    # }}}

    # {{{ test cache invalidation by versioning

    assert pdict[1].val == 1
    pdict2 = PersistentDict("pytools-in-memory-cache-test")
    pdict2[1] = PDictTestValue(5)
    assert pdict[1].val == 5

    # }}}

    # {{{ test cache invalidation by deletion

    del pdict2[1]
    pdict2[1] = PDictTestValue(10)
    assert pdict[1].val == 10

    # }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
