from __future__ import absolute_import, division, with_statement

import shutil
import sys  # noqa
import tempfile

import pytest
from six.moves import range, zip

from pytools.persistent_dict import (CollisionWarning, NoSuchEntryError,
        PersistentDict, ReadOnlyEntryError, WriteOncePersistentDict)


# {{{ type for testing

class PDictTestingKeyOrValue(object):

    def __init__(self, val, hash_key=None):
        self.val = val
        if hash_key is None:
            hash_key = val
        self.hash_key = hash_key

    def __getstate__(self):
        return {"val": self.val, "hash_key": self.hash_key}

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.hash_key)

    def __repr__(self):
        return "PDictTestingKeyOrValue(val=%r,hash_key=%r)" % (
                (self.val, self.hash_key))

    __str__ = __repr__

# }}}


def test_persistent_dict_storage_and_lookup():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir)

        from random import randrange

        def rand_str(n=20):
            return "".join(
                    chr(65+randrange(26))
                    for i in range(n))

        keys = [(randrange(2000), rand_str(), None) for i in range(20)]
        values = [randrange(2000) for i in range(20)]

        d = dict(list(zip(keys, values)))

        # {{{ check lookup

        for k, v in zip(keys, values):
            pdict[k] = v

        for k, v in d.items():
            assert d[k] == pdict[k]

        # }}}

        # {{{ check updating

        for k, v in zip(keys, values):
            pdict[k] = v + 1

        for k, v in d.items():
            assert d[k] + 1 == pdict[k]

        # }}}

        # {{{ check store_if_not_present

        for k, v in zip(keys, values):
            pdict.store_if_not_present(k, d[k] + 2)

        for k, v in d.items():
            assert d[k] + 1 == pdict[k]

        pdict.store_if_not_present(2001, 2001)
        assert pdict[2001] == 2001

        # }}}

        # check not found

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(3000)

    finally:
        shutil.rmtree(tmpdir)


def test_persistent_dict_deletion():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir)

        pdict[0] = 0
        del pdict[0]

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)

        with pytest.raises(NoSuchEntryError):
            del pdict[1]

    finally:
        shutil.rmtree(tmpdir)


def test_persistent_dict_synchronization():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict1 = PersistentDict("pytools-test", container_dir=tmpdir)
        pdict2 = PersistentDict("pytools-test", container_dir=tmpdir)

        # check lookup
        pdict1[0] = 1
        assert pdict2[0] == 1

        # check updating
        pdict1[0] = 2
        assert pdict2[0] == 2

        # check deletion
        del pdict1[0]
        with pytest.raises(NoSuchEntryError):
            pdict2.fetch(0)

    finally:
        shutil.rmtree(tmpdir)


def test_persistent_dict_cache_collisions():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir)

        key1 = PDictTestingKeyOrValue(1, hash_key=0)
        key2 = PDictTestingKeyOrValue(2, hash_key=0)

        pdict[key1] = 1

        # check lookup
        with pytest.warns(CollisionWarning):
            with pytest.raises(NoSuchEntryError):
                pdict.fetch(key2)

        # check deletion
        with pytest.warns(CollisionWarning):
            with pytest.raises(NoSuchEntryError):
                del pdict[key2]

        # check presence after deletion
        assert pdict[key1] == 1

        # check store_if_not_present
        pdict.store_if_not_present(key2, 2)
        assert pdict[key1] == 1

    finally:
        shutil.rmtree(tmpdir)


def test_persistent_dict_clear():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir)

        pdict[0] = 1
        pdict.fetch(0)
        pdict.clear()

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)

    finally:
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize("in_mem_cache_size", (0, 256))
def test_write_once_persistent_dict_storage_and_lookup(in_mem_cache_size):
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = WriteOncePersistentDict(
                "pytools-test", container_dir=tmpdir,
                in_mem_cache_size=in_mem_cache_size)

        # check lookup
        pdict[0] = 1
        assert pdict[0] == 1
        # do two lookups to test the cache
        assert pdict[0] == 1

        # check updating
        with pytest.raises(ReadOnlyEntryError):
            pdict[0] = 2

        # check not found
        with pytest.raises(NoSuchEntryError):
            pdict.fetch(1)

        # check store_if_not_present
        pdict.store_if_not_present(0, 2)
        assert pdict[0] == 1
        pdict.store_if_not_present(1, 1)
        assert pdict[1] == 1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_lru_policy():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = WriteOncePersistentDict(
                "pytools-test", container_dir=tmpdir, in_mem_cache_size=3)

        pdict[1] = PDictTestingKeyOrValue(1)
        pdict[2] = PDictTestingKeyOrValue(2)
        pdict[3] = PDictTestingKeyOrValue(3)
        pdict[4] = PDictTestingKeyOrValue(4)

        val1 = pdict.fetch(1)

        assert pdict.fetch(1) is val1
        pdict.fetch(2)
        assert pdict.fetch(1) is val1
        pdict.fetch(2)
        pdict.fetch(3)
        assert pdict.fetch(1) is val1
        pdict.fetch(2)
        pdict.fetch(3)
        pdict.fetch(2)
        assert pdict.fetch(1) is val1
        pdict.fetch(2)
        pdict.fetch(3)
        pdict.fetch(4)
        assert pdict.fetch(1) is not val1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_synchronization():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict1 = WriteOncePersistentDict("pytools-test", container_dir=tmpdir)
        pdict2 = WriteOncePersistentDict("pytools-test", container_dir=tmpdir)

        # check lookup
        pdict1[1] = 0
        assert pdict2[1] == 0

        # check updating
        with pytest.raises(ReadOnlyEntryError):
            pdict2[1] = 1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_cache_collisions():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = WriteOncePersistentDict("pytools-test", container_dir=tmpdir)

        key1 = PDictTestingKeyOrValue(1, hash_key=0)
        key2 = PDictTestingKeyOrValue(2, hash_key=0)
        pdict[key1] = 1

        # check lookup
        with pytest.warns(CollisionWarning):
            with pytest.raises(NoSuchEntryError):
                pdict.fetch(key2)

        # check update
        with pytest.raises(ReadOnlyEntryError):
            pdict[key2] = 1

        # check store_if_not_present
        pdict.store_if_not_present(key2, 2)
        assert pdict[key1] == 1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_clear():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = WriteOncePersistentDict("pytools-test", container_dir=tmpdir)

        pdict[0] = 1
        pdict.fetch(0)
        pdict.clear()

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
