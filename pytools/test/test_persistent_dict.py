import shutil
import sys  # noqa
import tempfile
from dataclasses import dataclass
from enum import Enum, IntEnum

import pytest

from pytools.persistent_dict import (
    CollisionWarning, KeyBuilder, NoSuchEntryCollisionError, NoSuchEntryError,
    PersistentDict, ReadOnlyEntryError, WriteOncePersistentDict)
from pytools.tag import Tag, tag_dataclass


# {{{ type for testing

class PDictTestingKeyOrValue:

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
        return "PDictTestingKeyOrValue(val={!r},hash_key={!r})".format(
                self.val, self.hash_key)

    __str__ = __repr__

# }}}


@tag_dataclass
class SomeTag(Tag):
    value: str


class MyEnum(Enum):
    YES = 1
    NO = 2


class MyIntEnum(IntEnum):
    YES = 1
    NO = 2


@dataclass
class MyStruct:
    name: str
    value: int


def test_persistent_dict_storage_and_lookup():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir)

        from random import randrange

        def rand_str(n=20):
            return "".join(
                    chr(65+randrange(26))
                    for i in range(n))

        keys = [
                (randrange(2000)-1000, rand_str(), None, SomeTag(rand_str()),
                    frozenset({"abc", 123}))
                for i in range(20)]
        values = [randrange(2000) for i in range(20)]

        d = dict(zip(keys, values))

        # {{{ check lookup

        for k, v in zip(keys, values):
            pdict[k] = v

        for k, v in d.items():
            assert d[k] == pdict[k]
            assert v == pdict[k]

        # }}}

        # {{{ check updating

        for k, v in zip(keys, values):
            pdict[k] = v + 1

        for k, v in d.items():
            assert d[k] + 1 == pdict[k]
            assert v + 1 == pdict[k]

        # }}}

        # {{{ check store_if_not_present

        for k, _ in zip(keys, values):
            pdict.store_if_not_present(k, d[k] + 2)

        for k, v in d.items():
            assert d[k] + 1 == pdict[k]
            assert v + 1 == pdict[k]

        pdict.store_if_not_present(2001, 2001)
        assert pdict[2001] == 2001

        # }}}

        # {{{ check dataclasses

        for v in [17, 18]:
            key = MyStruct("hi", v)
            pdict[key] = v

            # reuse same key, with stored hash
            assert pdict[key] == v

        with pytest.raises(NoSuchEntryError):
            pdict[MyStruct("hi", 19)]

        for v in [17, 18]:
            # make new key instances
            assert pdict[MyStruct("hi", v)] == v

        # }}}

        # {{{ check enums

        pdict[MyEnum.YES] = 1
        with pytest.raises(NoSuchEntryError):
            pdict[MyEnum.NO]
        assert pdict[MyEnum.YES] == 1

        pdict[MyIntEnum.YES] = 12
        with pytest.raises(NoSuchEntryError):
            pdict[MyIntEnum.NO]
        assert pdict[MyIntEnum.YES] == 12

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
            with pytest.raises(NoSuchEntryCollisionError):
                pdict.fetch(key2)

        # check deletion
        with pytest.warns(CollisionWarning):
            with pytest.raises(NoSuchEntryCollisionError):
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
            with pytest.raises(NoSuchEntryCollisionError):
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


def test_dtype_hashing():
    np = pytest.importorskip("numpy")

    keyb = KeyBuilder()
    assert keyb(np.float32) == keyb(np.float32)
    assert keyb(np.dtype(np.float32)) == keyb(np.dtype(np.float32))


def test_scalar_hashing():
    keyb = KeyBuilder()

    assert keyb(1) == keyb(1)
    assert keyb(2) != keyb(1)
    assert keyb(1.1) == keyb(1.1)
    assert keyb(1+4j) == keyb(1+4j)

    try:
        import numpy as np
    except ImportError:
        return

    assert keyb(np.int8(1)) == keyb(np.int8(1))
    assert keyb(np.int16(1)) == keyb(np.int16(1))
    assert keyb(np.int32(1)) == keyb(np.int32(1))
    assert keyb(np.int32(2)) != keyb(np.int32(1))
    assert keyb(np.int64(1)) == keyb(np.int64(1))
    assert keyb(1) == keyb(np.int64(1))
    assert keyb(1) != keyb(np.int32(1))

    assert keyb(np.longlong(1)) == keyb(np.longlong(1))

    assert keyb(np.float16(1.1)) == keyb(np.float16(1.1))
    assert keyb(np.float32(1.1)) == keyb(np.float32(1.1))
    assert keyb(np.float64(1.1)) == keyb(np.float64(1.1))
    if hasattr(np, "float128"):
        assert keyb(np.float128(1.1)) == keyb(np.float128(1.1))

    assert keyb(np.longfloat(1.1)) == keyb(np.longfloat(1.1))
    assert keyb(np.longdouble(1.1)) == keyb(np.longdouble(1.1))

    assert keyb(np.complex64(1.1+2.2j)) == keyb(np.complex64(1.1+2.2j))
    assert keyb(np.complex128(1.1+2.2j)) == keyb(np.complex128(1.1+2.2j))
    if hasattr(np, "complex256"):
        assert keyb(np.complex256(1.1+2.2j)) == keyb(np.complex256(1.1+2.2j))

    assert keyb(np.longcomplex(1.1+2.2j)) == keyb(np.longcomplex(1.1+2.2j))
    assert keyb(np.clongdouble(1.1+2.2j)) == keyb(np.clongdouble(1.1+2.2j))


def test_frozendict_hashing():
    pytest.importorskip("frozendict")
    from frozendict import frozendict

    keyb = KeyBuilder()

    d = {"a": 1, "b": 2}

    assert keyb(frozendict(d)) == keyb(frozendict(d))
    assert keyb(frozendict(d)) != keyb(frozendict({"a": 1, "b": 3}))
    assert keyb(frozendict(d)) == keyb(frozendict({"b": 2, "a": 1}))

    with pytest.raises(TypeError):
        keyb(d)


def test_immutabledict_hashing():
    pytest.importorskip("immutabledict")
    from immutabledict import immutabledict

    keyb = KeyBuilder()

    d = {"a": 1, "b": 2}

    assert keyb(immutabledict(d)) == keyb(immutabledict(d))
    assert keyb(immutabledict(d)) != keyb(immutabledict({"a": 1, "b": 3}))
    assert keyb(immutabledict(d)) == keyb(immutabledict({"b": 2, "a": 1}))


def test_frozenset_hashing():
    keyb = KeyBuilder()

    assert keyb(frozenset([1, 2, 3])) == keyb(frozenset([1, 2, 3]))
    assert keyb(frozenset([1, 2, 3])) != keyb(frozenset([1, 2, 4]))
    assert keyb(frozenset([1, 2, 3])) == keyb(frozenset([3, 2, 1]))


def test_frozenorderedset_hashing():
    pytest.importorskip("orderedsets")
    from orderedsets import FrozenOrderedSet
    keyb = KeyBuilder()

    assert (keyb(FrozenOrderedSet([1, 2, 3]))
            == keyb(FrozenOrderedSet([1, 2, 3]))
            == keyb(frozenset([1, 2, 3])))
    assert keyb(FrozenOrderedSet([1, 2, 3])) != keyb(FrozenOrderedSet([1, 2, 4]))
    assert keyb(FrozenOrderedSet([1, 2, 3])) == keyb(FrozenOrderedSet([3, 2, 1]))


def test_ABC_hashing():  # noqa: N802
    from abc import ABC, ABCMeta

    keyb = KeyBuilder()

    class MyABC(ABC):
        pass

    assert keyb(MyABC) != keyb(ABC)

    with pytest.raises(TypeError):
        keyb(MyABC())

    with pytest.raises(TypeError):
        keyb(ABC())

    class MyABC2(MyABC):
        def update_persistent_hash(self, key_hash, key_builder):
            key_builder.rec(key_hash, 42)

    assert keyb(MyABC2) != keyb(MyABC)
    assert keyb(MyABC2())

    class MyABC3(metaclass=ABCMeta):  # noqa: B024
        def update_persistent_hash(self, key_hash, key_builder):
            key_builder.rec(key_hash, 42)

    assert keyb(MyABC3) != keyb(MyABC) != keyb(MyABC3())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
