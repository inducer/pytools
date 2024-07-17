import shutil
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, Optional

import pytest

from pytools.persistent_dict import (
    CollisionWarning,
    KeyBuilder,
    NoSuchEntryCollisionError,
    NoSuchEntryError,
    PersistentDict,
    ReadOnlyEntryError,
    WriteOncePersistentDict,
)
from pytools.tag import Tag, tag_dataclass


# {{{ type for testing

class PDictTestingKeyOrValue:

    def __init__(self, val: Any, hash_key=None) -> None:
        self.val = val
        if hash_key is None:
            hash_key = val
        self.hash_key = hash_key

    def __getstate__(self) -> Dict[str, Any]:
        return {"val": self.val, "hash_key": self.hash_key}

    def __eq__(self, other: Any) -> bool:
        return self.val == other.val

    def update_persistent_hash(self, key_hash: Any, key_builder: KeyBuilder) -> None:
        key_builder.rec(key_hash, self.hash_key)

    def __repr__(self) -> str:
        return "PDictTestingKeyOrValue(val={!r},hash_key={!r})".format(
                self.val, self.hash_key)

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


def test_persistent_dict_storage_and_lookup() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: PersistentDict[Any, int] = PersistentDict("pytools-test",
                                                         container_dir=tmpdir,
                                                         safe_sync=False)

        from random import randrange

        def rand_str(n=20):
            return "".join(
                    chr(65+randrange(26))
                    for i in range(n))

        keys = [
                (randrange(2000)-1000, rand_str(), None,
                 SomeTag(rand_str()),
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


def test_persistent_dict_deletion() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: PersistentDict[int, int] = PersistentDict("pytools-test",
                                                         container_dir=tmpdir,
                                                         safe_sync=False)

        pdict[0] = 0
        del pdict[0]

        with pytest.raises(NoSuchEntryError):
            pdict.remove(0)

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)

        with pytest.raises(NoSuchEntryError):
            del pdict[1]

    finally:
        shutil.rmtree(tmpdir)


def test_persistent_dict_synchronization() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict1: PersistentDict[int, int] = PersistentDict("pytools-test",
                                                          container_dir=tmpdir,
                                                          safe_sync=False)
        pdict2: PersistentDict[int, int] = PersistentDict("pytools-test",
                                                          container_dir=tmpdir,
                                                          safe_sync=False)

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


def test_persistent_dict_cache_collisions() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: PersistentDict[PDictTestingKeyOrValue, int] = \
            PersistentDict("pytools-test", container_dir=tmpdir, safe_sync=False)

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


def test_persistent_dict_clear() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: PersistentDict[int, int] = PersistentDict("pytools-test",
                                                         container_dir=tmpdir,
                                                         safe_sync=False)

        pdict[0] = 1
        pdict.fetch(0)
        pdict.clear()

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)

    finally:
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize("in_mem_cache_size", (0, 256))
def test_write_once_persistent_dict_storage_and_lookup(in_mem_cache_size) -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: WriteOncePersistentDict[int, int] = WriteOncePersistentDict(
                "pytools-test", container_dir=tmpdir,
                in_mem_cache_size=in_mem_cache_size, safe_sync=False)

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


def test_write_once_persistent_dict_lru_policy() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: WriteOncePersistentDict[Any, Any] = WriteOncePersistentDict(
                "pytools-test", container_dir=tmpdir, in_mem_cache_size=3,
                safe_sync=False)

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

        # test clear_in_mem_cache
        val1 = pdict.fetch(1)
        pdict.clear_in_mem_cache()
        assert pdict.fetch(1) is not val1

        val1 = pdict.fetch(1)
        assert pdict.fetch(1) is val1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_synchronization() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict1: WriteOncePersistentDict[int, int] = \
            WriteOncePersistentDict("pytools-test", container_dir=tmpdir,
                                    safe_sync=False)
        pdict2: WriteOncePersistentDict[int, int] = \
            WriteOncePersistentDict("pytools-test", container_dir=tmpdir,
                                    safe_sync=False)

        # check lookup
        pdict1[1] = 0
        assert pdict2[1] == 0

        # check updating
        with pytest.raises(ReadOnlyEntryError):
            pdict2[1] = 1

    finally:
        shutil.rmtree(tmpdir)


def test_write_once_persistent_dict_cache_collisions() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: WriteOncePersistentDict[Any, int] = \
            WriteOncePersistentDict("pytools-test", container_dir=tmpdir,
                                    safe_sync=False)

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


def test_write_once_persistent_dict_clear() -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        pdict: WriteOncePersistentDict[int, int] = \
            WriteOncePersistentDict("pytools-test", container_dir=tmpdir,
                                    safe_sync=False)

        pdict[0] = 1
        pdict.fetch(0)
        pdict.clear()

        with pytest.raises(NoSuchEntryError):
            pdict.fetch(0)
    finally:
        shutil.rmtree(tmpdir)


def test_dtype_hashing() -> None:
    np = pytest.importorskip("numpy")

    keyb = KeyBuilder()
    assert keyb(np.float32) == keyb(np.float32)
    assert keyb(np.dtype(np.float32)) == keyb(np.dtype(np.float32))


def test_scalar_hashing() -> None:
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
    assert keyb(np.longdouble(1.1)) == keyb(np.longdouble(1.1))

    assert keyb(np.complex64(1.1+2.2j)) == keyb(np.complex64(1.1+2.2j))
    assert keyb(np.complex128(1.1+2.2j)) == keyb(np.complex128(1.1+2.2j))
    if hasattr(np, "complex256"):
        assert keyb(np.complex256(1.1+2.2j)) == keyb(np.complex256(1.1+2.2j))
    assert keyb(np.clongdouble(1.1+2.2j)) == keyb(np.clongdouble(1.1+2.2j))


@pytest.mark.parametrize("dict_impl", ("immutabledict", "frozendict",
                                       "constantdict",
                                       ("immutables", "Map"),
                                       ("pyrsistent", "pmap")))
def test_dict_hashing(dict_impl) -> None:
    if isinstance(dict_impl, str):
        dict_package = dict_impl
        dict_class = dict_impl
    else:
        dict_package = dict_impl[0]
        dict_class = dict_impl[1]

    pytest.importorskip(dict_package)
    import importlib
    dc = getattr(importlib.import_module(dict_package), dict_class)

    keyb = KeyBuilder()

    d = {"a": 1, "b": 2}

    assert keyb(dc(d)) == keyb(dc(d))
    assert keyb(dc(d)) != keyb(dc({"a": 1, "b": 3}))
    assert keyb(dc(d)) == keyb(dc({"b": 2, "a": 1}))


def test_frozenset_hashing() -> None:
    keyb = KeyBuilder()

    assert keyb(frozenset([1, 2, 3])) == keyb(frozenset([1, 2, 3]))
    assert keyb(frozenset([1, 2, 3])) != keyb(frozenset([1, 2, 4]))
    assert keyb(frozenset([1, 2, 3])) == keyb(frozenset([3, 2, 1]))


def test_frozenorderedset_hashing() -> None:
    pytest.importorskip("orderedsets")
    from orderedsets import FrozenOrderedSet
    keyb = KeyBuilder()

    assert (keyb(FrozenOrderedSet([1, 2, 3]))
            == keyb(FrozenOrderedSet([1, 2, 3]))
            == keyb(frozenset([1, 2, 3])))
    assert keyb(FrozenOrderedSet([1, 2, 3])) != keyb(FrozenOrderedSet([1, 2, 4]))
    assert keyb(FrozenOrderedSet([1, 2, 3])) == keyb(FrozenOrderedSet([3, 2, 1]))


def test_ABC_hashing() -> None:  # noqa: N802
    from abc import ABC, ABCMeta

    keyb = KeyBuilder()

    class MyABC(ABC):  # noqa: B024
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


def test_class_hashing() -> None:
    keyb = KeyBuilder()

    class WithUpdateMethod:
        def update_persistent_hash(self, key_hash, key_builder):
            # Only called for instances of this class, not for the class itself
            key_builder.rec(key_hash, 42)

    class TagClass(Tag):
        pass

    @tag_dataclass
    class TagClass2(Tag):
        pass

    assert keyb(WithUpdateMethod) != keyb(WithUpdateMethod())
    assert keyb(TagClass) != keyb(TagClass())
    assert keyb(TagClass2) != keyb(TagClass2())

    assert keyb(TagClass) != keyb(TagClass2)
    assert keyb(TagClass()) != keyb(TagClass2())

    assert keyb(TagClass()) == "7b3e4e66503438f6"
    assert keyb(TagClass2) == "690b86bbf51aad83"

    @tag_dataclass
    class TagClass3(Tag):
        s: str

    assert (keyb(TagClass3("foo")) == "cf1a33652cc75b9c")


def test_dataclass_hashing() -> None:
    keyb = KeyBuilder()

    @dataclass
    class MyDC:
        name: str
        value: int

    assert keyb(MyDC("hi", 1)) == "d1a1079f1c10aa4f"

    assert keyb(MyDC("hi", 1)) == keyb(MyDC("hi", 1))
    assert keyb(MyDC("hi", 1)) != keyb(MyDC("hi", 2))

    @dataclass
    class MyDC2:
        name: str
        value: int

    # Class types must be encoded in hash
    assert keyb(MyDC2("hi", 1)) != keyb(MyDC("hi", 1))


def test_attrs_hashing() -> None:
    attrs = pytest.importorskip("attrs")

    keyb = KeyBuilder()

    @attrs.define
    class MyAttrs:
        name: str
        value: int

    assert (keyb(MyAttrs("hi", 1)) == "5b6c5da60eb2bd0f")  # type: ignore[call-arg]

    assert keyb(MyAttrs("hi", 1)) == keyb(MyAttrs("hi", 1))  # type: ignore[call-arg]
    assert keyb(MyAttrs("hi", 1)) != keyb(MyAttrs("hi", 2))  # type: ignore[call-arg]

    @dataclass
    class MyDC:
        name: str
        value: int

    assert keyb(MyDC("hi", 1)) != keyb(MyAttrs("hi", 1))  # type: ignore[call-arg]

    @attrs.define
    class MyAttrs2:
        name: str
        value: int

    # Class types must be encoded in hash
    assert (keyb(MyAttrs2("hi", 1))  # type: ignore[call-arg]
            != keyb(MyAttrs("hi", 1)))  # type: ignore[call-arg]


def test_datetime_hashing() -> None:
    keyb = KeyBuilder()

    import datetime

    # {{{ date
    # No timezone info; date is always naive
    assert (keyb(datetime.date(2020, 1, 1))
            == keyb(datetime.date(2020, 1, 1))
            == "1c866ff10ff0d997")
    assert keyb(datetime.date(2020, 1, 1)) != keyb(datetime.date(2020, 1, 2))

    # }}}

    # {{{ time

    # Must distinguish between naive and aware time objects

    # Naive time
    assert (keyb(datetime.time(12, 0))
            == keyb(datetime.time(12, 0))
            == keyb(datetime.time(12, 0, 0))
            == keyb(datetime.time(12, 0, 0, 0))
            == "e523be74ebc6b227")
    assert keyb(datetime.time(12, 0)) != keyb(datetime.time(12, 1))

    # Aware time
    t1 = datetime.time(12, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.time(7, 0,
                            tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    t3 = datetime.time(7, 0,
                            tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))

    assert t1 == t2
    assert (keyb(t1)
            == keyb(t2)
            == "2041e7cd5b17b8eb")

    assert t1 != t3
    assert keyb(t1) != keyb(t3)

    # }}}

    # {{{ datetime

    # must distinguish between naive and aware datetime objects

    # Aware datetime
    dt1 = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    dt2 = datetime.datetime(2020, 1, 1, 7,
                            tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))

    assert dt1 == dt2
    assert (keyb(dt1)
            == keyb(dt2)
            == "8be96b9e739c7d8c")

    dt3 = datetime.datetime(2020, 1, 1, 7,
                            tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))

    assert dt1 != dt3
    assert keyb(dt1) != keyb(dt3)

    # Naive datetime
    dt4 = datetime.datetime(2020, 1, 1, 6)  # matches dt1 'naively'
    assert dt1 != dt4  # naive and aware datetime objects are never equal
    assert keyb(dt1) != keyb(dt4)

    assert (keyb(datetime.datetime(2020, 1, 1))
            == keyb(datetime.datetime(2020, 1, 1))
            == keyb(datetime.datetime(2020, 1, 1, 0, 0, 0, 0))
            == "215dbe82add7a55c"  # spellchecker: disable-line
            )
    assert keyb(datetime.datetime(2020, 1, 1)) != keyb(datetime.datetime(2020, 1, 2))
    assert (keyb(datetime.datetime(2020, 1, 1))
            != keyb(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)))

    # }}}

    # {{{ timezone

    tz1 = datetime.timezone(datetime.timedelta(hours=-4))
    tz2 = datetime.timezone(datetime.timedelta(hours=0))
    tz3 = datetime.timezone.utc

    assert tz1 != tz2
    assert keyb(tz1) != keyb(tz2)

    assert tz1 != tz3
    assert keyb(tz1) != keyb(tz3)

    assert tz2 == tz3
    assert (keyb(tz2)
            == keyb(tz3)
            == "5e1d46ab778c7ccf")

    # }}}


def test_xdg_cache_home() -> None:
    import os
    xdg_dir = "tmpdir_pytools_xdg_test"

    assert not os.path.exists(xdg_dir)

    old_xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    try:
        os.environ["XDG_CACHE_HOME"] = xdg_dir

        PersistentDict("pytools-test", safe_sync=False)

        assert os.path.exists(xdg_dir)
    finally:
        if old_xdg_cache_home is not None:
            os.environ["XDG_CACHE_HOME"] = old_xdg_cache_home
        else:
            del os.environ["XDG_CACHE_HOME"]

        shutil.rmtree(xdg_dir)


def test_speed():
    import time

    tmpdir = tempfile.mkdtemp()
    pdict = WriteOncePersistentDict("pytools-test", container_dir=tmpdir,
                                    safe_sync=False)

    start = time.time()
    for i in range(10000):
        pdict[i] = i
    end = time.time()
    print("persistent dict write time: ", end-start)

    start = time.time()
    for _ in range(5):
        for i in range(10000):
            pdict[i]
    end = time.time()
    print("persistent dict read time: ", end-start)

    shutil.rmtree(tmpdir)


def test_size():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir, safe_sync=False)

        for i in range(10000):
            pdict[f"foobarbazfoobbb{i}"] = i

        size = pdict.nbytes()
        print("sqlite size: ", size/1024/1024, " MByte")
        assert 1024*1024//2 < size < 2*1024*1024
    finally:
        shutil.rmtree(tmpdir)


def test_len():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir, safe_sync=False)

        assert len(pdict) == 0

        for i in range(10000):
            pdict[i] = i

        assert len(pdict) == 10000

        pdict.clear()

        assert len(pdict) == 0
    finally:
        shutil.rmtree(tmpdir)


def test_repr():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir, safe_sync=False)

        assert repr(pdict)[:15] == "PersistentDict("
    finally:
        shutil.rmtree(tmpdir)


def test_keys_values_items():
    try:
        tmpdir = tempfile.mkdtemp()
        pdict = PersistentDict("pytools-test", container_dir=tmpdir, safe_sync=False)

        for i in range(10000):
            pdict[i] = i

        # This also tests deterministic iteration order
        assert len(list(pdict.keys())) == 10000 == len(set(pdict.keys()))
        assert list(pdict.keys()) == list(range(10000))
        assert list(pdict.values()) == list(range(10000))
        assert list(pdict.items()) == list(zip(list(pdict.keys()), range(10000)))

        assert ([k for k in pdict.keys()]  # noqa: C416
                == list(pdict.keys())
                == list(pdict)
                == [k for k in pdict])  # noqa: C416

    finally:
        shutil.rmtree(tmpdir)


def global_fun():
    pass


def global_fun2():
    pass


def test_hash_function() -> None:
    keyb = KeyBuilder()

    # {{{ global functions

    assert keyb(global_fun) == keyb(global_fun) == "79efd03f9a38ed77"
    assert keyb(global_fun) != keyb(global_fun2)

    # }}}

    # {{{ closures

    def get_fun(x):
        def add_x(y):
            return x + y
        return add_x

    f1 = get_fun(1)
    f11 = get_fun(1)
    f2 = get_fun(2)

    fa = get_fun
    fb = get_fun

    assert fa == fb
    assert keyb(fa) == keyb(fb)

    assert f1 != f2
    assert keyb(f1) != keyb(f2)

    # FIXME: inconsistency!
    assert f1 != f11
    assert hash(f1) != hash(f11)
    assert keyb(f1) == keyb(f11)

    # }}}

    # {{{ local functions

    def local_fun():
        pass

    def local_fun2():
        pass

    assert keyb(local_fun) == keyb(local_fun) == "adc92e690b62dc2b"
    assert keyb(local_fun) != keyb(local_fun2)

    # }}}

    # {{{ methods

    class C1:
        def method(self):
            pass

    class C2:
        def method(self):
            pass

    assert keyb(C1.method) == keyb(C1.method) == "af19e056ad7749c4"
    assert keyb(C1.method) != keyb(C2.method)

    # }}}


# {{{ basic concurrency tests

def _conc_fn(tmpdir: Optional[str] = None,
             pdict: Optional[PersistentDict[int, int]] = None) -> None:
    import time

    assert (pdict is None) ^ (tmpdir is None)

    if pdict is None:
        pdict = PersistentDict("pytools-test",
                                container_dir=tmpdir,
                                safe_sync=False)
    n = 10000
    s = 0

    start = time.time()

    for i in range(n):
        if i % 1000 == 0:
            print(f"i={i}")

        if isinstance(pdict, WriteOncePersistentDict):
            try:
                pdict[i] = i
            except ReadOnlyEntryError:
                pass
        else:
            pdict[i] = i

        try:
            s += pdict[i]
        except NoSuchEntryError:
            # Someone else already deleted the entry
            pass

        if not isinstance(pdict, WriteOncePersistentDict):
            try:
                del pdict[i]
            except NoSuchEntryError:
                # Someone else already deleted the entry
                pass

    end = time.time()

    print(f"PersistentDict: time taken to write {n} entries to "
        f"{pdict.filename}: {end-start} s={s}")


def test_concurrency_processes() -> None:
    from multiprocessing import Process

    tmpdir = "_tmp_proc/"  # must be the same across all processes in this test

    try:
        # multiprocessing needs to pickle function arguments, so we can't pass
        # the PersistentDict object (which is unpicklable) directly.
        p = [Process(target=_conc_fn, args=(tmpdir, None)) for _ in range(4)]
        for pp in p:
            pp.start()
        for pp in p:
            pp.join()

        assert all(pp.exitcode == 0 for pp in p), [pp.exitcode for pp in p]
    finally:
        shutil.rmtree(tmpdir)


from threading import Thread


class RaisingThread(Thread):
    def run(self) -> None:
        self._exc = None
        try:
            super().run()
        except Exception as e:
            self._exc = e

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout=timeout)
        if self._exc:
            raise self._exc


def test_concurrency_threads() -> None:
    tmpdir = "_tmp_threads/"  # must be the same across all threads in this test

    try:
        # Share this pdict object among all threads to test thread safety
        pdict: PersistentDict[int, int] = PersistentDict("pytools-test",
                                                    container_dir=tmpdir,
                                                    safe_sync=False)
        t = [RaisingThread(target=_conc_fn, args=(None, pdict)) for _ in range(4)]
        for tt in t:
            tt.start()
        for tt in t:
            tt.join()
            # Threads will raise in join() if they encountered an exception
    finally:
        shutil.rmtree(tmpdir)

    try:
        # Share this pdict object among all threads to test thread safety
        pdict2: WriteOncePersistentDict[int, int] = WriteOncePersistentDict(
                                                    "pytools-test",
                                                    container_dir=tmpdir,
                                                    safe_sync=False)
        t = [RaisingThread(target=_conc_fn, args=(None, pdict2)) for _ in range(4)]
        for tt in t:
            tt.start()
        for tt in t:
            tt.join()
            # Threads will raise in join() if they encountered an exception
    finally:
        shutil.rmtree(tmpdir)

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
