from __future__ import annotations


"""Generic persistent, concurrent dictionary-like facility."""


__copyright__ = """
Copyright (C) 2011,2014 Andreas Kloeckner
Copyright (C) 2017 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import logging
import os
import pickle
import sqlite3
import sys
from dataclasses import fields as dc_fields, is_dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    cast,
)
from warnings import warn


class RecommendedHashNotFoundWarning(UserWarning):
    pass


try:
    from siphash24 import siphash13 as _default_hash
except ImportError:
    warn("Unable to import recommended hash 'siphash24.siphash13', "
         "falling back to 'hashlib.sha256'. "
         "Run 'python3 -m pip install siphash24' to install "
         "the recommended hash.",
         RecommendedHashNotFoundWarning, stacklevel=1)
    from hashlib import sha256 as _default_hash

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
    from typing_extensions import Self

try:
    import attrs
except ModuleNotFoundError:
    _HAS_ATTRS = False
else:
    _HAS_ATTRS = True


logger = logging.getLogger(__name__)

# NOTE: not always available so they get hardcoded here
SQLITE_BUSY = getattr(sqlite3, "SQLITE_BUSY", 5)
SQLITE_CONSTRAINT_PRIMARYKEY = getattr(sqlite3, "SQLITE_CONSTRAINT_PRIMARYKEY", 1555)

__doc__ = """
Persistent Hashing and Persistent Dictionaries
==============================================

This module contains functionality that allows hashing with keys that remain
valid across interpreter invocations, unlike Python's built-in hashes.

This module also provides a disk-backed dictionary that uses persistent hashing.

.. autoexception:: NoSuchEntryError
.. autoexception:: NoSuchEntryCollisionError
.. autoexception:: ReadOnlyEntryError

.. autoexception:: CollisionWarning

.. autoclass:: Hash
.. autoclass:: KeyBuilder
.. autoclass:: PersistentDict
.. autoclass:: WriteOncePersistentDict


Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: K

    A type variable for the key type of a :class:`PersistentDict`.

.. class:: V

    A type variable for the value type of a :class:`PersistentDict`.
"""


# {{{ key generation

class Hash(Protocol):
    """A protocol for the hashes from :mod:`hashlib`.

    .. automethod:: update
    .. automethod:: digest
    .. automethod:: hexdigest
    .. automethod:: copy
    """
    def update(self, data: ReadableBuffer) -> None:
        ...

    def digest(self) -> bytes:
        ...

    def hexdigest(self) -> str:
        ...

    def copy(self) -> Self:
        ...


class KeyBuilder:
    """A (stateless) object that computes persistent hashes of objects fed to it.
    Subclassing this class permits customizing the computation of hash keys.

    This class follows the same general rules as Python's built-in hashing:

    - Only immutable objects can be hashed.
    - If two objects compare equal, they must hash to the same value.
    - Objects with the same hash may or may not compare equal.

    In addition, hashes computed with :class:`KeyBuilder` have the following
    properties:

    - The hash is persistent across interpreter invocations.
    - The hash is the same across different Python versions and platforms.
    - The hash is invariant with respect to :envvar:`PYTHONHASHSEED`.
    - Hashes are computed using functionality from :mod:`hashlib`.

    Key builders of this type are used by :class:`PersistentDict`, but
    other uses are entirely allowable.

    .. automethod:: __call__
    .. automethod:: rec
    .. staticmethod:: new_hash()

        Return a new hash instance following the protocol of the ones
        from :mod:`hashlib`. This will permit switching to different
        hash algorithms in the future. Subclasses are expected to use
        this to create new hashes. Not doing so is deprecated and
        may stop working as early as 2022.

        .. versionadded:: 2021.2

    .. note::

        Some key-building uses system byte order, so the built keys may not match
        across different systems. It would be desirable to fix this, but this is
        not yet done.
    """

    # this exists so that we can (conceivably) switch algorithms at some point
    # down the road
    new_hash: Callable[..., Hash] = _default_hash

    def rec(self, key_hash: Hash, key: Any) -> Hash:
        """
        :arg key_hash: the hash object to be updated with the hash of *key*.
        :arg key: the (immutable) Python object to be hashed.
        :returns: the updated *key_hash*

        .. versionchanged:: 2021.2

            Now returns the updated *key_hash*.
        """

        digest = getattr(key, "_pytools_persistent_hash_digest", None)

        if digest is None and not isinstance(key, type):
            try:
                method = key.update_persistent_hash
            except AttributeError:
                pass
            else:
                inner_key_hash = self.new_hash()
                method(inner_key_hash, self)
                digest = inner_key_hash.digest()

        if digest is None:
            tp = type(key)
            tname = tp.__name__
            method = None
            try:
                method = getattr(self, "update_for_"+tname)
            except AttributeError:
                if "numpy" in sys.modules:
                    import numpy as np

                    # Hashing numpy dtypes
                    if (
                            # Handling numpy >= 1.20, for which
                            # type(np.dtype("float32")) -> "dtype[float32]"
                            tname.startswith("dtype[")
                            # Handling numpy >= 1.25, for which
                            # type(np.dtype("float32")) -> "Float32DType"
                            or tname.endswith("DType")
                            ):
                        if isinstance(key, np.dtype):
                            method = self.update_for_specific_dtype

                    # Hashing numpy scalars
                    elif isinstance(key, np.number):
                        # Non-numpy scalars are handled above in the try block.
                        method = self.update_for_numpy_scalar

                if method is None:
                    if issubclass(tp, Enum):
                        method = self.update_for_enum

                    elif is_dataclass(tp):
                        method = self.update_for_dataclass

                    elif _HAS_ATTRS and attrs.has(tp):
                        method = self.update_for_attrs

            if method is not None:
                inner_key_hash = self.new_hash()
                method(inner_key_hash, key)
                digest = inner_key_hash.digest()

        if digest is None:
            raise TypeError(
                    f"unsupported type for persistent hash keying: {type(key)}")

        if not isinstance(key, type):
            try:
                object.__setattr__(key, "_pytools_persistent_hash_digest", digest)
            except AttributeError:
                pass
            except TypeError:
                pass

        key_hash.update(digest)
        return key_hash

    def __call__(self, key: Any) -> str:
        """Return the hash of *key*."""
        key_hash = self.new_hash()
        self.rec(key_hash, key)
        return key_hash.hexdigest()

    # {{{ updaters

    @staticmethod
    def update_for_type(key_hash: Hash, key: type) -> None:
        key_hash.update(
            f"{key.__module__}.{key.__qualname__}.{key.__name__}".encode())

    update_for_ABCMeta = update_for_type  # noqa: N815

    @staticmethod
    def update_for_int(key_hash: Hash, key: int) -> None:
        sz = 8
        while True:
            try:
                # Must match system byte order so that numpy and this
                # generate the same string of bytes.
                # https://github.com/inducer/pytools/issues/259
                key_hash.update(key.to_bytes(sz, byteorder=sys.byteorder, signed=True))
                return
            except OverflowError:
                sz *= 2

    @classmethod
    def update_for_enum(cls, key_hash: Hash, key: Enum) -> None:
        cls.update_for_str(key_hash, str(key))

    @staticmethod
    def update_for_bool(key_hash: Hash, key: bool) -> None:
        key_hash.update(str(key).encode("utf8"))

    @staticmethod
    def update_for_float(key_hash: Hash, key: float) -> None:
        key_hash.update(key.hex().encode("utf8"))

    @staticmethod
    def update_for_complex(key_hash: Hash, key: float) -> None:
        key_hash.update(repr(key).encode("utf-8"))

    @staticmethod
    def update_for_str(key_hash: Hash, key: str) -> None:
        key_hash.update(key.encode("utf8"))

    @staticmethod
    def update_for_bytes(key_hash: Hash, key: bytes) -> None:
        key_hash.update(key)

    def update_for_tuple(self, key_hash: Hash, key: Tuple[Any, ...]) -> None:
        for obj_i in key:
            self.rec(key_hash, obj_i)

    def update_for_frozenset(self, key_hash: Hash, key: FrozenSet[Any]) -> None:
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), key_i).digest() for key_i in key),
            hash_constructor=self.new_hash)

    update_for_FrozenOrderedSet = update_for_frozenset  # noqa: N815

    @staticmethod
    def update_for_NoneType(key_hash: Hash, key: None) -> None:  # noqa: N802
        del key
        key_hash.update(b"<None>")

    @staticmethod
    def update_for_dtype(key_hash: Hash, key: Any) -> None:
        key_hash.update(key.str.encode("utf8"))

    # Handling numpy >= 1.20, for which
    # type(np.dtype("float32")) -> "dtype[float32]"
    # Introducing this method allows subclasses to specially handle all those
    # dtypes.
    @staticmethod
    def update_for_specific_dtype(key_hash: Hash, key: Any) -> None:
        key_hash.update(key.str.encode("utf8"))

    @staticmethod
    def update_for_numpy_scalar(key_hash: Hash, key: Any) -> None:
        import numpy as np
        if hasattr(np, "complex256") and key.dtype == np.dtype("complex256"):
            key_hash.update(repr(complex(key)).encode("utf8"))
        elif hasattr(np, "float128") and key.dtype == np.dtype("float128"):
            key_hash.update(repr(float(key)).encode("utf8"))
        else:
            key_hash.update(np.array(key).tobytes())

    def update_for_dataclass(self, key_hash: Hash, key: Any) -> None:
        self.rec(key_hash, f"{type(key).__qualname__}.{type(key).__name__}")

        for fld in dc_fields(key):
            self.rec(key_hash, fld.name)
            self.rec(key_hash, getattr(key, fld.name, None))

    def update_for_attrs(self, key_hash: Hash, key: Any) -> None:
        self.rec(key_hash, f"{type(key).__qualname__}.{type(key).__name__}")

        for fld in attrs.fields(key.__class__):
            self.rec(key_hash, fld.name)
            self.rec(key_hash, getattr(key, fld.name, None))

    def update_for_frozendict(self, key_hash: Hash, key: Mapping[Any, Any]) -> None:
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), (k, v)).digest() for k, v in key.items()),
            hash_constructor=self.new_hash)

    update_for_immutabledict = update_for_frozendict
    update_for_constantdict = update_for_frozendict
    update_for_PMap = update_for_frozendict  # noqa: N815
    update_for_Map = update_for_frozendict  # noqa: N815

    # {{{ date, time, datetime, timezone

    def update_for_date(self, key_hash: Hash, key: Any) -> None:
        # 'date' has no timezone information; it is always naive
        self.rec(key_hash, key.isoformat())

    def update_for_time(self, key_hash: Hash, key: Any) -> None:
        # 'time' should differentiate between naive and aware
        import datetime

        # Convert to datetime object
        self.rec(key_hash, datetime.datetime.combine(datetime.date.min, key))
        self.rec(key_hash, "<time>")

    def update_for_datetime(self, key_hash: Hash, key: Any) -> None:
        # 'datetime' should differentiate between naive and aware

        # https://docs.python.org/3.11/library/datetime.html#determining-if-an-object-is-aware-or-naive
        if key.tzinfo is not None and key.tzinfo.utcoffset(key) is not None:
            self.rec(key_hash, key.timestamp())
            self.rec(key_hash, "<aware>")
        else:
            from datetime import timezone
            self.rec(key_hash, key.replace(tzinfo=timezone.utc).timestamp())
            self.rec(key_hash, "<naive>")

    def update_for_timezone(self, key_hash: Hash, key: Any) -> None:
        self.rec(key_hash, repr(key))

    # }}}

    def update_for_function(self, key_hash: Hash, key: Any) -> None:
        self.rec(key_hash, key.__module__ + key.__qualname__)

        if key.__closure__:
            self.rec(key_hash, tuple(c.cell_contents for c in key.__closure__))

    # }}}

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    """Raised when an entry is not found in a :class:`PersistentDict`."""
    pass


class NoSuchEntryCollisionError(NoSuchEntryError):
    """Raised when an entry is not found in a :class:`PersistentDict`, but it
    contains an entry with the same hash key (hash collision)."""
    pass


class ReadOnlyEntryError(KeyError):
    """Raised when an attempt is made to overwrite an entry in a
    :class:`WriteOncePersistentDict`."""
    pass


class CollisionWarning(UserWarning):
    """Warning raised when a collision is detected in a :class:`PersistentDict`."""
    pass


def __getattr__(name: str) -> Any:
    if name in ("NoSuchEntryInvalidKeyError",
                "NoSuchEntryInvalidContentsError"):
        warn(f"pytools.persistent_dict.{name} has been removed.", stacklevel=2)
        return NoSuchEntryError

    raise AttributeError(name)


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class _PersistentDictBase(Mapping[K, V]):
    def __init__(self,
                 identifier: str,
                 key_builder: Optional[KeyBuilder] = None,
                 container_dir: Optional[str] = None,
                 enable_wal: bool = False,
                 safe_sync: Optional[bool] = None) -> None:
        self.identifier = identifier
        self.conn = None

        if key_builder is None:
            key_builder = KeyBuilder()

        self.key_builder = key_builder

        from os.path import join
        if container_dir is None:
            import platformdirs

            if sys.platform == "darwin" and os.getenv("XDG_CACHE_HOME") is not None:
                # platformdirs does not handle XDG_CACHE_HOME on macOS
                # https://github.com/platformdirs/platformdirs/issues/269
                container_dir = join(os.getenv("XDG_CACHE_HOME"), "pytools")
            else:
                container_dir = platformdirs.user_cache_dir("pytools", "pytools")

        self.filename = join(container_dir, f"pdict-v5-{identifier}"
                             + ".".join(str(i) for i in sys.version_info)
                             + ".sqlite")

        self.container_dir = container_dir
        self._make_container_dir()

        from threading import Lock
        self.mutex = Lock()

        # * isolation_level=None: enable autocommit mode
        #   https://www.sqlite.org/lang_transaction.html#implicit_versus_explicit_transactions
        # * check_same_thread=False: thread-level concurrency is handled by the
        #   mutex above
        self.conn = sqlite3.connect(self.filename,
                                    isolation_level=None,
                                    check_same_thread=False)

        self._exec_sql(
            "CREATE TABLE IF NOT EXISTS dict "
            "(keyhash TEXT NOT NULL PRIMARY KEY, key_value TEXT NOT NULL)"
            )

        # https://www.sqlite.org/wal.html
        if enable_wal:
            self._exec_sql("PRAGMA journal_mode = 'WAL'")

        # Note: the following configuration values were taken mostly from litedict:
        # https://github.com/litements/litedict/blob/377603fa597453ffd9997186a493ed4fd23e5399/litedict.py#L67-L70

        # Use in-memory temp store
        # https://www.sqlite.org/pragma.html#pragma_temp_store
        self._exec_sql("PRAGMA temp_store = 'MEMORY'")

        # fsync() can be extremely slow on some systems.
        # See https://github.com/inducer/pytools/issues/227 for context.
        # https://www.sqlite.org/pragma.html#pragma_synchronous
        if safe_sync is None or safe_sync:
            if safe_sync is None:
                warn(f"pytools.persistent_dict '{identifier}': "
                     "enabling safe_sync as default. "
                     "This provides strong protection against data loss, "
                     "but can be unnecessarily expensive for use cases such as "
                     "caches."
                     "Pass 'safe_sync=False' if occasional data loss is tolerable. "
                     "Pass 'safe_sync=True' to suppress this warning.",
                     stacklevel=3)
            self._exec_sql("PRAGMA synchronous = 'NORMAL'")
        else:
            self._exec_sql("PRAGMA synchronous = 'OFF'")

        # 64 MByte of cache
        # https://www.sqlite.org/pragma.html#pragma_cache_size
        self._exec_sql("PRAGMA cache_size = -64000")

    def __del__(self) -> None:
        with self.mutex:
            if self.conn:
                self.conn.close()

    def _collision_check(self, key: K, stored_key: K) -> None:
        if stored_key != key:
            # Key collision, oh well.
            warn(f"{self.identifier}: key collision in cache at "
                    f"'{self.container_dir}' -- these are sufficiently unlikely "
                    "that they're often indicative of a broken hash key "
                    "implementation (that is not considering some elements "
                    "relevant for equality comparison)",
                    CollisionWarning,
                    stacklevel=3
                 )

            # This is here so we can step through equality comparison to
            # see what is actually non-equal.
            stored_key == key  # noqa: B015
            raise NoSuchEntryCollisionError(key)

    def _exec_sql(self, *args: Any) -> sqlite3.Cursor:
        def execute() -> sqlite3.Cursor:
            assert self.conn is not None
            return self.conn.execute(*args)

        cursor = self._exec_sql_fn(execute)
        if not isinstance(cursor, sqlite3.Cursor):
            raise RuntimeError("Failed to execute SQL statement")

        return cursor

    def _exec_sql_fn(self, fn: Callable[[], T]) -> Optional[T]:
        n = 0

        with self.mutex:
            while True:
                n += 1
                try:
                    return fn()
                except sqlite3.OperationalError as e:
                    # If the database is busy, retry
                    if (hasattr(e, "sqlite_errorcode")
                        and e.sqlite_errorcode != SQLITE_BUSY):
                        raise
                    if n % 20 == 0:
                        warn(f"PersistentDict: database '{self.filename}' busy, {n} "
                             "retries", stacklevel=3)
                else:
                    break

    def store_if_not_present(self, key: K, value: V) -> None:
        """Store (*key*, *value*) if *key* is not already present."""
        self.store(key, value, _skip_if_present=True)

    def store(self, key: K, value: V, _skip_if_present: bool = False) -> None:
        """Store (*key*, *value*) in the dictionary."""
        raise NotImplementedError()

    def fetch(self, key: K) -> V:
        """Return the value associated with *key* in the dictionary."""
        raise NotImplementedError()

    def _make_container_dir(self) -> None:
        """Create the container directory to store the dictionary."""
        os.makedirs(self.container_dir, exist_ok=True)

    def __getitem__(self, key: K) -> V:
        """Return the value associated with *key* in the dictionary."""
        return self.fetch(key)

    def __setitem__(self, key: K, value: V) -> None:
        """Store (*key*, *value*) in the dictionary."""
        self.store(key, value)

    def __len__(self) -> int:
        """Return the number of entries in the dictionary."""
        result, = next(self._exec_sql("SELECT COUNT(*) FROM dict"))
        assert isinstance(result, int)
        return result

    def __iter__(self) -> Iterator[K]:
        """Return an iterator over the keys in the dictionary."""
        return self.keys()

    def keys(self) -> Iterator[K]:  # type: ignore[override]
        """Return an iterator over the keys in the dictionary."""
        for row in self._exec_sql("SELECT key_value FROM dict ORDER BY rowid"):
            yield pickle.loads(row[0])[0]

    def values(self) -> Iterator[V]:  # type: ignore[override]
        """Return an iterator over the values in the dictionary."""
        for row in self._exec_sql("SELECT key_value FROM dict ORDER BY rowid"):
            yield pickle.loads(row[0])[1]

    def items(self) -> Iterator[Tuple[K, V]]:  # type: ignore[override]
        """Return an iterator over the items in the dictionary."""
        for row in self._exec_sql("SELECT key_value FROM dict ORDER BY rowid"):
            yield pickle.loads(row[0])

    def nbytes(self) -> int:
        """Return the size of the dictionary in bytes."""
        result, = next(self._exec_sql("SELECT page_size * page_count FROM "
                                      "pragma_page_size(), pragma_page_count()"))
        assert isinstance(result, int)

        return result

    def __repr__(self) -> str:
        """Return a string representation of the dictionary."""
        return f"{type(self).__name__}({self.filename}, nitems={len(self)})"

    def clear(self) -> None:
        """Remove all entries from the dictionary."""
        self._exec_sql("DELETE FROM dict")


class WriteOncePersistentDict(_PersistentDictBase[K, V]):
    """A concurrent disk-backed dictionary that disallows overwriting/
    deletion (but allows removing all entries).

    Compared with :class:`PersistentDict`, this class has faster
    retrieval times because it uses an LRU cache to cache entries in memory.

    .. note::

        This class intentionally does not store all values with a certain
        key, based on the assumption that key conflicts are highly unlikely,
        and if they occur, almost always due to a bug in the hash key
        generation code (:class:`KeyBuilder`).

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: clear
    .. automethod:: clear_in_mem_cache
    .. automethod:: store
    .. automethod:: store_if_not_present
    .. automethod:: fetch
    """
    def __init__(self, identifier: str,
                 key_builder: Optional[KeyBuilder] = None,
                 container_dir: Optional[str] = None,
                 *,
                 enable_wal: bool = False,
                 safe_sync: Optional[bool] = None,
                 in_mem_cache_size: int = 256) -> None:
        """
        :arg identifier: a filename-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg container_dir: the directory in which to store this
            dictionary. If ``None``, the default cache directory from
            :func:`platformdirs.user_cache_dir` is used
        :arg enable_wal: enable write-ahead logging (WAL) mode. This mode
            is faster than the default rollback journal mode, but it is
            not compatible with network filesystems.
        :arg in_mem_cache_size: retain an in-memory cache of up to
            *in_mem_cache_size* items (with an LRU replacement policy)
        """
        super().__init__(identifier,
                         key_builder=key_builder,
                         container_dir=container_dir,
                         enable_wal=enable_wal,
                         safe_sync=safe_sync)

        from functools import lru_cache

        self._fetch = lru_cache(maxsize=in_mem_cache_size)(self._fetch_uncached)

    def clear_in_mem_cache(self) -> None:
        """
        Clear the in-memory cache of this dictionary.

        .. versionadded:: 2023.1.1
        """
        self._fetch.cache_clear()

    def store(self, key: K, value: V, _skip_if_present: bool = False) -> None:
        keyhash = self.key_builder(key)
        v = pickle.dumps((key, value))

        if _skip_if_present:
            self._exec_sql("INSERT OR IGNORE INTO dict VALUES (?, ?)",
                              (keyhash, v))
        else:
            try:
                self._exec_sql("INSERT INTO dict VALUES (?, ?)", (keyhash, v))
            except sqlite3.IntegrityError as e:
                if hasattr(e, "sqlite_errorcode"):
                    if e.sqlite_errorcode == SQLITE_CONSTRAINT_PRIMARYKEY:
                        raise ReadOnlyEntryError("WriteOncePersistentDict, "
                                                 "tried overwriting key") from e
                    else:
                        raise
                else:
                    raise ReadOnlyEntryError("WriteOncePersistentDict, "
                                             "tried overwriting key") from e

    def _fetch_uncached(self, keyhash: str) -> Tuple[K, V]:
        # This method is separate from fetch() to allow for LRU caching

        def fetch_inner() -> Optional[Tuple[Any]]:
            assert self.conn is not None

            # This is separate from fetch() so that the mutex covers the
            # fetchone() call
            c = self.conn.execute("SELECT key_value FROM dict WHERE keyhash=?",
                                (keyhash,))
            res = c.fetchone()
            assert res is None or isinstance(res, tuple)
            return res

        row = self._exec_sql_fn(fetch_inner)
        if row is None:
            raise KeyError

        key, value = pickle.loads(row[0])
        return key, value

    def fetch(self, key: K) -> V:
        keyhash = self.key_builder(key)

        try:
            stored_key, value = self._fetch(keyhash)
        except KeyError as err:
            raise NoSuchEntryError(key) from err
        else:
            self._collision_check(key, stored_key)
            return value

    def clear(self) -> None:
        super().clear()
        self._fetch.cache_clear()


class PersistentDict(_PersistentDictBase[K, V]):
    """A concurrent disk-backed dictionary.

    .. note::

        This class intentionally does not store all values with a certain
        key, based on the assumption that key conflicts are highly unlikely,
        and if they occur, almost always due to a bug in the hash key
        generation code (:class:`KeyBuilder`).

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: clear
    .. automethod:: store
    .. automethod:: store_if_not_present
    .. automethod:: fetch
    .. automethod:: remove
    """
    def __init__(self,
                 identifier: str,
                 key_builder: Optional[KeyBuilder] = None,
                 container_dir: Optional[str] = None,
                 *,
                 enable_wal: bool = False,
                 safe_sync: Optional[bool] = None) -> None:
        """
        :arg identifier: a filename-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg container_dir: the directory in which to store this
            dictionary. If ``None``, the default cache directory from
            :func:`platformdirs.user_cache_dir` is used
        :arg enable_wal: enable write-ahead logging (WAL) mode. This mode
            is faster than the default rollback journal mode, but it is
            not compatible with network filesystems.
        """
        super().__init__(identifier,
                         key_builder=key_builder,
                         container_dir=container_dir,
                         enable_wal=enable_wal,
                         safe_sync=safe_sync)

    def store(self, key: K, value: V, _skip_if_present: bool = False) -> None:
        keyhash = self.key_builder(key)
        v = pickle.dumps((key, value))

        mode = "IGNORE" if _skip_if_present else "REPLACE"

        self._exec_sql(f"INSERT OR {mode} INTO dict VALUES (?, ?)",
                              (keyhash, v))

    def fetch(self, key: K) -> V:
        keyhash = self.key_builder(key)

        def fetch_inner() -> Optional[Tuple[Any]]:
            assert self.conn is not None

            # This is separate from fetch() so that the mutex covers the
            # fetchone() call
            c = self.conn.execute("SELECT key_value FROM dict WHERE keyhash=?",
                                (keyhash,))
            res = c.fetchone()
            assert res is None or isinstance(res, tuple)
            return res

        row = self._exec_sql_fn(fetch_inner)

        if row is None:
            raise NoSuchEntryError(key)

        stored_key, value = pickle.loads(row[0])
        self._collision_check(key, stored_key)
        return cast(V, value)

    def remove(self, key: K) -> None:
        """Remove the entry associated with *key* from the dictionary."""
        keyhash = self.key_builder(key)

        def remove_inner() -> None:
            assert self.conn is not None
            self.conn.execute("BEGIN EXCLUSIVE TRANSACTION")
            try:
                # This is split into SELECT/DELETE to allow for a collision check
                c = self.conn.execute("SELECT key_value FROM dict WHERE "
                                        "keyhash=?", (keyhash,))
                row = c.fetchone()
                if row is None:
                    raise NoSuchEntryError(key)

                stored_key, _value = pickle.loads(row[0])
                self._collision_check(key, stored_key)

                self.conn.execute("DELETE FROM dict WHERE keyhash=?", (keyhash,))
                self.conn.execute("COMMIT")
            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

        self._exec_sql_fn(remove_inner)

    def __delitem__(self, key: K) -> None:
        """Remove the entry associated with *key* from the dictionary."""
        self.remove(key)

# }}}

# vim: foldmethod=marker
