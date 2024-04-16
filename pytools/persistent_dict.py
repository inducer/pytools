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

import errno
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import fields as dc_fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Protocol, TypeVar


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

__doc__ = """
Persistent Hashing and Persistent Dictionaries
==============================================

This module contains functionality that allows hashing with keys that remain
valid across interpreter invocations, unlike Python's built-in hashes.

This module also provides a disk-backed dictionary that uses persistent hashing.

.. autoexception:: NoSuchEntryError
.. autoexception:: NoSuchEntryInvalidKeyError
.. autoexception:: NoSuchEntryInvalidContentsError
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


# {{{ cleanup managers

class CleanupBase:
    pass


class CleanupManager(CleanupBase):
    def __init__(self):
        self.cleanups = []

    def register(self, c):
        self.cleanups.insert(0, c)

    def clean_up(self):
        for c in self.cleanups:
            c.clean_up()

    def error_clean_up(self):
        for c in self.cleanups:
            c.error_clean_up()


class LockManager(CleanupBase):
    def __init__(self, cleanup_m, lock_file, stacklevel=0):
        self.lock_file = lock_file

        attempts = 0
        while True:
            try:
                self.fd = os.open(self.lock_file,
                        os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                break
            except OSError:
                pass

            # This value was chosen based on the py-filelock package:
            # https://github.com/tox-dev/py-filelock/blob/a6c8fabc4192fa7a4ae19b1875ee842ec5eb4f61/src/filelock/_api.py#L113
            wait_time_seconds = 0.05

            # Warn every 10 seconds if not able to acquire lock
            warn_attempts = int(10/wait_time_seconds)

            # Exit after 60 seconds if not able to acquire lock
            exit_attempts = int(60/wait_time_seconds)

            from time import sleep
            sleep(wait_time_seconds)

            attempts += 1

            if attempts % warn_attempts == 0:
                from warnings import warn
                warn("could not obtain lock -- "
                        f"delete '{self.lock_file}' if necessary",
                        stacklevel=1 + stacklevel)

            if attempts > exit_attempts:
                raise RuntimeError("waited more than one minute "
                        f"on the lock file '{self.lock_file}' "
                        "-- something is wrong")

        cleanup_m.register(self)

    def clean_up(self):
        os.close(self.fd)
        os.unlink(self.lock_file)

    def error_clean_up(self):
        pass


class ItemDirManager(CleanupBase):
    def __init__(self, cleanup_m, path, delete_on_error):
        from os.path import isdir

        self.existed = isdir(path)
        self.path = path
        self.delete_on_error = delete_on_error

        cleanup_m.register(self)

    def reset(self):
        try:
            shutil.rmtree(self.path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def mkdir(self):
        from os import makedirs
        makedirs(self.path, exist_ok=True)

    def clean_up(self):
        pass

    def error_clean_up(self):
        if self.delete_on_error:
            self.reset()

# }}}


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
    """

    # this exists so that we can (conceivably) switch algorithms at some point
    # down the road
    new_hash = hashlib.sha256

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
                # pylint:disable=protected-access
                object.__setattr__(key, "_pytools_persistent_hash_digest",  digest)
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
            f"{key.__module__}.{key.__qualname__}.{key.__name__}".encode("utf-8"))

    update_for_ABCMeta = update_for_type  # noqa: N815

    @staticmethod
    def update_for_int(key_hash: Hash, key: int) -> None:
        sz = 8
        while True:
            try:
                key_hash.update(key.to_bytes(sz, byteorder="little", signed=True))
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

    def update_for_tuple(self, key_hash: Hash, key: tuple) -> None:
        for obj_i in key:
            self.rec(key_hash, obj_i)

    def update_for_frozenset(self, key_hash: Hash, key: frozenset) -> None:
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), key_i).digest() for key_i in key))

    update_for_FrozenOrderedSet = update_for_frozenset  # noqa: N815

    @staticmethod
    def update_for_NoneType(key_hash: Hash, key: None) -> None:  # noqa: N802
        del key
        key_hash.update(b"<None>")

    @staticmethod
    def update_for_dtype(key_hash, key):
        key_hash.update(key.str.encode("utf8"))

    # Handling numpy >= 1.20, for which
    # type(np.dtype("float32")) -> "dtype[float32]"
    # Introducing this method allows subclasses to specially handle all those
    # dtypes.
    @staticmethod
    def update_for_specific_dtype(key_hash, key):
        key_hash.update(key.str.encode("utf8"))

    @staticmethod
    def update_for_numpy_scalar(key_hash: Hash, key) -> None:
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

    def update_for_attrs(self, key_hash: Hash, key) -> None:
        self.rec(key_hash, f"{type(key).__qualname__}.{type(key).__name__}")

        for fld in attrs.fields(key.__class__):
            self.rec(key_hash, fld.name)
            self.rec(key_hash, getattr(key, fld.name, None))

    def update_for_frozendict(self, key_hash: Hash, key: Mapping) -> None:
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), (k, v)).digest() for k, v in key.items()))

    update_for_immutabledict = update_for_frozendict
    update_for_constantdict = update_for_frozendict
    update_for_PMap = update_for_frozendict  # noqa: N815
    update_for_Map = update_for_frozendict  # noqa: N815

    def update_for_datetime(self, key_hash: Hash, key: Any) -> None:
        self.rec(key_hash, key.isoformat(timespec="microseconds"))

    # }}}

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    """Raised when an entry is not found in a :class:`PersistentDict`."""
    pass


class NoSuchEntryInvalidKeyError(NoSuchEntryError):
    """Raised when an entry is not found in a :class:`PersistentDict` due to an
    invalid key file."""
    pass


class NoSuchEntryInvalidContentsError(NoSuchEntryError):
    """Raised when an entry is not found in a :class:`PersistentDict` due to an
    invalid contents file."""
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


K = TypeVar("K")
V = TypeVar("V")


class _PersistentDictBase(Generic[K, V]):
    def __init__(self, identifier: str,
                 key_builder: Optional[KeyBuilder] = None,
                 container_dir: Optional[str] = None) -> None:
        self.identifier = identifier

        if key_builder is None:
            key_builder = KeyBuilder()

        self.key_builder = key_builder

        from os.path import join
        if container_dir is None:
            import platformdirs

            if sys.platform == "darwin" and os.getenv("XDG_CACHE_HOME") is not None:
                # platformdirs does not handle XDG_CACHE_HOME on macOS
                # https://github.com/platformdirs/platformdirs/issues/269
                cache_dir = join(os.getenv("XDG_CACHE_HOME"), "pytools")
            else:
                cache_dir = platformdirs.user_cache_dir("pytools", "pytools")

            container_dir = join(
                    cache_dir,
                    "pdict-v4-{}-py{}".format(
                        identifier,
                        ".".join(str(i) for i in sys.version_info)))

        self.container_dir = container_dir

        self._make_container_dir()

    @staticmethod
    def _warn(msg: str, category: Any = UserWarning, stacklevel: int = 0) -> None:
        from warnings import warn
        warn(msg, category, stacklevel=1 + stacklevel)

    def store_if_not_present(self, key: K, value: V,
                             _stacklevel: int = 0) -> None:
        """Store (*key*, *value*) if *key* is not already present."""
        self.store(key, value, _skip_if_present=True, _stacklevel=1 + _stacklevel)

    def store(self, key: K, value: V, _skip_if_present: bool = False,
              _stacklevel: int = 0) -> None:
        """Store (*key*, *value*) in the dictionary."""
        raise NotImplementedError()

    def fetch(self, key: K, _stacklevel: int = 0) -> V:
        """Return the value associated with *key* in the dictionary."""
        raise NotImplementedError()

    @staticmethod
    def _read(path: str) -> V:
        from pickle import load
        with open(path, "rb") as inf:
            return load(inf)

    @staticmethod
    def _write(path: str, value: V) -> None:
        from pickle import HIGHEST_PROTOCOL, dump
        with open(path, "wb") as outf:
            dump(value, outf, protocol=HIGHEST_PROTOCOL)

    def _item_dir(self, hexdigest_key: str) -> str:
        from os.path import join

        # Some file systems limit the number of directories in a directory.
        # For ext4, that limit appears to be 64K for example.
        # This doesn't solve that problem, but it makes it much less likely

        return join(self.container_dir,
                hexdigest_key[:3],
                hexdigest_key[3:6],
                hexdigest_key[6:])

    def _key_file(self, hexdigest_key: str) -> str:
        from os.path import join
        return join(self._item_dir(hexdigest_key), "key")

    def _contents_file(self, hexdigest_key: str) -> str:
        from os.path import join
        return join(self._item_dir(hexdigest_key), "contents")

    def _lock_file(self, hexdigest_key: str) -> str:
        from os.path import join
        return join(self.container_dir, str(hexdigest_key) + ".lock")

    def _make_container_dir(self) -> None:
        """Create the container directory to store the dictionary."""
        os.makedirs(self.container_dir, exist_ok=True)

    def _collision_check(self, key: K, stored_key: K, _stacklevel: int) -> None:
        if stored_key != key:
            # Key collision, oh well.
            self._warn(f"{self.identifier}: key collision in cache at "
                    f"'{self.container_dir}' -- these are sufficiently unlikely "
                    "that they're often indicative of a broken hash key "
                    "implementation (that is not considering some elements "
                    "relevant for equality comparison)",
                    CollisionWarning,
                    1 + _stacklevel)

            # This is here so we can step through equality comparison to
            # see what is actually non-equal.
            stored_key == key  # pylint:disable=pointless-statement  # noqa: B015
            raise NoSuchEntryCollisionError(key)

    def __getitem__(self, key: K) -> V:
        """Return the value associated with *key* in the dictionary."""
        return self.fetch(key, _stacklevel=1)

    def __setitem__(self, key: K, value: V) -> None:
        """Store (*key*, *value*) in the dictionary."""
        self.store(key, value, _stacklevel=1)

    def clear(self) -> None:
        """Remove all entries from the dictionary."""
        try:
            shutil.rmtree(self.container_dir)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

        self._make_container_dir()


class WriteOncePersistentDict(_PersistentDictBase[K, V]):
    """A concurrent disk-backed dictionary that disallows overwriting/
    deletion (but allows removing all entries).

    Compared with :class:`PersistentDict`, this class has faster
    retrieval times because it uses an LRU cache to cache entries in memory.

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
                 in_mem_cache_size: int = 256) -> None:
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg container_dir: the directory in which to store this
            dictionary. If ``None``, the default cache directory from
            :func:`platformdirs.user_cache_dir` is used
        :arg in_mem_cache_size: retain an in-memory cache of up to
            *in_mem_cache_size* items (with an LRU replacement policy)
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)
        self._in_mem_cache_size = in_mem_cache_size
        from functools import lru_cache
        self._fetch = lru_cache(maxsize=in_mem_cache_size)(self._fetch)

    def clear_in_mem_cache(self) -> None:
        """
        Clear the in-memory cache of this dictionary.

        .. versionadded:: 2023.1.1
        """

        self._fetch.cache_clear()

    def _spin_until_removed(self, lock_file: str, stacklevel: int) -> None:
        from os.path import exists

        attempts = 0
        while exists(lock_file):
            from time import sleep
            sleep(1)

            attempts += 1

            if attempts > 10:
                self._warn(
                        f"waiting until unlocked--delete '{lock_file}' if necessary",
                        stacklevel=1 + stacklevel)

            if attempts > 3 * 60:
                raise RuntimeError("waited more than three minutes "
                        f"on the lock file '{lock_file}'"
                        "--something is wrong")

    def store(self, key: K, value: V, _skip_if_present: bool = False,
              _stacklevel: int = 0) -> None:
        hexdigest_key = self.key_builder(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, self._item_dir(hexdigest_key),
                        delete_on_error=False)

                if item_dir_m.existed:
                    if _skip_if_present:
                        return
                    raise ReadOnlyEntryError(key)

                item_dir_m.mkdir()

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                self._write(value_path, value)
                self._write(key_path, key)

                logger.debug("%s: disk cache store [key=%s]",
                        self.identifier, hexdigest_key)
            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def fetch(self, key: K, _stacklevel: int = 0) -> Any:
        hexdigest_key = self.key_builder(key)

        (stored_key, stored_value) = self._fetch(hexdigest_key, 1 + _stacklevel)

        self._collision_check(key, stored_key, 1 + _stacklevel)

        return stored_value

    def _fetch(self, hexdigest_key: str,  # pylint:disable=method-hidden
               _stacklevel: int = 0) -> V:
        # This is separate from fetch() to allow for LRU caching

        # {{{ check path exists and is unlocked

        item_dir = self._item_dir(hexdigest_key)

        from os.path import isdir
        if not isdir(item_dir):
            logger.debug("%s: disk cache miss [key=%s]",
                    self.identifier, hexdigest_key)
            raise NoSuchEntryError(hexdigest_key)

        lock_file = self._lock_file(hexdigest_key)
        self._spin_until_removed(lock_file, 1 + _stacklevel)

        # }}}

        key_file = self._key_file(hexdigest_key)
        contents_file = self._contents_file(hexdigest_key)

        # Note: Unlike PersistentDict, this doesn't autodelete invalid entires,
        # because that would lead to a race condition.

        # {{{ load key file and do equality check

        try:
            read_key = self._read(key_file)
        except Exception as e:
            self._warn(f"{type(self).__name__}({self.identifier}) "
                    f"encountered an invalid key file for key {hexdigest_key}. "
                    f"Remove the directory '{item_dir}' if necessary. "
                    f"(caught: {type(e).__name__}: {e})",
                    stacklevel=1 + _stacklevel)
            raise NoSuchEntryInvalidKeyError(hexdigest_key)

        # }}}

        logger.debug("%s: disk cache hit [key=%s]",
                self.identifier, hexdigest_key)

        # {{{ load contents

        try:
            read_contents = self._read(contents_file)
        except Exception as e:
            self._warn(f"{type(self).__name__}({self.identifier}) "
                    f"encountered an invalid contents file for key {hexdigest_key}. "
                    f"Remove the directory '{item_dir}' if necessary."
                    f"(caught: {type(e).__name__}: {e})",
                    stacklevel=1 + _stacklevel)
            raise NoSuchEntryInvalidContentsError(hexdigest_key)

        # }}}

        return (read_key, read_contents)

    def clear(self) -> None:
        _PersistentDictBase.clear(self)
        self._fetch.cache_clear()


class PersistentDict(_PersistentDictBase[K, V]):
    """A concurrent disk-backed dictionary.

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
                 container_dir: Optional[str] = None) -> None:
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg container_dir: the directory in which to store this
            dictionary. If ``None``, the default cache directory from
            :func:`platformdirs.user_cache_dir` is used
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)

    def store(self, key: K, value: V, _skip_if_present: bool = False,
              _stacklevel: int = 0) -> None:
        hexdigest_key = self.key_builder(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, self._item_dir(hexdigest_key),
                        delete_on_error=True)

                if item_dir_m.existed:
                    if _skip_if_present:
                        return
                    item_dir_m.reset()

                item_dir_m.mkdir()

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                self._write(value_path, value)
                self._write(key_path, key)

                logger.debug("%s: cache store [key=%s]",
                        self.identifier, hexdigest_key)
            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def fetch(self, key: K, _stacklevel: int = 0) -> V:
        hexdigest_key = self.key_builder(key)
        item_dir = self._item_dir(hexdigest_key)

        from os.path import isdir
        if not isdir(item_dir):
            logger.debug("%s: cache miss [key=%s]",
                    self.identifier, hexdigest_key)
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, item_dir, delete_on_error=False)

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                # {{{ load key

                try:
                    read_key = self._read(key_path)
                except Exception as e:
                    item_dir_m.reset()
                    self._warn(f"{type(self).__name__}({self.identifier}) "
                            "encountered an invalid key file for key "
                            f"{hexdigest_key}. Entry deleted."
                            f"(caught: {type(e).__name__}: {e})",
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryInvalidKeyError(key)

                self._collision_check(key, read_key, 1 + _stacklevel)

                # }}}

                logger.debug("%s: cache hit [key=%s]",
                        self.identifier, hexdigest_key)

                # {{{ load value

                try:
                    read_contents = self._read(value_path)
                except Exception as e:
                    item_dir_m.reset()
                    self._warn(f"{type(self).__name__}({self.identifier}) "
                            "encountered an invalid contents file for key "
                            f"{hexdigest_key}. Entry deleted."
                            f"(caught: {type(e).__name__}: {e})",
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryInvalidContentsError(key)

                return read_contents

                # }}}

            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def remove(self, key: K, _stacklevel: int = 0) -> None:
        """Remove the entry associated with *key* from the dictionary."""
        hexdigest_key = self.key_builder(key)

        item_dir = self._item_dir(hexdigest_key)
        from os.path import isdir
        if not isdir(item_dir):
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, item_dir, delete_on_error=False)
                key_file = self._key_file(hexdigest_key)

                # {{{ load key

                try:
                    read_key = self._read(key_file)
                except Exception as e:
                    item_dir_m.reset()
                    self._warn(f"{type(self).__name__}({self.identifier}) "
                            "encountered an invalid key file for key "
                            f"{hexdigest_key}. Entry deleted"
                            f"(caught: {type(e).__name__}: {e})",
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryInvalidKeyError(key)

                self._collision_check(key, read_key, 1 + _stacklevel)

                # }}}

                item_dir_m.reset()

            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def __delitem__(self, key: K) -> None:
        """Remove the entry associated with *key* from the dictionary."""
        self.remove(key, _stacklevel=1)

# }}}

# vim: foldmethod=marker
