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

import collections.abc as abc
import errno
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import fields as dc_fields, is_dataclass
from enum import Enum


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

.. autoclass:: KeyBuilder
.. autoclass:: PersistentDict
.. autoclass:: WriteOncePersistentDict
"""

# {{{ key generation

class KeyBuilder:
    """A (stateless) object that computes hashes of objects fed to it. Subclassing
    this class permits customizing the computation of hash keys.

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

    def rec(self, key_hash, key):
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

    def __call__(self, key):
        key_hash = self.new_hash()
        self.rec(key_hash, key)
        return key_hash.hexdigest()

    # {{{ updaters

    @staticmethod
    def update_for_type(key_hash, key):
        key_hash.update(
            f"{key.__module__}.{key.__qualname__}.{key.__name__}".encode("utf-8"))

    update_for_ABCMeta = update_for_type  # noqa: N815

    @staticmethod
    def update_for_int(key_hash, key):
        sz = 8
        while True:
            try:
                key_hash.update(key.to_bytes(sz, byteorder="little", signed=True))
                return
            except OverflowError:
                sz *= 2

    @classmethod
    def update_for_enum(cls, key_hash, key):
        cls.update_for_str(key_hash, str(key))

    @staticmethod
    def update_for_bool(key_hash, key):
        key_hash.update(str(key).encode("utf8"))

    @staticmethod
    def update_for_float(key_hash, key):
        key_hash.update(key.hex().encode("utf8"))

    @staticmethod
    def update_for_complex(key_hash, key):
        key_hash.update(repr(key).encode("utf-8"))

    @staticmethod
    def update_for_str(key_hash, key):
        key_hash.update(key.encode("utf8"))

    @staticmethod
    def update_for_bytes(key_hash, key):
        key_hash.update(key)

    def update_for_tuple(self, key_hash, key):
        for obj_i in key:
            self.rec(key_hash, obj_i)

    def update_for_frozenset(self, key_hash, key):
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), key_i).digest() for key_i in key))

    update_for_FrozenOrderedSet = update_for_frozenset  # noqa: N815

    @staticmethod
    def update_for_NoneType(key_hash, key):  # noqa: N802
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
    def update_for_numpy_scalar(key_hash, key):
        import numpy as np
        if hasattr(np, "complex256") and key.dtype == np.dtype("complex256"):
            key_hash.update(repr(complex(key)).encode("utf8"))
        elif hasattr(np, "float128") and key.dtype == np.dtype("float128"):
            key_hash.update(repr(float(key)).encode("utf8"))
        else:
            key_hash.update(np.array(key).tobytes())

    def update_for_dataclass(self, key_hash, key):
        self.rec(key_hash, f"{type(key).__qualname__}.{type(key).__name__}")

        for fld in dc_fields(key):
            self.rec(key_hash, fld.name)
            self.rec(key_hash, getattr(key, fld.name, None))

    def update_for_attrs(self, key_hash, key):
        self.rec(key_hash, f"{type(key).__qualname__}.{type(key).__name__}")

        for fld in attrs.fields(key.__class__):
            self.rec(key_hash, fld.name)
            self.rec(key_hash, getattr(key, fld.name, None))

    def update_for_frozendict(self, key_hash, key):
        from pytools import unordered_hash

        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), (k, v)).digest() for k, v in key.items()))

    update_for_immutabledict = update_for_frozendict
    update_for_constantdict = update_for_frozendict
    update_for_PMap = update_for_frozendict  # noqa: N815
    update_for_Map = update_for_frozendict  # noqa: N815

    # }}}

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    pass


class NoSuchEntryInvalidKeyError(NoSuchEntryError):
    pass


class NoSuchEntryInvalidContentsError(NoSuchEntryError):
    pass


class NoSuchEntryCollisionError(NoSuchEntryError):
    pass


class ReadOnlyEntryError(KeyError):
    pass


class CollisionWarning(UserWarning):
    pass


class _PersistentDictBase:
    def __init__(self, identifier, key_builder=None, container_dir=None,
                 compression: str = "lzma"):
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
                container_dir = join(os.getenv("XDG_CACHE_HOME"), "pytools")
            else:
                container_dir = platformdirs.user_cache_dir("pytools", "pytools")

            # container_dir = join(
            #         cache_dir,
            #         "pdict-v6-{}-py{}".format(
            #             identifier,
            #             ".".join(str(i) for i in sys.version_info)))
        filename = join(container_dir, f"pdict-v6-{identifier}" + ".".join(str(i) for i in sys.version_info) + ".sqlite")

        self.container_dir = container_dir

        self._make_container_dir()

        from sqlitedict import SqliteDict

        import blosc, pickle, sqlite3
        def my_encode(obj):
            return sqlite3.Binary(blosc.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))
        def my_decode(obj):
            return pickle.loads(blosc.decompress(bytes(obj)))

        self.db = SqliteDict(filename, encode=my_encode,
                                       decode=my_decode)

    @staticmethod
    def _warn(msg, category=UserWarning, stacklevel=0):
        from warnings import warn
        warn(msg, category, stacklevel=1 + stacklevel)

    def store_if_not_present(self, key, value, _stacklevel=0):
        self.store(key, value, _skip_if_present=True, _stacklevel=1 + _stacklevel)

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        raise NotImplementedError()

    def fetch(self, key, _stacklevel=0):
        raise NotImplementedError()

    # def _read(self, path):
    #     from pickle import load
    #     with self.compression.open(path, "rb") as inf:
    #         return load(inf)

    # def _write(self, path, value):
    #     from pickle import HIGHEST_PROTOCOL, dump
    #     with self.compression.open(path, "wb") as outf:
    #         dump(value, outf, protocol=HIGHEST_PROTOCOL)

    # def _item_dir(self, hexdigest_key):
    #     from os.path import join

    #     # Some file systems limit the number of directories in a directory.
    #     # For ext4, that limit appears to be 64K for example.
    #     # This doesn't solve that problem, but it makes it much less likely

    #     return join(self.container_dir,
    #             hexdigest_key[:1],
    #             hexdigest_key[1:])

    # def _key_file(self, hexdigest_key):
    #     from os.path import join
    #     return join(self._item_dir(hexdigest_key), "key")

    # def _contents_file(self, hexdigest_key):
    #     from os.path import join
    #     return join(self._item_dir(hexdigest_key), "contents")

    # def _lock_file(self, hexdigest_key):
    #     from os.path import join
    #     return join(self.container_dir, str(hexdigest_key) + ".lock")

    def _make_container_dir(self):
        os.makedirs(self.container_dir, exist_ok=True)

    def _collision_check(self, key, stored_key, _stacklevel):
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

    def __getitem__(self, key):
        return self.fetch(key, _stacklevel=1)

    def __setitem__(self, key, value):
        self.store(key, value, _stacklevel=1)

    def clear(self):
        # try:
        #     shutil.rmtree(self.container_dir)
        # except OSError as e:
        #     if e.errno != errno.ENOENT:
        #         raise

        self.db.clear()
        self._make_container_dir()


class WriteOncePersistentDict(_PersistentDictBase):
    """A concurrent disk-backed dictionary that disallows overwriting/deletion.

    Compared with :class:`PersistentDict`, this class has faster
    retrieval times.

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: clear
    .. automethod:: clear_in_mem_cache
    .. automethod:: store
    .. automethod:: store_if_not_present
    .. automethod:: fetch
    """
    def __init__(self, identifier, key_builder=None, container_dir=None,
             in_mem_cache_size=256):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg in_mem_cache_size: retain an in-memory cache of up to
            *in_mem_cache_size* items
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)
        # self._in_mem_cache_size = in_mem_cache_size
        # self.clear_in_mem_cache()

    # def clear_in_mem_cache(self) -> None:
    #     """
    #     .. versionadded:: 2023.1.1
    #     """

    #     self._cache = _LRUCache(self._in_mem_cache_size)

    # def _spin_until_removed(self, lock_file, stacklevel):
    #     from os.path import exists

    #     attempts = 0
    #     while exists(lock_file):
    #         from time import sleep
    #         sleep(1)

    #         attempts += 1

    #         if attempts > 10:
    #             self._warn(
    #                     f"waiting until unlocked--delete '{lock_file}' if necessary",
    #                     stacklevel=1 + stacklevel)

    #         if attempts > 3 * 60:
    #             raise RuntimeError("waited more than three minutes "
    #                     f"on the lock file '{lock_file}'"
    #                     "--something is wrong")

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        if hexdigest_key in self.db:
            if _skip_if_present:
                return
            raise ReadOnlyEntryError(key)
        self.db[hexdigest_key] = value
        self.db.commit()
        # try:
        #     try:
        #         LockManager(cleanup_m, self._lock_file(hexdigest_key),
        #                 1 + _stacklevel)
        #         item_dir_m = ItemDirManager(
        #                 cleanup_m, self._item_dir(hexdigest_key),
        #                 delete_on_error=False)

        #         if item_dir_m.existed:
        #             if _skip_if_present:
        #                 return
        #             raise ReadOnlyEntryError(key)

        #         item_dir_m.mkdir()

        #         key_path = self._key_file(hexdigest_key)
        #         value_path = self._contents_file(hexdigest_key)

        #         self._write(value_path, value)
        #         self._write(key_path, key)

        #         logger.debug("%s: disk cache store [key=%s]",
        #                 self.identifier, hexdigest_key)
        #     except Exception:
        #         cleanup_m.error_clean_up()
        #         raise
        # finally:
        #     cleanup_m.clean_up()

    def fetch(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        try:
            return self.db[hexdigest_key]
        except KeyError:
            raise NoSuchEntryError(key)

        # {{{ in memory cache

        # try:
        #     stored_key, stored_value = self._cache[hexdigest_key]
        # except KeyError:
        #     pass
        # else:
        #     logger.debug("%s: in mem cache hit [key=%s]",
        #             self.identifier, hexdigest_key)
        #     self._collision_check(key, stored_key, 1 + _stacklevel)
        #     return stored_value

        # }}}

        # {{{ check path exists and is unlocked

        # item_dir = self._item_dir(hexdigest_key)

        # from os.path import isdir
        # if not isdir(item_dir):
        #     logger.debug("%s: disk cache miss [key=%s]",
        #             self.identifier, hexdigest_key)
        #     raise NoSuchEntryError(key)

        # lock_file = self._lock_file(hexdigest_key)
        # self._spin_until_removed(lock_file, 1 + _stacklevel)

        # }}}

        # key_file = self._key_file(hexdigest_key)
        # contents_file = self._contents_file(hexdigest_key)

        # Note: Unlike PersistentDict, this doesn't autodelete invalid entires,
        # because that would lead to a race condition.

        # {{{ load key file and do equality check

        # try:
        #     read_key = self._read(key_file)
        # except Exception as e:
        #     self._warn(f"{type(self).__name__}({self.identifier}) "
        #             f"encountered an invalid key file for key {hexdigest_key}. "
        #             f"Remove the directory '{item_dir}' if necessary. "
        #             f"(caught: {type(e).__name__}: {e})",
        #             stacklevel=1 + _stacklevel)
        #     raise NoSuchEntryInvalidKeyError(key)

        # self._collision_check(key, read_key, 1 + _stacklevel)

        # }}}

        # logger.debug("%s: disk cache hit [key=%s]",
        #         self.identifier, hexdigest_key)

        # {{{ load contents

        # try:
        #     read_contents = self._read(contents_file)
        # except Exception as e:
        #     self._warn(f"{type(self).__name__}({self.identifier}) "
        #             f"encountered an invalid contents file for key {hexdigest_key}. "
        #             f"Remove the directory '{item_dir}' if necessary."
        #             f"(caught: {type(e).__name__}: {e})",
        #             stacklevel=1 + _stacklevel)
        #     raise NoSuchEntryInvalidContentsError(key)

        # }}}

        # self._cache[hexdigest_key] = (key, read_contents)
        # return read_contents

    def clear(self):
        _PersistentDictBase.clear(self)
        # self._cache.clear()


class PersistentDict(_PersistentDictBase):
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
    def __init__(self, identifier, key_builder=None, container_dir=None):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        if _skip_if_present and hexdigest_key in self.db:
            return

        self.db[hexdigest_key] = value
        self.db.commit()

        # cleanup_m = CleanupManager()
        # try:
        #     try:
        #         LockManager(cleanup_m, self._lock_file(hexdigest_key),
        #                 1 + _stacklevel)
        #         item_dir_m = ItemDirManager(
        #                 cleanup_m, self._item_dir(hexdigest_key),
        #                 delete_on_error=True)

        #         if item_dir_m.existed:
        #             if _skip_if_present:
        #                 return
        #             item_dir_m.reset()

        #         item_dir_m.mkdir()

        #         key_path = self._key_file(hexdigest_key)
        #         value_path = self._contents_file(hexdigest_key)

        #         self._write(value_path, value)
        #         self._write(key_path, key)

        #         logger.debug("%s: cache store [key=%s]",
        #                 self.identifier, hexdigest_key)
        #     except Exception:
        #         cleanup_m.error_clean_up()
        #         raise
        # finally:
        #     cleanup_m.clean_up()

    def fetch(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        # if hexdigest_key not in self.db:
        #     raise NoSuchEntryError(key)
        try:
            return self.db[hexdigest_key]
        except KeyError:
            raise NoSuchEntryError(key)
        # item_dir = self._item_dir(hexdigest_key)

        # from os.path import isdir
        # if not isdir(item_dir):
        #     logger.debug("%s: cache miss [key=%s]",
        #             self.identifier, hexdigest_key)
        #     raise NoSuchEntryError(key)

        # cleanup_m = CleanupManager()
        # try:
        #     try:
        #         LockManager(cleanup_m, self._lock_file(hexdigest_key),
        #                 1 + _stacklevel)
        #         item_dir_m = ItemDirManager(
        #                 cleanup_m, item_dir, delete_on_error=False)

        #         key_path = self._key_file(hexdigest_key)
        #         value_path = self._contents_file(hexdigest_key)

        #         # {{{ load key

        #         try:
        #             read_key = self._read(key_path)
        #         except Exception as e:
        #             item_dir_m.reset()
        #             self._warn(f"{type(self).__name__}({self.identifier}) "
        #                     "encountered an invalid key file for key "
        #                     f"{hexdigest_key}. Entry deleted."
        #                     f"(caught: {type(e).__name__}: {e})",
        #                     stacklevel=1 + _stacklevel)
        #             raise NoSuchEntryInvalidKeyError(key)

        #         self._collision_check(key, read_key, 1 + _stacklevel)

        #         # }}}

        #         logger.debug("%s: cache hit [key=%s]",
        #                 self.identifier, hexdigest_key)

        #         # {{{ load value

        #         try:
        #             read_contents = self._read(value_path)
        #         except Exception as e:
        #             item_dir_m.reset()
        #             self._warn(f"{type(self).__name__}({self.identifier}) "
        #                     "encountered an invalid contents file for key "
        #                     f"{hexdigest_key}. Entry deleted."
        #                     f"(caught: {type(e).__name__}: {e})",
        #                     stacklevel=1 + _stacklevel)
        #             raise NoSuchEntryInvalidContentsError(key)

        #         return read_contents

        #         # }}}

        #     except Exception:
        #         cleanup_m.error_clean_up()
        #         raise
        # finally:
        #     cleanup_m.clean_up()

    def remove(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        try:
            del self.db[hexdigest_key]
            self.db.commit()
        except KeyError:
            raise NoSuchEntryError(key)
        # item_dir = self._item_dir(hexdigest_key)
        # from os.path import isdir
        # if not isdir(item_dir):
        #     raise NoSuchEntryError(key)

        # cleanup_m = CleanupManager()
        # try:
        #     try:
        #         LockManager(cleanup_m, self._lock_file(hexdigest_key),
        #                 1 + _stacklevel)
        #         item_dir_m = ItemDirManager(
        #                 cleanup_m, item_dir, delete_on_error=False)
        #         key_file = self._key_file(hexdigest_key)

        #         # {{{ load key

        #         try:
        #             read_key = self._read(key_file)
        #         except Exception as e:
        #             item_dir_m.reset()
        #             self._warn(f"{type(self).__name__}({self.identifier}) "
        #                     "encountered an invalid key file for key "
        #                     f"{hexdigest_key}. Entry deleted"
        #                     f"(caught: {type(e).__name__}: {e})",
        #                     stacklevel=1 + _stacklevel)
        #             raise NoSuchEntryInvalidKeyError(key)

        #         self._collision_check(key, read_key, 1 + _stacklevel)

        #         # }}}

        #         item_dir_m.reset()

        #     except Exception:
        #         cleanup_m.error_clean_up()
        #         raise
        # finally:
        #     cleanup_m.clean_up()

    def __delitem__(self, key):
        self.remove(key, _stacklevel=1)

# }}}

# vim: foldmethod=marker
