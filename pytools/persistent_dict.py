"""Generic persistent, concurrent dictionary-like facility."""

from __future__ import division, with_statement, absolute_import

__copyright__ = "Copyright (C) 2011,2014 Andreas Kloeckner"

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
logger = logging.getLogger(__name__)


import six
import sys
import os
import errno

__doc__ = """
Persistent Hashing
==================

This module contains functionality that allows hashing with keys that remain
valid across interpreter invocations, unlike Python's built-in hashes.

.. autoexception:: NoSuchEntryError
.. autoclass:: KeyBuilder
.. autoclass:: PersistentDict
"""

try:
    import hashlib
    new_hash = hashlib.sha256
except ImportError:
    # for Python << 2.5
    import sha
    new_hash = sha.new


def _make_dir_recursively(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        from errno import EEXIST
        if e.errno != EEXIST:
            raise


def _erase_dir(dir):
    from os import listdir, unlink, rmdir
    from os.path import join, isdir
    for name in listdir(dir):
        sub_name = join(dir, name)
        if isdir(sub_name):
            _erase_dir(sub_name)
        else:
            unlink(sub_name)

    rmdir(dir)


def update_checksum(checksum, obj):
    if isinstance(obj, six.text_type):
        checksum.update(obj.encode("utf8"))
    else:
        checksum.update(obj)


# {{{ cleanup managers

class CleanupBase(object):
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
    def __init__(self, cleanup_m, container_dir):
        if container_dir is not None:
            self.lock_file = os.path.join(container_dir, "lock")

            attempts = 0
            while True:
                try:
                    self.fd = os.open(self.lock_file,
                            os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                    break
                except OSError:
                    pass

                from time import sleep
                sleep(1)

                attempts += 1

                if attempts > 10:
                    from warnings import warn
                    warn("could not obtain lock--delete '%s' if necessary"
                            % self.lock_file)
                if attempts > 3 * 60:
                    raise RuntimeError("waited more than three minutes "
                            "on the lock file '%s'"
                            "--something is wrong" % self.lock_file)

            cleanup_m.register(self)

    def clean_up(self):
        import os
        os.close(self.fd)
        os.unlink(self.lock_file)

    def error_clean_up(self):
        pass


class ItemDirManager(CleanupBase):
    def __init__(self, cleanup_m, path):
        from os import mkdir
        import errno

        self.path = path
        try:
            mkdir(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            self.existed = True
        else:
            cleanup_m.register(self)
            self.existed = False

    def sub(self, n):
        from os.path import join
        return join(self.path, n)

    def reset(self):
        try:
            _erase_dir(self.path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

        try:
            os.mkdir(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def clean_up(self):
        pass

    def error_clean_up(self):
        _erase_dir(self.path)

# }}}


# {{{ key generation

class KeyBuilder(object):
    def rec(self, key_hash, key):
        digest = None

        try:
            digest = key._pytools_persistent_hash_digest
        except AttributeError:
            pass

        if digest is None:
            try:
                method = key.update_persistent_hash
            except AttributeError:
                pass
            else:
                inner_key_hash = new_hash()
                method(inner_key_hash, self)
                digest = inner_key_hash.digest()

        if digest is None:
            try:
                method = getattr(self, "update_for_"+type(key).__name__)
            except AttributeError:
                pass
            else:
                inner_key_hash = new_hash()
                method(inner_key_hash, key)
                digest = inner_key_hash.digest()

        if digest is None:
            raise TypeError("unsupported type for persistent hash keying: %s"
                    % type(key))

        if not isinstance(key, type):
            try:
                key._pytools_persistent_hash_digest = digest
            except Exception:
                pass

        key_hash.update(digest)

    def __call__(self, key):
        key_hash = new_hash()
        self.rec(key_hash, key)
        return key_hash.hexdigest()

    # {{{ updaters

    def update_for_int(self, key_hash, key):
        key_hash.update(str(key).encode("utf8"))

    update_for_long = update_for_int
    update_for_bool = update_for_int

    def update_for_float(self, key_hash, key):
        key_hash.update(repr(key).encode("utf8"))

    if sys.version_info >= (3,):
        def update_for_str(self, key_hash, key):
            key_hash.update(key.encode('utf8'))

        def update_for_bytes(self, key_hash, key):
            key_hash.update(key)
    else:
        def update_for_str(self, key_hash, key):
            key_hash.update(key)

        def update_for_unicode(self, key_hash, key):
            key_hash.update(key.encode('utf8'))

    def update_for_tuple(self, key_hash, key):
        for obj_i in key:
            self.rec(key_hash, obj_i)

    def update_for_frozenset(self, key_hash, key):
        for set_key in sorted(key):
            self.rec(key_hash, set_key)

    def update_for_NoneType(self, key_hash, key):  # noqa
        key_hash.update("<None>".encode('utf8'))

    def update_for_dtype(self, key_hash, key):
        key_hash.update(key.str.encode('utf8'))

    # }}}

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    pass


class _LinkedList(object):
    """The list operates on nodes of the form [value, leftptr, rightpr]. To create a
    node of this form you can use `LinkedList.new_node().`

    Supports inserting at the left and deleting from an arbitrary location.
    """

    def __init__(self):
        self.count = 0
        self.head = None
        self.end = None

    @staticmethod
    def new_node(element):
        return [element, None, None]

    def __len__(self):
        return self.count

    def appendleft_node(self, node):
        self.count += 1

        if self.head is None:
            self.head = self.end = node
            return

        self.head[1] = node
        node[2] = self.head

        self.head = node

    def pop_node(self):
        end = self.end
        self.remove_node(end)
        return end

    def remove_node(self, node):
        self.count -= 1

        if self.head is self.end:
            self.head = self.end = None
            return

        left = node[1]
        right = node[2]

        if left is None:
            self.head = right
        else:
            left[2] = right

        if right is None:
            self.tail = left
        else:
            right[1] = left

        node[1] = node[2] = None


class _LRUCache(object):
    """A mapping that keeps at most *maxsize* items with an LRU replacement policy.
    """

    def __init__(self, maxsize):
        self.lru_order = _LinkedList()
        self.maxsize = maxsize
        self.cache = {}

    def __delitem__(self, item):
        node = self.cache[item]
        self.lru_order.remove_node(node)
        del self.cache[item]

    def __getitem__(self, item):
        node = self.cache[item]
        self.lru_order.remove_node(node)
        self.lru_order.appendleft_node(node)
        # A linked list node contains a tuple of the form (item, value).
        return node[0][1]

    def __contains__(self, item):
        return item in self.cache

    def discard(self, item):
        if item in self.cache:
            del self[item]

    def clear(self):
        self.cache.clear()
        del self.lru_order
        self.lru_order = _LinkedList()

    def __setitem__(self, item, value):
        if self.maxsize < 1:
            return

        try:
            node = self.cache[item]
            self.lru_order.remove_node(node)
        except KeyError:
            if len(self.lru_order) >= self.maxsize:
                # Make room for new elements.
                end_node = self.lru_order.pop_node()
                del self.cache[end_node[0][0]]

            node = self.lru_order.new_node((item, value))
            self.cache[item] = node

        self.lru_order.appendleft_node(node)
        return node[0]


class PersistentDict(object):
    def __init__(self, identifier, key_builder=None, container_dir=None,
            in_mem_cache_size=None):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg in_mem_cache_size: If not *None*, retain an in-memory cache of
             *in_mem_cache_size* items. The replacement policy is LRU.

        .. automethod:: __getitem__
        .. automethod:: __setitem__
        .. automethod:: __delitem__
        .. automethod:: clear
        """

        self.identifier = identifier

        if key_builder is None:
            key_builder = KeyBuilder()

        self.key_builder = key_builder

        from os.path import join
        if container_dir is None:
            import appdirs
            container_dir = join(
                    appdirs.user_cache_dir("pytools", "pytools"),
                    "pdict-v2-%s-py%s" % (
                        identifier,
                        ".".join(str(i) for i in sys.version_info),))

        self.container_dir = container_dir
        self.version_dir = join(container_dir, "version")

        self._make_container_dirs()

        if in_mem_cache_size is None:
            in_mem_cache_size = 0

        self._use_cache = (in_mem_cache_size >= 1)
        self._read_key_cache = _LRUCache(in_mem_cache_size)
        self._read_contents_cache = _LRUCache(in_mem_cache_size)

    def _make_container_dirs(self):
        _make_dir_recursively(self.container_dir)
        _make_dir_recursively(self.version_dir)

    def store(self, key, value, info_files={}):
        hexdigest_key = self.key_builder(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self.container_dir)

                from os.path import join
                item_dir_m = ItemDirManager(cleanup_m,
                        join(self.container_dir, hexdigest_key))

                if item_dir_m.existed:
                    item_dir_m.reset()

                for info_name, info_value in six.iteritems(info_files):
                    info_path = item_dir_m.sub("info_"+info_name)

                    with open(info_path, "wt") as outf:
                        outf.write(info_value)

                from six.moves.cPickle import dump, HIGHEST_PROTOCOL
                value_path = item_dir_m.sub("contents")
                with open(value_path, "wb") as outf:
                    dump(value, outf, protocol=HIGHEST_PROTOCOL)

                logger.debug("%s: cache store [key=%s]" % (
                    self.identifier, hexdigest_key))

                self._tick_version(hexdigest_key)

                # Write key last, so that if the reader below
                key_path = item_dir_m.sub("key")
                with open(key_path, "wb") as outf:
                    dump(key, outf, protocol=HIGHEST_PROTOCOL)

            except:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def _tick_version(self, hexdigest_key):
        from os.path import join
        version_path = join(self.version_dir, hexdigest_key)

        from six.moves.cPickle import load, dump, HIGHEST_PROTOCOL

        try:
            with open(version_path, "r+b") as versionf:
                version = 1 + load(versionf)
                versionf.seek(0)
                dump(version, versionf, protocol=HIGHEST_PROTOCOL)

        except (IOError, EOFError):
            _make_dir_recursively(self.version_dir)
            with open(version_path, "wb") as versionf:
                dump(0, versionf, protocol=HIGHEST_PROTOCOL)

    def _read_cached(self, file_name, version, cache):
        try:
            value, cached_version = cache[file_name]
            if version == cached_version:
                return value
        except KeyError:
            pass

        with open(file_name, "rb") as inf:
            from six.moves.cPickle import load
            value = load(inf)

        cache[file_name] = (value, version)

        return value

    def fetch(self, key):
        hexdigest_key = self.key_builder(key)

        from os.path import join, isdir
        item_dir = join(self.container_dir, hexdigest_key)
        if not isdir(item_dir):
            logger.debug("%s: cache miss [key=%s]" % (
                self.identifier, hexdigest_key))
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self.container_dir)

                item_dir_m = ItemDirManager(cleanup_m, item_dir)
                key_path = item_dir_m.sub("key")
                value_path = item_dir_m.sub("contents")
                version_path = join(self.version_dir, hexdigest_key)

                # {{{ read version

                version = None
                exc = None

                try:
                    with open(version_path, "rb") as versionf:
                        from six.moves.cPickle import load
                        version = load(versionf)
                except IOError:
                    # Not a fatal error - but we won't be able to use the cache.
                    self._read_key_cache.discard(key_path)
                    self._read_contents_cache.discard(value_path)
                except (OSError, EOFError) as e:
                    exc = e

                if version is None:
                    try:
                        # If the version doesn't exist, reset the version
                        # counter.
                        self._tick_version(hexdigest_key)
                    except (OSError, IOError, EOFError) as e:
                        exc = e

                if exc is not None:
                    item_dir_m.reset()
                    from warnings import warn
                    warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key))
                    raise NoSuchEntryError(key)

                # }}}

                # {{{ load key file

                exc = None

                try:
                    read_key = self._read_cached(key_path, version,
                            self._read_key_cache)
                except (OSError, IOError, EOFError) as e:
                    exc = e

                if exc is not None:
                    item_dir_m.reset()
                    from warnings import warn
                    warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key))
                    raise NoSuchEntryError(key)

                # }}}

                if read_key != key:
                    # Key collision, oh well.
                    from warnings import warn
                    warn("%s: key collision in cache at '%s' -- these are "
                            "sufficiently unlikely that they're often "
                            "indicative of a broken implementation "
                            "of equality comparison"
                            % (self.identifier, self.container_dir))
                    # This is here so we can debug the equality comparison
                    read_key == key
                    raise NoSuchEntryError(key)

                logger.debug("%s: cache hit [key=%s]" % (
                    self.identifier, hexdigest_key))

                # {{{ load value

                exc = None

                try:
                    read_contents = self._read_cached(value_path, version,
                            self._read_contents_cache)
                except (OSError, IOError, EOFError) as e:
                    exc = e

                if exc is not None:
                    item_dir_m.reset()
                    from warnings import warn
                    warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key))
                    raise NoSuchEntryError(key)

                # }}}

                return read_contents

            except:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def remove(self, key):
        hexdigest_key = self.key_builder(key)

        from os.path import join, isdir
        item_dir = join(self.container_dir, hexdigest_key)
        if not isdir(item_dir):
            raise NoSuchEntryError(key)

        key_file = join(item_dir, "key")
        contents_file = join(item_dir, "contents")

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self.container_dir)

                item_dir_m = ItemDirManager(cleanup_m, item_dir)
                item_dir_m.reset()

            except:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

        self._read_key_cache.discard(key_file)
        self._read_contents_cache.discard(contents_file)

    def __getitem__(self, key):
        return self.fetch(key)

    def __setitem__(self, key, value):
        return self.store(key, value)

    def __delitem__(self, key):
        self.remove(key)

    def clear(self):
        try:
            _erase_dir(self.container_dir)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

        self._make_container_dirs()

        self._read_key_cache.clear()
        self._read_contents_cache.clear()

# }}}

# vim: foldmethod=marker
