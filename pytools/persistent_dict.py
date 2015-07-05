"""Generic persistent, concurrent dictionary-like facility."""

from __future__ import division, with_statement
from __future__ import absolute_import
import six

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


import sys
import os
import errno

try:
    import hashlib
    new_hash = hashlib.sha256
except ImportError:
    # for Python << 2.5
    import sha
    new_hash = sha.new


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
        try:
            method = key.update_persistent_hash
        except AttributeError:
            pass
        else:
            method(key_hash, self)
            return

        try:
            method = getattr(self, "update_for_"+type(key).__name__)
        except AttributeError:
            pass
        else:
            method(key_hash, key)
            return

        raise TypeError("unsupported type for persistent hash keying: %s"
                % type(key))

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
        return key.str.encode("utf8")

    # }}}

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    pass


class PersistentDict(object):
    def __init__(self, identifier, key_builder=None, container_dir=None):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
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

        self._make_container_dir()

    def _make_container_dir(self):
        # {{{ ensure container directory exists

        try:
            os.makedirs(self.container_dir)
        except OSError as e:
            from errno import EEXIST
            if e.errno != EEXIST:
                raise

        # }}}

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

                # Write key last, so that if the reader below
                key_path = item_dir_m.sub("key")
                with open(key_path, "wb") as outf:
                    dump(key, outf, protocol=HIGHEST_PROTOCOL)

            except:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def fetch(self, key):
        hexdigest_key = self.key_builder(key)

        from os.path import join, isdir
        item_dir = join(self.container_dir, hexdigest_key)
        if not isdir(item_dir):
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self.container_dir)

                item_dir_m = ItemDirManager(cleanup_m, item_dir)
                key_path = item_dir_m.sub("key")
                value_path = item_dir_m.sub("contents")

                from six.moves.cPickle import load

                # {{{ load key file

                exc = None

                try:
                    with open(key_path, "rb") as inf:
                        read_key = load(inf)
                except IOError as e:
                    exc = e
                except EOFError as e:
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
                    logger.debug("key collsion in cache at '%s'"
                            % self.container_dir)
                    raise NoSuchEntryError(key)

                # {{{ load value

                exc = None

                try:
                    with open(value_path, "rb") as inf:
                        read_contents = load(inf)
                except IOError as e:
                    exc = e
                except EOFError as e:
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

        self._make_container_dir()

# }}}

# vim: foldmethod=marker
