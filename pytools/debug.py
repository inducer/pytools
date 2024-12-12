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


__doc__ = """
Debugging helpers
=================

.. autofunction:: make_unique_filesystem_object
.. autofunction:: open_unique_debug_file
.. autofunction:: refdebug
.. autofunction:: get_object_cycles
.. autofunction:: estimate_memory_usage

"""

import sys
from typing import Collection, List, Set

from pytools import memoize


# {{{ debug files -------------------------------------------------------------

def make_unique_filesystem_object(stem, extension="", directory="",
        creator=None):
    """
    :param extension: needs a leading dot.
    :param directory: must not have a trailing slash.
    """
    import os
    from os.path import join

    if creator is None:
        def default_creator(name):
            return os.fdopen(os.open(name,
                    os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o444), "w")
        creator = default_creator

    i = 0
    while True:
        fname = join(directory, f"{stem}-{i}{extension}")
        try:
            return creator(fname), fname
        except OSError:
            i += 1


@memoize
def get_run_debug_directory():
    def creator(name):
        from os import mkdir
        mkdir(name)
        return name

    return make_unique_filesystem_object("run-debug", creator=creator)[0]


def open_unique_debug_file(stem, extension=""):
    """
    :param extension: needs a leading dot.
    """
    return make_unique_filesystem_object(
            stem, extension, get_run_debug_directory())

# }}}


# {{{ refcount debugging ------------------------------------------------------

class RefDebugQuit(Exception):  # noqa: N818
    pass


def refdebug(obj, top_level=True, exclude=()):
    from types import FrameType

    def is_excluded(o):
        for ex in exclude:
            if o is ex:
                return True

        from sys import _getframe
        if isinstance(o, FrameType) and \
                o.f_code.co_filename == _getframe().f_code.co_filename:
            return True

        return False

    if top_level:
        try:
            refdebug(obj, top_level=False, exclude=exclude)
        except RefDebugQuit:
            pass
        return

    import gc
    print_head = True
    print("-------------->")
    try:
        reflist = [x for x in gc.get_referrers(obj)
                if not is_excluded(x)]

        idx = 0
        while True:
            if print_head:
                print("referring to", id(obj), type(obj), obj)
                print("----------------------")
                print_head = False
            r = reflist[idx]

            if isinstance(r, FrameType):
                s = str(r.f_code)
            else:
                s = str(r)

            print(f"{idx}/{len(reflist)}: ", id(r), type(r), s)

            if isinstance(r, dict):
                for k, v in r.items():
                    if v is obj:
                        print("...referred to from key", k)

            print("[d]ig, [n]ext, [p]rev, [e]val, [r]eturn, [q]uit?")

            response = input()

            if response == "d":
                refdebug(r, top_level=False, exclude=exclude+tuple(reflist))
                print_head = True
            elif response == "n":
                if idx + 1 < len(reflist):
                    idx += 1
            elif response == "p":
                if idx - 1 >= 0:
                    idx -= 1
            elif response == "e":
                print("type expression, obj is your object:")
                expr_str = input()
                try:
                    res = eval(expr_str, {"obj": r})  # pylint:disable=eval-used
                except Exception:  # pylint:disable=broad-except
                    from traceback import print_exc
                    print_exc()
                print(res)
            elif response == "r":
                return
            elif response == "q":
                raise RefDebugQuit
            else:
                print("WHAT YOU SAY!!! (invalid choice)")

    finally:
        print("<--------------")

# }}}


# {{{ Find circular references

# Based on https://code.activestate.com/recipes/523004-find-cyclical-references/

def get_object_cycles(objects: Collection[object]) -> List[List[object]]:
    """
    Find circular references in *objects*. This can be useful for example to debug
    why certain objects need to be freed via garbage collection instead of
    reference counting.

    :arg objects: A collection of objects to find cycles in. A potential way
        to find a list of objects potentially containing cycles from the garbage
        collector is the following code::

            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            gc.set_debug(0)
            obj_list = gc.garbage

            from pytools.debug import get_object_cycles
            print(get_object_cycles(obj_list))

    :returns: A :class:`list` in which each element contains a :class:`list`
        of objects forming a cycle.
    """
    def recurse(obj: object, start: object, all_objs: Set[object],
                current_path: List[object]) -> None:
        all_objs.add(id(obj))

        import gc
        from types import FrameType

        referents = gc.get_referents(obj)

        for referent in referents:
            # If we've found our way back to the start, this is
            # a cycle, so return it
            if referent is start:
                res.append(current_path)
                return

            # Don't go back through the original list of objects, or
            # through temporary references to the object, since those
            # are just an artifact of the cycle detector itself.
            elif referent is objects or isinstance(referent, FrameType):
                continue

            # We haven't seen this object before, so recurse
            elif id(referent) not in all_objs:
                recurse(referent, start, all_objs, current_path + [obj])

    res: List[List[object]] = []
    for obj in objects:
        recurse(obj, obj, set(), [])

    return res

# }}}


# {{{ interactive shell

def get_shell_hist_filename():
    import os
    home = os.environ.get("HOME", "/")
    return os.path.join(home, ".pytools-debug-shell-history")


def setup_readline():
    from os.path import exists
    hist_filename = get_shell_hist_filename()
    if exists(hist_filename):
        try:
            readline.read_history_file(hist_filename)
        except Exception:  # pylint:disable=broad-except
            # http://docs.python.org/3/howto/pyporting.html#capturing-the-currently-raised-exception
            import sys
            e = sys.exc_info()[1]

            from warnings import warn
            warn(f"Error opening readline history file: {e}", stacklevel=2)

    readline.parse_and_bind("tab: complete")


try:
    import readline
    import rlcompleter
    HAVE_READLINE = True
except ImportError:
    HAVE_READLINE = False
else:
    setup_readline()


class SetPropagatingDict(dict):
    def __init__(self, source_dicts, target_dict):
        dict.__init__(self)
        for s in source_dicts[::-1]:
            self.update(s)

        self.target_dict = target_dict

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.target_dict[key] = value

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        del self.target_dict[key]


def shell(locals_=None, globals_=None):
    from inspect import currentframe, getouterframes
    calling_frame = getouterframes(currentframe())[1][0]

    if locals_ is None:
        locals_ = calling_frame.f_locals
    if globals_ is None:
        globals_ = calling_frame.f_globals

    ns = SetPropagatingDict([locals_, globals_], locals_)

    if HAVE_READLINE:
        readline.set_completer(
                rlcompleter.Completer(ns).complete)

    from code import InteractiveConsole
    cons = InteractiveConsole(ns)
    cons.interact("")

    readline.write_history_file(get_shell_hist_filename())

# }}}


# {{{ estimate memory usage

def estimate_memory_usage(root, seen_ids=None):
    if seen_ids is None:
        seen_ids = set()

    id_root = id(root)
    if id_root in seen_ids:
        return 0

    seen_ids.add(id_root)

    result = sys.getsizeof(root)

    from gc import get_referents
    for ref in get_referents(root):
        result += estimate_memory_usage(ref, seen_ids=seen_ids)

    return result

# }}}

# vim: foldmethod=marker
