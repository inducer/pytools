from __future__ import absolute_import
from __future__ import print_function
from pytools import memoize
import six
from six.moves import input


# {{{ debug files -------------------------------------------------------------

def make_unique_filesystem_object(stem, extension="", directory="",
        creator=None):
    """
    :param extension: needs a leading dot.
    :param directory: must not have a trailing slash.
    """
    from os.path import join
    import os

    if creator is None:
        def creator(name):
            return os.fdopen(os.open(name,
                    os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o444), "w")

    i = 0
    while True:
        fname = join(directory, "%s-%d%s" % (stem, i, extension))
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

class RefDebugQuit(Exception):
    pass


def refdebug(obj, top_level=True, exclude=[]):
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
    else:
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

                print("%d/%d: " % (idx, len(reflist)), id(r), type(r), s)

                if isinstance(r, dict):
                    for k, v in six.iteritems(r):
                        if v is obj:
                            print("...referred to from key", k)

                print("[d]ig, [n]ext, [p]rev, [e]val, [r]eturn, [q]uit?")

                response = input()

                if response == "d":
                    refdebug(r, top_level=False, exclude=exclude+[reflist])
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
                        res = eval(expr_str, {"obj": r})
                    except:
                        from traceback import print_exc
                        print_exc()
                    print(res)
                elif response == "r":
                    return
                elif response == "q":
                    raise RefDebugQuit()
                else:
                    print("WHAT YOU SAY!!! (invalid choice)")

        finally:
            print("<--------------")

# }}}


# {{{ interactive shell

def get_shell_hist_filename():
    import os
    _home = os.environ.get('HOME', '/')
    return os.path.join(_home, ".pytools-debug-shell-history")


def setup_readline():
    from os.path import exists
    hist_filename = get_shell_hist_filename()
    if exists(hist_filename):
        try:
            readline.read_history_file(hist_filename)
        except Exception:
            # http://docs.python.org/3/howto/pyporting.html#capturing-the-currently-raised-exception  # noqa
            import sys
            e = sys.exc_info()[1]

            from warnings import warn
            warn("Error opening readline history file: %s" % e)

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


def shell(locals=None, globals=None):
    from inspect import currentframe, getouterframes
    calling_frame = getouterframes(currentframe())[1][0]

    if locals is None:
        locals = calling_frame.f_locals
    if globals is None:
        globals = calling_frame.f_globals

    ns = SetPropagatingDict([locals, globals], locals)

    if HAVE_READLINE:
        readline.set_completer(
                rlcompleter.Completer(ns).complete)

    from code import InteractiveConsole
    cons = InteractiveConsole(ns)
    cons.interact("")

    readline.write_history_file(get_shell_hist_filename())

# }}}

# vim: foldmethod=marker
