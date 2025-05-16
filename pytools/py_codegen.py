from __future__ import annotations


__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

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

import marshal
from functools import cached_property
from importlib.util import MAGIC_NUMBER as BYTECODE_VERSION
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from pytools.codegen import (  # noqa: F401
    CodeGenerator as CodeGeneratorBase,
    Indentation,
    remove_common_indentation,
)


class ExistingLineCacheWarning(Warning):
    """Warning for overwriting existing generated code in the linecache."""


class PythonCodeGenerator(CodeGeneratorBase):
    def _gen_unique_name(self, name: str) -> str:
        import sys
        if "line_profiler" in sys.modules or "line_profiler" in self.get():
            # The '<ipython-input-' prefix is for compatibility with
            # line_profiler: https://github.com/pyutils/line_profiler/blob/1630e7c9a295ace2feb1d2b188e68f4d2833fb20/line_profiler/line_profiler.py#L194-L210
            prefix = "<ipython-input- generated: '"
        else:
            prefix = "<generated: '"

        import linecache

        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator(
            existing_names=linecache.cache.keys(),
            forced_prefix=prefix,
            forced_suffix="'>")

        return name_gen(name)

    def get_module(self, name: str | None = None,
                   _from_get_function: bool = False) -> dict[str, Any]:
        if name is None:
            name = self._gen_unique_name("module")

        result_dict: dict[str, Any] = {}
        source_text = self.get()

        # {{{ Handle Python's linecache

        import linecache

        if name in linecache.cache:
            from warnings import warn
            warn(f"Overwriting existing generated code in linecache: '{name}'.",
                   ExistingLineCacheWarning,
                   stacklevel=3 if _from_get_function else 2)

        linecache.cache[name] = (None, None,  # type: ignore[assignment]  # pyright: ignore[reportArgumentType]
                                 [e+"\n" for e in source_text.split("\n")], None)

        # }}}

        exec(compile(
            source_text.rstrip()+"\n", name, "exec"),
                result_dict)

        return result_dict

    def get_picklable_module(self, name: str | None = None) -> PicklableModule:
        return PicklableModule(self.get_module(name=name))


class PythonFunctionGenerator(PythonCodeGenerator):
    def __init__(self, name: str, args: Iterable[str],
                 decorators: Iterable[str] = ()) -> None:
        PythonCodeGenerator.__init__(self)
        self.name = name

        for decorator in decorators:
            self(decorator)

        self("def {}({}):".format(name, ", ".join(args)))
        self.indent()

    @cached_property
    def _gen_filename(self) -> str:
        return self._gen_unique_name(self.name)

    def get_function(self) -> Callable[..., Any]:
        return self.get_module(name=self._gen_filename,  # pyright: ignore [reportAny]
                               _from_get_function=True)[self.name]

    def get_picklable_function(self) -> PicklableFunction:
        module = self.get_picklable_module(name=self._gen_filename)
        return PicklableFunction(module, self.name)


# {{{ pickling of binaries for generated code

def _get_empty_module_dict():
    result_dict = {}
    exec(compile("", "<generated function>", "exec"), result_dict)
    return result_dict


_empty_module_dict = _get_empty_module_dict()


class PicklableModule:
    def __init__(self, mod_globals):
        self.mod_globals = mod_globals

    def __getstate__(self):
        nondefault_globals = {}
        functions = {}
        modules = {}

        for k, v in self.mod_globals.items():
            if isinstance(v, FunctionType):
                functions[k] = (
                        v.__name__,
                        marshal.dumps(v.__code__),
                        v.__defaults__)
            elif isinstance(v, ModuleType):
                modules[k] = v.__name__
            elif k not in _empty_module_dict:
                nondefault_globals[k] = v

        return (1, BYTECODE_VERSION, functions, modules, nondefault_globals)

    def __setstate__(self, obj):
        if obj[0] == 0:
            magic, functions, nondefault_globals = obj[1:]
            modules = {}
        elif obj[0] == 1:
            magic, functions, modules, nondefault_globals = obj[1:]
        else:
            raise ValueError("unknown version of PicklableModule")

        if magic != BYTECODE_VERSION:
            raise ValueError(
                    "cannot unpickle function binary: incorrect magic value "
                    f"(got: {magic!r}, expected: {BYTECODE_VERSION!r})")

        mod_globals = _empty_module_dict.copy()
        mod_globals.update(nondefault_globals)

        from importlib import import_module
        for k, mod_name in modules.items():
            mod_globals[k] = import_module(mod_name)

        for k, (name, code_bytes, argdefs) in functions.items():
            f = FunctionType(
                    marshal.loads(code_bytes), mod_globals, name=name,
                    argdefs=argdefs)
            mod_globals[k] = f

        self.mod_globals = mod_globals

# }}}


# {{{ picklable function

class PicklableFunction:
    """Convenience class wrapping a function in a :class:`PicklableModule`.
    """

    def __init__(self, module, name):
        self._initialize(module, name)

    def _initialize(self, module, name):
        self.module = module
        self.name = name
        self.func = module.mod_globals[name]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return {"module": self.module, "name": self.name}

    def __setstate__(self, obj):
        self._initialize(obj["module"], obj["name"])

# }}}

# vim: foldmethod=marker
