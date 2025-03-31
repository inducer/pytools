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
from dataclasses import dataclass, field
from importlib.util import MAGIC_NUMBER as BYTECODE_VERSION
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from pytools.codegen import (
    CodeGenerator as CodeGeneratorBase,
    Indentation,
    remove_common_indentation,
)


__all__ = (
    "ExistingLineCacheWarning",
    "Indentation",
    "PicklableFunction",
    "PicklableModule",
    "PythonCodeGenerator",
    "PythonFunctionGenerator",
    "remove_common_indentation",
)


class ExistingLineCacheWarning(Warning):
    """Warning for overwriting existing generated code in the linecache."""


def _linecache_unique_name(name_prefix: str, source_text: str | None) -> str:
    import linecache

    if source_text is not None:
        from siphash24 import siphash13  # pyright: ignore[reportUnknownVariableType]
        src_digest = cast("str", siphash13(source_text.encode()).hexdigest())  # pyright: ignore[reportUnknownMemberType]

        name_prefix = f"{name_prefix}-{src_digest}"

    from pytools import UniqueNameGenerator
    name_gen = UniqueNameGenerator(
        existing_names=linecache.cache.keys(),
        forced_prefix="<generated: '",
        forced_suffix="'>")

    return name_gen(name_prefix)


def _make_module(
            source_text: str,
            name: str | None = None,
            *,
            name_prefix: str | None = None,
        ) -> dict[str, Any]:
    if name_prefix is None:
        name_prefix = "module"

    if name is None:
        name = _linecache_unique_name(name_prefix, source_text)

    result_dict: dict[str, Any] = {}

    # {{{ insert into linecache

    import linecache

    if name in linecache.cache:
        from warnings import warn
        warn(f"Overwriting existing generated code in linecache: '{name}'.",
               ExistingLineCacheWarning,
               stacklevel=2)

    linecache.cache[name] = (len(source_text), None,
                             [e+"\n" for e in source_text.split("\n")], name)

    # }}}

    code_obj = compile(
                source_text.rstrip()+"\n", name, "exec")
    result_dict["__code__"] = code_obj
    exec(code_obj, result_dict)

    return result_dict


class PythonCodeGenerator(CodeGeneratorBase):
    def get_module(self, name: str | None = None,
                   *,
                   name_prefix: str | None = None,
                   ) -> dict[str, Any]:
        return _make_module(self.get(), name=name, name_prefix=name_prefix)

    def get_picklable_module(self,
                name: str | None = None,
                name_prefix: str | None = None
            ) -> PicklableModule:
        return PicklableModule(self.get_module(name=name, name_prefix=name_prefix))


class PythonFunctionGenerator(PythonCodeGenerator):
    name: str

    def __init__(self, name: str, args: Iterable[str],
                 decorators: Iterable[str] = ()) -> None:
        super().__init__()
        self.name = name

        for decorator in decorators:
            self(decorator)

        self("def {}({}):".format(name, ", ".join(args)))
        self.indent()

    def get_function(self) -> Callable[..., Any]:
        return self.get_module(name_prefix=self.name)[self.name]  # pyright: ignore[reportAny]

    def get_picklable_function(self) -> PicklableFunction:
        return PicklableFunction(
                self.get_picklable_module(name_prefix=self.name), self.name)


# {{{ pickling of binaries for generated code

def _get_empty_module_dict(filename: str | None = None) -> dict[str, Any]:
    if filename is None:
        filename = "<generated code>"

    result_dict: dict[str, Any] = {}
    code_obj = compile("", filename, "exec")
    result_dict["__code__"] = code_obj
    exec(code_obj, result_dict)
    return result_dict


_empty_module_dict = _get_empty_module_dict()


_FunctionsType: TypeAlias = dict[str, tuple[str, bytes, tuple[object, ...] | None]]
_ModulesType: TypeAlias = dict[str, str]


@dataclass
class PicklableModule:
    mod_globals: dict[str, Any]
    name_prefix: str | None = field(kw_only=True, default=None)
    source_code: str | None = field(kw_only=True, default=None)

    def __getstate__(self):
        functions: _FunctionsType = {}
        modules: _ModulesType = {}
        nondefault_globals: dict[str, object] = {}

        for k, v in self.mod_globals.items():  # pyright: ignore[reportAny]
            if isinstance(v, FunctionType):
                functions[k] = (
                        v.__name__,
                        marshal.dumps(v.__code__),
                        v.__defaults__)
            elif isinstance(v, ModuleType):
                modules[k] = v.__name__
            elif k not in _empty_module_dict:
                nondefault_globals[k] = v

        return (2, BYTECODE_VERSION, functions, modules, nondefault_globals,
                self.name_prefix, self.source_code)

    def __setstate__(self, obj: (
                 tuple[Literal[0], bytes, _FunctionsType, dict[str, object]]
                 | tuple[Literal[1], bytes, _FunctionsType, _ModulesType,
                         dict[str, object]]
                 | tuple[Literal[2], bytes, _FunctionsType, _ModulesType,
                         dict[str, object], str | None, str | None]
                 )
             ):
        if obj[0] == 0:
            magic, functions, nondefault_globals = obj[1:]
            modules = {}
            name_prefix: str | None = None
            source_code: str | None = None
        elif obj[0] == 1:
            magic, functions, modules, nondefault_globals = obj[1:]
            name_prefix = None
            source_code = None
        elif obj[0] == 2:
            magic, functions, modules, nondefault_globals, name_prefix, source_code = \
                obj[1:]
        else:
            raise ValueError("unknown version of PicklableModule")

        if magic != BYTECODE_VERSION:
            raise ValueError(
                    "cannot unpickle function binary: incorrect magic value "
                    f"(got: {magic!r}, expected: {BYTECODE_VERSION!r})")

        unique_filename = _linecache_unique_name(
                        name_prefix if name_prefix else "module", source_code)
        mod_globals = _get_empty_module_dict(unique_filename)
        mod_globals.update(nondefault_globals)

        import linecache
        if source_code:
            linecache.cache[unique_filename] = (len(source_code), None,
                                     [e+"\n" for e in source_code.split("\n")],
                                     unique_filename)

        from importlib import import_module
        for k, mod_name in modules.items():
            mod_globals[k] = import_module(mod_name)

        for k, (name, code_bytes, argdefs) in functions.items():
            f = FunctionType(
                    marshal.loads(code_bytes), mod_globals, name=name,
                    argdefs=argdefs)
            mod_globals[k] = f

        self.mod_globals = mod_globals
        self.name_prefix = name_prefix
        self.source_code = source_code

# }}}


# {{{ picklable function

class PicklableFunction:
    """Convenience class wrapping a function in a :class:`PicklableModule`.
    """

    module: PicklableModule
    name: str

    def __init__(self, module: PicklableModule, name: str) -> None:
        self._initialize(module, name)

    def _initialize(self, module: PicklableModule, name: str) -> None:
        self.module = module
        self.name = name
        self._callable = cast("FunctionType", module.mod_globals[name])  # pyright: ignore[reportUnannotatedClassAttribute]

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._callable(*args, **kwargs)  # pyright: ignore[reportAny]

    def __getstate__(self):
        return {"module": self.module, "name": self.name}

    def __setstate__(self, obj):
        self._initialize(obj["module"], obj["name"])

# }}}

# vim: foldmethod=marker
