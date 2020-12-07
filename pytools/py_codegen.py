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
from types import FunctionType, ModuleType

from pytools.codegen import CodeGenerator as CodeGeneratorBase
from pytools.codegen import Indentation, remove_common_indentation  # noqa


from importlib.util import MAGIC_NUMBER as BYTECODE_VERSION


class PythonCodeGenerator(CodeGeneratorBase):
    def get_module(self, name="<generated code>"):
        result_dict = {}
        source_text = self.get()
        exec(compile(
            source_text.rstrip()+"\n", name, "exec"),
                result_dict)
        result_dict["_MODULE_SOURCE_CODE"] = source_text
        return result_dict

    def get_picklable_module(self):
        return PicklableModule(self.get_module())


class PythonFunctionGenerator(PythonCodeGenerator):
    def __init__(self, name, args):
        PythonCodeGenerator.__init__(self)
        self.name = name

        self("def {}({}):".format(name, ", ".join(args)))
        self.indent()

    def get_function(self):
        return self.get_module()[self.name]

    def get_picklable_function(self):
        module = self.get_picklable_module()
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
            raise ValueError("cannot unpickle function binary: "
                    "incorrect magic value (got: %r, expected: %r)"
                    % (magic, BYTECODE_VERSION))

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
    """Convience class wrapping a function in a :class:`PicklableModule`.
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
