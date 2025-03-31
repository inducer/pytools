from __future__ import annotations

import sys

import pytest

import pytools
import pytools.py_codegen as codegen


def test_pickling_with_module_import():
    cg = codegen.PythonCodeGenerator()
    cg("import pytools")
    cg("import math as m")

    import pickle
    mod = pickle.loads(pickle.dumps(cg.get_picklable_module()))

    assert mod.mod_globals["pytools"] is pytools

    import math
    assert mod.mod_globals["m"] is math


def test_picklable_function():
    cg = codegen.PythonFunctionGenerator("f", args=())
    cg("return 1")

    import pickle
    f = pickle.loads(pickle.dumps(cg.get_picklable_function()))

    assert f() == 1


def test_function_decorators(capfd):
    cg = codegen.PythonFunctionGenerator("f", args=(), decorators=["@staticmethod"])
    cg("return 42")

    assert cg.get_function()() == 42

    cg = codegen.PythonFunctionGenerator("f", args=(), decorators=["@classmethod"])
    cg("return 42")

    with pytest.raises(TypeError):
        cg.get_function()()

    cg = codegen.PythonFunctionGenerator("f", args=(),
                                         decorators=["@staticmethod", "@classmethod"])
    cg("return 42")

    with pytest.raises(TypeError):
        cg.get_function()()

    cg = codegen.PythonFunctionGenerator("f", args=("x"),
                decorators=["from functools import lru_cache", "@lru_cache"])
    cg("print('Hello World!')")
    cg("return 42")

    f = cg.get_function()

    assert f(0) == 42
    out, _err = capfd.readouterr()
    assert out == "Hello World!\n"

    assert f(0) == 42
    out, _err = capfd.readouterr()
    assert out == ""  # second print is not executed due to lru_cache


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
