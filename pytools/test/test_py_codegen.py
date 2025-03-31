from __future__ import annotations

import pickle
import sys
from typing import cast

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


def test_linecache_func() -> None:
    cg = codegen.PythonFunctionGenerator("f", args=())
    cg("return 42")

    func = cg.get_function()
    func()

    mod_name = func.__code__.co_filename

    import linecache

    assert linecache.getlines(mod_name) == [
        "def f():\n",
        "    return 42\n",
    ]

    assert linecache.getline(mod_name, 1) == "def f():\n"
    assert linecache.getline(mod_name, 2) == "    return 42\n"

    pkl = pickle.dumps(cg.get_picklable_function())

    pf = cast("codegen.PicklableFunction", pickle.loads(pkl))

    post_pickle_mod_name = pf._callable.__code__.co_filename

    assert post_pickle_mod_name != mod_name
    assert linecache.getlines(post_pickle_mod_name) == [
        "def f():\n",
        "    return 42\n",
    ]


def test_linecache_mod() -> None:
    cg2 = codegen.PythonCodeGenerator()
    cg2("def f():")
    cg2("    return 37")

    mod = cg2.get_module()
    mod["f"]()
    mod_name = cast("str", mod["__code__"].co_filename)

    assert mod_name

    import linecache
    assert linecache.getlines(mod_name) == [
        "def f():\n",
        "    return 37\n",
    ]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
