from __future__ import annotations


__copyright__ = "Copyright (C) 2024 University of Illinois Board of Trustees"

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


import sys

import pytest

from pytools import opt_frozen_dataclass


def test_opt_frozen_dataclass() -> None:
    # {{{ basic usage

    @opt_frozen_dataclass()
    class A:
        x: int

    a = A(1)
    assert a.x == 1

    # Needs to be hashable by default, not using object.__hash__
    hash(a)
    assert hash(a) == hash(A(1))
    assert a == A(1)

    # Needs to be frozen by default
    if __debug__:
        with pytest.raises(AttributeError):
            a.x = 2  # type: ignore[misc]
    else:
        a.x = 2  # type: ignore[misc]

    assert a.__dataclass_params__.frozen is __debug__  # type: ignore[attr-defined]  # pylint: disable=no-member

    # }}}

    with pytest.raises(TypeError):
        # Can't specify frozen parameter
        @opt_frozen_dataclass(frozen=False)  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg
        class B:
            x: int

    # {{{ eq=False

    @opt_frozen_dataclass(eq=False)
    class C:
        x: int

    c = C(1)

    # Hashing still works, but uses object.__hash__ (i.e., id())
    assert hash(c) != hash(C(1))

    # Equality is not defined and uses id()
    assert c != C(1)

    # }}}


def test_dataclass_weakref() -> None:
    if sys.version_info < (3, 11):
        pytest.skip("weakref support needs Python 3.11+")

    @opt_frozen_dataclass(weakref_slot=True, slots=True)
    class Weakref:
        x: int

    a = Weakref(1)
    assert a.x == 1

    import weakref
    ref = weakref.ref(a)

    _ = ref().x

    with pytest.raises(TypeError):
        @opt_frozen_dataclass(weakref_slot=True)  # needs slots=True to work
        class Weakref2:
            x: int


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
