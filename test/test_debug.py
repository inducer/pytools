__copyright__ = "Copyright (C) 2023 University of Illinois Board of Trustees"

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


def test_get_object_cycles():
    from pytools.debug import get_object_cycles
    assert len(get_object_cycles([])) == 0

    a = {}
    assert len(get_object_cycles([a])) == 0

    b = {"a": a}
    assert len(get_object_cycles([b])) == 0
    assert len(get_object_cycles([a, b])) == 0

    a["b"] = b

    assert len(get_object_cycles([a, b])) == 2
    assert len(get_object_cycles([a, b])) == 2
    assert len(get_object_cycles([a])) == 1

    a = {}

    assert len(get_object_cycles([a])) == 0

    b = [42, 4]
    a = [1, 2, 3, 4, 5, b]
    b.append(a)

    assert len(get_object_cycles([a, b])) == 2


def test_get_object_graph():
    from pytools.debug import get_object_graph

    assert get_object_graph([]) == {}

    a = (1,)
    b = (2,)
    c = (a, b)
    assert get_object_graph([a]) == {(1,): set()}
    assert get_object_graph([a, b]) == {(1,): set(), (2,): set()}
    assert get_object_graph([a, b, c]) == {((1,), (2,)): {(2,), (1,)},  # c: [a, b]
                                           (1,): set(),  # a: set()
                                           (2,): set()}  # b: set()

    a = {}
    b = {"a": a}
    a["b"] = b

    assert get_object_graph([a, b]) == {
        (id(a), "{'b': {'a': {...}}}"): {(id(b), "{'a': {'b': {...}}}")},
        (id(b), "{'a': {'b': {...}}}"): {(id(a), "{'b': {'a': {...}}}")}}

    b = [42, 4]
    a = [1, 2, 3, 4, 5, b]
    b.append(a)

    assert get_object_graph([a, b]) == {
        (id(a), "[1, 2, 3, 4, 5, [42, 4, [...]]]"):
            {(id(b), "[42, 4, [1, 2, 3, 4, 5, [...]]]")},
        (id(b), "[42, 4, [1, 2, 3, 4, 5, [...]]]"):
            {(id(a), "[1, 2, 3, 4, 5, [42, 4, [...]]]")}}
