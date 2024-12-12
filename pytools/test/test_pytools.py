from __future__ import annotations


__copyright__ = "Copyright (C) 2009-2021 Andreas Kloeckner"

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
import sys
from dataclasses import dataclass

import pytest

from pytools import Record
from pytools.tag import tag_dataclass


logger = logging.getLogger(__name__)


def test_memoize_method_clear():
    from pytools import memoize_method

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        @memoize_method
        def f(self):
            self.run_count += 1
            return 17

    sc = SomeClass()
    sc.f()
    sc.f()
    assert sc.run_count == 1

    sc.f.clear_cache(sc)


def test_keyed_memoize_method_with_uncached():
    from pytools import keyed_memoize_method

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        @keyed_memoize_method(key=lambda x, y, z: x)
        def f(self, x, y, z):
            del x, y, z
            self.run_count += 1
            return 17

    sc = SomeClass()
    sc.f(17, 18, z=19)
    sc.f(17, 19, z=20)
    assert sc.run_count == 1
    sc.f(18, 19, z=20)
    assert sc.run_count == 2

    sc.f.clear_cache(sc)


def test_memoize_in():
    from pytools import memoize_in

    class SomeClass:
        def __init__(self):
            self.run_count = 0

        def f(self):

            @memoize_in(self, (SomeClass.f,))
            def inner(x):
                self.run_count += 1
                return 2*x

            inner(5)
            inner(5)

    sc = SomeClass()
    sc.f()
    assert sc.run_count == 1


def test_p_convergence_verifier():
    pytest.importorskip("numpy")

    from pytools.convergence import PConvergenceVerifier

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 0.1**order)
    pconv_verifier()

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 0.5**order)
    pconv_verifier()

    pconv_verifier = PConvergenceVerifier()
    for order in [2, 3, 4, 5]:
        pconv_verifier.add_data_point(order, 2)
    with pytest.raises(AssertionError):
        pconv_verifier()


def test_memoize():
    from pytools import memoize
    count = [0]

    @memoize
    def f(i, j):
        count[0] += 1
        return i + j

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert count[0] == 1


def test_memoize_with_kwargs():
    from pytools import memoize
    count = [0]

    @memoize(use_kwargs=True)
    def f(i, j=1):
        count[0] += 1
        return i + j

    assert f(1) == 2
    assert f(1, 2) == 3
    assert f(2, j=3) == 5
    assert count[0] == 3
    assert f(1) == 2
    assert f(1, 2) == 3
    assert f(2, j=3) == 5
    assert count[0] == 3


def test_memoize_keyfunc():
    from pytools import memoize
    count = [0]

    @memoize(key=lambda i, j=(1,): (i, len(j)))
    def f(i, j=(1,)):
        count[0] += 1
        return i + len(j)

    assert f(1) == 2
    assert f(1, [2]) == 2
    assert f(2, j=[2, 3]) == 4
    assert count[0] == 2
    assert f(1) == 2
    assert f(1, (2,)) == 2
    assert f(2, j=(2, 3)) == 4
    assert count[0] == 2


def test_memoize_frozen() -> None:

    from pytools import memoize_method

    # {{{ check frozen dataclass

    @dataclass(frozen=True)
    class FrozenDataclass:
        value: int

        @memoize_method
        def double_value(self):
            return 2 * self.value

    c0 = FrozenDataclass(10)
    assert c0.double_value() == 20

    c0.double_value.clear_cache(c0)  # type: ignore[attr-defined]

    # }}}

    # {{{ check class with no setattr

    class FrozenClass:
        value: int

        def __init__(self, value):
            object.__setattr__(self, "value", value)

        def __setattr__(self, key, value):
            raise AttributeError(f"cannot set attribute {key}")

        @memoize_method
        def double_value(self):
            return 2 * self.value

    c1 = FrozenClass(10)
    assert c1.double_value() == 20

    c1.double_value.clear_cache(c1)  # type: ignore[attr-defined]

    # }}}


@pytest.mark.parametrize("dims", [2, 3])
def test_spatial_btree(dims, do_plot=False):
    pytest.importorskip("numpy")
    import numpy as np

    rng = np.random.default_rng()
    nparticles = 2000
    x = -1 + 2*rng.uniform(size=(dims, nparticles))
    x = np.sign(x)*np.abs(x)**1.9
    x = (1.4 + x) % 2 - 1

    bl = np.min(x, axis=-1)
    tr = np.max(x, axis=-1)
    print(bl, tr)

    from pytools.spatial_btree import SpatialBinaryTreeBucket
    tree = SpatialBinaryTreeBucket(bl, tr, max_elements_per_box=10)
    for i in range(nparticles):
        tree.insert(i, (x[:, i], x[:, i]))

    if do_plot:
        import matplotlib.pyplot as pt
        pt.gca().set_aspect("equal")
        pt.plot(x[0], x[1], "x")
        tree.plot(fill=None)
        pt.show()


def test_generate_numbered_unique_names():
    from pytools import generate_numbered_unique_names

    gen = generate_numbered_unique_names("a")
    assert next(gen) == (0, "a")
    assert next(gen) == (1, "a_0")

    gen = generate_numbered_unique_names("b", 6)
    assert next(gen) == (7, "b_6")


def test_cartesian_product():
    from pytools import cartesian_product

    expected_outputs = [
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 3, 5),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
    ]

    for i, output in enumerate(cartesian_product([0, 1], [2, 3], [4, 5])):
        assert output == expected_outputs[i]


def test_find_module_git_revision():
    import pytools
    print(pytools.find_module_git_revision(pytools.__file__, n_levels_up=1))


def test_reshaped_view():
    import pytools
    np = pytest.importorskip("numpy")

    a = np.zeros((10, 2))
    b = a.T
    c = pytools.reshaped_view(a, -1)
    assert c.shape == (20,)
    with pytest.raises(AttributeError):
        pytools.reshaped_view(b, -1)


def test_processlogger():
    logging.basicConfig(level=logging.INFO)

    from pytools import ProcessLogger
    plog = ProcessLogger(logger, "testing the process logger",
            long_threshold_seconds=0.01)

    from time import sleep
    with plog:
        sleep(0.3)


def test_table():
    import math

    from pytools import Table

    tbl = Table()
    tbl.add_row(("i", "i^2", "i^3", "sqrt(i)"))

    for i in range(8):
        tbl.add_row((i, i ** 2, i ** 3, math.sqrt(i)))

    print(tbl)
    print()
    print(tbl.latex())

    # {{{ test merging

    from pytools import merge_tables
    tbl = merge_tables(tbl, tbl, tbl, skip_columns=(0,))
    print(tbl.github_markdown())

    # }}}


def test_eoc():
    np = pytest.importorskip("numpy")

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    # {{{ test pretty_print

    for i in range(1, 8):
        eoc.add_data_point(1.0 / i, 10 ** (-i))

    p = eoc.pretty_print()
    print(p)
    print()

    p = eoc.pretty_print(
            abscissa_format="%.5e",
            error_format="%.5e",
            eoc_format="%5.2f")
    print(p)

    # }}}

    # {{{ test merging

    from pytools.convergence import stringify_eocs
    p = stringify_eocs(eoc, eoc, eoc, names=("First", "Second", "Third"))
    print(p)

    # }}}

    # {{{ test invalid inputs

    eoc = EOCRecorder()

    # scalar inputs are fine
    eoc.add_data_point(1, 1)
    eoc.add_data_point(1.0, 1.0)
    eoc.add_data_point(np.float32(1.0), 1.0)
    eoc.add_data_point(np.array(3), 1.0)
    eoc.add_data_point(1.0, np.array(3))

    # non-scalar inputs are not fine though
    with pytest.raises(TypeError):
        eoc.add_data_point(np.array([3]), 1.0)

    with pytest.raises(TypeError):
        eoc.add_data_point(1.0, np.array([3]))

    # }}}


def test_natsorted():
    from pytools import natorder, natsorted

    assert natorder("1.001") < natorder("1.01")

    assert natsorted(["x10", "x1", "x9"]) == ["x1", "x9", "x10"]
    assert natsorted(map(str, range(100))) == list(map(str, range(100)))
    assert natsorted(["x10", "x1", "x9"], reverse=True) == ["x10", "x9", "x1"]
    assert natsorted([10, 1, 9], key=lambda d: f"x{d}") == [1, 9, 10]


# {{{ object array iteration behavior

class FakeArray:
    nopes = 0

    def __len__(self):
        FakeArray.nopes += 1
        return 10

    def __getitem__(self, idx):
        FakeArray.nopes += 1
        if idx > 10:
            raise IndexError


def test_make_obj_array_iteration():
    pytest.importorskip("numpy")

    from pytools.obj_array import make_obj_array
    make_obj_array([FakeArray()])

    assert FakeArray.nopes == 0, FakeArray.nopes

# }}}


# {{{ test obj array vectorization and decorators

def test_obj_array_vectorize(c=1):
    np = pytest.importorskip("numpy")
    la = pytest.importorskip("numpy.linalg")

    # {{{ functions

    import pytools.obj_array as obj

    def add_one(ary):
        assert ary.dtype.char != "O"
        return ary + c

    def two_add_one(x, y):
        assert x.dtype.char != "O" and y.dtype.char != "O"
        return x * y + c

    @obj.obj_array_vectorized
    def vectorized_add_one(ary):
        assert ary.dtype.char != "O"
        return ary + c

    @obj.obj_array_vectorized_n_args
    def vectorized_two_add_one(x, y):
        assert x.dtype.char != "O" and y.dtype.char != "O"
        return x * y + c

    class Adder:
        def __init__(self, c):
            self.c = c

        def add(self, ary):
            assert ary.dtype.char != "O"
            return ary + self.c

        @obj.obj_array_vectorized_n_args
        def vectorized_add(self, ary):
            assert ary.dtype.char != "O"
            return ary + self.c

    adder = Adder(c)

    # }}}

    # {{{ check

    scalar_ary = np.ones(42, dtype=np.float64)
    object_ary = obj.make_obj_array([scalar_ary, scalar_ary, scalar_ary])

    for func, vectorizer, nargs in [
            (add_one, obj.obj_array_vectorize, 1),
            (two_add_one, obj.obj_array_vectorize_n_args, 2),
            (adder.add, obj.obj_array_vectorize, 1),
            ]:
        input_ary = [scalar_ary] * nargs
        result = vectorizer(func, *input_ary)
        error = la.norm(result - c - 1)
        print(error)

        input_ary = [object_ary] * nargs
        result = vectorizer(func, *input_ary)
        error = 0

    for func, nargs in [
            (vectorized_add_one, 1),
            (vectorized_two_add_one, 2),
            (adder.vectorized_add, 1),
            ]:
        input_ary = [scalar_ary] * nargs
        result = func(*input_ary)

        input_ary = [object_ary] * nargs
        result = func(*input_ary)

    # }}}

# }}}


def test_tag() -> None:
    from pytools.tag import (
        NonUniqueTagError,
        Tag,
        Taggable,
        UniqueTag,
        check_tag_uniqueness,
    )

    # Need a subclass that defines the copy function in order to test.
    @tag_dataclass
    class TaggableWithCopy(Taggable):
        tags: frozenset[Tag]

        def _with_new_tags(self, tags):
            return TaggableWithCopy(tags)

    class FairRibbon(Tag):
        pass

    class BlueRibbon(FairRibbon):
        pass

    class RedRibbon(FairRibbon):
        pass

    class ShowRibbon(FairRibbon, UniqueTag):
        pass

    class BestInShowRibbon(ShowRibbon):
        pass

    class ReserveBestInShowRibbon(ShowRibbon):
        pass

    class BestInClassRibbon(FairRibbon, UniqueTag):
        pass

    best_in_show_ribbon = BestInShowRibbon()
    reserve_best_in_show_ribbon = ReserveBestInShowRibbon()
    blue_ribbon = BlueRibbon()
    red_ribbon = RedRibbon()
    best_in_class_ribbon = BestInClassRibbon()

    # Test that input processing fails if there are multiple instances
    # of the same UniqueTag subclass
    with pytest.raises(NonUniqueTagError):
        check_tag_uniqueness(frozenset((
            best_in_show_ribbon,
            reserve_best_in_show_ribbon, blue_ribbon, red_ribbon)))

    # Test that input processing fails if any of the tags are not
    # a subclass of Tag
    with pytest.raises(TypeError):
        check_tag_uniqueness(frozenset((
            "I am not a tag", best_in_show_ribbon,  # type: ignore[arg-type]
            blue_ribbon, red_ribbon)))

    # Test that instantiation succeeds if there are multiple instances
    # Tag subclasses.
    t1 = TaggableWithCopy(frozenset([reserve_best_in_show_ribbon, blue_ribbon,
                                    red_ribbon]))
    assert t1.tags == frozenset((reserve_best_in_show_ribbon, red_ribbon,
                                    blue_ribbon))

    # Test that instantiation succeeds if there are multiple instances
    # of UniqueTag of different subclasses.
    t1 = TaggableWithCopy(frozenset([reserve_best_in_show_ribbon,
                            best_in_class_ribbon, blue_ribbon,
                                    blue_ribbon]))
    assert t1.tags == frozenset((reserve_best_in_show_ribbon, best_in_class_ribbon,
                                blue_ribbon))

    # Test tagged() function
    t2 = t1.tagged(red_ribbon)
    print(t2.tags)
    assert t2.tags == frozenset((reserve_best_in_show_ribbon, best_in_class_ribbon,
                                blue_ribbon, red_ribbon))

    # Test that tagged() fails if a UniqueTag of the same subclass
    # is already present
    with pytest.raises(NonUniqueTagError):
        t1.tagged(best_in_show_ribbon)

    # Test that tagged() fails if tags are not a FrozenSet of Tags
    with pytest.raises(TypeError):
        t1.tagged(tags=frozenset((1,)))  # type: ignore[arg-type]

    # Test without_tags() function
    t4 = t2.without_tags(red_ribbon)
    assert t4.tags == t1.tags

    # Test that without_tags() fails if the tag is not present.
    with pytest.raises(ValueError):
        t4.without_tags(red_ribbon)

    # Test DottedName comparison
    from pytools.tag import DottedName
    assert FairRibbon() == FairRibbon()
    assert (FairRibbon().tag_name
            == FairRibbon().tag_name
            == DottedName(("pytools", "test", "test_pytools", "FairRibbon")))
    assert FairRibbon() != BlueRibbon()
    assert FairRibbon().tag_name != BlueRibbon().tag_name


def test_unordered_hash():
    import hashlib
    import random

    # FIXME: Use randbytes once >=3.9 is OK
    lst = [bytes([random.randrange(256) for _ in range(20)])
            for _ in range(200)]
    lorig = lst[:]
    random.shuffle(lst)

    from pytools import unordered_hash
    assert (unordered_hash(hashlib.sha256(), lorig).digest()
            == unordered_hash(hashlib.sha256(), lst).digest())
    assert (unordered_hash(hashlib.sha256(), lorig).digest()
            == unordered_hash(hashlib.sha256(), lorig).digest())
    assert (unordered_hash(hashlib.sha256(), lorig).digest()
            != unordered_hash(hashlib.sha256(), lorig[:-1]).digest())
    lst[0] = b"aksdjfla;sdfjafd"
    assert (unordered_hash(hashlib.sha256(), lorig).digest()
            != unordered_hash(hashlib.sha256(), lst).digest())


# {{{ sphere sampling

@pytest.mark.parametrize("sampling", [
    "equidistant", "fibonacci", "fibonacci_min", "fibonacci_avg",
    ])
def test_sphere_sampling(sampling, visualize=False):
    from functools import partial

    from pytools import sphere_sample_equidistant, sphere_sample_fibonacci

    npoints = 128
    radius = 1.5

    if sampling == "equidistant":
        sampling_func = partial(sphere_sample_equidistant, r=radius)
    elif sampling == "fibonacci":
        sampling_func = partial(
                sphere_sample_fibonacci, r=radius, optimize=None)
    elif sampling == "fibonacci_min":
        sampling_func = partial(
                sphere_sample_fibonacci, r=radius, optimize="minimum")
    elif sampling == "fibonacci_avg":
        sampling_func = partial(
                sphere_sample_fibonacci, r=radius, optimize="average")
    else:
        raise ValueError(f"unknown sampling method: '{sampling}'")

    np = pytest.importorskip("numpy")
    points = sampling_func(npoints)
    assert np.all(np.linalg.norm(points, axis=0) < radius + 1.0e-15)

    if not visualize:
        return

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    import matplotlib.tri as mtri
    theta = np.arctan2(np.sqrt(points[0]**2 + points[1]**2), points[2])
    phi = np.arctan2(points[1], points[0])
    triangles = mtri.Triangulation(theta, phi)

    ax.plot_trisurf(points[0], points[1], points[2], triangles=triangles.triangles)
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    ax.margins(0.05, 0.05, 0.05)

    # plt.show()
    fig.savefig(f"sphere_sampling_{sampling}")
    plt.close(fig)

# }}}


def test_unique_name_gen_conflicting_ok():
    from pytools import UniqueNameGenerator

    ung = UniqueNameGenerator()
    ung.add_names({"a", "b", "c"})

    with pytest.raises(ValueError):
        ung.add_names({"a"})

    ung.add_names({"a", "b", "c"}, conflicting_ok=True)


def test_strtobool():
    from pytools import strtobool
    assert strtobool("true") is True
    assert strtobool("tRuE") is True
    assert strtobool("1") is True
    assert strtobool("t") is True
    assert strtobool("on") is True

    assert strtobool("false") is False
    assert strtobool("FaLse") is False
    assert strtobool("0") is False
    assert strtobool("f") is False
    assert strtobool("off") is False

    with pytest.raises(ValueError):
        strtobool("tru")
        strtobool("fal")
        strtobool("xxx")
        strtobool(".")

    assert strtobool(None, False) is False


def test_to_identifier() -> None:
    from pytools import to_identifier

    assert to_identifier("_a_123_") == "_a_123_"
    assert to_identifier("a_123") == "a_123"
    assert to_identifier("a 123") == "a123"
    assert to_identifier("123") == "_123"
    assert to_identifier("_123") == "_123"
    assert to_identifier("123A") == "_123A"
    assert to_identifier("") == "_"

    assert not "a 123".isidentifier()
    assert to_identifier("a 123").isidentifier()
    assert to_identifier("123").isidentifier()
    assert to_identifier("").isidentifier()


def test_typedump():
    from pytools import typedump
    assert typedump("") == "str"
    assert typedump("abcdefg") == "str"
    assert typedump(5) == "int"

    assert typedump((5.0, 4)) == "tuple(float,int)"
    assert typedump([5, 4]) == "list(int,int)"
    assert typedump({5, 4}) == "set(int,int)"
    assert typedump(frozenset((1, 2, 3))) == "frozenset(int,int,int)"

    assert typedump([5, 4, 3, 2, 1]) == "list(int,int,int,int,int)"
    assert typedump([5, 4, 3, 2, 1, 0]) == "list(int,int,int,int,int,...)"
    assert typedump([5, 4, 3, 2, 1, 0], max_seq=6) == "list(int,int,int,int,int,int)"

    assert typedump({5: 42, 7: 43}) == "{'5': int, '7': int}"

    class C:
        class D:
            pass

    assert typedump(C()) == "pytools.test.test_pytools.test_typedump.<locals>.C"
    assert typedump(C.D()) == "pytools.test.test_pytools.test_typedump.<locals>.C.D"
    assert typedump(C.D(), fully_qualified_name=False) == "D"

    from pytools.datatable import DataTable
    t = DataTable(column_names=[])

    assert typedump(t) == "pytools.datatable.DataTable()"
    assert typedump(t, special_handlers={type(t): lambda x: "foo"}) == "foo"


def test_unique():
    from pytools import unique, unique_difference, unique_intersection, unique_union

    assert list(unique([1, 2, 1])) == [1, 2]
    assert tuple(unique((1, 2, 1))) == (1, 2)

    assert list(range(1000)) == list(unique(range(1000)))
    assert list(unique(list(range(1000)) + list(range(1000)))) == list(range(1000))

    # Also test strings since their ordering would be thrown off by
    # set-based 'unique' implementations.
    assert list(unique(["a", "b", "a"])) == ["a", "b"]
    assert tuple(unique(("a", "b", "a"))) == ("a", "b")

    assert list(unique_difference(["a", "b", "c"], ["b", "c", "d"])) == ["a"]
    assert list(unique_difference(["a", "b", "c"], ["a", "b", "c", "d"])) == []
    assert list(unique_difference(["a", "b", "c"], ["a"], ["b"], ["c"])) == []

    assert list(unique_intersection(["a", "b", "a"], ["b", "c", "a"])) == ["a", "b"]
    assert list(unique_intersection(["a", "b", "a"], ["d", "c", "e"])) == []

    assert list(unique_union(["a", "b", "a"], ["b", "c", "b"])) == ["a", "b", "c"]
    assert list(unique_union(
        ["a", "b", "a"], ["b", "c", "b"], ["c", "d", "c"])) == ["a", "b", "c", "d"]
    assert list(unique(["a", "b", "a"])) == \
        list(unique_union(["a", "b", "a"])) == ["a", "b"]

    assert list(unique_intersection()) == []
    assert list(unique_difference()) == []
    assert list(unique_union()) == []


# This class must be defined globally to be picklable
class SimpleRecord(Record):
    pass


def test_record():
    r = SimpleRecord(c=3, b=2, a=1)

    assert r.a == 1
    assert r.b == 2
    assert r.c == 3

    # Fields are sorted alphabetically in records
    assert str(r) == "SimpleRecord(a=1, b=2, c=3)"

    # Unregistered fields are (silently) ignored for printing
    r.f = 6
    assert str(r) == "SimpleRecord(a=1, b=2, c=3)"

    # Registered fields are printed
    r.register_fields({"d", "e"})
    assert str(r) == "SimpleRecord(a=1, b=2, c=3)"

    r.d = 4
    r.e = 5
    assert str(r) == "SimpleRecord(a=1, b=2, c=3, d=4, e=5)"

    with pytest.raises(AttributeError):
        r.ff  # noqa: B018

    # Test pickling
    import pickle
    r_pickled = pickle.loads(pickle.dumps(r))
    assert r == r_pickled

    # }}}

    # {{{ __slots__, __dict__, __weakref__ handling

    class RecordWithEmptySlots(Record):
        __slots__ = []

    assert hasattr(RecordWithEmptySlots(), "__slots__")
    assert not hasattr(RecordWithEmptySlots(), "__dict__")
    assert not hasattr(RecordWithEmptySlots(), "__weakref__")

    class RecordWithUnsetSlots(Record):
        pass

    assert hasattr(RecordWithUnsetSlots(), "__slots__")
    assert hasattr(RecordWithUnsetSlots(), "__dict__")
    assert hasattr(RecordWithUnsetSlots(), "__weakref__")

    from pytools import ImmutableRecord

    class ImmutableRecordWithEmptySlots(ImmutableRecord):
        __slots__ = []

    assert hasattr(ImmutableRecordWithEmptySlots(), "__slots__")
    assert hasattr(ImmutableRecordWithEmptySlots(), "__dict__")
    assert hasattr(ImmutableRecordWithEmptySlots(), "__weakref__")

    class ImmutableRecordWithUnsetSlots(ImmutableRecord):
        pass

    assert hasattr(ImmutableRecordWithUnsetSlots(), "__slots__")
    assert hasattr(ImmutableRecordWithUnsetSlots(), "__dict__")
    assert hasattr(ImmutableRecordWithUnsetSlots(), "__weakref__")

    # }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
