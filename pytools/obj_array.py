# pyright: reportNoOverloadImplementation=none, reportUninitializedInstanceVariable=none

"""
Handling :mod:`numpy` Object Arrays
===================================

.. autoclass:: T
.. autoclass:: ResultT
.. autoclass:: ShapeT

.. autoclass:: ObjectArray
.. autoclass:: ObjectArray0D
.. autoclass:: ObjectArray1D
.. autoclass:: ObjectArray2D
.. autoclass:: ObjectArrayND

Creation
--------

.. autofunction:: from_numpy
.. autofunction:: to_numpy
.. autofunction:: make_obj_array
.. autofunction:: flat_obj_array

Mapping
-------

.. autofunction:: obj_array_vectorize
.. autofunction:: obj_array_vectorize_n_args

Numpy workarounds
-----------------
These functions work around a `long-standing, annoying numpy issue
<https://github.com/numpy/numpy/issues/1740>`__.

.. autofunction:: obj_array_real
.. autofunction:: obj_array_imag
.. autofunction:: obj_array_real_copy
.. autofunction:: obj_array_imag_copy

References
----------

.. currentmodule:: np

.. class:: ndarray

    See :class:`numpy.ndarray`.

.. class:: dtype

    See :class:`numpy.dtype`.
"""


from __future__ import annotations


__copyright__ = "Copyright (C) 2009-2020 Andreas Kloeckner"

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

from functools import partial, update_wrapper
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self, override


# NOTE: Importing this must not require importing numpy, so do not be tempted
# to move the numpy import out of the type checking block.


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence

    import numpy as np


T = TypeVar("T", covariant=True)

ResultT = TypeVar("ResultT")
ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])


class _ObjectArrayMetaclass(type):
    @override
    def __instancecheck__(cls, instance: object) -> bool:
        try:
            import numpy as np
        except ModuleNotFoundError:
            return False
        return isinstance(instance, np.ndarray) and instance.dtype == object


class ObjectArray(Generic[ShapeT, T], metaclass=_ObjectArrayMetaclass):
    """This is a fake type used to distinguish :class:`numpy.ndarray`
    instances with a *dtype* of *object* for static type checking.
    Unlike :class:`numpy.ndarray`, this "type" can straightforwardly
    keep track of its entry type.
    All instances of this "type" will actually be :class:`numpy.ndarray` at run time.
    Note that this creates a "universe" of object arrays that's separate
    from the usual numpy-generated one. In order to usefully interact
    in a type-safe manner, object arrays should be "converted" to this
    "type".

    Furthermore, :func:`isinstance` has been customized so as to be accurate
    for this "type".

    For now, there is precise typing support for arrays up to two dimensions.

    .. autoattribute:: shape
    .. autoattribute:: dtype

    The usual set of operations on :class:`numpy.ndarray` instances works
    at run time; a subset of these operations are typed. If what you
    need isn't typed, please submit a pull request.

    This type is intended to be immutable, and hence covariant in *T*.
    """
    shape: ShapeT
    dtype: np.dtype[Any]

    if TYPE_CHECKING:
        def __len__(self) -> int: ...

        @property
        def size(self) -> int: ...

        @overload
        def __getitem__(self, x: ShapeT, /) -> T: ...

        @overload
        def __getitem__(self: ObjectArray1D[T], x: int, /) -> T: ...

        @overload
        def __getitem__(self: ObjectArray2D[T], x: int, /) -> ObjectArray1D[T]: ...

        @overload
        def __getitem__(self: ObjectArrayND[T], x: int, /) -> ObjectArrayND[T] | T: ...

        @overload
        def __getitem__(
            self: ObjectArray1D[T],
            x: slice) -> ObjectArray1D[T]: ...

        @overload
        def __getitem__(
            self: ObjectArray2D[T],
            x: slice) -> ObjectArray2D[T]: ...

        @overload
        def __iter__(self: ObjectArray1D[T]) -> Iterator[T]: ...
        @overload
        def __iter__(self: ObjectArray2D[T]) -> Iterator[ObjectArray1D[T]]: ...

        def __pos__(self) -> Self: ...
        def __neg__(self) -> Self: ...
        def __abs__(self) -> Self: ...

        # Technically, arithmetic with T prohibits covariance, because T is a sink
        # here. Precisely typing arithmetic is thorny, and I'm willing to take
        # a gamble on this, I think. -AK, July 2025
        @overload
        def __add__(self, other: Self, /) -> Self: ...
        @overload
        def __add__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __add__(self, other: int, /) -> Self: ...
        @overload
        def __radd__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __radd__(self, other: int, /) -> Self: ...

        @overload
        def __sub__(self, other: Self, /) -> Self: ...
        @overload
        def __sub__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __sub__(self, other: int, /) -> Self: ...
        @overload
        def __rsub__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __rsub__(self, other: int, /) -> Self: ...

        @overload
        def __mul__(self, other: Self, /) -> Self: ...
        @overload
        def __mul__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __mul__(self, other: int, /) -> Self: ...
        @overload
        def __rmul__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __rmul__(self, other: int, /) -> Self: ...

        @overload
        def __truediv__(self, other: Self, /) -> Self: ...
        @overload
        def __truediv__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __truediv__(self, other: int, /) -> Self: ...
        @overload
        def __rtruediv__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __rtruediv__(self, other: int, /) -> Self: ...

        @overload
        def __pow__(self, other: Self, /) -> Self: ...
        @overload
        def __pow__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __pow__(self, other: int, /) -> Self: ...
        @overload
        def __rpow__(self, other: T, /) -> Self: ...  # pyright: ignore[reportGeneralTypeIssues]
        @overload
        def __rpow__(self, other: int, /) -> Self: ...

        @overload
        def __matmul__(
                    self: ObjectArray1D[T],
                    other: ObjectArray1D[T],
                /) -> T: ...
        @overload
        def __matmul__(
                    self: ObjectArray2D[T],
                    other: ObjectArray1D[T],
                /) -> ObjectArray1D[T]: ...
        @overload
        def __matmul__(
                    self: ObjectArray1D[T],
                    other: ObjectArray2D[T],
                /) -> ObjectArray1D[T]: ...
        @overload
        def __matmul__(
                    self: ObjectArray2D[T],
                    other: ObjectArray2D[T],
                /) -> ObjectArray2D[T]: ...

        @property
        def flat(self) -> ObjectArray1D[T]: ...

        @overload
        def tolist(self: ObjectArray0D[T]) -> T: ...
        @overload
        def tolist(self: ObjectArray1D[T]) -> list[T]: ...
        @overload
        def tolist(self: ObjectArray2D[T]) -> list[list[T]]: ...
        @overload
        def tolist(self) -> list[Any]: ...


ObjectArray0D: TypeAlias = ObjectArray[tuple[()], T]
ObjectArray1D: TypeAlias = ObjectArray[tuple[int], T]
ObjectArray2D: TypeAlias = ObjectArray[tuple[int, int], T]
ObjectArrayND: TypeAlias = ObjectArray[tuple[int, ...], T]


def to_numpy(ary: ObjectArray[ShapeT, T]) -> np.ndarray[ShapeT, Any]:
    return cast("np.ndarray[ShapeT, Any]", cast("object", ary))


@overload
def from_numpy(
            ary: np.ndarray[ShapeT, Any],
            tp: None = None,
            /
        ) -> ObjectArray[ShapeT, Any]: ...

@overload
def from_numpy(
            ary: np.ndarray[ShapeT, Any],
            tp: type[T],
            /
        ) -> ObjectArray[ShapeT, T]: ...


def from_numpy(
            ary: np.ndarray[ShapeT, Any],
            tp: type[T] | None = None,  # pyright: ignore[reportUnusedParameter]
            /
        ) -> ObjectArray[ShapeT, T]:
    if ary.dtype != object:  # pyright: ignore[reportAny]
        ary = ary.astype(object)
    return cast("ObjectArray[ShapeT, T]", cast("object", ary))


def make_obj_array(res_list: Sequence[T]) -> ObjectArray1D[T]:
    """Create a one-dimensional object array from *res_list*.
    This differs from ``numpy.array(res_list, dtype=object)``
    by whether it tries to determine its shape by descending
    into nested array-like objects. Consider the following example:

    .. doctest::

        >>> import numpy as np
        >>> a = np.array([np.arange(5), np.arange(5)], dtype=object)
        >>> a
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]], dtype=object)
        >>> a.shape
        (2, 5)
        >>> # meanwhile:
        >>> from pytools.obj_array import make_obj_array
        >>> b = make_obj_array([np.arange(5), np.arange(5)])
        >>> b
        array([array([0, 1, 2, 3, 4]), array([0, 1, 2, 3, 4])], dtype=object)
        >>> b.shape
        (2,)

    In some settings (such as when the sub-arrays are large and/or
    live on a GPU), the recursive behavior of :func:`numpy.array`
    can be undesirable.
    """
    import numpy as np
    result = np.empty((len(res_list),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for idx in range(len(res_list)):
        result[idx] = res_list[idx]

    return cast("ObjectArray1D[T]", cast("object", result))


def obj_array_to_hashable(ary: ObjectArray[ShapeT, T] | Hashable, /) -> Hashable:
    if isinstance(ary, ObjectArray):
        ary = cast("ObjectArray[ShapeT, T]", ary)
        return tuple(ary.flat.tolist())
    return ary


def flat_obj_array(*args: ObjectArray[ShapeT, T] | list[T] | T) -> ObjectArray1D[T]:
    """Return a one-dimensional flattened object array consisting of
    elements obtained by 'flattening' *args* as follows:

    - The first axis of any non-subclassed object arrays will be flattened
      into the result.
    - Instances of :class:`list` will be flattened into the result.
    - Any other type will appear in the list as-is.
    """
    import numpy as np
    res_list: list[T] = []
    for arg in args:
        if isinstance(arg, list):
            res_list.extend(cast("list[T]", arg))

        # Only flatten genuine, non-subclassed object arrays.
        elif isinstance(arg, ObjectArray) and type(arg) is np.ndarray:  # pyright: ignore[reportUnnecessaryComparison,reportUnknownArgumentType]
            res_list.extend(arg.flat)

        else:
            res_list.append(cast("T", arg))

    return make_obj_array(res_list)


@overload
def obj_array_vectorize(
            f: Callable[[T], ResultT],
            ary: ObjectArray[ShapeT, T],
        ) -> ObjectArray[ShapeT, ResultT]: ...

@overload
def obj_array_vectorize(
            f: Callable[[T], ResultT],
            ary: T,
        ) -> ResultT: ...


def obj_array_vectorize(
            f: Callable[[T], ResultT],
            ary: T | ObjectArray[ShapeT, T],
        ) -> ResultT | ObjectArray[ShapeT, ResultT]:
    """Apply the function *f* to all entries of the object array *ary*.
    Return an object array of the same shape consisting of the return
    values.
    If *ary* is not an object array, return ``f(ary)``.

    .. note ::

        This function exists because :class:`numpy.vectorize` suffers from the same
        issue described under :func:`make_obj_array`.
    """

    import numpy as np
    if isinstance(ary, ObjectArray):
        result = np.empty_like(ary)
        ary = cast("ObjectArray[ShapeT, T]", ary)
        for i in np.ndindex(ary.shape):
            result[i] = f(ary[i])
        return cast("ObjectArray[ShapeT, ResultT]", cast("object", result))

    return f(ary)


# FIXME: It'd be nice to do make this more precise (T->T, ObjArray->ObjArray),
# but I don't know how.
def obj_array_vectorized(
            f: Callable[[T], T]
        ) -> Callable[[T | ObjectArray[ShapeT, T]], T | ObjectArray[ShapeT, T]]:
    wrapper = partial(obj_array_vectorize, f)
    update_wrapper(wrapper, f)
    return wrapper  # pyright: ignore[reportReturnType]


# FIXME: I don't know that this function *can* be precisely typed.
# We could probably handle a few specific nesting levels, but is it worth it?
def rec_obj_array_vectorize(
            f: Callable[[object], object],
            ary: object | ObjectArray[ShapeT, object]
        ) -> object | ObjectArray[ShapeT, object]:
    """Apply the function *f* to all entries of the object array *ary*.
    Return an object array of the same shape consisting of the return
    values.
    If the elements of *ary* are further object arrays, recurse
    until non-object-arrays are found and then apply *f* to those
    entries.
    If *ary* is not an object array, return ``f(ary)``.

    .. note ::

        This function exists because :class:`numpy.vectorize` suffers from the same
        issue described under :func:`make_obj_array`.
    """
    if isinstance(ary, ObjectArray):
        import numpy as np
        ary = cast("ObjectArray[ShapeT, object]", ary)
        result = np.empty_like(ary)
        for i in np.ndindex(ary.shape):
            result[i] = rec_obj_array_vectorize(f, ary[i])
        return cast("ObjectArray[Any, ShapeT]", cast("object", result))

    return f(ary)


def rec_obj_array_vectorized(
            f: Callable[[object], ResultT]
        ) -> Callable[
                [object | ObjectArray[ShapeT, object]],
                object | ObjectArray[ShapeT, object]]:
    wrapper = partial(rec_obj_array_vectorize, f)
    update_wrapper(wrapper, f)
    return wrapper


def obj_array_vectorize_n_args(f, *args):
    """Apply the function *f* elementwise to all entries of any
    object arrays in *args*. All such object arrays are expected
    to have the same shape (but this is not checked).
    Equivalent to an appropriately-looped execution of::

        result[idx] = f(obj_array_arg1[idx], arg2, obj_array_arg3[idx])

    Return an object array of the same shape as the arguments consisting of the
    return values of *f*.

    .. note ::

        This function exists because :class:`numpy.vectorize` suffers from the same
        issue described under :func:`make_obj_array`.
    """
    import numpy as np
    oarray_arg_indices = []
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray) and arg.dtype.char == "O":
            oarray_arg_indices.append(i)

    if not oarray_arg_indices:
        return f(*args)

    leading_oa_index = oarray_arg_indices[0]

    template_ary = args[leading_oa_index]
    result = np.empty_like(template_ary)
    new_args = list(args)
    for i in np.ndindex(template_ary.shape):
        for arg_i in oarray_arg_indices:
            new_args[arg_i] = args[arg_i][i]
        result[i] = f(*new_args)
    return result


def obj_array_vectorized_n_args(f):
    # Unfortunately, this can't use partial(), as the callable returned by it
    # will not be turned into a bound method upon attribute access.
    # This may happen here, because the decorator *could* be used
    # on methods, since it can "look past" the leading `self` argument.
    # Only exactly function objects receive this treatment.
    #
    # Spec link:
    # https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy
    # (under "Instance Methods", quote as of Py3.9.4)
    # > Also notice that this transformation only happens for user-defined functions;
    # > other callable objects (and all non-callable objects) are retrieved
    # > without transformation.

    def wrapper(*args):
        return obj_array_vectorize_n_args(f, *args)

    update_wrapper(wrapper, f)
    return wrapper


# {{{ workarounds for https://github.com/numpy/numpy/issues/1740

def obj_array_real(ary):
    return rec_obj_array_vectorize(lambda x: x.real, ary)


def obj_array_imag(ary):
    return rec_obj_array_vectorize(lambda x: x.imag, ary)


def obj_array_real_copy(ary):
    return rec_obj_array_vectorize(lambda x: x.real.copy(), ary)


def obj_array_imag_copy(ary):
    return rec_obj_array_vectorize(lambda x: x.imag.copy(), ary)

# }}}


# vim: foldmethod=marker
