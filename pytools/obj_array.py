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
from warnings import warn

import numpy as np


__doc__ = """
Handling :mod:`numpy` Object Arrays
===================================

Creation
--------

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
"""


def make_obj_array(res_list):
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
    result = np.empty((len(res_list),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for idx in range(len(res_list)):
        result[idx] = res_list[idx]

    return result


def obj_array_to_hashable(f):
    if isinstance(f, np.ndarray) and f.dtype.char == "O":
        return tuple(f)
    return f


def flat_obj_array(*args):
    """Return a one-dimensional flattened object array consisting of
    elements obtained by 'flattening' *args* as follows:

    - The first axis of any non-subclassed object arrays will be flattened
      into the result.
    - Instances of :class:`list` will be flattened into the result.
    - Any other type will appear in the list as-is.
    """
    res_list = []
    for arg in args:
        if isinstance(arg, list):
            res_list.extend(arg)

        # Only flatten genuine, non-subclassed object arrays.
        elif type(arg) is np.ndarray:
            res_list.extend(arg.flat)

        else:
            res_list.append(arg)

    return make_obj_array(res_list)


def obj_array_vectorize(f, ary):
    """Apply the function *f* to all entries of the object array *ary*.
    Return an object array of the same shape consisting of the return
    values.
    If *ary* is not an object array, return ``f(ary)``.

    .. note ::

        This function exists because :class:`numpy.vectorize` suffers from the same
        issue described under :func:`make_obj_array`.
    """

    if isinstance(ary, np.ndarray) and ary.dtype.char == "O":
        result = np.empty_like(ary)
        for i in np.ndindex(ary.shape):
            result[i] = f(ary[i])
        return result
    return f(ary)


def obj_array_vectorized(f):
    wrapper = partial(obj_array_vectorize, f)
    update_wrapper(wrapper, f)
    return wrapper


def rec_obj_array_vectorize(f, ary):
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
    if isinstance(ary, np.ndarray) and ary.dtype.char == "O":
        result = np.empty_like(ary)
        for i in np.ndindex(ary.shape):
            result[i] = rec_obj_array_vectorize(f, ary[i])
        return result
    return f(ary)


def rec_obj_array_vectorized(f):
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


# {{{ deprecated junk

def is_obj_array(val):
    warn("is_obj_array is deprecated and will go away in 2022, "
            "just inline the check.", DeprecationWarning, stacklevel=2)

    try:
        return isinstance(val, np.ndarray) and val.dtype.char == "O"
    except AttributeError:
        return False


def log_shape(array):
    """Returns the "logical shape" of the array.

    The "logical shape" is the shape that's left when the node-depending
    dimension has been eliminated.
    """

    warn("log_shape is deprecated and will go away in 2021, "
            "use the actual object array shape.",
            DeprecationWarning, stacklevel=2)

    try:
        if array.dtype.char == "O":
            return array.shape
        return array.shape[:-1]
    except AttributeError:
        return ()


def join_fields(*args):
    warn("join_fields is deprecated and will go away in 2022, "
            "use flat_obj_array", DeprecationWarning, stacklevel=2)

    return flat_obj_array(*args)


def is_equal(a, b):
    warn("is_equal is deprecated and will go away in 2021, "
            "use numpy.array_equal", DeprecationWarning, stacklevel=2)

    if is_obj_array(a):
        return is_obj_array(b) and (a.shape == b.shape) and (a == b).all()
    return not is_obj_array(b) and a == b


is_field_equal = is_equal


def gen_len(expr):
    if is_obj_array(expr):
        return len(expr)
    return 1


def gen_slice(expr, slice_):
    warn("gen_slice is deprecated and will go away in 2021",
            DeprecationWarning, stacklevel=2)

    result = expr[slice_]
    if len(result) == 1:
        return result[0]
    return result


def obj_array_equal(a, b):
    warn("obj_array_equal is deprecated and will go away in 2021, "
            "use numpy.array_equal", DeprecationWarning, stacklevel=2)

    a_is_oa = is_obj_array(a)
    assert a_is_oa == is_obj_array(b)

    if a_is_oa:
        return np.array_equal(a, b)
    return a == b


def to_obj_array(ary):
    warn("to_obj_array is deprecated and will go away in 2021, "
            "use make_obj_array", DeprecationWarning,
            stacklevel=2)

    ls = log_shape(ary)
    result = np.empty(ls, dtype=object)

    for i in np.ndindex(ls):
        result[i] = ary[i]

    return result


def setify_field(f):
    warn("setify_field is deprecated and will go away in 2021",
            DeprecationWarning, stacklevel=2)

    if is_obj_array(f):
        return set(f)
    return {f}


def cast_field(field, dtype):
    warn("cast_field is deprecated and will go away in 2021",
            DeprecationWarning, stacklevel=2)

    return with_object_array_or_scalar(
            lambda f: f.astype(dtype), field)


def with_object_array_or_scalar(f, field, obj_array_only=False):
    warn("with_object_array_or_scalar is deprecated and will go away in 2022, "
            "use obj_array_vectorize", DeprecationWarning, stacklevel=2)

    if obj_array_only:
        if is_obj_array(field):
            ls = field.shape
        else:
            ls = ()
    else:
        ls = log_shape(field)
    if ls != ():
        result = np.zeros(ls, dtype=object)
        for i in np.ndindex(ls):
            result[i] = f(field[i])
        return result
    return f(field)


def as_oarray_func(f):
    wrapper = partial(with_object_array_or_scalar, f)
    update_wrapper(wrapper, f)
    return wrapper


def with_object_array_or_scalar_n_args(f, *args):
    warn("with_object_array_or_scalar_n_args is deprecated and "
            "will go away in 2022, "
            "use obj_array_vectorize_n_args", DeprecationWarning, stacklevel=2)

    oarray_arg_indices = []
    for i, arg in enumerate(args):
        if is_obj_array(arg):
            oarray_arg_indices.append(i)

    if not oarray_arg_indices:
        return f(*args)

    leading_oa_index = oarray_arg_indices[0]

    ls = log_shape(args[leading_oa_index])
    if ls != ():
        result = np.zeros(ls, dtype=object)

        new_args = list(args)
        for i in np.ndindex(ls):
            for arg_i in oarray_arg_indices:
                new_args[arg_i] = args[arg_i][i]

            result[i] = f(*new_args)
        return result
    return f(*args)


def as_oarray_func_n_args(f):
    wrapper = partial(with_object_array_or_scalar_n_args, f)
    update_wrapper(wrapper, f)
    return wrapper


def oarray_real(ary):
    warn("oarray_real is deprecated and will go away in 2022, "
            "use obj_array_real", DeprecationWarning, stacklevel=2)
    return obj_array_real(ary)


def oarray_imag(ary):
    warn("oarray_imag is deprecated and will go away in 2022, "
            "use obj_array_imag", DeprecationWarning, stacklevel=2)
    return obj_array_imag(ary)


def oarray_real_copy(ary):
    warn("oarray_real_copy is deprecated and will go away in 2022, "
            "use obj_array_real_copy", DeprecationWarning, stacklevel=2)
    return obj_array_real_copy(ary)


def oarray_imag_copy(ary):
    warn("oarray_imag_copy is deprecated and will go away in 2022, "
            "use obj_array_imag_copy", DeprecationWarning, stacklevel=2)
    return obj_array_imag_copy(ary)

# }}}

# vim: foldmethod=marker
