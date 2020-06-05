from __future__ import absolute_import, division

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

import numpy as np
from pytools import my_decorator as decorator, MovedFunctionDeprecationWrapper

__doc__ = """
Handling :mod:`numpy` Object Arrays
===================================

.. autofunction:: oarray_real
.. autofunction:: oarray_imag
.. autofunction:: oarray_real_copy
.. autofunction:: oarray_imag_copy

Creation
--------

.. autofunction:: make_obj_array

Mapping
-------

"""


    else:
        return 1


    else:
        return result

# {{{ workarounds for https://github.com/numpy/numpy/issues/1740

def oarray_real(ary):
    return rec_oarray_vectorize(lambda x: x.real, ary)


def oarray_imag(ary):
    return rec_oarray_vectorize(lambda x: x.imag, ary)


def oarray_real_copy(ary):
    return rec_oarray_vectorize(lambda x: x.real.copy(), ary)


def oarray_imag_copy(ary):
    return rec_oarray_vectorize(lambda x: x.imag.copy(), ary)

# }}}


# {{{ deprecated junk

def is_obj_array(val):
    warn("is_obj_array is deprecated and will go away in 2021, "
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
        else:
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
    else:
        return not is_obj_array(b) and a == b


is_field_equal = is_equal


def gen_len(expr):
    if is_obj_array(expr):
        return len(expr)
    else:
        return 1


def gen_slice(expr, slice_):
    warn("gen_slice is deprecated and will go away in 2021",
            DeprecationWarning, stacklevel=2)

    result = expr[slice_]
    if len(result) == 1:
        return result[0]
    else:
        return result


def obj_array_equal(a, b):
    warn("obj_array_equal is deprecated and will go away in 2021, "
            "use numpy.array_equal", DeprecationWarning, stacklevel=2)

    a_is_oa = is_obj_array(a)
    assert a_is_oa == is_obj_array(b)

    if a_is_oa:
        return np.array_equal(a, b)
    else:
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
    else:
        return set([f])


def cast_field(field, dtype):
    warn("cast_field is deprecated and will go away in 2021",
            DeprecationWarning, stacklevel=2)

    return with_object_array_or_scalar(
            lambda f: f.astype(dtype), field)


def with_object_array_or_scalar(f, field, obj_array_only=False):
    warn("with_object_array_or_scalar is deprecated and will go away in 2022, "
            "use oarray_vectorize", DeprecationWarning, stacklevel=2)

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
    else:
        return f(field)


as_oarray_func = decorator(with_object_array_or_scalar)


def with_object_array_or_scalar_n_args(f, *args):
    warn("with_object_array_or_scalar is deprecated and will go away in 2022, "
            "use oarray_vectorize_n_args", DeprecationWarning, stacklevel=2)

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
    else:
        return f(*args)


as_oarray_func_n_args = decorator(with_object_array_or_scalar_n_args)

# }}}

# vim: foldmethod=marker
