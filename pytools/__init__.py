from __future__ import annotations


__copyright__ = """
Copyright (C) 2009-2013 Andreas Kloeckner
Copyright (C) 2013- University of Illinois Board of Trustees
Copyright (C) 2020 Matt Wala
"""

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

import builtins
import logging
import operator
import re
import sys
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from functools import reduce, wraps
from sys import intern
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
)

from typing_extensions import dataclass_transform


# These are deprecated and will go away in 2022.
all = builtins.all
any = builtins.any


__doc__ = """
A Collection of Utilities
=========================

Math
----

.. autofunction:: levi_civita

Assertive accessors
-------------------

.. autofunction:: one
.. autofunction:: is_single_valued
.. autofunction:: all_roughly_equal
.. autofunction:: single_valued

Memoization
-----------

.. autofunction:: memoize
.. autofunction:: memoize_on_first_arg
.. autofunction:: memoize_method
.. autofunction:: memoize_in
.. autofunction:: keyed_memoize_on_first_arg
.. autofunction:: keyed_memoize_method
.. autofunction:: keyed_memoize_in

Argmin/max
----------

.. autofunction:: argmin2
.. autofunction:: argmax2
.. autofunction:: argmin
.. autofunction:: argmax

Cartesian products
------------------
.. autofunction:: cartesian_product
.. autofunction:: distinct_pairs

Permutations, Tuples, Integer sequences
---------------------------------------

.. autofunction:: wandering_element
.. autofunction:: generate_nonnegative_integer_tuples_below
.. autofunction:: generate_nonnegative_integer_tuples_summing_to_at_most
.. autofunction:: generate_all_integer_tuples_below
.. autofunction:: generate_permutations
.. autofunction:: generate_unique_permutations

Formatting
----------

.. autoclass:: Table
.. autofunction:: merge_tables
.. autofunction:: string_histogram
.. autofunction:: word_wrap

Debugging
---------

.. autofunction:: typedump
.. autofunction:: invoke_editor

Progress bars
-------------

.. autoclass:: ProgressBar

Name generation
---------------

.. autofunction:: generate_unique_names
.. autofunction:: generate_numbered_unique_names
.. autoclass:: UniqueNameGenerator

Deprecation Warnings
--------------------

.. autofunction:: deprecate_keyword
.. autofunction:: module_getattr_for_deprecations

Functions for dealing with (large) auxiliary files
--------------------------------------------------

.. autofunction:: download_from_web_if_not_present

Helpers for :mod:`numpy`
------------------------

.. autofunction:: reshaped_view


Timing data
-----------

.. autoclass:: ProcessTimer

Log utilities
-------------

.. autoclass:: ProcessLogger
.. autoclass:: DebugProcessLogger
.. autoclass:: log_process

Sorting in natural order
------------------------

.. autofunction:: natorder
.. autofunction:: natsorted

Backports of newer Python functionality
---------------------------------------

.. autofunction:: resolve_name

Hashing
-------

.. autofunction:: unordered_hash

Sampling
--------

.. autofunction:: sphere_sample_equidistant
.. autofunction:: sphere_sample_fibonacci

String utilities
----------------

.. autofunction:: strtobool
.. autofunction:: to_identifier

Set-like functions for iterables
--------------------------------

These functions provide set-like operations on iterables. In contrast to
Python's built-in set type, they maintain the internal order of elements.

.. autofunction:: unique
.. autofunction:: unique_difference
.. autofunction:: unique_intersection
.. autofunction:: unique_union

Functionality for dataclasses
-----------------------------

.. autofunction:: opt_frozen_dataclass

Type Variables Used
-------------------

.. class:: T
.. class:: R

    Generic unbound invariant :class:`typing.TypeVar`.

.. class:: F

    Generic invariant :class:`typing.TypeVar` bound to a :class:`typing.Callable`.

.. class:: P

    Generic unbound invariant :class:`typing.ParamSpec`.
"""

# {{{ type variables

T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")

# }}}


# {{{ code maintenance

# Undocumented on purpose for now, unclear that this is a great idea, given
# that typing.deprecated exists.
class MovedFunctionDeprecationWrapper:
    def __init__(self, f: F, deadline: int | str | None = None) -> None:
        if deadline is None:
            deadline = "the future"

        self.f = f
        self.deadline = deadline

    def __call__(self, *args, **kwargs):
        from warnings import warn
        warn(f"This function is deprecated and will go away in {self.deadline}. "
            f"Use {self.f.__module__}.{self.f.__name__} instead.",
            DeprecationWarning, stacklevel=2)

        return self.f(*args, **kwargs)


def deprecate_keyword(oldkey: str,
        newkey: str | None = None, *,
        deadline: str | None = None):
    """Decorator used to deprecate function keyword arguments.

    :arg oldkey: deprecated argument name.
    :arg newkey: new argument name that serves the same purpose, if any.
    :arg deadline: expected time frame for the removal of the deprecated argument.
    """
    from warnings import warn

    if deadline is None:
        deadline = "the future"

    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            if oldkey in kwargs:
                if newkey is None:
                    warn(f"The '{oldkey}' keyword is deprecated and will "
                            f"go away in {deadline}.",
                            DeprecationWarning, stacklevel=2)
                else:
                    warn(f"The '{oldkey}' keyword is deprecated and will "
                            f"go away in {deadline}. "
                            f"Use '{newkey}' instead.",
                            DeprecationWarning, stacklevel=2)

                    if newkey in kwargs:
                        raise ValueError(f"Cannot use '{oldkey}' "
                                f"and '{newkey}' in the same call.")

                    kwargs[newkey] = kwargs[oldkey]
                    del kwargs[oldkey]

            return func(*args, **kwargs)

        return inner_wrapper

    return wrapper


def module_getattr_for_deprecations(
            module_name: str,
            depr_name_to_replacement_and_obj: Mapping[
                str, tuple[str, object, str | int]
            ],
            name: str
        ) -> object:
    """A helper to construct module-level :meth:`object.__getattr__` functions
    so that deprecated names can still be found but raise a warning.

    The typical usage pattern is as follows::

        __getattr__ = partial(module_getattr_for_deprecations, __name__, {
            "OldName": ("NewName", NewName, 2026),
            })
    """

    replacement_and_obj = depr_name_to_replacement_and_obj.get(name, None)
    if replacement_and_obj is not None:
        replacement, obj, deadline = replacement_and_obj
        from warnings import warn

        warn(f"'{module_name}.{name}' is deprecated. "
                f"Use '{replacement}' instead. "
                f"'{module_name}.{name}' will continue to work until {deadline}.",
                DeprecationWarning, stacklevel=2)
        return obj
    raise AttributeError(name)

# }}}


# {{{ math

def delta(x, y):
    if x == y:
        return 1
    return 0


def levi_civita(tup: tuple[int, ...]) -> int:
    """Compute an entry of the Levi-Civita symbol for the indices *tuple*."""
    if len(tup) == 2:
        i, j = tup
        return j - i
    if len(tup) == 3:
        i, j, k = tup
        return (j-i) * (k-i) * (k-j) // 2
    raise NotImplementedError(f"Levi-Civita symbol in {len(tup)} dimensions")


def norm_1(iterable):
    return sum(abs(x) for x in iterable)


def norm_2(iterable):
    return sum(x**2 for x in iterable)**0.5


def norm_inf(iterable):
    return max(abs(x) for x in iterable)


def norm_p(iterable, p):
    return sum(i**p for i in iterable)**(1/p)


class Norm:
    def __init__(self, p):
        self.p = p

    def __call__(self, iterable):
        return sum(i**self.p for i in iterable)**(1/self.p)

# }}}


# {{{ data structures

# {{{ record

class RecordWithoutPickling:
    """An aggregate of named sub-variables. Assumes that each record sub-type
    will be individually derived from this class.
    """

    __slots__: ClassVar[list[str]] = []
    fields: ClassVar[set[str]]

    def __init__(self,
                 valuedict: Mapping[str, Any] | None = None,
                 exclude: Sequence[str] | None = None,
                 **kwargs: Any) -> None:
        assert self.__class__ is not Record

        if exclude is None:
            exclude = ["self"]

        try:
            fields = self.__class__.fields
        except AttributeError:
            self.__class__.fields = fields = set()

        if valuedict is not None:
            kwargs.update(valuedict)

        for key, value in kwargs.items():
            if key not in exclude:
                fields.add(key)
                setattr(self, key, value)

    def get_copy_kwargs(self, **kwargs):
        for f in self.__class__.fields:
            if f not in kwargs:
                try:
                    kwargs[f] = getattr(self, f)
                except AttributeError:
                    pass
        return kwargs

    def copy(self, **kwargs):
        return self.__class__(**self.get_copy_kwargs(**kwargs))

    def __repr__(self):
        return "{}({})".format(
                self.__class__.__name__,
                ", ".join(f"{fld}={getattr(self, fld)!r}"
                    for fld in sorted(self.__class__.fields)
                    if hasattr(self, fld)))

    def register_fields(self, new_fields):
        try:
            fields = self.__class__.fields
        except AttributeError:
            self.__class__.fields = fields = set()

        fields.update(new_fields)

    def __getattr__(self, name):
        # This method is implemented to avoid pylint 'no-member' errors for
        # attribute access.
        raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name))


class Record(RecordWithoutPickling):
    __slots__: ClassVar[list[str]] = []

    def __getstate__(self):
        return {
                key: getattr(self, key)
                for key in self.__class__.fields
                if hasattr(self, key)}

    def __setstate__(self, valuedict):
        try:
            fields = self.__class__.fields
        except AttributeError:
            self.__class__.fields = fields = set()

        for key, value in valuedict.items():
            fields.add(key)
            setattr(self, key, value)

    def __eq__(self, other):
        if self is other:
            return True
        return (self.__class__ == other.__class__
                and self.__getstate__() == other.__getstate__())

    def __ne__(self, other):
        return not self.__eq__(other)


class ImmutableRecordWithoutPickling(RecordWithoutPickling):
    """Hashable record. Does not explicitly enforce immutability."""
    def __init__(self, *args, **kwargs):
        RecordWithoutPickling.__init__(self, *args, **kwargs)
        self._cached_hash = None

    def __hash__(self):
        # This attribute may vanish during pickling.
        if getattr(self, "_cached_hash", None) is None:
            self._cached_hash = hash((
                    type(self),
                    *(getattr(self, field) for field in self.__class__.fields)
                    ))

        return self._cached_hash


class ImmutableRecord(ImmutableRecordWithoutPickling, Record):
    pass

# }}}


class Reference:
    def __init__(self, value):
        self.value = value

    def get(self):
        from warnings import warn
        warn("Reference.get() is deprecated -- use ref.value instead. "
             "This will stop working in 2025.", stacklevel=2)
        return self.value

    def set(self, value):
        self.value = value


class FakeList:
    def __init__(self, f, length):
        self._Length = length
        self._Function = f

    def __len__(self):
        return self._Length

    def __getitem__(self, index):
        try:
            return [self._Function(i)
                    for i in range(*index.indices(self._Length))]
        except AttributeError:
            return self._Function(index)


# {{{ dependent dictionary

class DependentDictionary:
    def __init__(self, f, start=None):
        if start is None:
            start = {}

        self._Function = f
        self._Dictionary = start.copy()

    def copy(self):
        return DependentDictionary(self._Function, self._Dictionary)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        try:
            return self._Dictionary[key]
        except KeyError:
            return self._Function(self._Dictionary, key)

    def __setitem__(self, key, value):
        self._Dictionary[key] = value

    def genuineKeys(self):  # noqa: N802
        return list(self._Dictionary.keys())

    def iteritems(self):
        return self._Dictionary.items()

    def iterkeys(self):
        return self._Dictionary.keys()

    def itervalues(self):
        return self._Dictionary.values()

# }}}

# }}}


# {{{ assertive accessors

def one(iterable: Iterable[T]) -> T:
    """Return the first entry of *iterable*. Assert that *iterable* has only
    that one entry.
    """
    it = iter(iterable)
    try:
        v = next(it)
    except StopIteration:
        raise ValueError("empty iterable passed to 'one()'") from None

    def no_more():
        try:
            next(it)
            raise ValueError("iterable with more than one entry passed to 'one()'")
        except StopIteration:
            return True

    assert no_more()

    return v


def is_single_valued(
        iterable: Iterable[T],
        equality_pred: Callable[[T, T], bool] = operator.eq
        ) -> bool:
    it = iter(iterable)
    try:
        first_item = next(it)
    except StopIteration:
        raise ValueError("empty iterable passed to 'single_valued()'") from None

    for other_item in it:
        if not equality_pred(other_item, first_item):
            return False
    return True


all_equal = is_single_valued


def all_roughly_equal(iterable, threshold):
    return is_single_valued(iterable,
            equality_pred=lambda a, b: abs(a-b) < threshold)


def single_valued(
        iterable: Iterable[T],
        equality_pred: Callable[[T, T], bool] = operator.eq
        ) -> T:
    """Return the first entry of *iterable*; Assert that other entries
    are the same with the first entry of *iterable*.
    """
    it = iter(iterable)
    try:
        first_item = next(it)
    except StopIteration:
        raise ValueError("empty iterable passed to 'single_valued()'") from None

    def others_same():
        for other_item in it:
            if not equality_pred(other_item, first_item):
                return False
        return True
    assert others_same()

    return first_item

# }}}


# {{{ memoization / attribute storage

def memoize(*args: F, **kwargs: Any) -> F:
    """Stores previously computed function values in a cache.

    Two keyword-only arguments are supported:

    :arg use_kwargs: Allows the caller to use keyword arguments. Defaults to
        ``False``. Setting this to ``True`` has a non-negligible performance
        impact.
    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.
    """

    use_kw = bool(kwargs.pop("use_kwargs", False))

    default_key_func: Callable[..., Any] | None

    if use_kw:
        def default_key_func(*inner_args, **inner_kwargs):
            return inner_args, frozenset(inner_kwargs.items())
    else:
        default_key_func = None

    key_func = kwargs.pop("key", default_key_func)

    if kwargs:
        raise TypeError(
            "memoize received unexpected keyword arguments: {}".format(
                ", ".join(kwargs.keys())))

    if key_func is not None:
        def _decorator(func):
            def wrapper(*args, **kwargs):
                key = key_func(*args, **kwargs)
                try:
                    return func._memoize_dic[key]
                except AttributeError:
                    # _memoize_dic doesn't exist yet.
                    result = func(*args, **kwargs)
                    func._memoize_dic = {key: result}
                    return result
                except KeyError:
                    result = func(*args, **kwargs)
                    func._memoize_dic[key] = result
                    return result

            from functools import update_wrapper
            update_wrapper(wrapper, func)
            return wrapper

    else:
        def _decorator(func):
            def wrapper(*args):
                try:
                    return func._memoize_dic[args]
                except AttributeError:
                    # _memoize_dic doesn't exist yet.
                    result = func(*args)
                    func._memoize_dic = {args: result}
                    return result
                except KeyError:
                    result = func(*args)
                    func._memoize_dic[args] = result
                    return result

            from functools import update_wrapper
            update_wrapper(wrapper, func)
            return wrapper

    if not args:
        return _decorator  # type: ignore
    if callable(args[0]) and len(args) == 1:
        return _decorator(args[0])
    raise TypeError(
        f"memoize received unexpected position arguments: {args}")


FunctionValueCache = memoize


class _HasKwargs:
    pass


def memoize_on_first_arg(
        function: Callable[Concatenate[T, P], R], *,
        cache_dict_name: str | None = None) -> Callable[Concatenate[T, P], R]:
    """Like :func:`memoize_method`, but for functions that take the object
    in which do memoization information is stored as first argument.

    Supports cache deletion via ``function_name.clear_cache(self)``.
    """

    if cache_dict_name is None:
        cache_dict_name = intern(
                f"_memoize_dic_{function.__module__}{function.__name__}"
                )

    def wrapper(obj: T, *args: P.args, **kwargs: P.kwargs) -> R:
        if kwargs:
            key = (_HasKwargs, frozenset(kwargs.items()), *args)
        else:
            key = args

        assert cache_dict_name is not None
        try:
            return getattr(obj, cache_dict_name)[key]
        except AttributeError:
            attribute_error = True
        except KeyError:
            attribute_error = False

        result = function(obj, *args, **kwargs)
        if attribute_error:
            object.__setattr__(obj, cache_dict_name, {key: result})
            return result
        getattr(obj, cache_dict_name)[key] = result
        return result

    def clear_cache(obj):
        object.__delattr__(obj, cache_dict_name)

    from functools import update_wrapper
    new_wrapper = update_wrapper(wrapper, function)

    # type-ignore because mypy has a point here, stuffing random attributes
    # into the function's dict is moderately sketchy.
    new_wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]

    return new_wrapper


def memoize_method(
        method: Callable[Concatenate[T, P], R]
        ) -> Callable[Concatenate[T, P], R]:
    """Supports cache deletion via ``method_name.clear_cache(self)``.

    .. versionchanged:: 2021.2

        Can memoize methods on classes that do not allow setting attributes
        (e.g. by overwriting ``__setattr__``), e.g. frozen :mod:`dataclasses`.
    """

    return memoize_on_first_arg(method,
            cache_dict_name=intern(f"_memoize_dic_{method.__name__}"))


class keyed_memoize_on_first_arg(Generic[T, P, R]):  # noqa: N801
    """Like :func:`memoize_method`, but for functions that take the object
    in which memoization information is stored as first argument.

    Supports cache deletion via ``function_name.clear_cache(self)``.

    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.
    :arg cache_dict_name: The name of the `dict` attribute in the instance
        used to hold the cache.

    .. versionadded :: 2020.3
    """

    def __init__(self,
            key: Callable[P, Hashable], *,
            cache_dict_name: str | None = None) -> None:
        self.key = key
        self.cache_dict_name = cache_dict_name

    def _default_cache_dict_name(self,
            function: Callable[Concatenate[T, P], R]) -> str:
        return intern(f"_memoize_dic_{function.__module__}{function.__name__}")

    def __call__(
            self, function: Callable[Concatenate[T, P], R]
            ) -> Callable[Concatenate[T, P], R]:
        cache_dict_name = self.cache_dict_name
        key = self.key

        if cache_dict_name is None:
            cache_dict_name = self._default_cache_dict_name(function)

        def wrapper(obj: T, *args: P.args, **kwargs: P.kwargs) -> R:
            cache_key = key(*args, **kwargs)

            assert cache_dict_name is not None
            try:
                return getattr(obj, cache_dict_name)[cache_key]
            except AttributeError:
                result = function(obj, *args, **kwargs)
                object.__setattr__(obj, cache_dict_name, {cache_key: result})
                return result
            except KeyError:
                result = function(obj, *args, **kwargs)
                getattr(obj, cache_dict_name)[cache_key] = result
                return result

        def clear_cache(obj):
            object.__delattr__(obj, cache_dict_name)

        from functools import update_wrapper
        new_wrapper = update_wrapper(wrapper, function)
        new_wrapper.clear_cache = clear_cache       # type: ignore[attr-defined]

        return new_wrapper


class keyed_memoize_method(keyed_memoize_on_first_arg):  # noqa: N801
    """Like :class:`memoize_method`, but additionally uses a function *key* to
    compute the key under which the function result is stored.

    Supports cache deletion via ``method_name.clear_cache(self)``.

    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.

    .. versionadded :: 2020.3

    .. versionchanged:: 2021.2

        Can memoize methods on classes that do not allow setting attributes
        (e.g. by overwriting ``__setattr__``), e.g. frozen :mod:`dataclasses`.
    """
    def _default_cache_dict_name(self, function):
        return intern(f"_memoize_dic_{function.__name__}")


class memoize_in:  # noqa: N801
    """Adds a cache to the function it decorates. The cache is attached
    to *container* and must be uniquely specified by *identifier* (i.e.
    all functions using the same *container* and *identifier* will be using
    the same cache). The decorated function may only receive positional
    arguments.

    .. note::

        This function works well on nested functions, which
        do not have stable global identifiers.

    .. versionchanged :: 2020.3

        *identifier* no longer needs to be a :class:`str`,
        but it needs to be hashable.

    .. versionchanged:: 2021.2.1

        Can now use instances of classes as *container* that do not allow
        setting attributes (e.g. by overwriting ``__setattr__``),
        e.g. frozen :mod:`dataclasses`.
    """

    def __init__(self, container: Any, identifier: Hashable) -> None:
        try:
            memoize_in_dict = container._pytools_memoize_in_dict
        except AttributeError:
            memoize_in_dict = {}
            object.__setattr__(container, "_pytools_memoize_in_dict",
                    memoize_in_dict)

        self.cache_dict = memoize_in_dict.setdefault(identifier, {})

    def __call__(self, inner: Callable[P, R]) -> Callable[P, R]:
        @wraps(inner)
        def new_inner(*args: P.args, **kwargs: P.kwargs) -> R:
            assert not kwargs

            try:
                return self.cache_dict[args]
            except KeyError:
                result = inner(*args, **kwargs)
                self.cache_dict[args] = result
                return result

        return new_inner


class keyed_memoize_in(Generic[P]):  # noqa: N801
    """Like :class:`memoize_in`, but additionally uses a function *key* to
    compute the key under which the function result is memoized.

    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.

    .. versionadded :: 2021.2.1
    """

    def __init__(self,
            container: Any, identifier: Hashable,
            key: Callable[P, Hashable]) -> None:
        try:
            memoize_in_dict = container._pytools_keyed_memoize_in_dict
        except AttributeError:
            memoize_in_dict = {}
            object.__setattr__(container, "_pytools_keyed_memoize_in_dict",
                    memoize_in_dict)

        self.cache_dict = memoize_in_dict.setdefault(identifier, {})
        self.key = key

    def __call__(self, inner: Callable[P, R]) -> Callable[P, R]:
        @wraps(inner)
        def new_inner(*args: P.args, **kwargs: P.kwargs) -> R:
            assert not kwargs
            key = self.key(*args, **kwargs)

            try:
                return self.cache_dict[key]
            except KeyError:
                result = inner(*args, **kwargs)
                self.cache_dict[key] = result
                return result

        return new_inner

# }}}


# {{{ syntactical sugar

class InfixOperator:
    """Pseudo-infix operators that allow syntax of the kind `op1 <<operator>> op2'.

    Following a recipe from
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/384122
    """
    def __init__(self, function):
        self.function = function

    def __rlshift__(self, other):
        return InfixOperator(lambda x: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def call(self, a, b):
        return self.function(a, b)


def monkeypatch_method(cls):
    # from GvR, http://mail.python.org/pipermail/python-dev/2008-January/076194.html
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


def monkeypatch_class(_name, bases, namespace):
    # from GvR, http://mail.python.org/pipermail/python-dev/2008-January/076194.html

    assert len(bases) == 1, "Exactly one base class required"
    base = bases[0]
    for name, value in namespace.items():
        if name != "__metaclass__":
            setattr(base, name, value)
    return base

# }}}


# {{{ generic utilities

def add_tuples(t1, t2):
    return tuple(t1v + t2v for t1v, t2v in zip(t1, t2, strict=True))


def negate_tuple(t1):
    return tuple(-t1v for t1v in t1)


def shift(vec, dist):
    """Return a copy of *vec* shifted by *dist* such that

    .. code:: python

        shift(a, i)[j] == a[(i+j) % len(a)]
    """

    result = vec[:]

    N = len(vec)  # noqa: N806
    dist = dist % N

    # modulo only returns positive distances!
    if dist > 0:
        result[dist:] = vec[:N-dist]
        result[:dist] = vec[N-dist:]

    return result


def len_iterable(iterable):
    return sum(1 for i in iterable)


def flatten(iterable):
    """For an iterable of sub-iterables, generate each member of each
    sub-iterable in turn, i.e. a flattened version of that super-iterable.

    Example: Turn [[a,b,c],[d,e,f]] into [a,b,c,d,e,f].
    """
    for sublist in iterable:
        yield from sublist


def general_sum(sequence):
    return reduce(operator.add, sequence)


def linear_combination(coefficients, vectors):
    result = coefficients[0] * vectors[0]
    for c, v in zip(coefficients[1:], vectors[1:], strict=True):
        result += c*v
    return result


def common_prefix(iterable, empty=None):
    it = iter(iterable)
    try:
        pfx = next(it)
    except StopIteration:
        return empty

    for v in it:
        for j, pfx_j in enumerate(pfx):
            if pfx_j != v[j]:
                pfx = pfx[:j]
                if j == 0:
                    return pfx
                break

    return pfx


def decorate(function, iterable):
    return [(x, function(x)) for x in iterable]


def partition(criterion, iterable):
    part_true = []
    part_false = []
    for i in iterable:
        if criterion(i):
            part_true.append(i)
        else:
            part_false.append(i)
    return part_true, part_false


def partition2(iterable):
    part_true = []
    part_false = []
    for pred, i in iterable:
        if pred:
            part_true.append(i)
        else:
            part_false.append(i)
    return part_true, part_false


def product(iterable: Iterable[Any]) -> Any:
    from operator import mul
    return reduce(mul, iterable, 1)


def reverse_dictionary(the_dict):
    result = {}
    for key, value in the_dict.items():
        if value in result:
            raise RuntimeError(
                    f"non-reversible mapping, duplicate key '{value}'")
        result[value] = key
    return result


def set_sum(set_iterable):
    from operator import or_
    return reduce(or_, set_iterable, set())


def div_ceil(nr, dr):
    return -(-nr // dr)


def uniform_interval_splitting(n, granularity, max_intervals):
    """ Return *(interval_size, num_intervals)* such that::

        num_intervals * interval_size >= n

    and::

        (num_intervals - 1) * interval_size < n

    and *interval_size* is a multiple of *granularity*.
    """
    # ported from Thrust

    grains = div_ceil(n, granularity)

    # one grain per interval
    if grains <= max_intervals:
        return granularity, grains

    grains_per_interval = div_ceil(grains, max_intervals)
    interval_size = grains_per_interval * granularity
    num_intervals = div_ceil(n, interval_size)

    return interval_size, num_intervals


def find_max_where(predicate, prec=1e-5, initial_guess=1, fail_bound=1e38):
    """Find the largest value for which a predicate is true,
    along a half-line. 0 is assumed to be the lower bound."""

    # {{{ establish bracket

    mag = initial_guess

    if predicate(mag):
        mag *= 2
        while predicate(mag):
            mag *= 2

            if mag > fail_bound:
                raise RuntimeError("predicate appears to be true "
                        f"everywhere, up to {fail_bound:g}")

        lower_true = mag/2
        upper_false = mag
    else:
        mag /= 2

        while not predicate(mag):
            mag /= 2

            if mag < prec:
                return mag

        lower_true = mag
        upper_false = mag*2

    # }}}

    # {{{ refine

    # Refine a bracket between *lower_true*, where the predicate is true,
    # and *upper_false*, where it is false, until *prec* is satisfied.

    assert predicate(lower_true)
    assert not predicate(upper_false)

    while abs(lower_true-upper_false) > prec:
        mid = (lower_true+upper_false)/2
        if predicate(mid):
            lower_true = mid
        else:
            upper_false = mid

    return lower_true

    # }}}

# }}}


# {{{ argmin, argmax

def argmin2(iterable, return_value=False):
    it = iter(iterable)
    try:
        current_argmin, current_min = next(it)
    except StopIteration:
        raise ValueError("argmin of empty iterable") from None

    for arg, item in it:
        if item < current_min:
            current_argmin = arg
            current_min = item

    if return_value:
        return current_argmin, current_min
    return current_argmin


def argmax2(iterable, return_value=False):
    it = iter(iterable)
    try:
        current_argmax, current_max = next(it)
    except StopIteration:
        raise ValueError("argmax of empty iterable") from None

    for arg, item in it:
        if item > current_max:
            current_argmax = arg
            current_max = item

    if return_value:
        return current_argmax, current_max
    return current_argmax


def argmin(iterable):
    return argmin2(enumerate(iterable))


def argmax(iterable):
    return argmax2(enumerate(iterable))

# }}}


# {{{ cartesian products etc.

def cartesian_product(*args):
    if len(args) == 1:
        for arg in args[0]:
            yield (arg,)
        return
    first = args[:-1]
    for prod in cartesian_product(*first):
        for i in args[-1]:
            yield (*prod, i)


def distinct_pairs(list1, list2):
    for i, xi in enumerate(list1):
        for j, yj in enumerate(list2):
            if i != j:
                yield (xi, yj)


def cartesian_product_sum(list1, list2):
    """This routine returns a list of sums of each element of
    list1 with each element of list2. Also works with lists.
    """
    for i in list1:
        for j in list2:
            yield i+j

# }}}


# {{{ elementary statistics

def average(iterable):
    """Return the average of the values in iterable.

    iterable may not be empty.
    """
    it = iterable.__iter__()

    try:
        s = next(it)
        count = 1
    except StopIteration:
        raise ValueError("empty average") from None

    for value in it:
        s = s + value
        count += 1

    return s/count


class VarianceAggregator:
    """Online variance calculator.
    See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    Adheres to pysqlite's aggregate interface.
    """
    def __init__(self, entire_pop):
        self.n = 0
        self.mean = 0
        self.m2 = 0

        self.entire_pop = entire_pop

    def step(self, x):
        self.n += 1
        delta_ = x - self.mean
        self.mean += delta_/self.n
        self.m2 += delta_*(x - self.mean)

    def finalize(self):
        if self.entire_pop:
            if self.n == 0:
                return None
            return self.m2/self.n
        if self.n <= 1:
            return None
        return self.m2/(self.n - 1)


def variance(iterable, entire_pop):
    v_comp = VarianceAggregator(entire_pop)

    for x in iterable:
        v_comp.step(x)

    return v_comp.finalize()


def std_deviation(iterable, finite_pop):
    from math import sqrt
    return sqrt(variance(iterable, finite_pop))

# }}}


# {{{ permutations, tuples, integer sequences

def wandering_element(length, wanderer=1, landscape=0):
    for i in range(length):
        yield i*(landscape,) + (wanderer,) + (length-1-i)*(landscape,)


def indices_in_shape(shape):
    from warnings import warn
    warn("indices_in_shape is deprecated. You should prefer numpy.ndindex.",
            DeprecationWarning, stacklevel=2)

    if isinstance(shape, int):
        shape = (shape,)

    if not shape:
        yield ()
    elif len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    else:
        remainder = shape[1:]
        for i in range(shape[0]):
            for rest in indices_in_shape(remainder):
                yield (i, *rest)


def generate_nonnegative_integer_tuples_below(n, length=None, least=0):
    """n may be a sequence, in which case length must be None."""
    if length is None:
        if not n:
            yield ()
            return

        my_n = n[0]
        n = n[1:]
        next_length = None
    else:
        my_n = n

        assert length >= 0
        if length == 0:
            yield ()
            return

        next_length = length-1

    for i in range(least, my_n):
        my_part = (i,)
        for base in generate_nonnegative_integer_tuples_below(n, next_length, least):
            yield my_part + base


def generate_decreasing_nonnegative_tuples_summing_to(
        n, length, min_value=0, max_value=None):
    if length == 0:
        yield ()
    elif length == 1:
        if n <= max_value:
            # print "MX", n, max_value
            yield (n,)
        else:
            return
    else:
        if max_value is None or n < max_value:
            max_value = n

        for i in range(min_value, max_value+1):
            # print "SIG", sig, i
            for remainder in generate_decreasing_nonnegative_tuples_summing_to(
                    n-i, length-1, min_value, i):
                yield (i, *remainder)


def generate_nonnegative_integer_tuples_summing_to_at_most(n, length):
    """Enumerate all non-negative integer tuples summing to at most n,
    exhausting the search space by varying the first entry fastest,
    and the last entry the slowest.
    """
    assert length >= 0
    if length == 0:
        yield ()
    else:
        for i in range(n+1):
            for remainder in generate_nonnegative_integer_tuples_summing_to_at_most(
                    n-i, length-1):
                yield (*remainder, i)


# backwards compatibility
generate_positive_integer_tuples_below = generate_nonnegative_integer_tuples_below


def _pos_and_neg_adaptor(tuple_iter):
    for tup in tuple_iter:
        nonzero_indices = [i for i in range(len(tup)) if tup[i] != 0]
        for do_neg_tup in generate_nonnegative_integer_tuples_below(
                2, len(nonzero_indices)):
            this_result = list(tup)
            for index, do_neg in enumerate(do_neg_tup):
                if do_neg:
                    this_result[nonzero_indices[index]] *= -1
            yield tuple(this_result)


def generate_all_integer_tuples_below(n, length, least_abs=0):
    return _pos_and_neg_adaptor(generate_nonnegative_integer_tuples_below(
        n, length, least_abs))


def generate_permutations(original):
    """Generate all permutations of the list *original*.

    Nicked from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/252178
    """
    if len(original) <= 1:
        yield original
    else:
        for perm_ in generate_permutations(original[1:]):
            for i in range(len(perm_)+1):
                # nb str[0:1] works in both string and list contexts
                yield perm_[:i] + original[0:1] + perm_[i:]


def generate_unique_permutations(original):
    """Generate all unique permutations of the list *original*.
    """

    had_those = set()

    for perm_ in generate_permutations(original):
        if perm_ not in had_those:
            had_those.add(perm_)
            yield perm_


def enumerate_basic_directions(dimensions):
    coordinate_list = [[0], [1], [-1]]
    return reduce(cartesian_product_sum, [coordinate_list] * dimensions)[1:]

# }}}


# {{{ graph algorithms

from pytools.graph import a_star as a_star_moved


a_star = MovedFunctionDeprecationWrapper(a_star_moved)

# }}}


# {{{ formatting

# {{{ table formatting

class Table:
    """An ASCII table generator.

    .. automethod:: __init__
    .. automethod:: add_row

    .. autoproperty:: nrows
    .. autoproperty:: ncolumns

    .. automethod:: __str__
    .. automethod:: github_markdown
    .. automethod:: csv
    .. automethod:: latex
    .. automethod:: text_without_markup
    """

    def __init__(self, alignments: tuple[str, ...] | None = None) -> None:
        """Create a new :class:`Table`.

        :arg alignments: A :class:`tuple` of alignments of each column:
            ``"l"``, ``"c"``, or ``"r"``, for left, center, and right
            alignment, respectively). Columns which have no alignment specifier
            will use the last specified alignment. For example, with
            ``alignments=("l", "r")``, the third and all following
            columns will use right alignment.
        """

        if alignments is None:
            alignments = ("l",)
        else:
            if any(a not in ("l", "c", "r") for a in alignments):
                raise ValueError(f"alignments are ('l', 'c', 'r'): {alignments}")

            alignments = tuple(alignments)

        self.rows: list[tuple[str, ...]] = []
        self.alignments = alignments

    @property
    def nrows(self) -> int:
        """The number of rows currently in the table."""
        return len(self.rows)

    @property
    def ncolumns(self) -> int:
        """The number of columns currently in the table."""
        return len(self.rows[0])

    def add_row(self, row: tuple[Any, ...]) -> None:
        """Add *row* to the table. Note that all rows must have the same number
        of columns."""
        if self.rows and len(row) != self.ncolumns:
            raise ValueError(
                    f"tried to add a row with {len(row)} columns to "
                    f"a table with {self.ncolumns} columns")

        self.rows.append(tuple(str(i) for i in row))

    def _get_alignments(self) -> tuple[str, ...]:
        # NOTE: If not all alignments were specified, extend alignments with the
        # last alignment specified
        return (
                self.alignments
                + (self.alignments[-1],) * (self.ncolumns - len(self.alignments))
                )[:self.ncolumns]

    def _get_column_widths(self, rows) -> tuple[int, ...]:
        return tuple(
            max(len(row[i]) for row in rows) for i in range(self.ncolumns)
            )

    def __str__(self) -> str:
        """
        Returns a string representation of the table.

        .. doctest ::

            >>> tbl = Table(alignments=['l', 'r', 'l'])
            >>> tbl.add_row([1, '|'])
            >>> tbl.add_row([10, '20||'])
            >>> print(tbl)
            1  |    |
            ---+------
            10 | 20||

        """
        if not self.rows:
            return ""

        alignments = self._get_alignments()
        col_widths = self._get_column_widths(self.rows)

        lines = [" | ".join([
            cell.center(cwidth) if align == "c"
            else cell.ljust(cwidth) if align == "l"
            else cell.rjust(cwidth)
            for cell, cwidth, align in zip(row, col_widths, alignments, strict=True)])
            for row in self.rows]
        lines[1:1] = ["+".join("-" * (cwidth + 1 + (i > 0))
            for i, cwidth in enumerate(col_widths))]

        return "\n".join(lines)

    def github_markdown(self) -> str:
        r"""Returns a string representation of the table formatted as
        `GitHub-Flavored Markdown.
        <https://docs.github.com/en/github/writing-on-github/organizing-information-with-tables>`__

        .. doctest ::

            >>> tbl = Table(alignments=['l', 'r', 'l'])
            >>> tbl.add_row([1, '|'])
            >>> tbl.add_row([10, '20||'])
            >>> print(tbl.github_markdown())
            1  |     \|
            :--|-------:
            10 | 20\|\|

        """
        if not self.rows:
            return ""

        def escape(cell: str) -> str:
            # Pipe symbols ('|') must be replaced
            return cell.replace("|", "\\|")

        rows = [tuple(escape(cell) for cell in row) for row in self.rows]
        alignments = self._get_alignments()
        col_widths = self._get_column_widths(rows)

        lines = [" | ".join([
            cell.center(cwidth) if align == "c"
            else cell.ljust(cwidth) if align == "l"
            else cell.rjust(cwidth)
            for cell, cwidth, align in zip(row, col_widths, alignments, strict=True)])
            for row in rows]
        lines[1:1] = ["|".join(
            (":" + "-" * (cwidth - 1 + (i > 0)) + ":") if align == "c"
            else (":" + "-" * (cwidth + (i > 0))) if align == "l"
            else ("-" * (cwidth + (i > 0)) + ":")
            for i, (cwidth, align) in enumerate(
                zip(col_widths, alignments, strict=True)))]

        return "\n".join(lines)

    def csv(self,
            dialect: str = "excel",
            csv_kwargs: dict[str, Any] | None = None) -> str:
        """Returns a string containing a CSV representation of the table.

        :arg dialect: String passed to :func:`csv.writer`.
        :arg csv_kwargs: Dict of arguments passed to :func:`csv.writer`.

        .. doctest ::

            >>> tbl = Table()
            >>> tbl.add_row([1, ","])
            >>> tbl.add_row([10, 20])
            >>> print(tbl.csv())
            1,","
            10,20
        """

        if not self.rows:
            return ""

        import csv
        import io

        if csv_kwargs is None:
            csv_kwargs = {}

        # Default is "\r\n"
        if "lineterminator" not in csv_kwargs:
            csv_kwargs["lineterminator"] = "\n"

        output = io.StringIO()
        writer = csv.writer(output, dialect, **csv_kwargs)
        writer.writerows(self.rows)

        return output.getvalue().rstrip(csv_kwargs["lineterminator"])

    def latex(self,
            skip_lines: int = 0,
            hline_after: tuple[int, ...] | None = None) -> str:
        r"""Returns a string containing the rows of a LaTeX representation of
        the table.

        :arg skip_lines: number of lines to skip at the start of the table.
        :arg hline_after: list of row indices after which to add an ``hline``
            (the indices must subtract *skip_lines*, if non-zero).

        .. doctest::

            >>> tbl = Table()
            >>> tbl.add_row([0, "skipped"])
            >>> tbl.add_row([1, "apple"])
            >>> tbl.add_row([2, "pear"])
            >>> print(tbl.latex(skip_lines=1))
            1 & apple \\
            2 & pear \\
        """
        if not self.rows:
            return ""

        if hline_after is None:
            hline_after = ()

        lines = []
        for row_nr, row in enumerate(self.rows[skip_lines:]):
            lines.append(fr"{' & '.join(row)} \\")
            if row_nr in hline_after:
                lines.append(r"\hline")

        return "\n".join(lines)

    def text_without_markup(self) -> str:
        """Returns a string representation of the table without markup.

        .. doctest::

            >>> tbl = Table()
            >>> tbl.add_row([0, "orange"])
            >>> tbl.add_row([1111, "apple"])
            >>> tbl.add_row([2, "pear"])
            >>> print(tbl.text_without_markup())
            0    orange
            1111 apple
            2    pear
        """
        if not self.rows:
            return ""

        alignments = self._get_alignments()
        col_widths = self._get_column_widths(self.rows)

        lines = [" ".join([
            cell.center(cwidth) if align == "c"
            else cell.ljust(cwidth) if align == "l"
            else cell.rjust(cwidth)
            for cell, cwidth, align in zip(row, col_widths, alignments, strict=True)])
            for row in self.rows]

        # Remove the extra space added by the last cell
        lines = [line.rstrip() for line in lines]

        return "\n".join(lines)


def merge_tables(*tables: Table,
        skip_columns: tuple[int, ...] | None = None) -> Table:
    """
    :arg skip_columns: a :class:`tuple` of column indices to skip in all the
        tables except the first one.
    """

    if len(tables) == 1:
        return tables[0]

    if any(tables[0].nrows != tbl.nrows for tbl in tables[1:]):
        raise ValueError("tables do not have the same number of rows")

    if isinstance(skip_columns, int):
        skip_columns = (skip_columns,)

    def remove_columns(i, row):
        if i == 0 or skip_columns is None:
            return row
        return tuple(
            entry for i, entry in enumerate(row) if i not in skip_columns
            )

    alignments = sum((
        remove_columns(i, tbl._get_alignments())
        for i, tbl in enumerate(tables)
        ), ())
    result = Table(alignments=alignments)

    for i in range(tables[0].nrows):
        row = []
        for j, tbl in enumerate(tables):
            row.extend(remove_columns(j, tbl.rows[i]))

        result.add_row(tuple(row))

    return result

# }}}


# {{{ histogram formatting

def string_histogram(
        iterable, min_value=None, max_value=None,
        bin_count=20, width=70, bin_starts=None, use_unicode=True):
    if bin_starts is None:
        if min_value is None or max_value is None:
            iterable = list(iterable)
            min_value = min(iterable)
            max_value = max(iterable)

        bin_width = (max_value - min_value)/bin_count
        bin_starts = [min_value+bin_width*i for i in range(bin_count)]

    bins = [0 for i in range(len(bin_starts))]

    from bisect import bisect
    for value in iterable:
        if (max_value is not None and value > max_value) or value < bin_starts[0]:
            from warnings import warn
            warn("string_histogram: out-of-bounds value ignored", stacklevel=2)
        else:
            bin_nr = bisect(bin_starts, value)-1
            try:
                bins[bin_nr] += 1
            except Exception:
                print(value, bin_nr, bin_starts)
                raise

    from math import ceil, floor
    if use_unicode:
        def format_bar(cnt):
            scaled = cnt*width/max_count
            full = floor(scaled)
            eighths = ceil((scaled-full)*8)
            if eighths:
                return full*chr(0x2588) + chr(0x2588+(8-eighths))
            return full*chr(0x2588)
    else:
        def format_bar(cnt):
            return ceil(cnt*width/max_count)*"#"

    max_count = max(bins)
    total_count = sum(bins)
    return "\n".join("{:9g} |{:9d} | {:3.0f} % | {}".format(
        bin_start,
        bin_value,
        bin_value/total_count*100,
        format_bar(bin_value))
        for bin_start, bin_value in zip(bin_starts, bins, strict=True))

# }}}


def word_wrap(text, width, wrap_using="\n"):
    # http://code.activestate.com/recipes/148061-one-liner-word-wrap-function/
    r"""
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (``\n``).
    """
    space_or_break = [" ", wrap_using]
    return reduce(lambda line, word: "{}{}{}".format(
        line,
        space_or_break[(
            len(line) - line.rfind("\n") - 1
            + len(word.split("\n", 1)[0])
            ) >= width],
        word),
        text.split(" ")
        )

# }}}


# {{{ command line interfaces

def _exec_arg(arg, execenv):
    import os
    if os.access(arg, os.F_OK):
        exec(compile(open(arg), arg, "exec"), execenv)
    else:
        exec(compile(arg, "<command line>", "exec"), execenv)


class CPyUserInterface:
    class Parameters(Record):
        pass

    def __init__(self, variables, constants=None, doc=None):
        if constants is None:
            constants = {}
        if doc is None:
            doc = {}
        self.variables = variables
        self.constants = constants
        self.doc = doc

    def show_usage(self, progname):
        print(f"usage: {progname} <FILE-OR-STATEMENTS>")
        print()
        print("FILE-OR-STATEMENTS may either be Python statements of the form")
        print("'variable1 = value1; variable2 = value2' or the name of a file")
        print("containing such statements. Any valid Python code may be used")
        print("on the command line or in a command file. If new variables are")
        print("used, they must start with 'user_' or just '_'.")
        print()
        print("The following variables are recognized:")
        for v in sorted(self.variables):
            print(f"  {v} = {self.variables[v]}")
            if v in self.doc:
                print(f"    {self.doc[v]}")

        print()
        print("The following constants are supplied:")
        for c in sorted(self.constants):
            print(f"  {c} = {self.constants[c]}")
            if c in self.doc:
                print(f"    {self.doc[c]}")

    def gather(self, argv=None):
        if argv is None:
            argv = sys.argv

        if len(argv) == 1 or (
                ("-h" in argv)
                or ("help" in argv)
                or ("-help" in argv)
                or ("--help" in argv)):
            self.show_usage(argv[0])
            sys.exit(2)

        execenv = self.variables.copy()
        execenv.update(self.constants)

        for arg in argv[1:]:
            _exec_arg(arg, execenv)

        # check if the user set invalid keys
        for added_key in (
                set(execenv.keys())
                - set(self.variables.keys())
                - set(self.constants.keys())):
            if not (added_key.startswith("user_") or added_key.startswith("_")):
                raise ValueError(
                        f"invalid setup key: '{added_key}' "
                        "(user variables must start with 'user_' or '_')")

        result = self.Parameters({key: execenv[key] for key in self.variables})
        self.validate(result)
        return result

    def validate(self, setup):
        pass

# }}}


# {{{ debugging

class StderrToStdout:
    def __enter__(self):
        self.stderr_backup = sys.stderr
        sys.stderr = sys.stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.stderr_backup
        del self.stderr_backup


def typedump(val: Any, max_seq: int = 5,
             special_handlers: Mapping[type, Callable] | None = None,
             fully_qualified_name: bool = True) -> str:
    """
    Return a string representation of the type of *val*, recursing into
    iterable objects.

    :arg val: The object for which the type should be returned.
    :arg max_seq: For iterable objects, the maximum number of elements to
        include in the return string. Lower this value if you get a
        :class:`RecursionError`.
    :arg special_handlers: An optional mapping of specific types to special
        handlers.
    :arg fully_qualified_name: Return fully qualified names, that is,
        include module names and use ``__qualname__`` instead of ``__name__``.

    :returns: A string representation of the type of *val*.
    """
    if special_handlers is None:
        special_handlers = {}

    try:
        hdlr = special_handlers[type(val)]
    except KeyError:
        pass
    else:
        return hdlr(val)

    def objname(obj: Any) -> str:
        if type(obj).__module__ == "builtins":
            if fully_qualified_name:
                return type(obj).__qualname__
            return type(obj).__name__

        if fully_qualified_name:
            return type(obj).__module__ + "." + type(obj).__qualname__
        return type(obj).__name__

    # Special handling for 'str' since it is also iterable
    if isinstance(val, str):
        return "str"

    try:
        len(val)
    except TypeError:
        return objname(val)
    else:
        if isinstance(val, dict):
            return "{%s}" % (
                    ", ".join(
                        f"{str(k)!r}: {typedump(v)}"
                        for k, v in val.items()))

        try:
            if len(val) > max_seq:
                t = ",".join(typedump(x, max_seq, special_handlers)
                            for x in val[:max_seq])
                return f"{objname(val)}({t},...)"
            t = ",".join(typedump(x, max_seq, special_handlers)
                        for x in val)
            return f"{objname(val)}({t})"

        except TypeError:
            return objname(val)


def invoke_editor(s, filename="edit.txt", descr="the file"):
    from tempfile import mkdtemp
    tempdir = mkdtemp()

    from os.path import join
    full_name = join(tempdir, filename)

    outf = open(full_name, "w")
    outf.write(str(s))
    outf.close()

    import os
    if "EDITOR" in os.environ:
        from subprocess import Popen
        p = Popen([os.environ["EDITOR"], full_name])
        os.waitpid(p.pid, 0)
    else:
        print("(Set the EDITOR environment variable to be "
                "dropped directly into an editor next time.)")
        input(f"Edit {descr} at {full_name} now, then hit [Enter]:")

    inf = open(full_name)
    result = inf.read()
    inf.close()

    return result

# }}}


# {{{ progress bars

class ProgressBar:
    """
    .. automethod:: draw
    .. automethod:: progress
    .. automethod:: set_progress
    .. automethod:: finished
    .. automethod:: __enter__
    .. automethod:: __exit__
    """
    def __init__(self, descr, total, initial=0, length=40):
        import time
        self.description = descr
        self.total = total
        self.done = initial
        self.length = length
        self.last_squares = -1
        self.start_time = time.time()
        self.last_update_time = self.start_time

        self.speed_meas_start_time = self.start_time
        self.speed_meas_start_done = initial

        self.time_per_step = None

    def draw(self):
        import time

        now = time.time()
        squares = int(self.done/self.total*self.length)
        if squares != self.last_squares or now-self.last_update_time > 0.5:
            if (self.done != self.speed_meas_start_done
                    and now-self.speed_meas_start_time > 3):
                new_time_per_step = (now-self.speed_meas_start_time) \
                        / (self.done-self.speed_meas_start_done)
                if self.time_per_step is not None:
                    self.time_per_step = (new_time_per_step + self.time_per_step)/2
                else:
                    self.time_per_step = new_time_per_step

                self.speed_meas_start_time = now
                self.speed_meas_start_done = self.done

            if self.time_per_step is not None:
                eta_str = "{:7.1f}s ".format(
                        max(0, (self.total-self.done) * self.time_per_step)
                        )
            else:
                eta_str = "?"

            sys.stderr.write("{:<20} [{}] ETA {}\r".format(
                self.description,
                squares*"#"+(self.length-squares)*" ",
                eta_str))

            self.last_squares = squares
            self.last_update_time = now

    def progress(self, steps=1):
        self.set_progress(self.done + steps)

    def set_progress(self, done):
        self.done = done
        self.draw()

    def finished(self):
        self.set_progress(self.total)
        sys.stderr.write("\n")

    def __enter__(self):
        self.draw()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished()

# }}}


# {{{ file system related

def assert_not_a_file(name):
    import os
    if os.access(name, os.F_OK):
        raise OSError(f"file `{name}' already exists")


def add_python_path_relative_to_script(rel_path):
    from os.path import abspath, dirname, join

    script_name = sys.argv[0]
    rel_script_dir = dirname(script_name)

    sys.path.append(abspath(join(rel_script_dir, rel_path)))

# }}}


# {{{ numpy dtype mangling

def common_dtype(dtypes, default=None):
    dtypes = list(dtypes)
    if dtypes:
        return argmax2((dtype, dtype.num) for dtype in dtypes)
    if default is not None:
        return default
    raise ValueError(
            "cannot find common dtype of empty dtype list")


def to_uncomplex_dtype(dtype):
    import numpy as np
    return np.array(1, dtype=dtype).real.dtype.type


def match_precision(dtype, dtype_to_match):
    import numpy

    tgt_is_double = dtype_to_match in [
            numpy.float64, numpy.complex128]

    dtype_is_complex = dtype.kind == "c"
    if dtype_is_complex:
        if tgt_is_double:
            return numpy.dtype(numpy.complex128)
        return numpy.dtype(numpy.complex64)
    if tgt_is_double:
        return numpy.dtype(numpy.float64)
    return numpy.dtype(numpy.float32)

# }}}


# {{{ unique name generation

def generate_unique_names(prefix):
    yield prefix

    try_num = 0
    while True:
        yield f"{prefix}_{try_num}"
        try_num += 1


UNIQUE_NAME_GEN_COUNTER_RE = re.compile(r"^(?P<based_on>\w+)_(?P<counter>\d+)$")


def generate_numbered_unique_names(
        prefix: str, num: int | None = None) -> Iterable[tuple[int, str]]:
    if num is None:
        yield (0, prefix)
        num = 0

    while True:
        name = f"{prefix}_{num}"
        num += 1
        yield (num, name)


generate_unique_possibilities = MovedFunctionDeprecationWrapper(
        generate_unique_names)


class UniqueNameGenerator:
    """
    Class that creates a new :class:`str` on each :meth:`__call__` that is
    unique to the generator.

    .. automethod:: __init__
    .. automethod:: is_name_conflicting
    .. automethod:: add_name
    .. automethod:: add_names
    .. automethod:: __call__
    """
    def __init__(self,
            existing_names: Collection[str] | None = None,
            forced_prefix: str = ""):
        """
        Create a new :class:`UniqueNameGenerator`.

        :arg existing_names: a :class:`set` of existing names that will be
            skipped when generating new names.
        :arg forced_prefix: all generated :class:`str` have this prefix.
        """
        if existing_names is None:
            existing_names = set()

        self.existing_names = set(existing_names)
        self.forced_prefix = forced_prefix
        self.prefix_to_counter: dict[str, int] = {}

    def is_name_conflicting(self, name: str) -> bool:
        """Returns *True* if *name* conflicts with an existing :class:`str`."""
        return name in self.existing_names

    def _name_added(self, name: str) -> None:
        """Callback to alert subclasses when a name has been added.

        .. note::

            This will not get called for the names in the *existing_names*
            argument to :meth:`__init__`.
        """

    def add_name(self, name: str, *, conflicting_ok: bool = False) -> None:
        """
        :arg conflicting_ok: A flag to dictate the behavior when *name* is
            conflicting with the set of existing names. If *True*, a conflict
            is silently passed. If *False*, a :class:`ValueError` is raised on
            encountering a conflict.
        """
        if (not conflicting_ok) and self.is_name_conflicting(name):
            raise ValueError(f"name '{name}' conflicts with existing names")

        if not name.startswith(self.forced_prefix):
            raise ValueError(
                    f"name '{name}' does not start with required prefix "
                    f"'{self.forced_prefix}'")

        self.existing_names.add(name)
        self._name_added(name)

    def add_names(self, names: Iterable[str],
                  *,
                  conflicting_ok: bool = False) -> None:
        """
        :arg conflicting_ok: Plainly passed to :meth:`UniqueNameGenerator.add_name`.
        """
        for name in names:
            self.add_name(name, conflicting_ok=conflicting_ok)

    def __call__(self, based_on: str = "id") -> str:
        """Returns a new unique name."""
        based_on = self.forced_prefix + based_on

        counter = self.prefix_to_counter.get(based_on, None)

        # {{{ try to get counter from based_on if not already present

        if counter is None:
            counter_match = UNIQUE_NAME_GEN_COUNTER_RE.match(based_on)

            if counter_match:
                based_on = counter_match.groupdict()["based_on"]
                counter = int(counter_match.groupdict()["counter"])

        # }}}

        for counter, var_name in generate_numbered_unique_names(based_on, counter):  # noqa: B020,B007
            if not self.is_name_conflicting(var_name):
                break

        self.prefix_to_counter[based_on] = counter

        var_name = intern(var_name)  # pylint: disable=undefined-loop-variable

        self.existing_names.add(var_name)
        self._name_added(var_name)
        return var_name

# }}}


# {{{ recursion limit

class MinRecursionLimit:
    def __init__(self, min_rec_limit):
        self.min_rec_limit = min_rec_limit

    def __enter__(self):
        self.prev_recursion_limit = sys.getrecursionlimit()
        new_limit = max(self.prev_recursion_limit, self.min_rec_limit)
        sys.setrecursionlimit(new_limit)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Deep recursion can produce deeply nested data structures
        # (or long chains of to-be gc'd generators) that cannot be
        # undergo garbage collection with a lower recursion limit.
        #
        # As a result, it doesn't seem possible to lower the recursion limit
        # again after it has been raised without causing reliability issues.
        #
        # See https://gitlab.tiker.net/inducer/sumpy/issues/31 for
        # context.

        pass

# }}}


# {{{ download from web if not present

def download_from_web_if_not_present(url, local_name=None):
    """
    .. versionadded:: 2017.5
    """

    from os.path import basename, exists
    if local_name is None:
        local_name = basename(url)

    if not exists(local_name):
        from urllib.request import Request, urlopen

        from pytools.version import VERSION_TEXT
        req = Request(url, headers={
            "User-Agent": f"pytools/{VERSION_TEXT}"
            })

        with urlopen(req) as inf:
            contents = inf.read()

            with open(local_name, "wb") as outf:
                outf.write(contents)

# }}}


# {{{ find git revisions

def find_git_revision(tree_root):
    # Keep this routine self-contained so that it can be copy-pasted into
    # setup.py.

    from os.path import abspath, exists, join
    tree_root = abspath(tree_root)

    if not exists(join(tree_root, ".git")):
        return None

    # construct minimal environment
    # stolen from
    # https://github.com/numpy/numpy/blob/055ce3e90b50b5f9ef8cf1b8641c42e391f10735/setup.py#L70-L92
    import os
    env = {}
    for k in ["SYSTEMROOT", "PATH", "HOME"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"

    from subprocess import PIPE, STDOUT, Popen
    p = Popen(["git", "rev-parse", "HEAD"], shell=False,
              stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
              cwd=tree_root, env=env)
    (git_rev, _) = p.communicate()

    git_rev = git_rev.decode()

    git_rev = git_rev.rstrip()

    retcode = p.returncode
    assert retcode is not None
    if retcode != 0:
        from warnings import warn
        warn("unable to find git revision", stacklevel=1)
        return None

    return git_rev


def find_module_git_revision(module_file, n_levels_up):
    from os.path import dirname, join
    tree_root = join(*([dirname(module_file), ".." * n_levels_up]))

    return find_git_revision(tree_root)

# }}}


# {{{ create a reshaped view of a numpy array

def reshaped_view(a, newshape):
    """ Create a new view object with shape ``newshape`` without copying the data of
    ``a``. This function is different from ``numpy.reshape`` by raising an
    exception when data copy is necessary.

    :arg a: a :class:`numpy.ndarray` object.
    :arg newshape: an ``int`` object or a tuple of ``int`` objects.

    .. versionadded:: 2018.4
    """

    newview = a.view()
    newview.shape = newshape
    return newview

# }}}


# {{{ process timer

SUPPORTS_PROCESS_TIME = True


class ProcessTimer:
    """Measures elapsed wall time and process time.

    .. automethod:: __enter__
    .. automethod:: __exit__
    .. automethod:: done

    Timing data attributes:

    .. attribute:: wall_elapsed
    .. attribute:: process_elapsed

    .. versionadded:: 2018.5
    """

    def __init__(self):
        import time
        self.perf_counter_start = time.perf_counter()
        self.process_time_start = time.process_time()

        self.wall_elapsed = None
        self.process_elapsed = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()

    def done(self):
        import time
        self.wall_elapsed = time.perf_counter() - self.perf_counter_start
        self.process_elapsed = time.process_time() - self.process_time_start

    def __str__(self):
        cpu = self.process_elapsed / self.wall_elapsed
        return f"{self.wall_elapsed:.2f}s wall {cpu:.2f}x CPU"

    def __repr__(self):
        wall = self.wall_elapsed
        process = self.process_elapsed

        return (f"{type(self).__name__}"
                f"(wall_elapsed={wall!r}s, process_elapsed={process!r}s)")

# }}}


# {{{ log utilities

def _log_start_if_long(logger, sleep_duration, done_indicator,
                       noisy_level, description):
    from time import sleep
    sleep(sleep_duration)

    if not done_indicator[0]:
        logger.log(
                noisy_level, "%s: started %.gs ago",
                description,
                sleep_duration)


class ProcessLogger:
    """Logs the completion time of a (presumably) lengthy process to :mod:`logging`.
    Only uses a high log level if the process took perceptible time.

    .. automethod:: __init__
    .. automethod:: done
    .. automethod:: __enter__
    .. automethod:: __exit__
    """

    default_noisy_level = logging.INFO

    def __init__(
            self, logger, description,
            silent_level=None, noisy_level=None, long_threshold_seconds=None):
        self.logger = logger
        self.description = description
        self.silent_level = silent_level or logging.DEBUG
        self.noisy_level = noisy_level or self.default_noisy_level
        self.long_threshold_seconds = (
                # 0 is a valid value that should override the default
                0.3 if long_threshold_seconds is None else long_threshold_seconds)

        self.logger.log(self.silent_level, "%s: start", self.description)
        self._done_indicator = [False]

        import threading
        self.late_start_log_thread = threading.Thread(
                target=_log_start_if_long,
                args=(logger, 10*self.long_threshold_seconds, self._done_indicator,
                      self.noisy_level, self.description),

                # Do not delay interpreter exit if thread not finished.
                daemon=True)

        # https://github.com/firedrakeproject/firedrake/issues/1422
        # Starting a thread may irrecoverably break various environments,
        # e.g. MPI.
        #
        # Since the late-start logging is an optional 'quality-of-life'
        # feature for interactive use, do not do it unless there is (weak)
        # evidence of interactive use.
        import sys
        if sys.stdin is None:
            # Can happen, e.g., if pudb is controlling the console.
            use_late_start_logging = False
        elif hasattr(sys.stdin, "closed") and not sys.stdin.closed:
            # can query stdin.isatty() only if stdin's open
            use_late_start_logging = sys.stdin.isatty()
        else:
            use_late_start_logging = False

        import os
        if os.environ.get("PYTOOLS_LOG_NO_THREADS", ""):
            use_late_start_logging = False

        if use_late_start_logging:
            try:
                self.late_start_log_thread.start()
            except RuntimeError:
                # https://github.com/firedrakeproject/firedrake/issues/1422
                #
                # Starting a thread may fail in various environments, e.g. MPI.
                # Since the late-start logging is an optional 'quality-of-life'
                # feature for interactive use, tolerate failures of it without
                # warning.
                pass

        self.timer = ProcessTimer()

    def done(
            self, extra_msg=None, *extra_fmt_args):
        self.timer.done()
        self._done_indicator[0] = True

        completion_level = (
                self.noisy_level
                if self.timer.wall_elapsed > self.long_threshold_seconds
                else self.silent_level)

        msg = "%s: completed (%s)"
        fmt_args = [self.description, str(self.timer)]

        if extra_msg:
            msg = f"{msg}: {extra_msg}"
            fmt_args.extend(extra_fmt_args)

        self.logger.log(completion_level, msg, *fmt_args)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()


class DebugProcessLogger(ProcessLogger):
    default_noisy_level = logging.DEBUG


class log_process:  # noqa: N801
    """A decorator that uses :class:`ProcessLogger` to log data about calls
    to the wrapped function.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, logger, description=None, long_threshold_seconds=None):
        self.logger = logger
        self.description = description
        self.long_threshold_seconds = long_threshold_seconds

    def __call__(self, wrapped):
        def wrapper(*args, **kwargs):
            with ProcessLogger(
                    self.logger,
                    self.description or wrapped.__name__,
                    long_threshold_seconds=self.long_threshold_seconds):
                return wrapped(*args, **kwargs)

        from functools import update_wrapper
        new_wrapper = update_wrapper(wrapper, wrapped)

        return new_wrapper

# }}}


# {{{ sorting in natural order

def natorder(item):
    """Return a key for natural order string comparison.

    See :func:`natsorted`.

    .. versionadded:: 2020.1
    """
    import re
    result = []
    for (int_val, string_val) in re.findall(r"(\d+)|(\D+)", item):
        if int_val:
            result.append(int(int_val))
            # Tie-breaker in case of leading zeros in *int_val*.  Longer values
            # compare smaller to preserve order of numbers in decimal notation,
            # e.g., "1.001" < "1.01"
            # (cf. https://github.com/sourcefrog/natsort)
            result.append(-len(int_val))
        else:
            result.append(string_val)
    return result


def natsorted(iterable, key=None, reverse=False):
    """Sort using natural order [1]_, as opposed to lexicographic order.

    Example::

        >>> sorted(["_10", "_1", "_9"]) == ["_1", "_10", "_9"]
        True
        >>> natsorted(["_10", "_1", "_9"]) == ["_1", "_9", "_10"]
        True

    :arg iterable: an iterable to be sorted. It must only have strings, unless
        *key* is specified.
    :arg key: if provided, a key function that returns strings for ordering
        using natural order.
    :arg reverse: if *True*, sorts in descending order.

    :returns: a sorted list

    .. [1] https://en.wikipedia.org/wiki/Natural_sort_order

    .. versionadded:: 2020.1
    """
    if key is None:
        def key(x):
            return x
    return sorted(iterable, key=lambda y: natorder(key(y)), reverse=reverse)

# }}}


# {{{ resolve_name

# https://github.com/python/cpython/commit/1ed61617a4a6632905ad6a0b440cd2cafb8b6414

_DOTTED_WORDS = r"[a-z_]\w*(\.[a-z_]\w*)*"
_NAME_PATTERN = re.compile(f"^({_DOTTED_WORDS})(:({_DOTTED_WORDS})?)?$", re.I)
del _DOTTED_WORDS


def resolve_name(name):
    """A backport of :func:`pkgutil.resolve_name` (added in Python 3.9).

    .. versionadded:: 2021.1.2
    """
    from warnings import warn

    warn("'pytools.resolve_name' is deprecated and will be removed in 2024. "
         "Use 'pkgutil.resolve_name' from the standard library instead.",
         DeprecationWarning, stacklevel=2)

    import pkgutil
    return pkgutil.resolve_name(name)

# }}}


# {{{ unordered_hash

def unordered_hash(hash_instance: Any,
                   iterable: Iterable[Any],
                   hash_constructor: Callable[[], Any] | None = None) -> Any:
    """Using a hash algorithm given by the parameter-less constructor
    *hash_constructor*, return a hash object whose internal state
    depends on the entries of *iterable*, but not their order. If *hash*
    is the instance returned by evaluating ``hash_constructor()``, then
    the each entry *i* of the iterable must permit ``hash.update(i)`` to
    succeed. An example of *hash_constructor* is ``hashlib.sha256``
    from :mod:`hashlib`.  ``hash.digest_size`` must also be defined.
    If *hash_constructor* is not provided, ``hash_instance.name`` is
    used to deduce it.

    :returns: the updated *hash_instance*.

    .. warning::

        The construction used in this function is likely not cryptographically
        secure. Do not use this function in a security-relevant context.

    .. versionadded:: 2021.2
    """

    if hash_constructor is None:
        import hashlib
        from functools import partial
        hash_constructor = partial(hashlib.new, hash_instance.name)

    assert hash_constructor is not None

    h_int = 0
    for i in iterable:
        h_i = hash_constructor()
        h_i.update(i)
        # Using sys.byteorder (for efficiency) here technically makes the
        # hash system-dependent (which it should not be), however the
        # effect of this is undone by the to_bytes conversion below, while
        # left invariant by the intervening XOR operations (which do not
        # mix adjacent bits).
        h_int = h_int ^ int.from_bytes(h_i.digest(), sys.byteorder)

    hash_instance.update(h_int.to_bytes(hash_instance.digest_size, sys.byteorder))
    return hash_instance

# }}}


# {{{ sphere_sample

def sphere_sample_equidistant(npoints_approx: int, r: float = 1.0):
    """Generate points regularly distributed on a sphere
    based on https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf.

    :returns: an :class:`~numpy.ndarray` of shape ``(3, npoints)``, where
        ``npoints`` does not generally equal *npoints_approx*.
    """

    import numpy as np
    points: list[np.ndarray] = []

    count = 0
    a = 4 * np.pi / npoints_approx
    d = a ** 0.5
    M_theta = int(np.ceil(np.pi / d))  # noqa: N806
    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.ceil(2 * np.pi * np.sin(theta) / d_phi))  # noqa: N806
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            points.append(np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
                ]))
            count += 1

    # add poles
    for i in range(3):
        for sign in [-1, +1]:
            pole = np.zeros(3)
            pole[i] = r * sign
            points.append(pole)

    return np.array(points).T.copy()


# NOTE: each tuple contains ``(epsilon, max_npoints)``
_SPHERE_FIBONACCI_OFFSET = (
        (0.33, 24), (1.33, 177), (3.33, 890),
        (10, 11000), (27, 39000), (75, 600000), (214, float("inf")),
        )


def sphere_sample_fibonacci(
        npoints: int, r: float = 1.0, *,
        optimize: str | None = None):
    """Generate points on a sphere based on an offset Fibonacci lattice from [2]_.

    .. [2] http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param optimize: takes the values: *None* to use the standard Fibonacci
        lattice, ``"minimum"`` to minimize the nearest neighbor distances in the
        lattice and ``"average"`` to minimize the average distances in the
        lattice.

    :returns: an :class:`~numpy.ndarray` of shape ``(3, npoints)``.
    """

    import numpy as np
    if optimize is None:
        epsilon = 0.5
    elif optimize == "minimum":
        epsilon, _ = next(o for o in _SPHERE_FIBONACCI_OFFSET if npoints < o[1])
    elif optimize == "average":
        epsilon = 0.36
    else:
        raise ValueError(f"unknown 'optimize' choice: '{optimize}'")

    golden_ratio = (1 + np.sqrt(5)) / 2
    n = np.arange(npoints)

    phi = 2.0 * np.pi * n / golden_ratio
    theta = np.arccos(1.0 - 2.0 * (n + epsilon) / (npoints + 2 * epsilon - 1))

    return np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
        ])

# }}}


# {{{ strtobool

def strtobool(val: str | None, default: bool | None = None) -> bool:
    """Convert a string representation of truth to True or False.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Uppercase versions are
    also accepted. If *default* is None, raises ValueError if *val* is anything
    else. If *val* is None and *default* is not None, returns *default*.
    Based on :func:`distutils.util.strtobool`.

    :param val: Value to convert.
    :param default: Value to return if *val* is None.

    :returns: Truth value of *val*.
    """
    if val is None and default is not None:
        return default

    if val is None:
        raise ValueError(f"invalid truth value '{val}'. "
                          "Valid values are ('y', 'yes', 't', 'true', 'on', '1') "
                          "for 'True' and ('n', 'no', 'f', 'false', 'off', '0') "
                          "for 'False'. Uppercase versions are also accepted.")

    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value '{val}'. "
                      "Valid values are ('y', 'yes', 't', 'true', 'on', '1') "
                      "for 'True' and ('n', 'no', 'f', 'false', 'off', '0') "
                      "for 'False'. Uppercase versions are also accepted.")

# }}}


# {{{ to_identifier

def to_identifier(s: str) -> str:
    """Convert a string to a valid Python identifier, by removing
    non-alphanumeric, non-underscore characters, and prepending an underscore
    if the string starts with a numeric character.

    :param s: The string to convert to an identifier.

    :returns: The converted string.
    """
    if s.isidentifier():
        return s

    s = "".join(c for c in s if c.isalnum() or c == "_")

    if len(s) == 0:
        return "_"

    if s[0].isdigit():
        s = "_" + s

    return s

# }}}


# {{{ unique

def unique(seq: Iterable[T]) -> Collection[T]:
    """Return unique elements in *seq*, removing all duplicates. The internal
    order of the elements is preserved. See also
    :func:`itertools.groupby` (which removes consecutive duplicates)."""
    return dict.fromkeys(seq)


def unique_difference(*args: Iterable[T]) -> Collection[T]:
    r"""Return unique elements that are in the first iterable in *\*args* but not
    in any of the others. The internal order of the elements is preserved."""
    if not args:
        return []

    res = dict.fromkeys(args[0])
    for seq in args[1:]:
        for item in seq:
            if item in res:
                del res[item]

    return res


def unique_intersection(*args: Iterable[T]) -> Collection[T]:
    r"""Return unique elements that are common to all iterables in *\*args*.
    The internal order of the elements is preserved."""
    if not args:
        return []

    res = dict.fromkeys(args[0])
    for seq in args[1:]:
        seq = set(seq)
        res = {item: None for item in res if item in seq}

    return res


def unique_union(*args: Iterable[T]) -> Collection[T]:
    r"""Return unique elements that are in any iterable in *\*args*.
    The internal order of the elements is preserved."""
    if not args:
        return []

    res: dict[T, None] = {}
    for seq in args:
        for item in seq:
            if item not in res:
                res[item] = None

    return res

# }}}


@dataclass_transform(frozen_default=True)
def opt_frozen_dataclass(
            *,
            init: bool = True,
            repr: bool = True,
            eq: bool = True,
            order: bool = False,
            unsafe_hash: bool | None = None,
            match_args: bool = True,
            kw_only: bool = False,
            slots: bool = False,
            # Added in 3.11.
            weakref_slot: bool = False,
         ) -> Callable[[type[T]], type[T]]:
    """Like :func:`dataclasses.dataclass`, but marks the dataclass frozen
    only if :data:`__debug__` is active. Frozen dataclasses have a ~20%
    cost penalty (on creation, from having to call :meth:`object.__setattr__`) that
    this decorator avoids when the interpreter runs with "optimization"
    enabled.

    The resulting dataclass supports hashing, even when it is not actually frozen,
    if *unsafe_hash* is left at the default or set to *True*.

    .. note::

        Python prevents non-frozen dataclasses from inheriting from frozen ones,
        and vice versa. To ensure frozen-ness is applied predictably in all
        scenarios (mainly :data:`__debug__` on and off), it is strongly recommended
        that all dataclasses inheriting from ones with this decorator *also*
        use this decorator. There are no run-time checks to make sure of this.

    .. versionadded:: 2024.1.18
    """
    def map_cls(cls: type[T]) -> type[T]:
        # This ensures that the resulting dataclass is hashable with and without
        # __debug__, unless the user overrides unsafe_hash or provides their own
        # __hash__ method.
        if unsafe_hash is None:
            if (eq
                    and not __debug__
                    and "__hash__" not in cls.__dict__):
                loc_unsafe_hash = True
            else:
                loc_unsafe_hash = False
        else:
            loc_unsafe_hash = unsafe_hash

        dc_extra_kwargs: dict[str, bool] = {}
        if weakref_slot:
            if sys.version_info < (3, 11):
                raise TypeError("weakref_slot is not available before Python 3.11")
            dc_extra_kwargs["weakref_slot"] = weakref_slot

        from dataclasses import dataclass
        return dataclass(
             init=init,
             repr=repr,
             eq=eq,
             order=order,
             unsafe_hash=loc_unsafe_hash,
             frozen=__debug__,
             match_args=match_args,
             kw_only=kw_only,
             slots=slots,
             **dc_extra_kwargs,
        )(cls)

    return map_cls


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

# vim: foldmethod=marker
