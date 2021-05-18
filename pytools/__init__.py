# pylint:  disable=too-many-lines
# (Yes, it has a point!)


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


import re
from functools import reduce, wraps
import operator
import sys
import logging
from typing import (
        Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar)
import builtins


from sys import intern


# These are deprecated and will go away in 2022.
all = builtins.all
any = builtins.any


__doc__ = """
A Collection of Utilities
=========================

Math
----

.. autofunction:: levi_civita
.. autofunction:: perm
.. autofunction:: comb

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

Type Variables Used
-------------------

.. class:: T

    Any type.

.. class:: F

    Any callable.
"""

# {{{ type variables

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# }}}


# {{{ code maintenance

class MovedFunctionDeprecationWrapper:
    def __init__(self, f, deadline=None):
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
        newkey: Optional[str] = None, *,
        deadline: Optional[str] = None):
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

# }}}


# {{{ math --------------------------------------------------------------------

def delta(x, y):
    if x == y:
        return 1
    else:
        return 0


def levi_civita(tup):
    """Compute an entry of the Levi-Civita tensor for the indices *tuple*."""
    if len(tup) == 2:
        i, j = tup
        return j-i
    if len(tup) == 3:
        i, j, k = tup
        return (j-i)*(k-i)*(k-j)/2
    else:
        raise NotImplementedError


def factorial(n):
    from operator import mul
    assert n == int(n)
    return reduce(mul, (i for i in range(1, n+1)), 1)


def perm(n, k):
    """Return P(n, k), the number of permutations of length k drawn from n
    choices.
    """
    result = 1
    assert k > 0
    while k:
        result *= n
        n -= 1
        k -= 1

    return result


def comb(n, k):
    """Return C(n, k), the number of combinations (subsets)
    of length k drawn from n choices.
    """
    return perm(n, k)//factorial(k)


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

    __slots__: List[str] = []

    def __init__(self, valuedict=None, exclude=None, **kwargs):
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
                ", ".join("{}={!r}".format(fld, getattr(self, fld))
                    for fld in self.__class__.fields
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
    __slots__: List[str] = []

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
        if self._cached_hash is None:
            self._cached_hash = hash(
                (type(self),) + tuple(getattr(self, field)
                    for field in self.__class__.fields))

        return self._cached_hash


class ImmutableRecord(ImmutableRecordWithoutPickling, Record):
    pass

# }}}


class Reference:
    def __init__(self, value):
        self.value = value

    def get(self):
        from warnings import warn
        warn("Reference.get() is deprecated -- use ref.value instead")
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


# {{{ dependent dictionary ----------------------------------------------------

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
            self[key]  # pylint: disable=pointless-statement
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

    def genuineKeys(self):  # noqa
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
        raise ValueError("empty iterable passed to 'one()'")

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
        raise ValueError("empty iterable passed to 'single_valued()'")

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
        raise ValueError("empty iterable passed to 'single_valued()'")

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

    default_key_func: Optional[Callable[..., Any]]

    if use_kw:
        def default_key_func(*inner_args, **inner_kwargs):  # noqa pylint:disable=function-redefined
            return inner_args, frozenset(inner_kwargs.items())
    else:
        default_key_func = None

    key_func = kwargs.pop("key", default_key_func)

    if kwargs:
        raise TypeError(
            "memoize received unexpected keyword arguments: %s"
            % ", ".join(list(kwargs.keys())))

    if key_func is not None:
        def _decorator(func):
            def wrapper(*args, **kwargs):
                key = key_func(*args, **kwargs)
                try:
                    return func._memoize_dic[key]  # noqa: E501 # pylint: disable=protected-access
                except AttributeError:
                    # _memoize_dic doesn't exist yet.
                    result = func(*args, **kwargs)
                    func._memoize_dic = {key: result}  # noqa: E501 # pylint: disable=protected-access
                    return result
                except KeyError:
                    result = func(*args, **kwargs)
                    func._memoize_dic[key] = result  # noqa: E501 # pylint: disable=protected-access
                    return result

            from functools import update_wrapper
            update_wrapper(wrapper, func)
            return wrapper

    else:
        def _decorator(func):
            def wrapper(*args):
                try:
                    return func._memoize_dic[args]  # noqa: E501 # pylint: disable=protected-access
                except AttributeError:
                    # _memoize_dic doesn't exist yet.
                    result = func(*args)
                    func._memoize_dic = {args: result}  # noqa: E501 # pylint:disable=protected-access
                    return result
                except KeyError:
                    result = func(*args)
                    func._memoize_dic[args] = result  # noqa: E501 # pylint: disable=protected-access
                    return result

            from functools import update_wrapper
            update_wrapper(wrapper, func)
            return wrapper

    if not args:
        return _decorator  # type: ignore
    if callable(args[0]) and len(args) == 1:
        return _decorator(args[0])
    raise TypeError(
        "memoize received unexpected position arguments: %s" % args)


FunctionValueCache = memoize


class _HasKwargs:
    pass


def memoize_on_first_arg(function, cache_dict_name=None):
    """Like :func:`memoize_method`, but for functions that take the object
    in which do memoization information is stored as first argument.

    Supports cache deletion via ``function_name.clear_cache(self)``.
    """

    if cache_dict_name is None:
        cache_dict_name = intern(
                f"_memoize_dic_{function.__module__}{function.__name__}"
                )

    def wrapper(obj, *args, **kwargs):
        if kwargs:
            key = (_HasKwargs, frozenset(kwargs.items())) + args
        else:
            key = args

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
        else:
            getattr(obj, cache_dict_name)[key] = result
            return result

    def clear_cache(obj):
        object.__delattr__(obj, cache_dict_name)

    from functools import update_wrapper
    new_wrapper = update_wrapper(wrapper, function)
    new_wrapper.clear_cache = clear_cache

    return new_wrapper


def memoize_method(method: F) -> F:
    """Supports cache deletion via ``method_name.clear_cache(self)``.

    .. versionchanged:: 2021.2

        Can memoize methods on classes that do not allow setting attributes
        (e.g. by overwritting ``__setattr__``), e.g. frozen :mod:`dataclasses`.
    """

    return memoize_on_first_arg(method,
            intern(f"_memoize_dic_{method.__name__}"))


class keyed_memoize_on_first_arg:  # noqa: N801
    """Like :func:`memoize_method`, but for functions that take the object
    in which memoization information is stored as first argument.

    Supports cache deletion via ``function_name.clear_cache(self)``.

    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.
    :arg cache_dict_name: The name of the `dict` attribute in the instance
        used to hold the cache.

    .. versionadded :: 2020.3
    """

    def __init__(self, key, cache_dict_name=None):
        self.key = key
        self.cache_dict_name = cache_dict_name

    def _default_cache_dict_name(self, function):
        return intern(f"_memoize_dic_{function.__module__}{function.__name__}")

    def __call__(self, function):
        cache_dict_name = self.cache_dict_name
        key = self.key

        if cache_dict_name is None:
            cache_dict_name = self._default_cache_dict_name(function)

        def wrapper(obj, *args, **kwargs):
            cache_key = key(*args, **kwargs)

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
        new_wrapper.clear_cache = clear_cache

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
        (e.g. by overwritting ``__setattr__``), e.g. frozen :mod:`dataclasses`.
    """
    def _default_cache_dict_name(self, function):
        return intern(f"_memoize_dic_{function.__name__}")


def memoize_method_with_uncached(uncached_args=None, uncached_kwargs=None):
    """Supports cache deletion via ``method_name.clear_cache(self)``.

    :arg uncached_args: a list of argument numbers
        (0-based, not counting 'self' argument)
    """
    from warnings import warn
    warn("memoize_method_with_uncached is deprecated and will go away in 2022. "
            "Use memoize_method_with_key instead",
            DeprecationWarning,
            stacklevel=2)

    if uncached_args is None:
        uncached_args = []
    if uncached_kwargs is None:
        uncached_kwargs = set()

    # delete starting from the end
    uncached_args = sorted(uncached_args, reverse=True)
    uncached_kwargs = list(uncached_kwargs)

    def parametrized_decorator(method):
        cache_dict_name = intern(f"_memoize_dic_{method.__name__}")

        def wrapper(self, *args, **kwargs):
            cache_args = list(args)
            cache_kwargs = kwargs.copy()

            for i in uncached_args:
                if i < len(cache_args):
                    cache_args.pop(i)

            cache_args = tuple(cache_args)

            if kwargs:
                for name in uncached_kwargs:
                    cache_kwargs.pop(name, None)

                key = (
                        (_HasKwargs, frozenset(cache_kwargs.items()))
                        + cache_args)
            else:
                key = cache_args

            try:
                return getattr(self, cache_dict_name)[key]
            except AttributeError:
                result = method(self, *args, **kwargs)
                object.__setattr__(self, cache_dict_name, {key: result})
                return result
            except KeyError:
                result = method(self, *args, **kwargs)
                getattr(self, cache_dict_name)[key] = result
                return result

        def clear_cache(self):
            object.__delattr__(self, cache_dict_name)

        if sys.version_info >= (2, 5):
            from functools import update_wrapper
            new_wrapper = update_wrapper(wrapper, method)
            new_wrapper.clear_cache = clear_cache

        return new_wrapper

    return parametrized_decorator


class memoize_in:  # noqa
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
        setting attributes (e.g. by overwritting ``__setattr__``),
        e.g. frozen :mod:`dataclasses`.
    """

    def __init__(self, container, identifier):
        try:
            memoize_in_dict = container._pytools_memoize_in_dict
        except AttributeError:
            memoize_in_dict = {}
            object.__setattr__(container, "_pytools_memoize_in_dict",
                    memoize_in_dict)

        self.cache_dict = memoize_in_dict.setdefault(identifier, {})

    def __call__(self, inner):
        @wraps(inner)
        def new_inner(*args):
            try:
                return self.cache_dict[args]
            except KeyError:
                result = inner(*args)
                self.cache_dict[args] = result
                return result

        return new_inner


class keyed_memoize_in:  # noqa
    """Like :class:`memoize_in`, but additionally uses a function *key* to
    compute the key under which the function result is memoized.

    :arg key: A function receiving the same arguments as the decorated function
        which computes and returns the cache key.

    .. versionadded :: 2021.2.1
    """

    def __init__(self, container, identifier, key):
        try:
            memoize_in_dict = container._pytools_keyed_memoize_in_dict
        except AttributeError:
            memoize_in_dict = {}
            object.__setattr__(container, "_pytools_keyed_memoize_in_dict",
                    memoize_in_dict)

        self.cache_dict = memoize_in_dict.setdefault(identifier, {})
        self.key = key

    def __call__(self, inner):
        @wraps(inner)
        def new_inner(*args):
            key = self.key(*args)
            try:
                return self.cache_dict[key]
            except KeyError:
                result = inner(*args)
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
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])


def negate_tuple(t1):
    return tuple([-t1v for t1v in t1])


def shift(vec, dist):
    """Return a copy of C{vec} shifted by C{dist}.

    @postcondition: C{shift(a, i)[j] == a[(i+j) % len(a)]}
    """

    result = vec[:]

    N = len(vec)  # noqa
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
    for c, v in zip(coefficients[1:], vectors[1:]):
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
                    "non-reversible mapping, duplicate key '%s'" % value)
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
                        "everywhere, up to %g" % fail_bound)

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
        raise ValueError("argmin of empty iterable")

    for arg, item in it:
        if item < current_min:
            current_argmin = arg
            current_min = item

    if return_value:
        return current_argmin, current_min
    else:
        return current_argmin


def argmax2(iterable, return_value=False):
    it = iter(iterable)
    try:
        current_argmax, current_max = next(it)
    except StopIteration:
        raise ValueError("argmax of empty iterable")

    for arg, item in it:
        if item > current_max:
            current_argmax = arg
            current_max = item

    if return_value:
        return current_argmax, current_max
    else:
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
            yield prod + (i,)


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
        raise ValueError("empty average")

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
            else:
                return self.m2/self.n
        else:
            if self.n <= 1:
                return None
            else:
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
        for i in range(0, shape[0]):
            yield (i,)
    else:
        remainder = shape[1:]
        for i in range(0, shape[0]):
            for rest in indices_in_shape(remainder):
                yield (i,)+rest


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
            #print "MX", n, max_value
            yield (n,)
        else:
            return
    else:
        if max_value is None or n < max_value:
            max_value = n

        for i in range(min_value, max_value+1):
            #print "SIG", sig, i
            for remainder in generate_decreasing_nonnegative_tuples_summing_to(
                    n-i, length-1, min_value, i):
                yield (i,) + remainder


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
                yield remainder + (i,)


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
                #nb str[0:1] works in both string and list contexts
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


# {{{ index mangling

def get_read_from_map_from_permutation(original, permuted):
    """With a permutation given by *original* and *permuted*,
    generate a list *rfm* of indices such that
    ``permuted[i] == original[rfm[i]]``.

    Requires that the permutation can be inferred from
    *original* and *permuted*.

    .. doctest ::

        >>> for p1 in generate_permutations(list(range(5))):
        ...     for p2 in generate_permutations(list(range(5))):
        ...         rfm = get_read_from_map_from_permutation(p1, p2)
        ...         p2a = [p1[rfm[i]] for i in range(len(p1))]
        ...         assert p2 == p2a
    """
    from warnings import warn
    warn("get_read_from_map_from_permutation is deprecated and will be "
            "removed in 2019", DeprecationWarning, stacklevel=2)

    assert len(original) == len(permuted)
    where_in_original = {
            original[i]: i for i in range(len(original))}
    assert len(where_in_original) == len(original)
    return tuple(where_in_original[pi] for pi in permuted)


def get_write_to_map_from_permutation(original, permuted):
    """With a permutation given by *original* and *permuted*,
    generate a list *wtm* of indices such that
    ``permuted[wtm[i]] == original[i]``.

    Requires that the permutation can be inferred from
    *original* and *permuted*.

    .. doctest ::

        >>> for p1 in generate_permutations(list(range(5))):
        ...     for p2 in generate_permutations(list(range(5))):
        ...         wtm = get_write_to_map_from_permutation(p1, p2)
        ...         p2a = [0] * len(p2)
        ...         for i, oi in enumerate(p1):
        ...             p2a[wtm[i]] = oi
        ...         assert p2 == p2a
    """
    from warnings import warn
    warn("get_write_to_map_from_permutation is deprecated and will be "
            "removed in 2019", DeprecationWarning, stacklevel=2)

    assert len(original) == len(permuted)

    where_in_permuted = {
            permuted[i]: i for i in range(len(permuted))}

    assert len(where_in_permuted) == len(permuted)
    return tuple(where_in_permuted[oi] for oi in original)

# }}}


# {{{ graph algorithms

from pytools.graph import a_star as a_star_moved
a_star = MovedFunctionDeprecationWrapper(a_star_moved)

# }}}


# {{{ formatting

# {{{ table formatting

class Table:
    """An ASCII table generator.

    :arg alignments: List of alignments of each column ('l', 'c', or 'r',
        for left, center, and right alignment, respectively). Columns which
        have no alignment specifier will use the last specified alignment. For
        example, with `alignments=['l', 'r']`, the third and all following
        columns will use 'r' alignment.

    .. automethod:: add_row
    .. automethod:: __str__
    .. automethod:: latex
    .. automethod:: github_markdown
    """

    def __init__(self, alignments=None):
        self.rows = []
        if alignments is not None:
            self.alignments = alignments
        else:
            self.alignments = ["l"]

    def add_row(self, row):
        self.rows.append([str(i) for i in row])

    def __str__(self):
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

        columns = len(self.rows[0])
        col_widths = [max(len(row[i]) for row in self.rows)
                      for i in range(columns)]

        alignments = self.alignments
        # If not all alignments were specified, extend alignments with the
        # last alignment specified:
        alignments += self.alignments[-1] * (columns - len(self.alignments))

        lines = [" | ".join([
            cell.center(col_width) if align == "c"
            else cell.ljust(col_width) if align == "l"
            else cell.rjust(col_width)
            for cell, col_width, align in zip(row, col_widths, alignments)])
            for row in self.rows]
        lines[1:1] = ["+".join("-" * (col_width + 1 + (i > 0))
            for i, col_width in enumerate(col_widths))]

        return "\n".join(lines)

    def github_markdown(self):
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

        """  # noqa: W605
        # Pipe symbols ('|') must be replaced
        rows = [[w.replace("|", "\\|") for w in r] for r in self.rows]

        columns = len(rows[0])
        col_widths = [max(len(row[i]) for row in rows)
                      for i in range(columns)]

        alignments = self.alignments
        # If not all alignments were specified, extend alignments with the
        # last alignment specified:
        alignments += self.alignments[-1] * (columns - len(self.alignments))

        lines = [" | ".join([
            cell.center(col_width) if align == "c"
            else cell.ljust(col_width) if align == "l"
            else cell.rjust(col_width)
            for cell, col_width, align in zip(row, col_widths, alignments)])
            for row in rows]
        lines[1:1] = ["|".join(
            ":" + "-" * (col_width - 1 + (i > 0)) + ":" if align == "c"
            else ":" + "-" * (col_width + (i > 0)) if align == "l"
            else "-" * (col_width + (i > 0)) + ":"
            for i, (col_width, align) in enumerate(zip(col_widths, alignments)))]

        return "\n".join(lines)

    def csv(self, dialect="excel", csv_kwargs=None):
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

    def latex(self, skip_lines=0, hline_after=None):
        if hline_after is None:
            hline_after = []
        lines = []
        for row_nr, row in list(enumerate(self.rows))[skip_lines:]:
            lines.append(" & ".join(row)+r" \\")
            if row_nr in hline_after:
                lines.append(r"\hline")

        return "\n".join(lines)

# }}}


# {{{ histogram formatting

def string_histogram(  # pylint: disable=too-many-arguments,too-many-locals
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
        if max_value is not None and value > max_value or value < bin_starts[0]:
            from warnings import warn
            warn("string_histogram: out-of-bounds value ignored")
        else:
            bin_nr = bisect(bin_starts, value)-1
            try:
                bins[bin_nr] += 1
            except Exception:
                print(value, bin_nr, bin_starts)
                raise

    from math import floor, ceil
    if use_unicode:
        def format_bar(cnt):
            scaled = cnt*width/max_count
            full = int(floor(scaled))
            eighths = int(ceil((scaled-full)*8))
            if eighths:
                return full*chr(0x2588) + chr(0x2588+(8-eighths))
            else:
                return full*chr(0x2588)
    else:
        def format_bar(cnt):
            return int(ceil(cnt*width/max_count))*"#"

    max_count = max(bins)
    total_count = sum(bins)
    return "\n".join("%9g |%9d | %3.0f %% | %s" % (
        bin_start,
        bin_value,
        bin_value/total_count*100,
        format_bar(bin_value))
        for bin_start, bin_value in zip(bin_starts, bins))

# }}}


def word_wrap(text, width, wrap_using="\n"):
    # http://code.activestate.com/recipes/148061-one-liner-word-wrap-function/
    r"""
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (``\n``).
    """
    space_or_break = [" ", wrap_using]
    return reduce(lambda line, word: "%s%s%s" %
            (line,
                space_or_break[(len(line)-line.rfind("\n")-1
                    + len(word.split("\n", 1)[0])
                    >= width)],
                    word),
            text.split(" ")
            )

# }}}


# {{{ command line interfaces -------------------------------------------------

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
        print("usage: %s <FILE-OR-STATEMENTS>" % progname)
        print()
        print("FILE-OR-STATEMENTS may either be Python statements of the form")
        print("'variable1 = value1; variable2 = value2' or the name of a file")
        print("containing such statements. Any valid Python code may be used")
        print("on the command line or in a command file. If new variables are")
        print("used, they must start with 'user_' or just '_'.")
        print()
        print("The following variables are recognized:")
        for v in sorted(self.variables):
            print("  {} = {}".format(v, self.variables[v]))
            if v in self.doc:
                print("    %s" % self.doc[v])

        print()
        print("The following constants are supplied:")
        for c in sorted(self.constants):
            print("  {} = {}".format(c, self.constants[c]))
            if c in self.doc:
                print("    %s" % self.doc[c])

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
                        "invalid setup key: '%s' "
                        "(user variables must start with 'user_' or '_')"
                        % added_key)

        result = self.Parameters({key: execenv[key] for key in self.variables})
        self.validate(result)
        return result

    def validate(self, setup):
        pass

# }}}


# {{{ debugging

class StderrToStdout:
    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.stderr_backup = sys.stderr
        sys.stderr = sys.stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.stderr_backup
        del self.stderr_backup


def typedump(val, max_seq=5, special_handlers=None):
    if special_handlers is None:
        special_handlers = {}

    try:
        hdlr = special_handlers[type(val)]
    except KeyError:
        pass
    else:
        return hdlr(val)

    try:
        len(val)
    except TypeError:
        return type(val).__name__
    else:
        if isinstance(val, dict):
            return "{%s}" % (
                    ", ".join(
                        "{!r}: {}".format(str(k), typedump(v))
                        for k, v in val.items()))

        try:
            if len(val) > max_seq:
                return "{}({},...)".format(
                        type(val).__name__,
                        ",".join(typedump(x, max_seq, special_handlers)
                            for x in val[:max_seq]))
            else:
                return "{}({})".format(
                        type(val).__name__,
                        ",".join(typedump(x, max_seq, special_handlers)
                            for x in val))
        except TypeError:
            return val.__class__.__name__


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
        input("Edit %s at %s now, then hit [Enter]:"
                % (descr, full_name))

    inf = open(full_name)
    result = inf.read()
    inf.close()

    return result

# }}}


# {{{ progress bars

class ProgressBar:  # pylint: disable=too-many-instance-attributes
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
                eta_str = "%7.1fs " % max(
                        0, (self.total-self.done) * self.time_per_step)
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
        raise OSError("file `%s' already exists" % name)


def add_python_path_relative_to_script(rel_path):
    from os.path import dirname, join, abspath

    script_name = sys.argv[0]
    rel_script_dir = dirname(script_name)

    sys.path.append(abspath(join(rel_script_dir, rel_path)))

# }}}


# {{{ numpy dtype mangling

def common_dtype(dtypes, default=None):
    dtypes = list(dtypes)
    if dtypes:
        return argmax2((dtype, dtype.num) for dtype in dtypes)
    else:
        if default is not None:
            return default
        else:
            raise ValueError(
                    "cannot find common dtype of empty dtype list")


def to_uncomplex_dtype(dtype):
    import numpy
    if dtype == numpy.complex64:
        return numpy.float32
    elif dtype == numpy.complex128:
        return numpy.float64
    if dtype == numpy.float32:
        return numpy.float32
    elif dtype == numpy.float64:
        return numpy.float64
    else:
        raise TypeError("unrecgonized dtype '%s'" % dtype)


def match_precision(dtype, dtype_to_match):
    import numpy

    tgt_is_double = dtype_to_match in [
            numpy.float64, numpy.complex128]

    dtype_is_complex = dtype.kind == "c"
    if dtype_is_complex:
        if tgt_is_double:
            return numpy.dtype(numpy.complex128)
        else:
            return numpy.dtype(numpy.complex64)
    else:
        if tgt_is_double:
            return numpy.dtype(numpy.float64)
        else:
            return numpy.dtype(numpy.float32)

# }}}


# {{{ unique name generation

def generate_unique_names(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1


UNIQUE_NAME_GEN_COUNTER_RE = re.compile(r"^(?P<based_on>\w+)_(?P<counter>\d+)$")


def generate_numbered_unique_names(
        prefix: str, num: Optional[int] = None) -> Iterable[Tuple[int, str]]:
    if num is None:
        yield (0, prefix)
        num = 0

    while True:
        name = "%s_%d" % (prefix, num)
        num += 1
        yield (num, name)


generate_unique_possibilities = MovedFunctionDeprecationWrapper(
        generate_unique_names)


class UniqueNameGenerator:
    """
    .. automethod:: is_name_conflicting
    .. automethod:: add_name
    .. automethod:: add_names
    .. automethod:: __call__
    """
    def __init__(self,
            existing_names: Optional[Set[str]] = None,
            forced_prefix: str = ""):
        if existing_names is None:
            existing_names = set()

        self.existing_names = existing_names.copy()
        self.forced_prefix = forced_prefix
        self.prefix_to_counter: Dict[str, int] = {}

    def is_name_conflicting(self, name: str) -> bool:
        return name in self.existing_names

    def _name_added(self, name: str) -> None:
        """Callback to alert subclasses when a name has been added.

        .. note::

            This will not get called for the names in the *existing_names*
            argument to :meth:`__init__`.
        """
        pass

    def add_name(self, name: str) -> None:
        if self.is_name_conflicting(name):
            raise ValueError("name '%s' conflicts with existing names" % name)
        if not name.startswith(self.forced_prefix):
            raise ValueError("name '%s' does not start with required prefix '%s'"
                             % (name, self.forced_prefix))

        self.existing_names.add(name)
        self._name_added(name)

    def add_names(self, names: Iterable[str]) -> None:
        for name in names:
            self.add_name(name)

    def __call__(self, based_on: str = "id") -> str:
        based_on = self.forced_prefix + based_on

        counter = self.prefix_to_counter.get(based_on, None)

        # {{{ try to get counter from based_on if not already present

        if counter is None:
            counter_match = UNIQUE_NAME_GEN_COUNTER_RE.match(based_on)

            if counter_match:
                based_on = counter_match.groupdict()["based_on"]
                counter = int(counter_match.groupdict()["counter"])

        # }}}

        for counter, var_name in generate_numbered_unique_names(based_on, counter):  # noqa: B007,E501
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
        # pylint: disable=attribute-defined-outside-init

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
        from pytools.version import VERSION_TEXT
        from urllib.request import Request, urlopen
        req = Request(url, headers={
            "User-Agent": f"pytools/{VERSION_TEXT}"
            })

        with urlopen(req) as inf:
            contents = inf.read()

            with open(local_name, "wb") as outf:
                outf.write(contents)

# }}}


# {{{ find git revisions

def find_git_revision(tree_root):  # pylint: disable=too-many-locals
    # Keep this routine self-contained so that it can be copy-pasted into
    # setup.py.

    from os.path import join, exists, abspath
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

    from subprocess import Popen, PIPE, STDOUT
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
        warn("unable to find git revision")
        return None

    return git_rev


def find_module_git_revision(module_file, n_levels_up):
    from os.path import dirname, join
    tree_root = join(*([dirname(module_file)] + [".." * n_levels_up]))

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

class ProcessLogger:  # pylint: disable=too-many-instance-attributes
    """Logs the completion time of a (presumably) lengthy process to :mod:`logging`.
    Only uses a high log level if the process took perceptible time.

    .. automethod:: __init__
    .. automethod:: done
    .. automethod:: __enter__
    .. automethod:: __exit__
    """

    default_noisy_level = logging.INFO

    def __init__(  # pylint: disable=too-many-arguments
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
        self.is_done = False

        import threading
        self.late_start_log_thread = threading.Thread(target=self._log_start_if_long)

        # Do not delay interpreter exit if thread not finished.
        self.late_start_log_thread.daemon = True

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
        else:
            if hasattr(sys.stdin, "closed") and not sys.stdin.closed:
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

    def _log_start_if_long(self):
        from time import sleep

        sleep_duration = 10*self.long_threshold_seconds
        sleep(sleep_duration)

        if not self.is_done:
            self.logger.log(
                    self.noisy_level, "%s: started %.gs ago",
                    self.description,
                    sleep_duration)

    def done(  # pylint: disable=keyword-arg-before-vararg
            self, extra_msg=None, *extra_fmt_args):
        self.timer.done()
        self.is_done = True

        completion_level = (
                self.noisy_level
                if self.timer.wall_elapsed > self.long_threshold_seconds
                else self.silent_level)

        msg = "%s: completed (%s)"
        fmt_args = [self.description, str(self.timer)]

        if extra_msg:
            msg += ": " + extra_msg
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
    """

    def __init__(self, logger, description=None):
        self.logger = logger
        self.description = description

    def __call__(self, wrapped):
        def wrapper(*args, **kwargs):
            with ProcessLogger(
                    self.logger,
                    self.description or wrapped.__name__):
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
        key = lambda x: x
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
    # Delete the tail of the function and deprecate this once we require Python 3.9.
    if sys.version_info >= (3, 9):
        # use the official version
        import pkgutil
        return pkgutil.resolve_name(name)  # pylint: disable=no-member

    import importlib

    m = _NAME_PATTERN.match(name)
    if not m:
        raise ValueError(f"invalid format: {name!r}")
    groups = m.groups()
    if groups[2]:
        # there is a colon - a one-step import is all that's needed
        mod = importlib.import_module(groups[0])
        parts = groups[3].split(".") if groups[3] else []
    else:
        # no colon - have to iterate to find the package boundary
        parts = name.split(".")
        modname = parts.pop(0)
        # first part *must* be a module/package.
        mod = importlib.import_module(modname)
        while parts:
            p = parts[0]
            s = f"{modname}.{p}"
            try:
                mod = importlib.import_module(s)
                parts.pop(0)
                modname = s
            except ImportError:
                break
    # if we reach this point, mod is the module, already imported, and
    # parts is the list of parts in the object hierarchy to be traversed, or
    # an empty list if just the module is wanted.
    result = mod
    for p in parts:
        result = getattr(result, p)
    return result

# }}}


# {{{ unordered_hash

def unordered_hash(hash_instance, iterable, hash_constructor=None):
    """Using a hash algorithm given by the parameter-less constructor
    *hash_constructor*, return a hash object whose internal state
    depends on the entries of *iterable*, but not their order. If *hash*
    is the instance returned by evaluating ``hash_constructor()``, then
    the each entry *i* of the iterable must permit ``hash.upate(i)`` to
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
        from functools import partial
        import hashlib
        hash_constructor = partial(hashlib.new, hash_instance.name)

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


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

# vim: foldmethod=marker
