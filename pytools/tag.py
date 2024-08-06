"""
Tag Interface
---------------
.. ``normalize_tags`` undocumented for now. (Not ready to commit.)

.. autofunction:: check_tag_uniqueness
.. autoclass:: Taggable
.. autoclass:: Tag
.. autoclass:: UniqueTag

Supporting Functionality
------------------------

.. autoclass:: DottedName
.. autoclass:: NonUniqueTagError


Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: TagT

    A type variable with lower bound :class:`Tag`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    FrozenSet,
    Iterable,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from warnings import warn


if TYPE_CHECKING:
    # NOTE: mypy seems to be confused by the `try.. except` below when called with
    #   python -m mypy --python-version 3.8 ...
    # see https://github.com/python/mypy/issues/14220
    from typing_extensions import Self, dataclass_transform
else:
    try:
        from typing import Self, dataclass_transform
    except ImportError:
        from typing_extensions import Self, dataclass_transform

from pytools import memoize, memoize_method


__copyright__ = """
Copyright (C) 2020 Andreas Klöckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
Copyright (C) 2020 Nicholas Christensen
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


#  {{{ dotted name

class DottedName:
    """
    .. attribute:: name_parts

        A tuple of strings, each of which is a valid
        Python identifier. No name part may start with
        a double underscore.

    The name (at least morally) exists in the
    name space defined by the Python module system.
    It need not necessarily identify an importable
    object.

    .. automethod:: from_class
    """

    def __init__(self, name_parts: Tuple[str, ...]) -> None:
        if len(name_parts) == 0:
            raise ValueError("empty name parts")

        for p in name_parts:
            if not p.isidentifier():
                raise ValueError(f"{p} is not a Python identifier")

        self.name_parts = name_parts

    @classmethod
    def from_class(cls, argcls: Any) -> DottedName:
        name_parts = tuple(
                [str(part) for part in argcls.__module__.split(".")]
                + [str(argcls.__name__)])
        if not all(not npart.startswith("__") for npart in name_parts):
            raise ValueError(f"some name parts of {'.'.join(name_parts)} "
                             "start with double underscores")
        return cls(name_parts)

    def __repr__(self) -> str:
        return self.__class__.__name__ + repr(self.name_parts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DottedName):
            return self.name_parts == other.name_parts
        else:
            return False


# }}}


# {{{ tag

T = TypeVar("T")


@dataclass_transform(eq_default=True, frozen_default=True)
def tag_dataclass(cls: type[T]) -> type[T]:
    return dataclass(init=True, frozen=True, eq=True, repr=True)(cls)


@tag_dataclass
class Tag:
    """
    Generic metadata, applied to, among other things,
    pytato Arrays.

    .. attribute:: tag_name

        A fully qualified :class:`DottedName` that reflects
        the class name of the tag.

    Instances of this type must be immutable, hashable,
    picklable, and have a reasonably concise :meth:`__repr__`
    of the form ``dotted.name(attr1=value1, attr2=value2)``.
    Positional arguments are not allowed.

   .. automethod:: __repr__
    """

    @property
    def tag_name(self) -> DottedName:
        return DottedName.from_class(type(self))

# }}}


# {{{ unique tag

@tag_dataclass
class UniqueTag(Tag):
    """
    A superclass for tags that are unique on each :class:`Taggable`.

    Each instance of :class:`Taggable` may have no more than one
    instance of each subclass of :class:`UniqueTag` in its
    set of `tags`. Multiple `UniqueTag` instances of
    different (immediate) subclasses are allowed.
    """

# }}}


ToTagSetConvertible = Union[Iterable[Tag], Tag, None]
TagT = TypeVar("TagT", bound="Tag")


# {{{ UniqueTag rules checking

@memoize
def _immediate_unique_tag_descendants(cls: type[Tag]) -> FrozenSet[type[Tag]]:
    if UniqueTag in cls.__bases__:
        return frozenset([cls])
    else:
        result: FrozenSet[type[Tag]] = frozenset()
        for base in cls.__bases__:
            result = result | _immediate_unique_tag_descendants(base)
        return result


class NonUniqueTagError(ValueError):
    """
    Raised when a :class:`Taggable` object is instantiated with more
    than one :class:`UniqueTag` instances of the same subclass in
    its set of tags.
    """
    pass


def check_tag_uniqueness(tags: FrozenSet[Tag]) -> FrozenSet[Tag]:
    """Ensure that *tags* obeys the rules set forth in :class:`UniqueTag`.
    If not, raise :exc:`NonUniqueTagError`. If any *tags* are not
    subclasses of :class:`Tag`, a :exc:`TypeError` will be raised.

    :returns: *tags*
    """
    unique_tag_descendants: Set[type[Tag]] = set()
    for tag in tags:
        if not isinstance(tag, Tag):
            raise TypeError(f"'{tag}' is not an instance of pytools.tag.Tag")
        tag_unique_tag_descendants = _immediate_unique_tag_descendants(
                type(tag))
        intersection = unique_tag_descendants & tag_unique_tag_descendants
        if intersection:
            raise NonUniqueTagError("Multiple tags are direct subclasses of "
                    "the following UniqueTag(s): "
                    f"{', '.join(d.__name__ for d in intersection)}")
        else:
            unique_tag_descendants.update(tag_unique_tag_descendants)

    return tags

# }}}


def normalize_tags(tags: ToTagSetConvertible) -> FrozenSet[Tag]:
    if isinstance(tags, Tag):
        tags = frozenset([tags])
    elif tags is None:
        tags = frozenset()
    else:
        tags = frozenset(tags)
    return tags


# {{{ taggable

class Taggable:
    """
    Parent class for objects with a `tags` attribute.

    .. autoattribute:: tags

    .. automethod:: _with_new_tags
    .. automethod:: tagged
    .. automethod:: without_tags
    .. automethod:: tags_of_type
    .. automethod:: tags_not_of_type

    .. versionadded:: 2021.1
    """

    if not TYPE_CHECKING:
        def __init__(self, tags: FrozenSet[Tag] = frozenset()):
            warn("The Taggable constructor is deprecated. "
                 "Subclasses must declare their own storage for .tags. "
                 "The constructor will disappear in 2025.x.",
                 DeprecationWarning, stacklevel=2)

            self.tags = tags

    # ReST references in docstrings must be fully qualified, as docstrings may
    # be inherited and appear in different contexts.

    # type-checking only so that self.tags = ... in subclasses still works
    if TYPE_CHECKING:
        @property
        def tags(self) -> FrozenSet[Tag]:
            ...

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> Self:
        """
        Returns a copy of *self* with the specified tags. This method
        should be overridden by subclasses.
        """
        raise NotImplementedError

    def tagged(self, tags: ToTagSetConvertible) -> Self:
        """
        Return a copy of *self* with the specified
        tag or tags added to the set of tags. If the resulting set of
        tags violates the rules on :class:`pytools.tag.UniqueTag`,
        an error is raised.

        :arg tags: An instance of :class:`~pytools.tag.Tag` or
            an iterable with instances therein.
        """
        return self._with_new_tags(
                tags=check_tag_uniqueness(normalize_tags(tags) | self.tags))

    def without_tags(self,
                     tags: ToTagSetConvertible, verify_existence: bool = True
                     ) -> Self:
        """
        Return a copy of *self* without the specified tags.

        :arg tags: An instance of :class:`~pytools.tag.Tag` or an iterable with
            instances therein.
        :arg verify_existence: If set to `True`, this method raises
            an exception if not all tags specified for removal are
            present in the original set of tags. Default `True`.
        """

        to_remove = normalize_tags(tags)
        new_tags = self.tags - to_remove
        if verify_existence and len(new_tags) > len(self.tags) - len(to_remove):
            raise ValueError("A tag specified for removal was not present.")

        return self._with_new_tags(tags=check_tag_uniqueness(new_tags))

    @memoize_method
    def tags_of_type(self, tag_t: Type[TagT]) -> FrozenSet[TagT]:
        """
        Returns *self*'s tags of type *tag_t*.
        """
        return frozenset({tag
                         for tag in self.tags
                         if isinstance(tag, tag_t)})

    @memoize_method
    def tags_not_of_type(self, tag_t: Type[TagT]) -> FrozenSet[Tag]:
        """
        Returns *self*'s tags that are not of type *tag_t*.
        """
        return frozenset({tag
                         for tag in self.tags
                         if not isinstance(tag, tag_t)})

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Taggable):
            return self.tags == other.tags
        else:
            return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.tags)

# }}}


# {{{ deprecation

_depr_name_to_replacement_and_obj = {
        "TagsType": (
            "FrozenSet[Tag]",
            FrozenSet[Tag], 2023),
        "TagOrIterableType": (
            "ToTagSetConvertible",
            ToTagSetConvertible, 2023),
        "T_co": (
            "Self (i.e. the self type from Python 3.11)",
            TypeVar("TaggableT", bound="Taggable"), 2023),
        }


def __getattr__(name: str) -> Any:
    replacement_and_obj = _depr_name_to_replacement_and_obj.get(name, None)
    if replacement_and_obj is not None:
        replacement, obj, year = replacement_and_obj
        from warnings import warn
        warn(f"'pytools.tag.{name}' is deprecated. "
                f"Use '{replacement}' instead. "
                f"'pytools.tag.{name}' will continue to work until {year}.",
                DeprecationWarning, stacklevel=2)
        return obj
    else:
        raise AttributeError(name)

# }}}


# vim: foldmethod=marker
