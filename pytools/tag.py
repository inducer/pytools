from dataclasses import dataclass
from typing import Tuple, Any, FrozenSet, Union, Iterable, TypeVar
from pytools import memoize

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


# {{{ docs

__doc__ = """

Tag Interface
---------------

.. autoclass:: Taggable
.. autoclass:: Tag
.. autoclass:: UniqueTag

Supporting Functionality
------------------------

.. autoclass:: DottedName
.. autoclass:: NonUniqueTagError

"""

#  }}}


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

    def __init__(self, name_parts: Tuple[str, ...]):
        if len(name_parts) == 0:
            raise ValueError("empty name parts")

        for p in name_parts:
            if not p.isidentifier():
                raise ValueError(f"{p} is not a Python identifier")

        self.name_parts = name_parts

    @classmethod
    def from_class(cls, argcls: Any) -> "DottedName":
        name_parts = tuple(
                [str(part) for part in argcls.__module__.split(".")]
                + [str(argcls.__name__)])
        if not all(not npart.startswith("__") for npart in name_parts):
            raise ValueError(f"some name parts of {'.'.join(name_parts)} "
                             "start with double underscores")
        return cls(name_parts)


# }}}


# {{{ tag

tag_dataclass = dataclass(init=True, eq=True, frozen=True, repr=True)


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

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.__class__.__qualname__)

        from dataclasses import fields
        # Fields are ordered consistently, so ordered hashing is OK.
        #
        # No need to dispatch to superclass: fields() automatically gives us
        # fields from the entire class hierarchy.
        for f in fields(self):
            key_builder.rec(key_hash, getattr(self, f.name))

# }}}


# {{{ unique tag

class UniqueTag(Tag):
    """
    A superclass for tags that are unique on each :class:`Taggable`.

    Each instance of :class:`Taggable` may have no more than one
    instance of each subclass of :class:`UniqueTag` in its
    set of `tags`. Multiple `UniqueTag` instances of
    different (immediate) subclasses are allowed.
    """
    pass

# }}}


TagsType = FrozenSet[Tag]
TagOrIterableType = Union[Iterable[Tag], Tag, None]
T_co = TypeVar("T_co", bound="Taggable")


# {{{ taggable

@memoize
def _immediate_unique_tag_descendants(cls):
    if UniqueTag in cls.__bases__:
        return frozenset([cls])
    else:
        result = frozenset()
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


class Taggable:
    """
    Parent class for objects with a `tags` attribute.

    .. attribute:: tags

        A :class:`frozenset` of :class:`Tag` instances

    .. method:: copy
    .. method:: tagged
    .. method:: without_tags

    .. versionadded:: 2021.1
    """

    def __init__(self, tags: TagsType = frozenset()):
        # For performance we assert rather than
        # normalize the input.
        assert isinstance(tags, FrozenSet)
        assert all(isinstance(tag, Tag) for tag in tags)
        self.tags = tags
        self._check_uniqueness()

    def _normalize_tags(self, tags: TagOrIterableType) -> TagsType:
        if isinstance(tags, Tag):
            t = frozenset([tags])
        elif tags is None:
            t = frozenset()
        else:
            t = frozenset(tags)
        return t

    def _check_uniqueness(self):
        unique_tag_descendants = set()
        for tag in self.tags:
            tag_unique_tag_descendants = _immediate_unique_tag_descendants(
                    type(tag))
            intersection = unique_tag_descendants & tag_unique_tag_descendants
            if intersection:
                raise NonUniqueTagError("Multiple tags are direct subclasses of "
                        "the following UniqueTag(s): "
                        f"{', '.join(d.__name__ for d in intersection)}")
            else:
                unique_tag_descendants.update(tag_unique_tag_descendants)

    def copy(self: T_co, **kwargs: Any) -> T_co:
        """
        Returns of copy of *self* with the specified tags. This method
        should be overridden by subclasses.
        """
        raise NotImplementedError("The copy function is not implemented.")

    def tagged(self: T_co, tags: TagOrIterableType) -> T_co:
        """
        Return a copy of *self* with the specified
        tag or tags unioned. If *tags* is a :class:`pytools.tag.UniqueTag`
        and other tags of this type are already present, an error is raised
        Assumes `self.copy(tags=<NEW VALUE>)` is implemented.

        :arg tags: An instance of :class:`Tag` or
        an iterable with instances therein.
        """
        new_tags = self._normalize_tags(tags)
        union_tags = self.tags | new_tags
        cpy = self.copy(tags=union_tags)
        return cpy

    def without_tags(self: T_co,
            tags: TagOrIterableType, verify_existence: bool = True) -> T_co:
        """
        Return a copy of *self* without the specified tags.
        `self.copy(tags=<NEW VALUE>)` is implemented.

        :arg tags: An instance of :class:`Tag` or an iterable with instances
        therein.
        :arg verify_existence: If set
        to `True`, this method raises an exception if not all tags specified
        for removal are present in the original set of tags. Default `True`
        """

        to_remove = self._normalize_tags(tags)
        new_tags = self.tags - to_remove
        if verify_existence and len(new_tags) > len(self.tags) - len(to_remove):
            raise ValueError("A tag specified for removal was not present.")

        return self.copy(tags=new_tags)

# }}}

# vim: foldmethod=marker
