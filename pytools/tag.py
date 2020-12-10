from dataclasses import dataclass
from typing import Tuple, Any, FrozenSet
import importlib
from lark import Lark, Transformer
from mako.template import Template

__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
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

.. autoclass:: Tag
.. autoclass:: UniqueTag
.. autoclass:: ToPythonObjectMapper
.. autofunction:: parse_tag

Supporting Functionality
------------------------

.. autoclass:: DottedName

"""

# }}}


# {{{ dotted name


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


class UniqueTag(Tag):
    """
    Only one instance of this type of tag may be assigned
    to a single tagged object.
    """
    pass


TagsType = FrozenSet[Tag]

# }}}


# {{{ parse

def build_tag_grammar(shortcuts):
    tag_grammar_tmplt = """
    tag: tag_class "(" params ")" -> map_tag
    % if shortcuts:
       | SHORTCUT                 -> map_shortcut
    % endif

    params:                       -> map_empty_args_params
          | args                  -> map_args_only_params
          | kwargs                -> map_kwargs_only_params
          | args "," kwargs       -> map_args_kwargs_params

    ?kwargs: kwarg
           | kwargs "," kwarg     -> map_kwargs

    args: arg                     -> map_singleton_args
        | args "," arg            -> map_args

    kwarg: name "=" arg           -> map_kwarg

    ?arg: tag
        | INT                     -> map_int
        | ESCAPED_STRING          -> map_string

    tag_class: module "." name    -> map_tag_class

    module: name                  -> map_top_level_module
          | module "." name       -> map_nested_module

    name: CNAME                   -> map_name

    % if shortcuts:
    SHORTCUT: (${" | ".join('"' + name + '"' for name in shortcuts)})
    % endif

    %%import common.INT
    %%import common.ESCAPED_STRING
    %%import common.CNAME
    %%import common.WS
    %%ignore WS
    """
    return Template(tag_grammar_tmplt).render(shortcuts=shortcuts).replace("%%", "%")


class CallParams:
    """
    Intermediate data structure for :class:`ToPythonObjectMapper`.
    """
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class ToPythonObjectMapper(Transformer):
    """
    Map a parsed tree to pythonic objects.
    """
    def __init__(self, shortcuts):
        super().__init__()
        self.shortcuts = shortcuts

    def _call_userfunc(self, tree, new_children=None):
        # Assumes tree is already transformed
        children = new_children if new_children is not None else tree.children
        try:
            f = getattr(self, tree.data)
        except AttributeError:
            return self.__default__(tree.data, children, tree.meta)
        else:
            # flatten the args
            return f(*children)

    def map_tag(self, cls, params):
        return cls(*params.args, **params.kwargs)

    def map_empty_args_params(self):
        return CallParams((), {})

    def map_args_only_params(self, args):
        return CallParams(args, {})

    def map_kwargs_only_params(self, kwargs):
        return CallParams((), kwargs)

    def map_args_kwargs_params(self, args, kwargs):
        return CallParams(args, kwargs)

    def map_name(self, tok):
        return tok.value

    def map_top_level_module(self, modulename):
        return importlib.import_module(modulename)

    def map_nested_module(self, module, child):
        return getattr(module, child)

    def map_tag_class(self, module, classname):
        return getattr(module, classname)

    def map_string(self, text):
        return str(text)

    def map_int(self, text):
        return int(text)

    def map_singleton_args(self, arg):
        return (arg,)

    def map_args(self, args, arg):
        return args + (arg,)

    def map_kwarg(self, name, arg):
        return {name: arg}

    def map_kwargs(self, kwargs, kwarg):
        assert len(kwarg) == 1
        (key, val), = kwarg.items()
        if key in kwargs:
            # FIXME: This should probably be GrammarError
            raise ValueError("keyword argument '{key}' repeated")

        updated_kwargs = kwargs.copy()
        updated_kwargs[key] = val
        return updated_kwargs

    def map_shortcut(self, name):
        return self.shortcuts[name.value]


def parse_tag(tag_text, shortcuts={}):
    """
    Parses a :class:`Tag` from a provided dotted name.
    """
    tag_grammar = build_tag_grammar(shortcuts)
    parser = Lark(tag_grammar, start="tag")
    tag = ToPythonObjectMapper(shortcuts).transform(parser.parse(tag_text))

    return tag

# }}}

# vim: foldmethod=marker
