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

__doc__ = """
Tools for Source Code Generation
================================

.. autoclass:: CodeGenerator
.. autoclass:: Indentation
.. autofunction:: remove_common_indentation
"""

from typing import Any, List


# {{{ code generation

# loosely based on
# http://effbot.org/zone/python-code-generator.htm

class CodeGenerator:
    """Language-agnostic functionality for source code generation.

    .. automethod:: extend
    .. automethod:: get
    .. automethod:: add_to_preamble
    .. automethod:: __call__
    .. automethod:: indent
    .. automethod:: dedent
    """
    def __init__(self) -> None:
        self.preamble: List[str] = []
        self.code: List[str] = []
        self.level = 0
        self.indent_amount = 4

    def extend(self, sub_generator: "CodeGenerator") -> None:
        for line in sub_generator.code:
            self.code.append(" "*(self.indent_amount*self.level) + line)

    def get(self) -> str:
        result = "\n".join(self.code)
        if self.preamble:
            result = "\n".join(self.preamble) + "\n" + result
        return result

    def add_to_preamble(self, s: str) -> None:
        self.preamble.append(s)

    def __call__(self, s: str) -> None:
        if not s.strip():
            self.code.append("")
        else:
            if "\n" in s:
                s = remove_common_indentation(s)

            for line in s.split("\n"):
                self.code.append(" "*(self.indent_amount*self.level) + line)

    def indent(self) -> None:
        self.level += 1

    def dedent(self) -> None:
        if self.level == 0:
            raise RuntimeError("cannot decrease indentation level")
        self.level -= 1


class Indentation:
    """A context manager for indentation for use with :class:`CodeGenerator`.

    .. attribute:: generator
    .. automethod:: __enter__
    .. automethod:: __exit__
    """
    def __init__(self, generator: CodeGenerator):
        self.generator = generator

    def __enter__(self) -> None:
        self.generator.indent()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.generator.dedent()

# }}}


# {{{ remove common indentation

def remove_common_indentation(code: str, require_leading_newline: bool = True):
    r"""Remove leading indentation from one or more lines of code.

    Removes an amount of indentation equal to the indentation level of the first
    nonempty line in *code*.

    :param code: Input string.
    :param require_leading_newline: If *True*, only remove indentation if *code*
        starts with ``\n``.

    :returns: A copy of *code* stripped of leading common indentation.
    """
    if "\n" not in code:
        return code

    if require_leading_newline and not code.startswith("\n"):
        return code

    lines = code.split("\n")
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation")

    return "\n".join(line[base_indent:] for line in lines)

# }}}

# vim: foldmethod=marker
