from __future__ import annotations


__copyright__ = """
Copyright (C) 2009-2013 Andreas Kloeckner
Copyright (C) 2013- University of Illinois Board of Trustees
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


import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn, TypeAlias, cast

from typing_extensions import Self, final, overload, override


@final
class RuleError(RuntimeError):
    def __init__(self, rule: LexRule) -> None:
        RuntimeError.__init__(self)
        self.Rule = rule

    @override
    def __str__(self) -> str:
        return repr(self.Rule)


@final
class InvalidTokenError(RuntimeError):
    def __init__(self, s: str, str_index: int) -> None:
        RuntimeError.__init__(self)
        self.string = s
        self.index = str_index

    @override
    def __str__(self) -> str:
        return "at index {}: ...{}...".format(
                self.index, self.string[self.index:self.index+20])


@final
class ParseError(RuntimeError):
    def __init__(self, msg: str, s: str, token: Lexed | None) -> None:
        RuntimeError.__init__(self)
        self.message = msg
        self.string = s
        self.Token = token

    @override
    def __str__(self) -> str:
        if self.Token is None:
            return f"{self.message} at end of input"
        return "{} at index {}: ...{}...".format(
                self.message, self.Token[2],
                self.string[self.Token[2]:self.Token[2]+20])


@final
class RE:
    def __init__(self, s: str, flags: int = 0) -> None:
        self.Content = s
        self.RE = re.compile(s, flags)

    @override
    def __repr__(self) -> str:
        return f"RE({self.Content})"


LexRule: TypeAlias = tuple[str, RE | tuple[str | RE, ...]]
LexTable: TypeAlias = Sequence[LexRule]
BasicLexed: TypeAlias = tuple[str, str, int]
LexedWithMatch: TypeAlias = tuple[str, str, int, re.Match[str]]
Lexed: TypeAlias = BasicLexed | LexedWithMatch


def _matches_rule(
            rule: RE | str | tuple[str | RE, ...],
            s: str,
            start: int,
            rule_dict: Mapping[str, RE | tuple[str | RE, ...]],
            debug: bool = False
        ) -> tuple[int, re.Match[str] | None]:
    if debug:
        print("Trying", rule, "on", s[start:])

    if isinstance(rule, tuple):
        if rule[0] == "|":
            for subrule in rule[1:]:
                length, match_obj = _matches_rule(
                        subrule, s, start, rule_dict, debug)
                if not length:
                    continue
                return length, match_obj
        else:
            my_match_length = 0
            for subrule in rule:
                length, _ = _matches_rule(
                        subrule, s, start, rule_dict, debug)
                if not length:
                    break
                my_match_length += length
                start += length
            else:
                return my_match_length, None
        return 0, None

    if isinstance(rule, str):
        return _matches_rule(rule_dict[rule], s, start, rule_dict, debug)

    if isinstance(rule, RE):
        match_obj = rule.RE.match(s, start)
        if match_obj:
            return match_obj.end()-start, match_obj
        return 0, None

    raise RuleError(rule)


@overload
def lex(
            lex_table: LexTable,
            s: str,
            *,
            debug: bool = ...,
            match_objects: Literal[False] = False,
        ) -> Sequence[BasicLexed]:
    ...


@overload
def lex(
            lex_table: LexTable,
            s: str,
            *,
            debug: bool = ...,
            match_objects: Literal[True],
        ) -> Sequence[LexedWithMatch]:
    ...


def lex(
            lex_table: LexTable,
            s: str,
            *,
            debug: bool = False,
            match_objects: bool = False
        ) -> Sequence[Lexed]:
    rule_dict = dict(lex_table)
    result: list[Lexed] = []
    i = 0
    while i < len(s):
        for name, rule in lex_table:
            length, match_obj = _matches_rule(rule, s, i, rule_dict, debug)
            if length:
                if match_objects:
                    assert match_obj
                    result.append((name, s[i:i+length], i, match_obj))
                else:
                    result.append((name, s[i:i+length], i))
                i += length
                break
        else:
            raise InvalidTokenError(s, i)
    return result


@dataclass
class LexIterator:
    lexed: Sequence[tuple[str, str, int] | tuple[str, str, int, re.Match[str]]]
    raw_string: str
    index: int = 0

    def copy(self) -> Self:
        return type(self)(self.lexed, self.raw_string, self.index)

    def assign(self, rhs: LexIterator) -> None:
        assert self.lexed is rhs.lexed
        assert self.raw_string is rhs.raw_string

        self.index = rhs.index

    def next_tag(self, i: int = 0) -> str:
        return self.lexed[self.index + i][0]

    def next_str(self, i: int = 0) -> str:
        return self.lexed[self.index + i][1]

    def next_match_obj(self) -> re.Match[str]:
        _tok, _s, _i, match = cast("LexedWithMatch", self.lexed[self.index])
        return match

    def next_str_and_advance(self) -> str:
        result = self.next_str()
        self.advance()
        return result

    def advance(self) -> None:
        self.index += 1

    def is_at_end(self, i: int = 0) -> bool:
        return self.index + i >= len(self.lexed)

    def is_next(self, tag: str, i: int = 0) -> bool:
        return (
                self.index + i < len(self.lexed)
                and self.next_tag(i) is tag)

    def raise_parse_error(self, msg: str) -> NoReturn:
        if self.is_at_end():
            raise ParseError(msg, self.raw_string, None)

        raise ParseError(msg, self.raw_string, self.lexed[self.index])

    def expected(self, what_expected: str) -> NoReturn:
        if self.is_at_end():
            self.raise_parse_error(
                    f"{what_expected} expected, end of input found instead")
        else:
            self.raise_parse_error(
                    f"{what_expected} expected, {self.next_tag()} found instead")

    def expect_not_end(self) -> None:
        if self.is_at_end():
            self.raise_parse_error("unexpected end of input")

    def expect(self, tag: str) -> None:
        self.expect_not_end()
        if not self.is_next(tag):
            self.expected(str(tag))
