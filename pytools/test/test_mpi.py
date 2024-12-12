from __future__ import annotations


__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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

import pytest


def test_pytest_raises_on_rank():
    from pytools.mpi import pytest_raises_on_rank

    def fail(my_rank: int, fail_rank: int) -> None:
        if my_rank == fail_rank:
            raise ValueError("test failure")

    with pytest.raises(ValueError):
        fail(0, 0)

    fail(0, 1)

    with pytest_raises_on_rank(0, 0, ValueError):
        # Generates an exception, and pytest_raises_on_rank
        # expects one.
        fail(0, 0)

    with pytest_raises_on_rank(0, 1, ValueError):
        # Generates no exception, and pytest_raises_on_rank
        # does not expect one.
        fail(0, 1)
