__copyright__ = """
Copyright (C) 2009-2019 Andreas Kloeckner
Copyright (C) 2022 University of Illinois Board of Trustees
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

__doc__ = """
MPI helper functionality
========================

.. autofunction:: check_for_mpi_relaunch
.. autofunction:: run_with_mpi_ranks
.. autofunction:: pytest_raises_on_rank
"""

from contextlib import AbstractContextManager, contextmanager
from typing import Generator, Tuple, Type, Union


def check_for_mpi_relaunch(argv):
    if argv[1] != "--mpi-relaunch":
        return

    from pickle import loads
    f, args, kwargs = loads(argv[2])

    f(*args, **kwargs)
    import sys
    sys.exit()


def run_with_mpi_ranks(py_script, ranks, callable_, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}

    import os
    import sys
    newenv = os.environ.copy()
    newenv["PYTOOLS_RUN_WITHIN_MPI"] = "1"

    from pickle import dumps
    callable_and_args = dumps((callable_, args, kwargs))

    from subprocess import check_call
    check_call(["mpirun", "-np", str(ranks),
        sys.executable, py_script, "--mpi-relaunch", callable_and_args],
        env=newenv)


@contextmanager
def pytest_raises_on_rank(my_rank: int, fail_rank: int,
        expected_exception: Union[Type[BaseException],
                                  Tuple[Type[BaseException], ...]]) \
                -> Generator[AbstractContextManager, None, None]:
    """
    Like :func:`pytest.raises`, but only expect an exception on rank *fail_rank*.
    """
    from contextlib import nullcontext

    import pytest

    if my_rank == fail_rank:
        cm: AbstractContextManager = pytest.raises(expected_exception)
    else:
        cm = nullcontext()

    with cm as exc:
        yield exc
