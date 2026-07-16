"""See pytools.prefork for this module's reason for being."""
from __future__ import annotations

import mpi4py.rc  # pylint:disable=import-error


mpi4py.rc.initialize = False

from mpi4py.MPI import *  # ruff:ignore[undefined-local-with-import-star] pylint:disable=wildcard-import,wrong-import-position

import pytools.prefork  # pylint:disable=wrong-import-position


pytools.prefork.enable_prefork()


if Is_initialized():  # ruff:ignore[undefined-local-with-import-star-usage]
    raise RuntimeError("MPI already initialized before MPI wrapper import")


def InitWithAutoFinalize(*args, **kwargs):  # ruff:ignore[invalid-function-name]
    result = Init(*args, **kwargs)  # ruff:ignore[undefined-local-with-import-star-usage]
    import atexit
    atexit.register(Finalize)  # # ruff:ignore[undefined-local-with-import-star-usage]
    return result
