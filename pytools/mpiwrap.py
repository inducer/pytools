"""See pytools.prefork for this module's reason for being."""
from __future__ import annotations

import mpi4py.rc  # pylint:disable=import-error


mpi4py.rc.initialize = False

from mpi4py.MPI import *  # noqa: F403 pylint:disable=wildcard-import,wrong-import-position

import pytools.prefork  # pylint:disable=wrong-import-position


pytools.prefork.enable_prefork()


if Is_initialized():  # type: ignore[name-defined] # noqa: F405
    raise RuntimeError("MPI already initialized before MPI wrapper import")


def InitWithAutoFinalize(*args, **kwargs):  # noqa: N802
    result = Init(*args, **kwargs)  # noqa: F405
    import atexit
    atexit.register(Finalize)  # # noqa: F405
    return result
