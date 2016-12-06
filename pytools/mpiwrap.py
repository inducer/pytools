"""See pytools.prefork for this module's reason for being."""
from __future__ import absolute_import

import mpi4py.rc

mpi4py.rc.initialize = False

import pytools.prefork
pytools.prefork.enable_prefork()

from mpi4py.MPI import *  # noqa


if Is_initialized():  # noqa
    raise RuntimeError("MPI already initialized before MPI wrapper import")


def InitWithAutoFinalize(*args, **kwargs):  # noqa
    result = Init(*args, **kwargs)  # noqa
    import atexit
    atexit.register(Finalize)  # noqa
    return result
