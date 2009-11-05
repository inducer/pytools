import mpi4py.rc

mpi4py.rc.initialize = False

import pytools.prefork
pytools.prefork.enable_prefork()

from mpi4py.MPI import *

if Is_initialized():
    raise RuntimeError("MPI already initialized before MPI wrapper import")
