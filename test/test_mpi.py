from __future__ import division
from __future__ import absolute_import

from pytools.mpi import (  # noqa
        make_mpi_executor,
        MPIExecError,
        MPIExecutorParams,
        MPIExecutorParamError)

import pytest
import os
from functools import partial


def get_test_mpi_executor():
    if "MPI_EXECUTOR_TYPE" not in os.environ:
        pytest.skip("No MPI executor specified.")
    return make_mpi_executor(os.environ["MPI_EXECUTOR_TYPE"])


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_mpi_launch(num_tasks):
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    exit_code = mpi_exec(["true"], exec_params=MPIExecutorParams(
                num_tasks=num_tasks))
    assert exit_code == 0

    exit_code = mpi_exec(["false"], exec_params=MPIExecutorParams(
                num_tasks=num_tasks))
    assert exit_code != 0


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_mpi_execute(num_tasks):
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    exit_code = mpi_exec.execute("from mpi4py import MPI; "
                + "MPI.COMM_WORLD.Barrier(); assert True",
                exec_params=MPIExecutorParams(num_tasks=num_tasks))
    assert exit_code == 0

    exit_code = mpi_exec.execute("from mpi4py import MPI; "
                + "MPI.COMM_WORLD.Barrier(); assert False",
                exec_params=MPIExecutorParams(num_tasks=num_tasks))
    assert exit_code != 0


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_mpi_check_execute(num_tasks):
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    mpi_exec.check_execute("from mpi4py import MPI; "
                + "MPI.COMM_WORLD.Barrier(); assert True",
                exec_params=MPIExecutorParams(num_tasks=num_tasks))

    with pytest.raises(MPIExecError):
        mpi_exec.check_execute("from mpi4py import MPI; "
                    + "MPI.COMM_WORLD.Barrier(); assert False",
                    exec_params=MPIExecutorParams(num_tasks=num_tasks))


def _test_mpi_func(arg):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    assert arg == "hello"


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_mpi_call(num_tasks):
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    exit_code = mpi_exec.call(partial(_test_mpi_func, "hello"),
                exec_params=MPIExecutorParams(num_tasks=num_tasks))
    assert exit_code == 0

    exit_code = mpi_exec.call(partial(_test_mpi_func, "goodbye"),
                exec_params=MPIExecutorParams(num_tasks=num_tasks))
    assert exit_code != 0


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_mpi_check_call(num_tasks):
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    mpi_exec.check_call(partial(_test_mpi_func, "hello"),
                exec_params=MPIExecutorParams(num_tasks=num_tasks))

    with pytest.raises(MPIExecError):
        mpi_exec.check_call(partial(_test_mpi_func, "goodbye"),
                    exec_params=MPIExecutorParams(num_tasks=num_tasks))


def test_mpi_unsupported_param():
    pytest.importorskip("mpi4py")

    mpi_exec = get_test_mpi_executor()

    try:
        mpi_exec.call(partial(_test_mpi_func, "hello"),
                    exec_params=MPIExecutorParams(num_tasks=2, gpus_per_task=1))
        pytest.skip("Oops. Unsupported param is actually supported.")
    except MPIExecutorParamError as e:
        assert e.param_name == "gpus_per_task"

# }}}
