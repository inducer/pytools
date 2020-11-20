from __future__ import absolute_import

import abc


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

    import sys
    import os
    newenv = os.environ.copy()
    newenv["PYTOOLS_RUN_WITHIN_MPI"] = "1"

    from pickle import dumps
    callable_and_args = dumps((callable_, args, kwargs))

    from subprocess import check_call
    check_call(["mpirun", "-np", str(ranks),
        sys.executable, py_script, "--mpi-relaunch", callable_and_args],
        env=newenv)


# {{{ MPI executors

class MPIExecutorParams:
    """Collection of parameters to pass to the MPI launcher."""
    def __init__(self, num_tasks=None, num_nodes=None, tasks_per_node=None,
                gpus_per_task=None):
        """
        Possible arguments are:

        :arg num_tasks: The number of MPI ranks to launch.
        :arg num_nodes: The number of nodes on which to run.
        :arg tasks_per_nodes: The number of MPI ranks to launch per node.
        :arg gpus_per_task: The number of GPUs to assign per task.

        Note: A given executor may not support all of these arguments. If it is
        passed an unsupported argument, it will raise an instance of
        :class:`MPIExecutorParamError`.
        """
        self._create_param_dict(num_tasks=num_tasks, num_nodes=num_nodes,
                    tasks_per_node=tasks_per_node, gpus_per_task=gpus_per_task)

    def _create_param_dict(self, **kwargs):
        self.param_dict = {}
        for name, value in kwargs.items():
            if value is not None:
                self.param_dict[name] = value


class MPIExecutorParamError(RuntimeError):
    def __init__(self, param_name):
        self.param_name = param_name
        super().__init__("MPI executor does not support parameter "
                    + f"'{self.param_name}'.")


class MPIExecError(RuntimeError):
    def __init__(self, exit_code):
        self.exit_code = exit_code
        super().__init__(f"MPI execution failed with exit code {exit_code}.")


class MPIExecutor(metaclass=abc.ABCMeta):
    """Base class for a general MPI launcher."""
    @abc.abstractmethod
    def get_mpi_command(self, command, exec_params=None):
        """
        Returns a list of strings representing the MPI command that will be executed
        to launch *command*.
        """
        pass

    @abc.abstractmethod
    def __call__(self, command, exec_params=None):
        """Executes *command* with MPI."""
        pass

    def execute(self, code_string, exec_params=None):
        """Executes Python code stored in *code_string* with MPI."""
        import sys
        return self.__call__([sys.executable, "-m", "mpi4py", "-c", "\'"
                    + code_string + "\'"], exec_params)

    def check_execute(self, code_string, exec_params=None):
        """
        Executes Python code stored in *code_string* with MPI and raises an instance
        of :class:`MPIExecError` if the execution fails.
        """
        exit_code = self.execute(code_string, exec_params)
        if exit_code != 0:
            raise MPIExecError(exit_code)

    def call(self, func, exec_params=None):
        """Calls *func* with MPI. Note: *func* must be picklable."""
        import pickle
        calling_code = ('import sys; import pickle; pickle.loads(bytes.fromhex("'
                    + pickle.dumps(func).hex() + '"))()')
        return self.execute(calling_code, exec_params)

    def check_call(self, func, exec_params=None):
        """
        Calls *func* with MPI and raises an instance of :class:`MPIExecError` if
        the execution fails. Note: *func* must be picklable.
        """
        exit_code = self.call(func, exec_params)
        if exit_code != 0:
            raise MPIExecError(exit_code)


class BasicMPIExecutor(MPIExecutor):
    """Simple `mpiexec` executor."""
    def get_mpi_command(self, command, exec_params=None):
        mpi_command = ["mpiexec"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                mpi_command += ["-n", str(value)]
            else:
                raise MPIExecutorParamError(name)
        mpi_command += command
        return mpi_command

    def __call__(self, command, exec_params=None):
        mpi_command = self.get_mpi_command(command, exec_params)
        import subprocess
        return subprocess.call(" ".join(mpi_command), shell=True)


class SlurmMPIExecutor(MPIExecutor):
    """Executor for Slurm-based platforms."""
    def get_mpi_command(self, command, exec_params=None):
        mpi_command = ["srun"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                mpi_command += ["-n", str(value)]
            elif name == "num_nodes":
                mpi_command += ["-N", str(value)]
            elif name == "tasks_per_node":
                mpi_command += [f"--ntasks-per-node={value}"]
            else:
                raise MPIExecutorParamError(name)
        mpi_command += command
        return mpi_command

    def __call__(self, command, exec_params=None):
        mpi_command = self.get_mpi_command(command, exec_params)
        import subprocess
        return subprocess.call(" ".join(mpi_command), shell=True)


class LCLSFMPIExecutor(MPIExecutor):
    """Executor for Livermore wrapper around IBM LSF."""
    def get_mpi_command(self, command, exec_params=None):
        mpi_command = ["lrun"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                mpi_command += ["-n", str(value)]
            elif name == "num_nodes":
                mpi_command += ["-N", str(value)]
            elif name == "tasks_per_node":
                mpi_command += ["-T", str(value)]
            elif name == "gpus_per_task":
                mpi_command += ["-g", str(value)]
            else:
                raise MPIExecutorParamError(name)
        mpi_command += command
        return mpi_command

    def __call__(self, command, exec_params=None):
        mpi_command = self.get_mpi_command(command, exec_params)
        import subprocess
        return subprocess.call(" ".join(mpi_command), shell=True)


def make_mpi_executor(executor_type_name):
    """
    Returns an instance of a class derived from :class:`MPIExecutor` given an
    executor type name as input.

    :arg executor_type_name: The executor type name. Can be one of `'basic'`,
        `'slurm'`, or `'lclsf'`.
    """
    type_name_map = {
        "basic": BasicMPIExecutor,
        "slurm": SlurmMPIExecutor,
        "lclsf": LCLSFMPIExecutor
    }
    return type_name_map[executor_type_name]()

# }}}

# vim: foldmethod=marker
