def _cp(src, dest):
    from pytools import assert_not_a_file
    assert_not_a_file(dest)

    inf = open(src, "rb")
    try:
        outf = open(dest, "wb")
        try:
            outf.write(inf.read())
        finally:
            outf.close()
    finally:
        inf.close()


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


class BatchJob:
    def __init__(self, moniker, main_file, aux_files=(), timestamp=None):
        import os
        import os.path

        if timestamp is None:
            timestamp = get_timestamp()

        self.moniker = (
                moniker
                .replace("/", "-")
                .replace("-$DATE", "")
                .replace("$DATE-", "")
                .replace("$DATE", "")
                )
        self.subdir = moniker.replace("$DATE", timestamp)
        self.path = os.path.join(
                os.getcwd(),
                self.subdir)

        os.makedirs(self.path)

        runscript = open(f"{self.path}/run.sh", "w")
        import sys
        runscript.write(f"{sys.executable} {main_file} setup.cpy")
        runscript.close()

        from os.path import basename

        if not main_file.startswith("-m "):
            _cp(main_file, os.path.join(self.path, basename(main_file)))

        for aux_file in aux_files:
            _cp(aux_file, os.path.join(self.path, basename(aux_file)))

    def write_setup(self, lines):
        import os.path
        setup = open(os.path.join(self.path, "setup.cpy"), "w")
        setup.write("\n".join(lines))
        setup.close()


class INHERIT:
    pass


class GridEngineJob(BatchJob):
    def submit(self, env=(("LD_LIBRARY_PATH", INHERIT), ("PYTHONPATH", INHERIT)),
            memory_megs=None, extra_args=()):
        from subprocess import Popen
        args = [
            "-N", self.moniker,
            "-cwd",
            ]

        from os import getenv
        env = dict(env)
        for var, value in env.items():
            if value is INHERIT:
                value = getenv(var)

            args += ["-v", f"{var}={value}"]

        if memory_megs is not None:
            args.extend(["-l", f"mem={memory_megs}"])

        args.extend(extra_args)

        subproc = Popen(["qsub", *args, "run.sh"], cwd=self.path)
        if subproc.wait() != 0:
            raise RuntimeError(f"Process submission of {self.moniker} failed")


class PBSJob(BatchJob):
    def submit(self, env=(("LD_LIBRARY_PATH", INHERIT), ("PYTHONPATH", INHERIT)),
            memory_megs=None, extra_args=()):
        from subprocess import Popen
        args = [
            "-N", self.moniker,
            "-d", self.path,
            ]

        if memory_megs is not None:
            args.extend(["-l", f"pmem={memory_megs}mb"])

        from os import getenv

        env = dict(env)
        for var, value in env.items():
            if value is INHERIT:
                value = getenv(var)

            args += ["-v", f"{var}={value}"]

        args.extend(extra_args)

        subproc = Popen(["qsub", *args, "run.sh"], cwd=self.path)
        if subproc.wait() != 0:
            raise RuntimeError(f"Process submission of {self.moniker} failed")


def guess_job_class():
    from subprocess import PIPE, STDOUT, Popen
    qstat_helplines = Popen(["qstat", "--help"],
            stdout=PIPE, stderr=STDOUT).communicate()[0].split("\n")
    if qstat_helplines[0].startswith("GE"):
        return GridEngineJob
    else:
        return PBSJob


class ConstructorPlaceholder:
    def __init__(self, classname, *args, **kwargs):
        self.classname = classname
        self.args = args
        self.kwargs = kwargs

    def arg(self, i):
        return self.args[i]

    def kwarg(self, name):
        return self.kwargs[name]

    def __str__(self):
        return "{}({})".format(self.classname,
                ",".join(
                    [str(arg) for arg in self.args]
                    + [f"{kw}={val!r}"
                        for kw, val in self.kwargs.items()]
                    )
                )
    __repr__ = __str__
