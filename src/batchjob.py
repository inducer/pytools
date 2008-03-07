from __future__ import with_statement




def _cp(src, dest):
    from pytools import assert_not_a_file
    assert_not_a_file(dest)

    with open(src, "rb") as inf:
        with open(dest, "wb") as outf:
            outf.write(inf.read())




class BatchJob(object):
    def __init__(self, moniker, main_file, aux_files=[]):
        from datetime import datetime
        import os
        import os.path

        self.moniker = moniker
        self.subdir = os.path.join(
                os.getcwd(),
                "%s-%s" % (moniker, datetime.now().strftime("%Y-%m-%d-%H%M%S")))
        os.mkdir(self.subdir)

        with open("%s/run.sh" % self.subdir, "w") as runscript:
            import sys
            runscript.write("%s %s setup.cpy" 
                    % (sys.executable, main_file))

        _cp(main_file, os.path.join(self.subdir, main_file))
        for aux_file in aux_files:
            _cp(aux_file, os.path.join(self.subdir, aux_file))

    def write_setup(self, lines):
        import os.path
        with open(os.path.join(self.subdir, "setup.cpy"), "w") as setup:
            setup.write("\n".join(lines))




class INHERIT(object):
    pass




class GridEngineJob(BatchJob):
    def submit(self, env={"LD_LIBRARY_PATH": INHERIT, "PYTHONPATH": INHERIT}):
        from subprocess import Popen
        args = [
            "-N", self.moniker,
            "-cwd",
            ]

        from os import getenv

        for var, value in env.iteritems():
            if value is INHERIT:
                value = getenv(var)

            args += ["-v", "%s=%s" % (var, value)]

        subproc = Popen(["qsub"] + args + ["run.sh"], cwd=self.subdir)
        if subproc.wait() != 0:
            raise RuntimeError("Process submission of %s failed" % self.moniker)




class PBSJob(BatchJob):
    def submit(self, env={"LD_LIBRARY_PATH": INHERIT, "PYTHONPATH": INHERIT}):
        from subprocess import Popen
        args = [
            "-N", self.moniker,
            "-d", self.subdir,
            ]

        from os import getenv

        for var, value in env.iteritems():
            if value is INHERIT:
                value = getenv(var)

            args += ["-v", "%s=%s" % (var, value)]

        subproc = Popen(["qsub"] + args + ["run.sh"], cwd=self.subdir)
        if subproc.wait() != 0:
            raise RuntimeError("Process submission of %s failed" % self.moniker)




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
        return "%s(%s)" % (self.classname,
                ",".join(
                    [str(arg) for arg in self.args]
                    + ["%s=%s" % (kw, val) for kw, val in self.kwargs.iteritems()]
                    )
                )

