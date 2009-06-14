def run_with_mpi_ranks(py_script, ranks, callable, *args, **kwargs):
    import os
    if "BOOSTMPI_RUN_WITHIN_MPI" in os.environ:
        callable(*args, **kwargs)
    else:
        import sys
        newenv = os.environ.copy()
        newenv["BOOSTMPI_RUN_WITHIN_MPI"] = "1"

        from subprocess import check_call
        check_call(["mpirun", "-np", str(ranks), 
            sys.executable, py_script], env=newenv)

