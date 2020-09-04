Time series logging
===================

How do I use this?
------------------

Consider this example of a 'simulation' logging various quantities:

.. literalinclude:: ../examples/log.py

(You can find this example as
:download:`examples/log.py <../examples/log.py>` in the pytools
source distribution.)

Running this script will give you a file ``mylog.dat``. Note how log quantities
given as "watches" are also periodically logged to the console.

To analyze this data, we need some tools that are distributed as part of the
`'datapyle' package on Github <https://github.com/inducer/datapyle>`__, which
we need to install first::

    $ pip install git+https://github.com/inducer/datapyle.git

To prepare this data set for analysis, run ``runalyzer-gather`` as a script::

    $ runalyzer-gather summary.dat mylog.dat
    Scanning...          [########################################] ETA ?
    Importing...         [########################################] ETA ?

This step may look unnecessary here, but it serves two important functions:

- If you would like to compare data from multiple runs, you can simply
  add data files from multiple runs and analyze them side-by-side.
- For distributed-memory runs, you can likewise gve data sets from
  multiple ranks to be gathered in a single file for centralized analysis.
- As well as, naturally, any combination of the above.

After this, you may use ``runalyzer`` (also in ``datapyle``) to analyze
the summary file::

    # -m enables '$quantity' in SQL
    $ runalyzer -m summary.dat
    Runalyzer running on Python 3.8.5 (default, Aug  2 2020, 15:09:07)
    [GCC 10.2.0]
    Copyright (c) Andreas Kloeckner 2008
    Run .help to see help for 'magic' commands
    >>>

This is a normal Python console, but it has a few extra tricks up its
sleeve. ``.help`` gives an overview. Here, we'll focus on demonstrating
two interesting features. First, lets get an overview of all the
logged quantities::

    >>> .quantities
    id | name    | unit | description                 | rank_aggregator
    ---+---------+------+-----------------------------+-----------------
    7  | dt      | s    | Simulation Timestep         | None
    9  | fifteen | None | None                        | None
    5  | step    | 1    | Timesteps                   | None
    2  | t_2step | s    | Step-to-step duration       | None
    3  | t_cpu   | s    | Wall time                   | None
    4  | t_log   | s    | Time spent updating the log | None
    6  | t_sim   | s    | Simulation Time             | None
    1  | t_step  | s    | Time step duration          | None
    8  | t_vis   | s    | Time spent visualizing      | None

Next, we could plot, e.g. time step duration against simulation time::

    >>> .plot select $t_sim, $t_2step

Next, a fact that the ``select`` may have given away: The argument to
``.plot`` is a SQL query that is passed on to :mod:`sqlite3`, with support
for some additional syntax. You can  see what this query expanded to internally::

    >>> mangle_sql("select $t_sim, $t_2step")
    'select t_sim.value, t_2step.value from runs  inner join t_sim on (t_sim.run_id = runs.id)  inner join t_2step on (t_2step.run_id = runs.id and t_sim.step = t_2step.step and t_sim.rank=t_2step.rank) '

As you can see, by default these queries return data from all runs.
You can filter which runs you are interested in through run properties::

    >>> .runprops
    cmdline
    date
    dirname
    filename
    id
    is_parallel
    machine
    myconst
    rank_count
    schema_version
    unique_run_id
    unixtime

simply by adding a SQL ``where`` clause referring to these properties.

As you can see from the demo, you can of course add your own,
application-focused run properties.

This means that you have the full power of SQL available to analyze
your time series data, for example:

- Plot the solution 2-norms of all runs that were run on a Tuesday between time
  step 2000 and 3000
- Make a scatterplot correlating memory transfer amounts and simulation time

.. automodule:: pytools.log

.. no-doctest
