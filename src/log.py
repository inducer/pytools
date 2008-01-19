from __future__ import division




# timing function -------------------------------------------------------------
def time():
    """Return elapsed CPU time, as a float, in seconds."""
    import os
    time_opt = os.environ.get("PYTOOLS_LOG_TIME")
    if time_opt == "wall":
        from time import time
        return time()
    else:
        from resource import getrusage, RUSAGE_SELF
        return getrusage(RUSAGE_SELF).ru_utime




# abstract logging interface --------------------------------------------------
class LogQuantity(object):
    """A source of loggable scalars."""
    def __init__(self, name, unit=None, description=None):
        self.name = name
        self.unit = unit
        self.description = description
    
    @property
    def default_aggregator(self): return None

    def __call__(self):
        """Return the current value of the diagnostic represented by this 
        L{LogQuantity}."""
        raise NotImplementedError




class MultiLogQuantity(object):
    """A source of multiple loggable scalars."""
    def __init__(self, names, units=None, descriptions=None):
        self.names = names
        self.units = units
        self.descriptions = descriptions
    
    @property
    def default_aggregators(self): return [None] * len(self.names)

    def __call__(self):
        """Return an iterable of the current values of the diagnostic represented 
        by this L{MultiLogQuantity}."""
        raise NotImplementedError




class DtConsumer(object):
    def __init__(self, dt):
        self.dt = dt

    def set_dt(self, dt):
        self.dt = dt




class SimulationLogQuantity(LogQuantity, DtConsumer):
    """A source of loggable scalars that needs to know the simulation timestep."""

    def __init__(self, dt, name, unit=None, description=None):
        LogQuantity.__init__(self, name, unit, description)
        DtConsumer.__init__(self, dt)
    




class CallableLogQuantityAdapter(LogQuantity):
    """Adapt a 0-ary callable as a L{LogQuantity}."""
    def __init__(self, callable, name, unit=None, description=None):
        self.callable = callable
        LogQuantity.__init__(self, name, unit, description)

    def __call__(self):
        return self.callable()




# manager functionality -------------------------------------------------------
class _GatherDescriptor(object):
    def __init__(self, quantity, interval):
        self.quantity = quantity
        self.interval = interval




class _QuantityData(object):
    def __init__(self, unit, description, default_aggregator, table=None):
        self.unit = unit
        self.description = description
        self.default_aggregator = default_aggregator

        if table is None:
            from pytools.datatable import DataTable
            self.table = DataTable(["step", "rank", "value"])
        else:
            self.table = table.copy()




def _join_by_first_of_tuple(list_of_iterables):
    loi = [i.__iter__() for i in list_of_iterables]
    if not loi:
        return
    key_vals = [iter.next() for iter in loi]
    keys = [kv[0] for kv in key_vals]
    values = [kv[1] for kv in key_vals]
    target_key = max(keys)

    force_advance = False

    i = 0
    while True:
        while keys[i] < target_key or force_advance:
            try:
                new_key, new_value = loi[i].next()
            except StopIteration:
                return
            assert keys[i] < new_key
            keys[i] = new_key
            values[i] = new_value
            if new_key > target_key:
                target_key = new_key

            force_advance = False

        i += 1
        if i >= len(loi):
            i = 0

        if min(keys) == target_key:
            yield target_key, values[:]
            force_advance = True




class LogManager(object):
    """A parallel-capable diagnostic time-series logging facility.

    A C{LogManager} logs any number of named time series of floats to 
    a file. Non-time-series data, in the form of constants, is also 
    supported and saved.

    If MPI parallelism is used, the "head rank" below always refers to
    rank 0.

    A command line tool called C{logtool} is available for looking at the
    data in a saved log.
    """

    def __init__(self, filename=None, mpi_comm=None):
        """Initialize this log manager instance.

        @arg filename: If given, the log is periodically written to this file.
        @arg mpi_comm: A C{boost.mpi} communicator. If given, logs are periodically
          synchronized to the head node, which then writes them out to disk.
        """
        self.quantity_data = {}
        self.gather_descriptors = []
        self.tick_count = 0
        self.filename = filename

        self.constants = {}

        if filename is not None:
            from os import access, R_OK
            if access(self.filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % self.filename

        # self-timing
        self.start_time = time()
        self.t_log = 0

        # parallel support
        self.head_rank = 0
        self.mpi_comm = mpi_comm
        self.is_parallel = mpi_comm is not None
        if mpi_comm is None:
            self.rank = 0
            self.last_checkpoint = self.start_time
        else:
            self.rank = mpi_comm.rank
            self.next_sync_tick = 10
            self.head_rank = 0

        # watch stuff
        self.watches = []
        self.next_watch_tick = 1

    def add_watches(self, watches):
        """Add quantities that are printed after every time step."""

        from pytools import Record

        for watch in watches:
            parsed = self._parse_expr(watch)
            parsed, dep_data = self._get_expr_dep_data(parsed)

            from pymbolic import compile
            compiled = compile(parsed, [dd.varname for dd in dep_data])

            watch_info = Record(expr=watch, parsed=parsed, dep_data=dep_data,
                    compiled=compiled)

            self.watches.append(watch_info)

    def set_constant(self, name, value):
        """Make a named, constant value available in the log."""
        self.constants[name] = value

    def tick(self):
        """Record data points from each added L{LogQuantity}.

        May also checkpoint data to disk, and/or synchronize data points 
        to the head rank.
        """
        start_time = time()

        for gd in self.gather_descriptors:
            if self.tick_count % gd.interval == 0:
                q_value = gd.quantity()
                if isinstance(gd.quantity, MultiLogQuantity):
                    for name, value in zip(gd.quantity.names, q_value):
                        self.quantity_data[name].table.insert_row(
                                (self.tick_count, self.rank, value))
                else:
                    self.quantity_data[gd.quantity.name].table.insert_row(
                                (self.tick_count, self.rank, q_value))
        self.tick_count += 1

        end_time = time()
        self.t_log = end_time - start_time

        # print watches
        if self.tick_count == self.next_watch_tick:
            self._watch_tick()

        # synchronize logs with parallel peers, if necessary
        if self.mpi_comm is not None:
            # parallel-case : sync, then checkpoint
            if self.tick_count == self.next_sync_tick:
                if self.filename is not None:
                    # implicitly synchronizes
                    self.save()
                else:
                    self.synchronize_logs()

                # figure out next sync tick, broadcast to peers
                ticks_per_20_sec = 20*self.tick_count/max(1, end_time-self.start_time)
                next_sync_tick = self.tick_count + int(max(10, ticks_per_20_sec))
                from boost.mpi import broadcast
                self.next_sync_tick = broadcast(self.mpi_comm, next_sync_tick, self.head_rank)
        else:
            # non-parallel-case : checkpoint log to disk, if necessary
            if self.filename is not None:
                if end_time - self.last_checkpoint > 10:
                    self.save()
                    self.last_checkpoint = end_time

    def synchronize_logs(self):
        """Transfer data from client ranks to the head rank.
        
        Must be called on all ranks simultaneously."""
        if self.mpi_comm is None:
            return

        from boost.mpi import gather
        if self.mpi_comm.rank == self.head_rank:
            for rank_data in gather(self.mpi_comm, None, self.head_rank)[1:]:
                for name, rows in rank_data:
                    self.quantity_data[name].table.insert_rows(rows)
        else:
            # send non-head data away
            gather(self.mpi_comm, 
                    [(name, qdat.table.data)
                        for name, qdat in self.quantity_data.iteritems()], 
                self.head_rank)

            # and erase it
            for qdat in self.quantity_data.itervalues():
                qdat.table.clear()

    def add_quantity(self, quantity, interval=1):
        """Add an object derived from L{LogQuantity} to this manager."""
        self.gather_descriptors.append(_GatherDescriptor(quantity, interval))
        if isinstance(quantity, MultiLogQuantity):
            for name, unit, description, def_agg in zip(
                    quantity.names,
                    quantity.units,
                    quantity.descriptions,
                    quantity.default_aggregators):
                self.quantity_data[name] = _QuantityData(
                        unit, description, def_agg)
        else:
            self.quantity_data[quantity.name] = _QuantityData(
                        quantity.unit, quantity.description, 
                        quantity.default_aggregator)

    def get_expr_dataset(self, expression, description=None, unit=None):
        """Prepare a time-series dataset for a given expression.

        @arg expression: A C{pymbolic} expression that may involve
          the time-series variables and the constants in this L{LogManager}.
          If there is data from multiple ranks for a quantity occuring in
          this expression, an aggregator may have to be specified.
        @return: C{(description, unit, table)}, where C{table} 
          is a list of tuples C{(tick_nbr, value)}.

        Aggregators are specified as follows:
        - C{qty.min}, C{qty.max}, C{qty.avg}, C{qty.sum}, C{qty.norm2}
        - C{qty[rank_nbr]
        """

        parsed = self._parse_expr(expression)
        parsed, dep_data = self._get_expr_dep_data(parsed)

        # aggregate table data
        for dd in dep_data:
            table = self.quantity_data[dd.name].table
            table.sort(["step"])
            dd.table = table.aggregated(["step"], "value", dd.agg_func).data

        # evaluate unit and description, if necessary
        if unit is None:
            from pymbolic import substitute, parse

            unit = substitute(parsed,
                    dict((dd.varname, parse(dd.qdat.unit)) for dd in dep_data)
                    )

        if description is None:
            description = expression

        # compile and evaluate
        from pymbolic import compile
        compiled = compile(parsed, [dd.varname for dd in dep_data])

        return (description,
                unit,
                [(key, compiled(*values))
                    for key, values in _join_by_first_of_tuple(
                        dd.table for dd in dep_data)
                    ])

    def get_joint_dataset(self, expressions):
        """Return a joint data set for a list of expressions.

        @arg expressions: a list of either strings representing
          expressions directly, or triples (descr, unit, expr).
          In the former case, the description and the unit are
          found automatically, if possible. In the latter case,
          they are used as specified.
        @return: A triple C{(descriptions, units, table)}, where
        C{table} is a a list of C{[(tstep, (val_expr1, val_expr2,...)...]}.
        """

        # dubs is a list of (desc, unit, table) triples as
        # returned by get_expr_dataset
        dubs = []
        for expr in expressions:
            if isinstance(expr, str):
                dub = self.get_expr_dataset(expr)
            else:
                expr_descr, expr_unit, expr_str = expr
                dub = get_expr_dataset(
                        expr_str,
                        description=expr_descr,
                        unit=expr_unit)

            dubs.append(dub)

        zipped_dubs = list(zip(*dubs))
        zipped_dubs[2] = list(
                _join_by_first_of_tuple(zipped_dubs[2]))

        return zipped_dubs

    def save(self, filename=None):
        """Save log data to a file.

        L{synchronize_logs} is called before saving.

        @arg filename: Specify the file name. If not given, the globally set name
          is used.
        """
        self.synchronize_logs()
        if self.mpi_comm and self.mpi_comm.rank != self.head_rank:
            return

        if filename is not None:
            from os import access, R_OK
            if access(filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % filename
        else:
            filename = self.filename

        from cPickle import dump, HIGHEST_PROTOCOL
        dump((self.quantity_data, self.constants, self.is_parallel), 
                open(filename, "w"), protocol=HIGHEST_PROTOCOL)

    def load(self, filename):
        """Load saved log data from C{filename}."""
        if self.mpi_comm and self.mpi_comm.rank != self.head_rank:
            return

        from cPickle import load
        self.quantity_data, self.constants, self.is_parallel = load(open(filename))

    def get_plot_data(self, expr_x, expr_y, min_step=None, max_step=None):
        """Generate plot-ready data.

        @return: C{(data_x, descr_x, unit_x), (data_y, descr_y, unit_y)}
        """
        (descr_x, descr_y), (unit_x, unit_y), data = \
                self.get_joint_dataset([expr_x, expr_y])
        if min_step is not None:
            data = [(step, tup) for step, tup in data if min_step <= step]
        if max_step is not None:
            data = [(step, tup) for step, tup in data if step <= max_step]

        stepless_data = [tup for step, tup in data]

        data_x, data_y = zip(*stepless_data)
        return (data_x, descr_x, unit_x), \
               (data_y, descr_y, unit_y)

    def plot_gnuplot(self, gp, expr_x, expr_y, **kwargs):
        """Plot data to Gnuplot.py.

        @arg gp: a Gnuplot.Gnuplot instance to which the plot is sent.
        @arg expr_x: an allowed argument to L{get_joint_dataset}.
        @arg expr_y: an allowed argument to L{get_joint_dataset}.
        @arg kwargs: keyword arguments that are directly passed on to 
          C{Gnuplot.Data}.
        """
        (data_x, descr_x, unit_x), (data_y, descr_y, unit_y) = \
                self.get_plot_data(expr_x, expr_y)

        gp.xlabel("%s [%s]" % (descr_x, unit_x))
        gp.ylabel("%s [%s]" % (descr_y, unit_y))
        gp.plot(Data(data_x, data_y, **kwargs))

    def write_datafile(self, filename, expr_x, expr_y):
        (data_x, label_x), (data_y, label_y) = self.get_plot_data(
                expr_x, expr_y)

        outf = open(filename, "w")
        outf.write("# %s [%s] vs. %s [%s]" % 
                (descr_x, unit_x, descr_y, unit_y))
        for dx, dy in zip(data_x, data_y):
            outf.write("%s\t%s\n" % (repr(dx), repr(dy)))
        outf.close()

    def plot_matplotlib(self, expr_x, expr_y):
        from pylab import xlabel, ylabel, plot

        (data_x, descr_x, unit_x), (data_y, descr_y, unit_y) = \
                self.get_plot_data(expr_x, expr_y)

        xlabel("%s [%s]" % (descr_x, unit_x))
        ylabel("%s [%s]" % (descr_y, unit_y))
        xlabel(label_x)
        ylabel(label_y)
        plot(data_x, data_y)

    # private functionality ---------------------------------------------------
    def _parse_expr(self, expr):
        from pymbolic import parse, substitute
        parsed = parse(expr)

        # substitute in global constants
        parsed = substitute(parsed, self.constants)

        return parsed

    def _get_expr_dep_data(self, parsed):
        from pymbolic import get_dependencies

        deps = get_dependencies(parsed)

        # gather information on aggregation expressions
        dep_data = []
        from pymbolic.primitives import Variable, Lookup, Subscript
        for dep_idx, dep in enumerate(deps):
            if isinstance(dep, Variable):
                name = dep.name
                agg_func = self.quantity_data[name].default_aggregator
                if agg_func is None: 
                    if self.is_parallel:
                        raise ValueError, "must specify explicit aggregator for '%s'" % name
                    else:
                        agg_func = lambda lst: lst[0]
            elif isinstance(dep, Lookup):
                assert isinstance(dep.aggregate, Variable)
                name = dep.aggregate.name
                agg_name = dep.name
                if agg_name == "min":
                    agg_func = min
                elif agg_name == "max":
                    agg_func = max
                elif agg_name == "avg":
                    from pytools import average
                    agg_func = average
                elif agg_name == "sum":
                    agg_func = sum
                elif agg_name == "norm2":
                    from math import sqrt
                    agg_func = lambda iterable: sqrt(
                            sum(entry**2 for entry in iterable))
                else:
                    raise ValueError, "invalid rank aggregator '%s'" % agg_name
            elif isinstance(dep, Subscript):
                assert isinstance(dep.aggregate, Variable)
                name = dep.aggregate.name

                class Nth:
                    def __init__(self, n):
                        self.n = n

                    def __call__(self, lst):
                        return lst[self.n]

                from pymbolic import evaluate
                agg_func = Nth(evaluate(dep.index))

            qdat = self.quantity_data[name]

            from pytools import Record
            this_dep_data = Record(name=name, qdat=qdat, agg_func=agg_func,
                    varname="logvar%d" % dep_idx, expr=dep)
            dep_data.append(this_dep_data)

        # substitute in the "logvar" variable names
        from pymbolic import var, substitute
        parsed = substitute(parsed, 
                dict((dd.expr, var(dd.varname)) for dd in dep_data))

        return parsed, dep_data

    def _watch_tick(self):
        def get_last_value(table):
            if table:
                return table.data[-1][2]
            else:
                return 0

        data_block = dict((name, get_last_value(qdat.table))
                for name, qdat in self.quantity_data.iteritems())

        if self.mpi_comm is not None:
            from boost.mpi import broadcast, gather

            gathered_data = gather(self.mpi_comm, data_block, self.head_rank)
        else:
            gathered_data = [data_block]

        if self.rank == self.head_rank:
            values = {}
            for data_block in gathered_data:
                for name, value in data_block.iteritems():
                    values.setdefault(name, []).append(value)

            print " | ".join(
                    "%s=%g" % (watch.expr, watch.compiled(
                       *[dd.agg_func(values[dd.name]) for dd in watch.dep_data]))
                    for watch in self.watches
                    )

        ticks_per_sec = self.tick_count/max(1, time()-self.start_time)
        self.next_watch_tick = self.tick_count + int(max(1, ticks_per_sec))

        if self.mpi_comm is not None:
            self.next_watch_tick = broadcast(self.mpi_comm, 
                    self.next_watch_tick, self.head_rank)





# actual data loggers ---------------------------------------------------------
class IntervalTimer(LogQuantity):
    """Records the elapsed time between L{start} and L{stop} calls."""
    def __init__(self, name="interval", description=None):
        LogQuantity.__init__(self, name, "s", description)

        self.elapsed = 0

    def start(self):
        self.start_time = time()

    def stop(self):
        self.elapsed += time() - self.start_time
        del self.start_time

    def __call__(self):
        result = self.elapsed
        self.elapsed = 0
        return result




class LogUpdateDuration(LogQuantity):
    """Records how long the last L{LogManager.tick} invocation took."""
    def __init__(self, mgr, name="t_log"):
        LogQuantity.__init__(self, name, "s", "Time spent updating the log")
        self.log_manager = mgr

    def __call__(self):
        return self.log_manager.t_log



class EventCounter(LogQuantity):
    """Counts events signaled by L{add}."""
    
    def __init__(self, name="interval", description=None):
        LogQuantity.__init__(self, name, "1", description)
        self.events = 0

    def add(self, n=1):
        self.events += n

    def transfer(self, counter):
        self.events += counter.pop()

    def __call__(self):
        result = self.events
        self.events = 0
        return result




class TimestepCounter(LogQuantity):
    """Counts the number of times L{LogManager.tick} is called."""

    def __init__(self, name="step"):
        LogQuantity.__init__(self, name, "1", "Timesteps")
        self.steps = 0

    def __call__(self):
        result = self.steps
        self.steps += 1
        return result




class TimestepDuration(LogQuantity):
    """Records the CPU time between invocations of L{LogManager.tick}."""

    def __init__(self, name="t_step"):
        LogQuantity.__init__(self, name, "s", "Time step duration")

        self.last_start = time()

    def __call__(self):
        now = time()
        result = now - self.last_start
        self.last_start = now
        return result




class CPUTime(LogQuantity):
    """Records (monotonically increasing) CPU time."""
    def __init__(self, name="t_cpu"):
        LogQuantity.__init__(self, name, "s", "Wall time")

        self.start = time()

    def __call__(self):
        return time()-self.start




class ETA(LogQuantity):
    """Records an estimate of how long the computation will still take."""
    def __init__(self, total_steps, name="t_eta"):
        LogQuantity.__init__(self, name, "s", "Estimated remaining duration")

        self.steps = 0
        self.total_steps = total_steps
        self.start = time()

    def __call__(self):
        fraction_done = self.steps/self.total_steps
        self.steps += 1
        time_spent = time()-self.start
        if fraction_done > 1e-9:
            return time_spent/fraction_done-time_spent
        else:
            return 0




def add_general_quantities(mgr):
    """Add generally applicable L{LogQuantity} objects to C{mgr}."""

    mgr.add_quantity(TimestepDuration())
    mgr.add_quantity(CPUTime())
    mgr.add_quantity(LogUpdateDuration(mgr))
    mgr.add_quantity(TimestepCounter())




class SimulationTime(SimulationLogQuantity):
    """Record (monotonically increasing) simulation time."""

    def __init__(self, dt, name="t_sim", start=0):
        SimulationLogQuantity.__init__(self, dt, name, "s", "Simulation Time")
        self.t = 0

    def __call__(self):
        result = self.t
        self.t += self.dt
        return result




class Timestep(SimulationLogQuantity):
    """Record the magnitude of the simulated time step."""

    def __init__(self, dt, name="dt"):
        SimulationLogQuantity.__init__(self, dt, name, "s", "Simulation Timestep")

    def __call__(self):
        return self.dt



def set_dt(mgr, dt):
    """Set the simulation timestep on L{LogManager} C{mgr} to C{dt}."""

    for qdat in mgr.quantity_data.itervalues():
        if isinstance(qdat.quantity, DtConsumer):
            qdat.quantity.set_dt(dt)




def add_simulation_quantities(mgr, dt):
    """Add L{LogQuantity} objects relating to simulation time."""
    mgr.add_quantity(SimulationTime(dt))
    mgr.add_quantity(Timestep(dt))

    add_general_quantities(mgr)




def add_run_info(mgr):
    """Add generic run metadata, such as command line, host, and time."""

    import sys
    mgr.set_constant("cmdline", " ".join(sys.argv))
    from socket import gethostname
    mgr.set_constant("machine", gethostname())
    from time import localtime, strftime
    mgr.set_constant("date", strftime("%a, %d %b %Y %H:%M:%S %Z", localtime()))
