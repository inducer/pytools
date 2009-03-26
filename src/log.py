from __future__ import division




# timing function -------------------------------------------------------------
def time():
    """Return elapsed CPU time, as a float, in seconds."""
    import os
    time_opt = os.environ.get("PYTOOLS_LOG_TIME") or "wall"
    if time_opt == "wall":
        from time import time
        return time()
    elif time_opt == "rusage":
        from resource import getrusage, RUSAGE_SELF
        return getrusage(RUSAGE_SELF).ru_utime
    else:
        raise RuntimeError, "invalid timing method '%s'" % time_opt




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
        L{LogQuantity} or None if no value is available."""
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
    




class PushLogQuantity(LogQuantity):
    def __init__(self, name, unit=None, description=None):
        LogQuantity.__init__(self, name, unit, description)
        self.value = None

    def push_value(self, value):
        if self.value is not None:
            raise RuntimeError, "can't push two values per cycle"
        self.value = value

    def __call__(self):
        v = self.value
        self.value = None
        return v

        


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
    def __init__(self, unit, description, default_aggregator):
        self.unit = unit
        self.description = description
        self.default_aggregator = default_aggregator




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

    def __init__(self, filename=None, mode="r", mpi_comm=None, capture_warnings=True):
        """Initialize this log manager instance.

        @arg filename: If given, the filename to which this log is bound.
          If this database exists, the current state is loaded from it.
        @arg mode: One of "w", "r" for write, read. "w" assumes that the 
          database is initially empty.
        @arg mpi_comm: An C{boostmpi} communicator. If given, logs are periodically
          synchronized to the head node, which then writes them out to disk.
        @arg capture_warnings: Tap the Python warnings facility and save warnings
          to the log file.
        """

        assert isinstance(mode, basestring), "mode must be a string"
        mode = mode[0]
        assert mode in ["w", "r"], "invalid mode"

        self.quantity_data = {}
        self.quantity_table = {}
        self.gather_descriptors = []
        self.tick_count = 0

        self.constants = {}

        # self-timing
        self.start_time = time()
        self.t_log = 0

        # parallel support
        self.head_rank = 0
        self.mpi_comm = mpi_comm
        self.is_parallel = mpi_comm is not None

        if mpi_comm is None:
            self.rank = 0
        else:
            self.rank = mpi_comm.rank
            self.head_rank = 0

        self.next_sync_tick = 10

        # watch stuff
        self.watches = []
        self.next_watch_tick = 1

        # database binding
        if filename is not None and self.rank == self.head_rank:
            try:
                import sqlite3 as sqlite
            except ImportError:
                try:
                    from pysqlite2 import dbapi2 as sqlite
                except ImportError:
                    raise ImportError, "could not find a usable version of sqlite."

            self.db_conn = sqlite.connect(filename, timeout=30)
            self.mode = mode
            try:
                self.db_conn.execute("select * from quantities;")
                if mode == "w":
                    raise RuntimeError, "Log database '%s' already exists" % filename
                self._load()
            except sqlite.OperationalError:
                if mode == "r":
                    raise RuntimeError, "Log database '%s' not found" % filename

                # initialize new database
                self.db_conn.execute("""
                  create table quantities (
                    name text, 
                    unit text, 
                    description text,
                    default_aggregator blob)""")
                self.db_conn.execute("""
                  create table constants (
                    name text, 
                    value blob)""")
                self.db_conn.execute("""
                  create table warnings (
                    step integer,
                    message text, 
                    category text,
                    filename text,
                    lineno integer
                    )""")
                self.set_constant("is_parallel", self.is_parallel)
                self.schema_version = 1
                self.set_constant("schema_version", self.schema_version)
        else:
            self.db_conn = None

        self.old_showwarning = None
        if capture_warnings:
            self.capture_warnings(True)

    def capture_warnings(self, enable=True):
        # FIXME warning capture on multiple processors

        import warnings
        if enable:
            if self.old_showwarning is None:
                pass
                self.old_showwarning = warnings.showwarning
                warnings.showwarning = self._showwarning
            else:
                raise RuntimeError, "Warnings capture was enabled twice"
        else:
            if self.old_showwarning is None:
                raise RuntimeError, "Warnings capture was disabled, but never enabled"
            else:
                warnings.showwarning = self.old_showwarning
                self.old_showwarning = None

    def _showwarning(self, message, category, filename, line):
        self.old_showwarning(message, category, filename, line)

        if (self.db_conn is not None 
                and self.schema_version >= 1 
                and self.mode == "w"):
            self.db_conn.execute("insert into warnings values (?,?,?,?,?)",
                    (self.tick_count, message.message, str(category), filename, lineno))

    def _load(self):
        if self.mpi_comm and self.mpi_comm.rank != self.head_rank:
            return

        from pickle import loads
        for name, value in self.db_conn.execute("select name, value from constants"):
            self.constants[name] = loads(value)

        self.schema_version = self.constants.get("schema_version", 0)

        self.is_parallel = self.constants["is_parallel"]

        for name, unit, description, def_agg in self.db_conn.execute(
                "select name, unit, description, default_aggregator from quantities"):
            qdat = self.quantity_data[name] = _QuantityData(
                    unit, description, loads(def_agg))

    def close(self):
        if self.db_conn is not None:
            self.db_conn.close()

    def get_table(self, q_name):
        if q_name not in self.quantity_data:
            raise KeyError, "invalid quantity name '%s'" % q_name

        try:
            return self.quantity_table[q_name]
        except KeyError:
            from pytools.datatable import DataTable
            result = self.quantity_table[q_name] = DataTable(["step", "rank", "value"])

            if self.db_conn is not None:
                for row in self.db_conn.execute(
                        "select step, rank, value from %s" % q_name):
                    result.insert_row(row)

            return result

    def get_warnings(self):
        from pytools.datatable import DataTable
        result = DataTable(["step", "message", "category", "filename", "lineno"])

        if self.schema_version >= 1 and self.db_conn is not None:
            for row in self.db_conn.execute(
                    "select step, message, category, filename, lineno from warnings"):
                result.insert_row(row)

        return result

    def add_watches(self, watches):
        """Add quantities that are printed after every time step."""

        from pytools import Record
        class WatchInfo(Record): pass

        for watch in watches:
            if isinstance(watch, tuple):
                display, expr = watch
            else:
                display = watch
                expr = watch

            parsed = self._parse_expr(expr)
            parsed, dep_data = self._get_expr_dep_data(parsed)

            from pymbolic import compile
            compiled = compile(parsed, [dd.varname for dd in dep_data])

            watch_info = WatchInfo(display=display, parsed=parsed, dep_data=dep_data,
                    compiled=compiled)

            self.watches.append(watch_info)

    def set_constant(self, name, value):
        """Make a named, constant value available in the log."""
        existed = name in self.constants
        self.constants[name] = value

        if self.db_conn is not None:
            from pickle import dumps
            value = buffer(dumps(value))

            if existed:
                self.db_conn.execute("update constants set value = ? where name = ?",
                        (value, name))
            else:
                self.db_conn.execute("insert into constants values (?,?)",
                        (name, value))

            self.db_conn.commit()

    def tick(self):
        """Record data points from each added L{LogQuantity}.

        May also checkpoint data to disk, and/or synchronize data points 
        to the head rank.
        """
        start_time = time()

        def insert_datapoint(name, value):
            if value is None:
                return

            self.get_table(name).insert_row(
                (self.tick_count, self.rank, value))
            if self.db_conn is not None:
                try:
                    self.db_conn.execute("insert into %s values (?,?,?)" % name,
                            (self.tick_count, self.rank, value))
                except:
                    print "while adding datapoint for '%s':" % name
                    raise

        for gd in self.gather_descriptors:
            if self.tick_count % gd.interval == 0:
                q_value = gd.quantity()
                if isinstance(gd.quantity, MultiLogQuantity):
                    for name, value in zip(gd.quantity.names, q_value):
                        insert_datapoint(name, value)
                else:
                    insert_datapoint(gd.quantity.name, q_value)
        self.tick_count += 1

        end_time = time()

        # print watches
        if self.tick_count == self.next_watch_tick:
            self._watch_tick()

        if self.tick_count == self.next_sync_tick:
            # sync every few seconds:
            self.save()

            # figure out next sync tick, broadcast to peers
            ticks_per_10_sec = 10*self.tick_count/max(1, end_time-self.start_time)
            self.next_sync_tick = self.tick_count + int(max(50, ticks_per_10_sec))
            if self.mpi_comm is not None:
                from boostmpi import broadcast
                self.next_sync_tick = broadcast(self.mpi_comm, self.next_sync_tick, self.head_rank)

        self.t_log = time() - start_time

    def save(self):
        self.synchronize_logs()

        if self.db_conn is not None:
            # then, to disk
            self.db_conn.commit()

    def synchronize_logs(self):
        """Transfer data from client ranks to the head rank.
        
        Must be called on all ranks simultaneously."""
        if self.mpi_comm is None:
            return

        from boostmpi import gather
        if self.mpi_comm.rank == self.head_rank:
            for rank_data in gather(self.mpi_comm, None, self.head_rank)[1:]:
                for name, rows in rank_data:
                    self.get_table(name).insert_rows(rows)
                    if self.db_conn is not None:
                        for row in rows:
                            self.db_conn.execute(
                                    "insert into %s values (?,?,?)" % name, row)
        else:
            # send non-head data away
            gather(self.mpi_comm, 
                    [(name, self.get_table(name).data)
                        for name, qdat in self.quantity_data.iteritems()], 
                self.head_rank)

            # and erase it
            for qname in self.quantity_data.iterkeys():
                self.get_table(qname).clear()

    def add_quantity(self, quantity, interval=1):
        """Add an object derived from L{LogQuantity} to this manager."""
        def add_internal(name, unit, description, def_agg):
            if name in self.quantity_data:
                raise RuntimeError("cannot add the same quantity '%s' twice" % name)
            self.quantity_data[name] = _QuantityData(
                    unit, description, def_agg)

            if self.db_conn is not None:
                from pickle import dumps
                self.db_conn.execute("""insert into quantities values (?,?,?,?)""", (
                      name, unit, description,
                      buffer(dumps(def_agg))))
                self.db_conn.execute("""create table %s 
                  (step integer, rank integer, value real)""" % name)

                self.db_conn.commit()

        self.gather_descriptors.append(_GatherDescriptor(quantity, interval))

        if isinstance(quantity, MultiLogQuantity):
            for name, unit, description, def_agg in zip(
                    quantity.names,
                    quantity.units,
                    quantity.descriptions,
                    quantity.default_aggregators):
                add_internal(name, unit, description, def_agg)
        else:
            add_internal(quantity.name, 
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
            table = self.get_table(dd.name)
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

        data = []

        for key, values in _join_by_first_of_tuple(dd.table for dd in dep_data):
            try:
                data.append((key, compiled(*values)))
            except ZeroDivisionError:
                pass

        return (description, unit, data)

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

        if stepless_data:
            data_x, data_y = zip(*stepless_data)
        else:
            data_x = []
            data_y = []

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
        from pymbolic.mapper.dependency import DependencyMapper

        deps = DependencyMapper()(parsed)

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
            class DependencyData(Record): pass

            this_dep_data = DependencyData(name=name, qdat=qdat, agg_func=agg_func,
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

        data_block = dict((qname, get_last_value(self.get_table(qname)))
                for qname in self.quantity_data.iterkeys())

        if self.mpi_comm is not None:
            from boostmpi import broadcast, gather

            gathered_data = gather(self.mpi_comm, data_block, self.head_rank)
        else:
            gathered_data = [data_block]

        if self.rank == self.head_rank:
            values = {}
            for data_block in gathered_data:
                for name, value in data_block.iteritems():
                    values.setdefault(name, []).append(value)

            def compute_watch_str(watch):
                try:
                    return "%s=%g" % (watch.display, watch.compiled(
                        *[dd.agg_func(values[dd.name]) 
                            for dd in watch.dep_data]))
                except ZeroDivisionError:
                    return "%s:div0" % watch.display

            if self.watches:
                print " | ".join(
                        compute_watch_str(watch)
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

    def add_time(self, t):
        self.elapsed += t

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





def time_and_count_function(f, timer, counter=None, increment=1):
    def inner_f(*args, **kwargs):
        if counter is not None:
            counter.add(increment)
        timer.start()
        try:
            return f(*args, **kwargs)
        finally:
            timer.stop()

    return inner_f






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




def add_run_info(mgr):
    """Add generic run metadata, such as command line, host, and time."""

    import sys
    mgr.set_constant("cmdline", " ".join(sys.argv))
    from socket import gethostname
    mgr.set_constant("machine", gethostname())
    from time import localtime, strftime, time
    mgr.set_constant("date", strftime("%a, %d %b %Y %H:%M:%S %Z", localtime()))
    mgr.set_constant("unixtime", time())
