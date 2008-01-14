from __future__ import division




# abstract logging interface --------------------------------------------------
class LogQuantity:
    def __init__(self, name, unit=None, description=None):
        self.name = name
        self.unit = unit
        self.description = description

    def __call__(self):
        raise NotImplementedError

class CallableLogQuantityAdapter(LogQuantity):
    def __init__(self, callable, name, unit=None, description=None):
        self.callable = callable
        LogQuantity.__init__(self, name, unit, description)

    def __call__(self):
        return self.callable()




# manager functionality -------------------------------------------------------
class _QuantityData:
    def __init__(self, quantity, interval=1, table=None):
        self.quantity = quantity
        self.interval = interval

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




class LogManager:
    def __init__(self, filename=None, mpi_comm=None):
        """Initialize this log manager instance.

        @arg filename: If given, the log is periodically written to this file.
        @arg mpi_comm: A C{boost.mpi} communicator. If given, logs are periodically
          synchronized to the head node, which then writes them out to disk.
        """
        self.quantity_data = {}
        self.tick_count = 0
        self.filename = filename

        self.variables = {}

        if filename is not None:
            from os import access, R_OK
            if access(self.filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % self.filename

        from time import time
        self.last_checkpoint = time()

        self.mpi_comm = mpi_comm
        if mpi_comm is None:
            self.rank = 0
        else:
            self.rank = mpi_comm.rank
            self.last_sync = self.last_checkpoint

        self.t_log = 0

    def set_variable(self, name, value):
        self.variables[name] = value

    def tick(self):
        from time import time
        start_time = time()

        for qbuf in self.quantity_data.itervalues():
            if self.tick_count % qbuf.interval == 0:
                qbuf.table.insert_row((self.tick_count, self.rank, qbuf.quantity()))
        self.tick_count += 1

        end_time = time()
        self.t_log = end_time - start_time

        # synchronize logs with parallel peers, if necessary
        if self.mpi_comm is not None:
            if end_time - self.last_sync > 10:
                self.synchronize_logs()
                self.last_sync = end_time

        # checkpoint log to disk, if necessary
        if self.filename is not None:
            if end_time - self.last_checkpoint > 10:
                self.save()
                self.last_checkpoint = end_time

    def synchronize_logs(self):
        """Send logs to head node."""
        if self.mpi_comm is None:
            return

        from boost.mpi import gather
        root = 0
        if self.mpi_comm.rank == root:
            for rank_data in gather(self.mpi_comm, None, root)[1:]:
                for name, rows in rank_data:
                    self.quantity_data[name].insert_rows(rows)
        else:
            gather(self.mpi_comm, 
                    [(name, qdat.table.data)
                        for name, qdat in self.quantity_data.iteritems()], 
                root)

    def add_quantity(self, quantity, interval=1):
        """Add an object derived from L{LogQuantity} to this manager."""
        self.quantity_data[quantity.name] = _QuantityData(quantity, interval)

    def get_expr_dataset(self, expression, description=None, unit=None):
        """Return a triple C{(description, unit, table)} for the given expression.

        C{table} is a list of tuples C{(tick_nbr, value)}.
        """

        from pymbolic import parse, get_dependencies, substitute

        parsed = parse(expression)
        deps = get_dependencies(parsed)

        # gather information on aggregation expressions
        dep_data = []
        from pymbolic.primitives import Variable, Lookup, Subscript
        for dep_idx, dep in enumerate(deps):
            if isinstance(dep, Variable):
                name = dep.name
                from pytools import average
                agg_func = average
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

            quantity = self.quantity_data[name].quantity
            table = self.quantity_data[name].table
            table.sort(["step"])
            table = table.aggregated(["step"], "value", agg_func).data

            from pytools import Record
            this_dep_data = Record(table=table, quantity=quantity, 
                    varname="logvar%d" % dep_idx, expr=dep)
            dep_data.append(this_dep_data)

        # evaluate unit and description, if necessary
        if unit is None:
            unit = substitute(parsed,
                    dict((dd.expr, parse(dd.quantity.unit)) for dd in dep_data)
                    )

        if description is None:
            description = expression

        # substitute in the "logvar" variable names
        from pymbolic import var
        parsed = substitute(parsed, 
                dict((dd.expr, var(dd.varname)) for dd in dep_data))

        # substitute in global variables
        parsed = substitute(parsed, self.variables)

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
        self.synchronize_logs()
        if self.mpi_comm and not self.mpi_comm.rank != 0:
            return

        if filename is not None:
            from os import access, R_OK
            if access(filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % filename
        else:
            filename = self.filename

        save_buffers = dict(
                (name, _QuantityData(
                    LogQuantity(
                        qbuf.quantity.name,
                        qbuf.quantity.unit,
                        qbuf.quantity.description,
                        ),
                    qbuf.interval,
                    qbuf.table))
                for name, qbuf in self.quantity_data.iteritems())

        from cPickle import dump, HIGHEST_PROTOCOL
        dump((save_buffers, self.variables), 
                open(filename, "w"), protocol=HIGHEST_PROTOCOL)

    def load(self, filename):
        if self.mpi_comm and not self.mpi_comm.rank != 0:
            return

        from cPickle import load
        self.quantity_data, self.variables = load(open(filename))

    def get_plot_data(self, expr_x, expr_y):
        """Generate plot-ready data.

        @return: C{(data_x, descr_x, unit_x), (data_y, descr_y, unit_y)}
        """
        (descr_x, descr_y), (unit_x, unit_y), data = \
                self.get_joint_dataset([expr_x, expr_y])
        _, stepless_data = zip(*data)
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



# actual data loggers ---------------------------------------------------------
class IntervalTimer(LogQuantity):
    def __init__(self, name="interval", description=None):
        LogQuantity.__init__(self, name, "s", description)

        self.elapsed = 0

    def start(self):
        from time import time
        self.start_time = time()

    def stop(self):
        from time import time
        self.elapsed += time() - self.start_time
        del self.start_time

    def __call__(self):
        result = self.elapsed
        self.elapsed = 0
        return result




class LogUpdateDuration(LogQuantity):
    def __init__(self, mgr, name="t_log"):
        LogQuantity.__init__(self, name, "s", "Time spent updating the log")
        self.log_manager = mgr

    def __call__(self):
        return self.log_manager.t_log



class EventCounter(LogQuantity):
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
    def __init__(self, name="step"):
        LogQuantity.__init__(self, name, "1", "Timesteps")
        self.steps = 0

    def __call__(self):
        result = self.steps
        self.steps += 1
        return result




class TimestepDuration(LogQuantity):
    def __init__(self, name="t_step"):
        LogQuantity.__init__(self, name, "s", "Time step duration")

        from time import time
        self.last_start = time()

    def __call__(self):
        from time import time
        now = time()
        result = now - self.last_start
        self.last_start = now
        return result




class WallTime(LogQuantity):
    def __init__(self, name="t_wall"):
        LogQuantity.__init__(self, name, "s", "Wall time")

        from time import time
        self.start = time()

    def __call__(self):
        from time import time
        return time()-self.start




class SimulationTime(LogQuantity):
    def __init__(self, dt, name="t_sim", start=0):
        LogQuantity.__init__(self, name, "s", "Simulation Time")
        self.dt = dt
        self.t = 0

    def set_dt(self, dt):
        self.dt = dt

    def __call__(self):
        result = self.t
        self.t += self.dt
        return result




class Timestep(LogQuantity):
    def __init__(self, dt, name="dt"):
        LogQuantity.__init__(self, name, "s", "Simulation Timestep")
        self.dt = dt

    def set_dt(self, dt):
        self.dt = dt

    def __call__(self):
        return self.dt



def set_dt(mgr, dt):
    mgr.quantity_data["dt"].quantity.set_dt(dt)
    mgr.quantity_data["t_sim"].quantity.set_dt(dt)




def add_general_quantities(mgr, dt):
    mgr.add_quantity(TimestepDuration())
    mgr.add_quantity(WallTime())
    mgr.add_quantity(SimulationTime(dt))
    mgr.add_quantity(Timestep(dt))
    mgr.add_quantity(LogUpdateDuration(mgr))
    mgr.add_quantity(TimestepCounter())



def add_run_info(mgr):
    import sys
    mgr.set_variable("cmdline", " ".join(sys.argv))
    from socket import gethostname
    mgr.set_variable("machine", gethostname())
    from time import localtime, strftime
    mgr.set_variable("date", strftime("%a, %d %b %Y %H:%M:%S %Z", localtime()))
