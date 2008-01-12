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
class _QuantityBuffer:
    def __init__(self, quantity, interval=1, buffer=[]):
        self.quantity = quantity
        self.interval = interval
        self.buffer = buffer[:]

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
    def __init__(self, filename=None):
        self.quantity_buffers = {}
        self.tick_count = 0
        self.filename = filename

        self.variables = {}

        if filename is not None:
            from os import access, R_OK
            if access(self.filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % self.filename

        from time import time
        self.last_checkpoint = time()

    def set_variable(self, name, value):
        self.variables[name] = value

    def tick(self):
        for qbuf in self.quantity_buffers.itervalues():
            if self.tick_count % qbuf.interval == 0:
                qbuf.buffer.append((self.tick_count, qbuf.quantity()))
        self.tick_count += 1

        if self.filename is not None:
            from time import time
            now = time()
            if now - self.last_checkpoint > 10:
                self.save()
                self.last_checkpoint = now

    def add_quantity(self, quantity, interval=1):
        """Add an object derived from L{LogQuantity} to this manager."""
        self.quantity_buffers[quantity.name] = _QuantityBuffer(quantity, interval)

    def get_expr_dataset(self, expression, description=None, unit=None):
        """Return a triple C{(description, unit, buffer)} for the given expression.

        C{buffer} consists of a list of tuples C{(tick_nbr, value)}.
        """
        try:
            qbuf = self.quantity_buffers[expression]
        except KeyError:
            from pymbolic import parse, get_dependencies, evaluate, \
                    var, substitute
            parsed = parse(expression)
            deps = [dep.name for dep in get_dependencies(parsed)]

            if unit is None:
                unit = substitute(parsed,
                        dict((name, 
                            var(self.quantity_buffers[name].quantity.unit))
                            for name in deps))
            if description is None:
                description = expression

            def make_eval_context(seq):
                ctx = dict(seq)
                ctx.update(self.variables)
                return ctx

            return (description,
                    unit,
                    [(key, evaluate(parsed,
                        make_eval_context((name, value) 
                            for name, value in zip(deps, values))
                        ))

                        for key, values in _join_by_first_of_tuple(
                            self.quantity_buffers[dep].buffer for dep in deps)
                        ])
        else:
            return (description or qbuf.quantity.description,
                    unit or qbuf.quantity.unit,
                    qbuf.buffer)

    def get_joint_dataset(self, expressions):
        """Return a joint data set for a list of expressions.

        @arg expressions: a list of either strings representing
          expressions directly, or triples (descr, unit, expr).
          In the former case, the description and the unit are
          found automatically, if possible. In the latter case,
          they are used as specified.
        @return: A triple C{(descriptions, units, buffer)}, where
        C{buffer} is a a list of C{[(tstep, (val_expr1, val_expr2,...)...]}.
        """

        # dubs is a list of (desc, unit, buffer) triples as
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
        if filename is not None:
            from os import access, R_OK
            if access(filename, R_OK):
                raise IOError, "cowardly refusing to overwrite '%s'" % filename
        else:
            filename = self.filename

        save_buffers = dict(
                (name, _QuantityBuffer(
                    LogQuantity(
                        qbuf.quantity.name,
                        qbuf.quantity.unit,
                        qbuf.quantity.description,
                        ),
                    qbuf.interval,
                    qbuf.buffer))
                for name, qbuf in self.quantity_buffers.iteritems())

        from cPickle import dump, HIGHEST_PROTOCOL
        dump((save_buffers, self.variables), 
                open(filename, "w"), protocol=HIGHEST_PROTOCOL)

    def load(self, filename):
        from cPickle import load
        self.quantity_buffers, self.variables = load(open(filename))

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
    mgr.quantity_buffers["dt"].quantity.set_dt(dt)
    mgr.quantity_buffers["t_sim"].quantity.set_dt(dt)




def add_general_quantities(mgr, dt):
    mgr.add_quantity(TimestepDuration())
    mgr.add_quantity(WallTime())
    mgr.add_quantity(SimulationTime(dt))
    mgr.add_quantity(Timestep(dt))
    mgr.add_quantity(TimestepCounter())




