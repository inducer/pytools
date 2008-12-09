#! /usr/bin/env python




import code




try:
    import readline
    import rlcompleter
    HAVE_READLINE = True
except ImportError:
    HAVE_READLINE = False




from pytools import cartesian_product, Record
class PlotStyle(Record):
    pass




PLOT_STYLES = [
        PlotStyle(dashes=dashes, color=color)
        for dashes, color in cartesian_product(
            [(), (12, 2), (4, 2),  (2,2), (2,8) ],
            ["blue", "green", "red", "magenta", "cyan"],
            )]




class RunDB(object):
    def __init__(self, db, interactive):
        self.db = db
        self.interactive = interactive

    def q(self, qry, *extra_args):
        return self.db.execute(self.mangle_sql(qry), *extra_args)

    def mangle_sql(self, qry):
        return qry

    def scatter_cursor(self, cursor, *args, **kwargs):
        from pylab import scatter, show

        data_args = tuple(zip(*list(cursor)))
        scatter(*(data_args + args), **kwargs)

        if self.interactive:
            show()
        
    def plot_cursor(self, cursor, *args, **kwargs):
        from pylab import plot, show, legend

        auto_style = kwargs.pop("auto_style", True)

        if len(cursor.description) == 2:
            if auto_style:
                style = PLOT_STYLES[0]
                kwargs["dashes"] = style.dashes
                kwargs["color"] = style.color

            x, y = zip(*list(cursor))
            plot(x, y, *args, **kwargs)
        elif len(cursor.description) > 2:
            small_legend = kwargs.pop("small_legend", True)

            def format_label(kv_pairs):
                return " ".join("%s:%s" % (column, value)
                            for column, value in kv_pairs)
            format_label = kwargs.pop("format_label", format_label)

            def do_plot():
                my_kwargs = kwargs.copy()
                style = PLOT_STYLES[style_idx[0] % len(PLOT_STYLES)]
                if auto_style:
                    my_kwargs.setdefault("dashes", style.dashes)
                    my_kwargs.setdefault("color", style.color)

                my_kwargs.setdefault("label",
                        format_label(zip(
                            (col[0] for col in cursor.description[2:]), 
                            last_rest)))

                plot(x, y, hold=True, *args, **my_kwargs)
                style_idx[0] += 1
                del x[:]
                del y[:]

            style_idx = [0]
            x = []
            y = []
            last_rest = None
            for row in cursor:
                row_tuple = tuple(row)
                row_rest = row_tuple[2:]

                if last_rest is None:
                    last_rest = row_rest

                if row_rest != last_rest:
                    do_plot()
                    last_rest = row_rest

                x.append(row_tuple[0])
                y.append(row_tuple[1])
            if x:
                do_plot()

            if small_legend:
                from matplotlib.font_manager import FontProperties
                legend(pad=0.04, prop=FontProperties(size=8), loc="best",
                        labelsep=0)
        else:
            raise ValueError, "invalid number of columns"

        if self.interactive:
            show()
        
    def table_from_cursor(self, cursor):
        from pytools import Table
        tbl = Table()
        tbl.add_row([column[0] for column in cursor.description])
        for row in cursor:
            tbl.add_row(row)
        return tbl

    def print_cursor(self, cursor):
        print self.table_from_cursor(cursor)




class MagicRunDB(RunDB):
    def mangle_sql(self, qry):
        up_qry = qry.upper()
        if "FROM" in up_qry and not "$$" in up_qry:
            return qry

        magic_columns = set()

        def replace_magic_column(match):
            qty_name = match.group(1)
            magic_columns.add(qty_name)
            return "%s.value" % qty_name

        import re
        magic_column_re = re.compile(r"\$([a-zA-Z][A-Za-z0-9_]*)")
        qry = magic_column_re.sub(replace_magic_column, qry)

        other_clauses = ["UNION",  "INTERSECT", "EXCEPT", "WHERE", "GROUP",
                "HAVING", "ORDER", "LIMIT", ";"]

        from_clause = "from runs "
        last_tbl = None
        for tbl in magic_columns:
            if last_tbl is not None:
                addendum = " and %s.step = %s.step" % (last_tbl, tbl)
            else:
                addendum = ""

            from_clause += " inner join %s on (%s.run_id = runs.id%s) " % (
                    tbl, tbl, addendum)
            last_tbl = tbl
        
        if "$$" in qry:
            return qry.replace("$$"," %s " % from_clause)
        else:
            first_clause_idx = len(qry)+1
            up_qry = qry.upper()
            for clause in other_clauses:
                clause_match = re.search(r"\b%s\b" % clause, up_qry)
                if clause_match is not None and clause_match.start() < first_clause_idx:
                    first_clause_idx = clause_match.start()
            if first_clause_idx > len(qry):
                from_clause = " "+from_clause
            return (
                    qry[:first_clause_idx]
                    +from_clause
                    +qry[first_clause_idx:])




class RunalyzerConsole(code.InteractiveConsole):
    def __init__(self, db):
        self.db = db
        symbols = {
                "__name__": "__console__",
                "__doc__": None,
                "db": db,
                "mangle_sql": db.mangle_sql,
                "q": db.q,
                "dbplot": db.plot_cursor,
                "dbscatter": db.scatter_cursor,
                "dbprint": db.print_cursor,
                }
        code.InteractiveConsole.__init__(self, symbols)

        try:
            import pylab
            import matplotlib
            self.runsource("from pylab import *")
        except ImportError:
            pass

        if HAVE_READLINE:
            import os
            import atexit

            histfile = os.path.join(os.environ["HOME"], ".runalyzerhist")
            if os.access(histfile, os.R_OK):
                readline.read_history_file(histfile)
            atexit.register(readline.write_history_file, histfile)
            readline.parse_and_bind("tab: complete")

        self.last_push_result = False

    def push(self, cmdline):
        if cmdline.startswith("."):
            try:
                self.execute_magic(cmdline)
            except:
                import traceback
                traceback.print_exc()
        else:
            self.last_push_result = code.InteractiveConsole.push(self, cmdline)

        return self.last_push_result


    def execute_magic(self, cmdline):
        cmd_end = cmdline.find(" ")
        if cmd_end == -1:
            cmd = cmdline[1:]
            args = ""
        else:
            cmd = cmdline[1:cmd_end]
            args = cmdline[cmd_end+1:]

        if cmd == "help":
            print """
Commands:
 .help        show this help message
 .q SQL       execute a (potentially mangled) query
 .runprops    show a list of run properties
 .quantities  show a list of time-dependent quantites

Plotting:
 .plot SQL    plot results of (potentially mangled) query.
              result sets can be (x,y) or (x,y,descr1,descr2,...),
              in which case a new plot will be started for each
              tuple (descr1, descr2, ...)
 .scatter SQL make scatterplot results of (potentially mangled) query.
              result sets can have between two and four columns
              for (x,y,size,color).

SQL mangling, if requested ("MagicSQL"):
    select $quantity where pred(feature)

Custom SQLite aggregates:
    stddev, var, norm1, norm2

Available Python symbols:
    db: the SQLite database
    mangle_sql(query_str): mangle the SQL query string query_str
    q(query_str): get db cursor for mangled query_str
    dbplot(cursor): plot result of cursor
    dbscatter(cursor): make scatterplot result of cursor
    dbprint(cursor): print result of cursor
"""
        elif cmd == "q":
            self.db.print_cursor(self.db.q(args))

        elif cmd == "runprops":
            cursor = self.db.db.execute("select * from runs")
            columns = [column[0] for column in cursor.description]
            columns.sort()
            for col in columns:
                print col
        elif cmd == "quantities":
            self.db.print_cursor(self.db.q("select * from quantities order by name"))
        elif cmd == "title":
            from pylab import title
            title(args)
        elif cmd == "plot":
            self.db.plot_cursor(self.db.db.execute(
                self.db.mangle_sql(args)))
        elif cmd == "scatter":
            self.db.scatter_cursor(self.db.db.execute(
                self.db.mangle_sql(args)))
        else:
            print "invalid magic command"




# custom aggregates -----------------------------------------------------------
from pytools import VarianceAggregator
class Variance(VarianceAggregator):
    def __init__(self):
        VarianceAggregator.__init__(self, entire_pop=True)

class StdDeviation(Variance):
    def finalize(self):
        result = Variance.finalize(self)

        if result is None:
            return None
        else:
            from math import sqrt
            return sqrt(result)

class Norm1:
    def __init__(self):
        self.abs_sum = 0

    def step(self, value):
        self.abs_sum += abs(value)

    def finalize(self):
        return self.abs_sum

class Norm2:
    def __init__(self):
        self.square_sum = 0

    def step(self, value):
        self.square_sum += value**2

    def finalize(self):
        from math import sqrt
        return sqrt(self.square_sum)

def my_sprintf(format, arg):
    return format % arg




# main program ----------------------------------------------------------------
def make_wrapped_db(filename, interactive, mangle):
    import sqlite3
    db = sqlite3.connect(filename)
    db.create_aggregate("stddev", 1, StdDeviation)
    db.create_aggregate("var", 1, Variance)
    db.create_aggregate("norm1", 1, Norm1)
    db.create_aggregate("norm2", 1, Norm2)

    db.create_function("sprintf", 2, my_sprintf)

    if mangle:
        db_wrap_class = MagicRunDB
    else:
        db_wrap_class = RunDB

    return db_wrap_class(db, interactive=interactive)
