from __future__ import division
import math
import sys
import operator
import types
from pytools.decorator import decorator




def delta(x,y):
    if x == y:
        return 1
    else:
        return 0




def factorial(n):
    from operator import mul
    assert n == int(n)
    return reduce(mul, (i for i in xrange(1,n+1)), 1)




def norm_1(iterable):
    return sum(abs(x) for x in iterable)

def norm_2(iterable):
    return sum(x**2 for x in iterable)**0.5

def norm_inf(iterable):
    return max(abs(x) for x in iterable)

def norm_p(iterable, p):
    return sum(i**p for i in iterable)**(1/p)




# Data structures ------------------------------------------------------------
class Record(object):
    def __init__(self, valuedict=None, exclude=["self"], **kwargs):
        try:
            fields = self.__class__.fields
        except AttributeError:
            self.__class__.fields = fields = set()

        if valuedict is not None:
            kwargs.update(valuedict)

        for key, value in kwargs.iteritems():
            if not key in exclude:
                fields.add(key)
                setattr(self, key, value)

    def copy(self, **kwargs):
        for f in self.__class__.fields:
            if f not in kwargs:
                kwargs[f] = getattr(self, f)
        return self.__class__(**kwargs)

    def __getstate__(self):
        return dict(
                (key, getattr(self, key))
                for key in self.__class__.fields)

    def __setstate__(self, valuedict):
        try:
            fields = self.__class__.fields
        except AttributeError:
            self.__class__.fields = fields = set()

        for key, value in valuedict.iteritems():
            fields.add(key)
            setattr(self, key, value)



class Reference(object):
    def __init__( self, value ):
        self.V = value
    def get( self ):
        return self.V
    def set( self, value ):
        self.V = value





def _getattr_(obj, name, default_thunk):
    "Similar to .setdefault in dictionaries."
    try:
        return getattr(obj, name)
    except AttributeError:
        default = default_thunk()
        setattr(obj, name, default)
        return default




@decorator
def memoize(func, *args):
    # by Michele Simionato
    # http://www.phyast.pitt.edu/~micheles/python/

    dic = _getattr_(func, "memoize_dic", dict)

    # memoize_dic is created at the first call
    if args in dic:
        return dic[args]
    else:
        result = func(*args)
        dic[args] = result
        return result
FunctionValueCache = memoize




class DictionaryWithDefault(object):
    def __init__(self, default_value_generator, start = {}):
        self._Dictionary = dict(start)
        self._DefaultGenerator = default_value_generator

    def __getitem__(self, index):
        try:
            return self._Dictionary[index]
        except KeyError:
            value = self._DefaultGenerator(index)
            self._Dictionary[index] = value
            return value

    def __setitem__(self, index, value):
        self._Dictionary[index] = value

    def __contains__(self, item):
        return True

    def iterkeys(self):
        return self._Dictionary.iterkeys()

    def __iter__(self):
        return self._Dictionary.__iter__()

    def iteritems(self):
        return self._Dictionary.iteritems()


    
class FakeList(object):
    def __init__(self, f, length):
        self._Length = length
        self._Function = f

    def __len__(self):
        return self._Length

    def __getitem__(self, index):
        try:
            return [self._Function(i)
                    for i in range(*index.indices(self._Length))]
        except AttributeError:
            return self._Function(index)




class DependentDictionary(object):
    def __init__(self, f, start = {}):
        self._Function = f
        self._Dictionary = start.copy()

    def copy(self):
        return DependentDictionary(self._Function, self._Dictionary)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        try:
            return self._Dictionary[key]
        except KeyError:
            return self._Function(self._Dictionary, key)

    def __setitem__(self, key, value):
        self._Dictionary[key] = value
    
    def genuineKeys(self):
        return self._Dictionary.keys()

    def iteritems(self):
        return self._Dictionary.iteritems()

    def iterkeys(self):
        return self._Dictionary.iterkeys()

    def itervalues(self):
        return self._Dictionary.itervalues()




def add_tuples(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

def negate_tuple(t1):
    return tuple([-t1v for t1v in t1])





def shift(vec, dist):
    """Return a copy of C{vec} shifted by C{dist}. 

    @postcondition: C{shift(a, i)[j] == a[(i+j) % len(a)]}
    """

    result = vec[:]

    N = len(vec)
    dist = dist % N

    # modulo only returns positive distances!
    if dist > 0:
        result[dist:] = vec[:N-dist]
        result[:dist] = vec[N-dist:]

    return result




# plotting --------------------------------------------------------------------
def write_1d_gnuplot_graph(f, a, b, steps=100, fname=",,f.data", progress = False):
    h = float(b - a)/steps
    gnuplot_file = file(fname, "w")

    def do_plot(func):
        for n in range(steps):
            if progress:
                sys.stdout.write(".")
                sys.stdout.flush()
            x = a + h * n
            gnuplot_file.write("%f\t%f\n" % (x, func(x)))

    do_plot(f)
    if progress:
        sys.stdout.write("\n")

def write_1d_gnuplot_graphs(f, a, b, steps=100, fnames=None, progress=False):
    h = float(b - a)/steps
    if not fnames:
        result_count = len(f(a))
        fnames = [",,f%d.data" % i for i in range(result_count)]

    gnuplot_files = [file(fname, "w") for fname in fnames]

    for n in range(steps):
        if progress:
            sys.stdout.write(".")
            sys.stdout.flush()
        x = a + h * n
        for gpfile, y in zip(gnuplot_files, f(x)):
            gpfile.write("%f\t%f\n" % (x, y))
    if progress:
        sys.stdout.write("\n")



def write_2d_gnuplot_graph(f, (x0, y0), (x1, y1), (xsteps, ysteps)=(100, 100), fname=",,f.data"):
    hx = float(x1 - x0)/xsteps
    hy = float(y1 - y0)/ysteps
    gnuplot_file = file(fname, "w")

    for ny in range(ysteps):
        for nx in range(xsteps):
            x = x0 + hx * nx
            y = y0 + hy * ny
            gnuplot_file.write("%g\t%g\t%g\n" % (x, y, f(x, y)))

        gnuplot_file.write("\n")


def write_gnuplot_graph(f, a, b, steps = 100, fname = ",,f.data", progress = False):
    h = float(b - a)/steps
    gnuplot_file = file(fname, "w")

    def do_plot(func):
        for n in range(steps):
            if progress:
                sys.stdout.write(".")
                sys.stdout.flush()
            x = a + h * n
            gnuplot_file.write("%f\t%f\n" % (x, func(x)))

    if isinstance(f, types.ListType):
        for f_index, real_f in enumerate(f):
            if progress:
                sys.stdout.write("function %d: " % f_index)
            do_plot(real_f)
            gnuplot_file.write("\n")
            if progress:
                sys.stdout.write("\n")
    else:
        do_plot(f)
        if progress:
            sys.stdout.write("\n")




# syntactical sugar -----------------------------------------------------------
class InfixOperator:
    """Pseudo-infix operators that allow syntax of the kind `op1 <<operator>> op2'.
    
    Following a recipe from
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/384122
    """
    def __init__(self, function):
        self.function = function
    def __rlshift__(self, other):
        return InfixOperator(lambda x: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def call(self, a, b):
        return self.function(a, b)




def monkeypatch_method(cls):
    # from GvR, http://mail.python.org/pipermail/python-dev/2008-January/076194.html
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator




def monkeypatch_class(name, bases, namespace):
    # from GvR, http://mail.python.org/pipermail/python-dev/2008-January/076194.html

    assert len(bases) == 1, "Exactly one base class required"
    base = bases[0]
    for name, value in namespace.iteritems():
        if name != "__metaclass__":
            setattr(base, name, value)
    return base




# Generic utilities ----------------------------------------------------------
def len_iterable(iterable):
    return sum(1 for i in iterable)




def flatten(list):
    """For an iterable of sub-iterables, generate each member of each 
    sub-iterable in turn, i.e. a flattened version of that super-iterable.

    Example: Turn [[a,b,c],[d,e,f]] into [a,b,c,d,e,f].
    """
    for sublist in list:
        for j in sublist:
            yield j




def general_sum(sequence):
    return reduce(operator.add, sequence)




def linear_combination(coefficients, vectors):
    result = coefficients[0] * vectors[0]
    for c,v in zip(coefficients, vectors)[1:]:
        result += c*v
    return result




def average(iterable):
    """Return the average of the values in iterable.

    iterable may not be empty.
    """
    it = iterable.__iter__()
    
    try:
        sum = it.next()
        count = 1
    except StopIteration:
        raise ValueError, "empty average"

    for value in it:
        sum += value
        count += 1

    return sum/count




def all_equal(iterable):
    it = iterable.__iter__()
    try:
        value = it.next()
    except StopIteration:
        return True # empty sequence

    for i in it:
        if i != value:
            return False
    return True




def all_roughly_equal(iterable, threshold):
    it = iterable.__iter__()
    try:
        value = it.next()
    except StopIteration:
        return True # empty sequence

    for i in it:
        if abs(i - value) > threshold:
            return False
    return True




def decorate(function, list):
    return map(lambda x: (x, function(x)), list)




def partition(criterion, list):
    part_true = []
    part_false = []
    for i in list:
        if criterion(i):
            part_true.append(i)
        else:
            part_false.append(i)
    return part_true, part_false




def product(iterable):
    from operator import mul
    return reduce(mul, iterable, 1)




def argmin_f(list, f = lambda x: x):
    # deprecated -- the function has become unnecessary because of
    # generator expressions
    current_min_index = -1
    current_min = f(list[0])

    for idx, item in enumerate(list[1:]):
        value = f(item)
        if value < current_min:
            current_min_index = idx
            current_min = value
    return current_min_index+1




def argmax_f(list, f = lambda x: x):
    # deprecated -- the function has become unnecessary because of
    # generator expressions
    current_max_index = -1
    current_max = f(list[0])

    for idx, item in enumerate(list[1:]):
        value = f(item)
        if value > current_max:
            current_max_index = idx
            current_max = value
    return current_max_index+1




def argmin(list):
    current_min_index = -1
    it = list.__iter__()
    try:
        current_min = it.next()
    except StopIteration:
        raise ValueError, "argmin of empty iterable"
    current_min_index = -1

    for idx, item in enumerate(it):
        value = item
        if value < current_min:
            current_min_index = idx
            current_min = value
    return current_min_index+1




def argmax(list):
    it = list.__iter__()
    try:
        current_max = it.next()
    except StopIteration:
        raise ValueError, "argmax of empty iterable"
    current_max_index = -1

    for idx, item in enumerate(it):
        value = item
        if value > current_max:
            current_max_index = idx
            current_max = value
    return current_max_index+1




def cartesian_product(list1, list2):
    for i in list1:
        for j in list2:
            yield (i,j)





def distinct_pairs(list1, list2):
    for i, xi in enumerate(list1):
        for j, yj in enumerate(list2):
            if i != j:
                yield (xi, yj)




def cartesian_product_sum(list1, list2):
    """This routine returns a list of sums of each element of
    list1 with each element of list2. Also works with lists.
    """
    for i in list1:
        for j in list2:
            yield i+j




def reverse_dictionary(the_dict):
    result = {}
    for key, value in the_dict.iteritems():
        if value in result:
            raise RuntimeError, "non-reversible mapping"
        result[value] = key
    return result




def wandering_element(length, wanderer=1, landscape=0):
    for i in range(length):
        yield i*(landscape,) + (wanderer,) + (length-1-i)*(landscape,)




def generate_nonnegative_integer_tuples_below(n, length, least=0):
    assert length >= 0
    if length == 0:
        yield ()
    else:
        for i in range(least, n):
            for base in generate_nonnegative_integer_tuples_below(n, length-1, least):
                yield (i,) + base

def generate_decreasing_nonnegative_tuples_summing_to(n, length, min=0, max=None):
    sig = (n,length,max)
    if length == 0:
        yield ()
    elif length == 1:
        if n <= max:
            #print "MX", n, max
            yield (n,)
        else:
            return
    else:
        if max is None or n < max:
            max = n

        for i in range(min, max+1):
            #print "SIG", sig, i
            for remainder in generate_decreasing_nonnegative_tuples_summing_to(
                    n-i, length-1, min, i):
                yield (i,) + remainder


def generate_nonnegative_integer_tuples_summing_to_at_most(n, length):
    """Enumerate all non-negative integer tuples summing to at most n,
    exhausting the search space by varying the first entry fastest,
    and the last entry the slowest.
    """
    assert length >= 0
    if length == 0:
        yield ()
    else:
        for i in range(n+1):
            for remainder in generate_nonnegative_integer_tuples_summing_to_at_most(
                    n-i, length-1):
                yield remainder + (i,)

def generate_all_nonnegative_integer_tuples(length, least=0):
    assert length >= 0
    current_max = least
    while True:
        for max_pos in range(length):
            for prebase in generate_nonnegative_integer_tuples_below(current_max, max_pos, least):
                for postbase in generate_nonnegative_integer_tuples_below(current_max+1, length-max_pos-1, least):
                    yield prebase + [current_max] + postbase
        current_max += 1

 # backwards compatibility
generate_positive_integer_tuples_below = generate_nonnegative_integer_tuples_below
generate_all_positive_integer_tuples = generate_all_nonnegative_integer_tuples

def _pos_and_neg_adaptor(tuple_iter):
    for tup in tuple_iter:
        nonzero_indices = [i for i in range(len(tup)) if tup[i] != 0]
        for do_neg_tup in generate_nonnegative_integer_tuples_below(2, len(nonzero_indices)):
            this_result = list(tup)
            for index, do_neg in enumerate(do_neg_tup):
                if do_neg:
                    this_result[nonzero_indices[index]] *= -1
            yield tuple(this_result)

def generate_all_integer_tuples_below(n, length, least_abs=0):
    return _pos_and_neg_adaptor(generate_nonnegative_integer_tuples_below(
        n, length, least_abs))

def generate_all_integer_tuples(length, least_abs=0):
    return _pos_and_neg_adaptor(generate_all_nonnegative_integer_tuples(
        length, least_abs))





def generate_permutations(original):
    """Generate all permutations of the list `original'.

    Nicked from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/252178
    """
    if len(original) <=1:
        yield original
    else:
        for perm in generate_permutations(original[1:]):
            for i in range(len(perm)+1):
                #nb str[0:1] works in both string and list contexts
                yield perm[:i] + original[0:1] + perm[i:]


            



def generate_unique_permutations(original):
    """Generate all unique permutations of the list `original'.
    """

    had_those = set()

    for perm in generate_permutations(original):
        if perm not in had_those:
            had_those.add(perm)
            yield perm




class Table:
    """An ASCII table generator."""
    def __init__(self):
        self.Rows = []

    def add_row(self, row):
        self.Rows.append([str(i) for i in row])

    def __str__(self):
        columns = len(self.Rows[0])
        col_widths = [max(len(row[i]) for row in self.Rows)
                      for i in range(columns)]

        lines = [
            "|".join([cell.ljust(col_width)
                      for cell, col_width in zip(row, col_widths)])
            for row in self.Rows]
        lines[1:1] = ["+".join("-"*col_width
                              for col_width in col_widths)]
        return "\n".join(lines)





# command line interfaces -----------------------------------------------------
def gather_parameters_from_user(variables, constants={}, doc={},):
    import sys

    def show_usage():
        print "usage: %s <FILE-OR-STATEMENTS>" % sys.argv[0]
        print
        print "FILE-OR-STATEMENTS may either be Python statements of the form"
        print "'variable1 = value1; variable2 = value2' or the name of a file"
        print "containing such statements. Any valid Python code may be used"
        print "on the command line or in a command file. If new variables are"
        print "used, they must start with 'user_'."
        print
        print "The following variables are recognized:"
        for v in sorted(variables):
            print "  %s = %s" % (v, variables[v])
            if v in doc:
                print "    %s" % doc[v]

        print
        print "The following constants are supplied:"
        for c in sorted(constants):
            print "  %s = %s" % (c, constants[c])
            if c in doc:
                print "    %s" % doc[c]

        sys.exit(2)

    if len(sys.argv) != 2 or sys.argv[1] in ["-h", "-help", "--help"]:
        show_usage()

    execenv = variables.copy()
    execenv.update(constants)

    import os
    if os.access(sys.argv[1], os.F_OK):
        exec open(sys.argv[1], "r") in execenv
    else:
        exec sys.argv[1] in execenv

    # check if the user set invalid keys 
    for added_key in set(execenv.keys()) - set(variables.keys()) - set(constants.keys()):
        if not (added_key.startswith("user_") or added_key == "__builtins__"):
            raise ValueError( 
                    "invalid setup key: '%s' "
                    "(user variables must start with 'user_')" % added_key)

    return Record(dict((key, execenv[key]) for key in variables))




# obscure stuff --------------------------------------------------------------
def enumerate_basic_directions(dimensions):
    coordinate_list = [[0], [1], [-1]]
    return reduce(cartesian_product_sum, [coordinate_list] * dimensions)[1:]




# file system related ---------------------------------------------------------
def assert_not_a_file(name):
    import os
    if os.access(name, os.F_OK):
        raise IOError, "file `%s' already exists" % name

