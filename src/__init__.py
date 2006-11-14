import math, sys, operator, types




def delta(x,y):
    if x == y:
        return 1
    else:
        return 0




# Data structures ------------------------------------------------------------
class Reference(object):
    def __init__( self, value ):
        self.V = value
    def get( self ):
        return self.V
    def set( self, value ):
        self.V = value




class FunctionValueCache(object):
    def __init__(self, function):
        self.Function = function
        self.ResultMap = {}

    def __call__(self, arg):
        try:
            return self.ResultMap[arg]
        except KeyError:
            result = self.Function(arg)
            self.ResultMap[arg] = result
            return result

    


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




class DictionaryOfArithmeticTypes(dict):
    """Allows arithmetic operations on dictionaries
    which propagate to its elements.
    """

    def _get_empty_self(self):
        return DictionaryOfArithmeticTypes()

    def assert_same_keys(self, other):
        for key in self:
            assert key in other
        for key in other:
            assert key in self

    def unary_operator(self, operator):
        result = self._get_empty_self()
        for key in self:
            result[key] = operator(self[key])
        return result

    def binary_operator(self, other, operator):
        try:
            self.assert_same_keys(other)
            result = self._get_empty_self()
            for key in self:
                result[key] = operator(self[key], other[key])
            return result
        except TypeError:
            result = self._get_empty_self()
            for key in self:
                result[key] = operator(self[key], other)
            return result

    def reverse_binary_operator(self, other, operator):
        try:
            self.assert_same_keys(other)
            result = self._get_empty_self()
            for key in self:
                result[key] = operator(other[key], self[key])
            return result
        except TypeError:
            result = self._get_empty_self()
            for key in self:
                result[key] = operator(other, self[key])
            return result

    def __neg__(self): return self.unary_operator(operator.neg)
    def __pos__(self): return self.unary_operator(operator.pos)
    def __abs__(self): return self.unary_operator(operator.abs)
    def __invert__(self): return self.unary_operator(operator.invert)

    def __add__(self, other): return self.binary_operator(other, operator.add)
    def __sub__(self, other): return self.binary_operator(other, operator.sub)
    def __mul__(self, other): return self.binary_operator(other, operator.mul)
    def __div__(self, other): return self.binary_operator(other, operator.div)
    def __mod__(self, other): return self.binary_operator(other, operator.mod)
    def __pow__(self, other): return self.binary_operator(other, operator.pow)
    def __lshift__(self, other): return self.binary_operator(other, operator.lshift)
    def __rshift__(self, other): return self.binary_operator(other, operator.rshift)
    def __and__(self, other): return self.binary_operator(other, operator.and_)
    def __or__(self, other): return self.binary_operator(other, operator.or_)
    def __xor__(self, other): return self.binary_operator(other, operator.xor)

    def __radd__(self, other): return self.reverse_binary_operator(other, operator.add)
    def __rsub__(self, other): return self.reverse_binary_operator(other, operator.sub)
    def __rmul__(self, other): return self.reverse_binary_operator(other, operator.mul)
    def __rdiv__(self, other): return self.reverse_binary_operator(other, operator.div)
    def __rmod__(self, other): return self.reverse_binary_operator(other, operator.mod)
    def __rpow__(self, other): return self.reverse_binary_operator(other, operator.pow)
    def __rlshift__(self, other): return self.reverse_binary_operator(other, operator.lshift)
    def __rrshift__(self, other): return self.reverse_binary_operator(other, operator.rshift)
    def __rand__(self, other): return self.reverse_binary_operator(other, operator.and_)
    def __ror__(self, other): return self.reverse_binary_operator(other, operator.or_)
    def __rxor__(self, other): return self.reverse_binary_operator(other, operator.xor)

    def __iadd__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] += other[key]
        return self

    def __isub__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] -= other[key]
        return self

    def __imul__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] *= other[key]
        return self

    def __idiv__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] /= other[key]
        return self

    def __imod__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] %= other[key]
        return self

    def __ipow__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] **= other[key]
        return self

    def __ilshift__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] <<= other[key]
        return self

    def __irshift__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] >>= other[key]
        return self

    def __iand__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] &= other[key]
        return self

    def __ior__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] |= other[key]
        return self

    def __ixor__(self, other): 
        self.assert_same_keys(other)
        for key in self: 
            self[key] ^= other[key]
        return self



# Generic utilities ----------------------------------------------------------
def flatten(list):
    result = []
    for i in list:
        result += i
    return result




def sum_over(function, arguments):
    result = 0
    for i in arguments:
        result += function(i)
    return result




def general_sum(sequence):
    return reduce(operator.add, sequence)




def linear_combination(coefficients, vectors):
    result = coefficients[0] * vectors[0]
    for c,v in zip(coefficients, vectors)[1:]:
        result += c*v
    return result




def average(sequence):
    return general_sum(sequence)/float(len(sequence))



def all_equal(sequence):
    item = sequence[0]
    for i in sequence[1:]:
        if i != item:
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




def product(list):
    return reduce(lambda x,y: x*y, list, 1)




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
    current_min = it.next()

    for idx, item in enumerate(it):
        value = item
        if value < current_min:
            current_min_index = idx
            current_min = value
    return current_min_index+1




def argmax(list):
    it = list.__iter__()
    current_max = it.next()

    for idx, item in enumerate(it):
        value = item
        if value > current_max:
            current_max_index = idx
            current_max = value
    return current_max_index+1




def cartesian_product(list1, list2):
    result = []
    for i in list1:
        for j in list2:
            result.append((i,j))




def cartesian_product_sum(list1, list2):
    """This routine returns a list of sums of each element of
    list1 with each element of list2. Also works with lists.
    """
    result = []
    for i in list1:
        for j in list2:
            result.append(i+j)
    return result




def reverse_dictionary(the_dict):
    result = {}
    for key, value in the_dict.iteritems():
        if value in result:
            raise RuntimeError, "non-reversible mapping"
        result[value] = key
    return result




def generate_positive_integer_tuples_below(n, length, least = 0):
    assert length >= 0
    if length == 0:
        yield []
    else:
        for i in range(least, n):
            for base in generate_positive_integer_tuples_below(n, length-1, least):
                yield [i] + base

def generate_all_positive_integer_tuples(length, least = 0):
    assert length >= 0
    current_max = least
    while True:
        for max_pos in range(length):
            for prebase in generate_positive_integer_tuples_below(current_max, max_pos, least):
                for postbase in generate_positive_integer_tuples_below(current_max+1, length-max_pos-1, least):
                    yield prebase + [current_max] + postbase
        current_max += 1

def _pos_and_neg_adaptor(tuple_iter):
    for tup in tuple_iter:
        nonzero_indices = [i for i in range(len(tup)) if tup[i] != 0]
        for do_neg_tup in generate_positive_integer_tuples_below(2, len(nonzero_indices)):
            this_result = list(tup)
            for index, do_neg in enumerate(do_neg_tup):
                if do_neg:
                    this_result[nonzero_indices[index]] *= -1
            yield tuple(this_result)

def generate_all_integer_tuples_below(n, length, least_abs = 0):
    return _pos_and_neg_adaptor(generate_positive_integer_tuples_below(
        n, length, least_abs))

def generate_all_integer_tuples(length, least_abs = 0):
    return _pos_and_neg_adaptor(generate_all_positive_integer_tuples(
        length, least_abs))
            



# Obscure stuff --------------------------------------------------------------
def enumerate_basic_directions(dimensions):
    coordinate_list = [[0], [1], [-1]]
    return reduce(cartesian_product_sum, [coordinate_list] * dimensions)[1:]



