import pytools

import pylinear.array as num
import pylinear.linear_algebra as la




class LexicographicSequencer(object):
    def __init__(self, container, limits):
        self._Container = container
        self._Dimensions = [high-low for low, high in limits]
        self._Low = [low for low, high in limits]

    def __len__(self):
        return pytools.product(self._Dimensions)

    def translate_single_index(self, index):
        indices = []
        remaining_size = len(self)
        if not (0 <= index < remaining_size):
            raise IndexError, "invalid subscript to sequencer object"
        for i,sz in enumerate(self._Dimensions):
            remaining_size /= sz
            quotient, index = divmod(index, remaining_size)
            indices.append(quotient + self._Low[i])
        return tuple(indices)

    def get_all_indices(self):
        return [self.translate_single_index(i) for i in range(len(self))]

    def __getitem__(self, index):
        return self._Container[self.translate_single_index(index)]

  



class Grid(object):
    def __init__(self, origin, grid_vectors):
        self._Origin = origin
        self._GridVectors = grid_vectors

    def grid_vectors(self):
        return self._GridVectors

    def __getitem__(self, index):
        result = self._Origin.copy()
        for i, gv in zip(index, self._GridVectors):
            result += i * gv
        return result

    def find_closest_grid_point_index(self, point):
        tmat = num.array(self._GridVectors).T
        float_coords = tmat <<num.solve>> (point - self._Origin)
        return tuple([int(round(c)) for c in float_coords])

    def interpolate_prid_point_index(self, point):
        tmat = num.array(self._GridVectors).T
        float_coords = tmat <<num.solve>> (point - self._Origin)
        rounded_down_int_coords = [int(math.floor(c)) for c in float_coords]
        neighbors = [rounded_down_int_coords]
        for d in range(len(self._GridVectors)):
            new_neighbors = []
            for item in neighbors:
                new_neighbors.append(item) 
                new_neighbor = item[:]
                new_neighbor[d] += 1
                new_neighbors.append(new_neighbor)
            neighbors = new_neighbors
        weights = []
        for neighbor in neighbors:
            weight = product([1-abs(a-b) for a,b in zip(float_coords, neighbor)])
            if abs(weight) >= 1e-5:
                weights.append((weight, tuple(neighbor)))
        return weights




class FiniteGrid(Grid):
    def __init__(self, origin, grid_vectors, limits):
        """Instantiates a finite grid. The limits are specified as a
        list of tuples of (low, high) values, one for each grid vector.
        For the index of a dimension, we assert, as is usual in Python:

        low <= index < high,

        such that there are (high-low) gridpoints and (high-low-1)
        grid intervals
        """
        assert len(grid_vectors) == len(limits)
        
        Grid.__init__(self, origin, grid_vectors)
        self._Limits = limits

    def limits(self):
        return self._Limits

    def __iter__(self):
        return iter(self.as_sequence().get_all_indices())

    def iterkeys(self):
        return self.__iter__()

    def grid_point_counts(self):
        """Returns the number of grid intervals in each direction.
        """
        return [high-low for low, high in self._Limits]

    def grid_point_count(self):
        """Returns the number of grid intervals in each direction.
        """
        return pytools.product(self.grid_point_counts())

    def is_within_bounds(self, key):
        for el, (low, high) in zip(key, self._Limits):
            if not (low <= el < high):
                return False
        return True

    def as_sequence(self):
        return LexicographicSequencer(self, self._Limits)

    def chop_upper_boundary(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low, high-by) for low, high in self._Limits])

    def chop_lower_boundary(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low+by, high) for low, high in self._Limits])

    def chop_both_boundaries(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low+by, high-by) for low, high in self._Limits])

    def enlarge_at_upper_boundary(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low, high+by) for low, high in self._Limits])

    def enlarge_at_lower_boundary(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low-by, high) for low, high in self._Limits])

    def enlarge_at_both_boundaries(self, by = 1):
        return FiniteGrid(self._Origin, self._GridVectors,
                          [(low-by, high+by) for low, high in self._Limits])

    def reduce_periodically(self, key):
        return tuple([
            el % (high-low) for el, (low, high) in zip(key, self._Limits)])

    def reduce_to_closest(self, key):
        return tuple([
            max(min(high-1, el), low) for el, (low, high) in zip(key, self._Limits)])
  



def make_subdivision_grid(origin, grid_vectors, limits):
    interval_counts = [high - low - 1 for low, high in limits]
    my_gvs = [gv / float(ivs) for gv, ivs in zip(grid_vectors, interval_counts)]
    return FiniteGrid(origin, my_gvs, limits)
    



def make_cell_centered_grid(origin, grid_vectors, limits):
    my_gvs = [gv / float(high - low) for gv, (low, high) in zip(grid_vectors, limits)]
    return FiniteGrid(origin + pytools.general_sum(my_gvs) * 0.5,
                       my_gvs, limits)
    


