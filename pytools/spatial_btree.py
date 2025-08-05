"""
Spatial Binary Tree
===================

.. autoclass:: Point
.. autoclass:: Element
.. autoclass:: BoundingBox
.. autoclass:: SpatialBinaryTree

.. autoclass:: SpatialBinaryTreeBucket
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator

Point: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]
Element: TypeAlias = Any
BoundingBox: TypeAlias = tuple[Point, Point]
SpatialBinaryTree: TypeAlias = (
    "SpatialBinaryTreeBucket | tuple[SpatialBinaryTree, SpatialBinaryTree]")


def do_boxes_intersect(bl: BoundingBox, tr: BoundingBox) -> bool:
    (bl1, tr1) = bl
    (bl2, tr2) = tr
    (dimension,) = bl1.shape
    return all(max(bl1[i], bl2[i]) <= min(tr1[i], tr2[i]) for i in range(dimension))


def make_buckets(
        bottom_left: Point,
        top_right: Point,
        allbuckets: list[SpatialBinaryTreeBucket],
        max_elements_per_box: int
    ) -> SpatialBinaryTree:
    (dimensions,) = bottom_left.shape
    half = (top_right - bottom_left) / 2.

    def do(dimension: int, pos: Point) -> SpatialBinaryTree:
        if dimension == dimensions:
            origin = bottom_left + pos * half
            bucket = SpatialBinaryTreeBucket(
                    origin,
                    origin + half,
                    max_elements_per_box=max_elements_per_box)

            allbuckets.append(bucket)
            return bucket

        pos[dimension] = 0
        first = do(dimension + 1, pos)

        pos[dimension] = 1
        second = do(dimension + 1, pos)

        return first, second

    return do(0, np.zeros((dimensions,), dtype=bottom_left.dtype))


class SpatialBinaryTreeBucket:
    """This class represents one bucket in a spatial binary tree.
    It automatically decides whether it needs to create more subdivisions
    beneath itself or not.

    .. autoattribute:: elements

    .. autoattribute:: bottom_left
    .. autoattribute:: top_right
    .. autoattribute:: center
    .. autoattribute:: max_elements_per_box

    .. automethod:: insert
    .. automethod:: generate_matches
    .. automethod:: visualize
    .. automethod:: plot
    """

    elements: list[tuple[Element, BoundingBox]]
    """A list of tuples *(element, bbox)* where bbox is again a tuple
    *(lower_left, upper_right)* of :class:`numpy.ndarray` instances
    satisfying ``(lower_right <= upper_right).all()``.
    """

    bottom_left: Point
    top_right: Point
    center: Point
    max_elements_per_box: int

    buckets: SpatialBinaryTree | None
    all_buckets: list[SpatialBinaryTreeBucket] | None

    def __init__(self,
                 bottom_left: Point,
                 top_right: Point,
                 max_elements_per_box: int | None = None) -> None:
        """
        :param bottom_left: a :mod:`numpy.ndarray` of the minimal coordinates
            of the box being partitioned.
        :param top_right: a :mod:`numpy.ndarray` array of the maximal coordinates
            of the box being partitioned.
        """

        self.bottom_left = bottom_left
        self.top_right = top_right
        self.center = (bottom_left + top_right) / 2

        # As long as buckets is None, there are no subdivisions
        self.elements = []
        self.buckets = None
        self.all_buckets = None

        if max_elements_per_box is None:
            dimensions, = self.bottom_left.shape
            max_elements_per_box = cast("int", 8 * 2**dimensions)

        self.max_elements_per_box = max_elements_per_box

    def insert(self, element: Element, bbox: BoundingBox) -> None:
        """Insert an element into the spatial tree.

        :param element: the element to be stored in the retrieval data
            structure. It is treated as opaque and no assumptions are made on it.

        :param bbox: a bounding box supplied as a tuple ``(bottom_left, top_right)``
            of :mod:`numpy` vectors, such that ``(bottom_left <= top_right).all()``.
            Despite these names, the bounding box (and this entire data structure)
            may be of any dimension.
        """

        def insert_into_subdivision(element: Element, bbox: BoundingBox) -> None:
            assert self.all_buckets is not None

            bucket_matches = [
                ibucket
                for ibucket, bucket in enumerate(self.all_buckets)
                if do_boxes_intersect((bucket.bottom_left, bucket.top_right), bbox)]

            from random import uniform
            if len(bucket_matches) > len(self.all_buckets) // 2:
                # Would go into more than half of all buckets--keep it here
                self.elements.append((element, bbox))
            elif len(bucket_matches) > 1 and uniform(0, 1) > 0.95:
                # Would go into more than one bucket and therefore may recurse
                # indefinitely. Keep it here with a low probability.
                self.elements.append((element, bbox))
            else:
                for ibucket_match in bucket_matches:
                    self.all_buckets[ibucket_match].insert(element, bbox)

        if self.buckets is None:
            # No subdivisions yet.
            if len(self.elements) > self.max_elements_per_box:
                # Too many elements. Need to subdivide.
                self.all_buckets = []
                self.buckets = make_buckets(
                        self.bottom_left, self.top_right,
                        self.all_buckets,
                        max_elements_per_box=self.max_elements_per_box)

                old_elements = self.elements
                self.elements = []

                # Move all elements from the full bucket into the new finer ones
                for el, el_bbox in old_elements:
                    insert_into_subdivision(el, el_bbox)

                insert_into_subdivision(element, bbox)
            else:
                # Simple:
                self.elements.append((element, bbox))
        else:
            # Go find which sudivision to place element
            insert_into_subdivision(element, bbox)

    def generate_matches(self, point: Point) -> Iterator[Element]:
        if self.buckets:
            # We have subdivisions. Use them.
            (dimensions,) = point.shape
            bucket = self.buckets
            for dim in range(dimensions):
                assert isinstance(bucket, tuple)
                bucket = bucket[0] if point[dim] < self.center[dim] else bucket[1]

            assert isinstance(bucket, SpatialBinaryTreeBucket)
            yield from bucket.generate_matches(point)

        # Perform linear search.
        for el, _ in self.elements:
            yield el

    def visualize(self, file: TextIO) -> None:
        file.write(f"{self.bottom_left[0]:f} {self.bottom_left[1]:f}\n")
        file.write(f"{self.top_right[0]:f} {self.bottom_left[1]:f}\n")
        file.write(f"{self.top_right[0]:f} {self.top_right[1]:f}\n")
        file.write(f"{self.bottom_left[0]:f} {self.top_right[1]:f}\n")
        file.write(f"{self.bottom_left[0]:f} {self.bottom_left[1]:f}\n\n")

        if self.buckets:
            assert self.all_buckets is not None
            for i in self.all_buckets:
                i.visualize(file)

    def plot(self, **kwargs: Any) -> None:
        """
        :param ax: an :class:`~matplotlib.axes.Axes` object on which to plot the tree.
        :param kwargs: any remaining arguments are passed to
            :class:`matplotlib.patches.PathPatch`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        ax = kwargs.pop("ax", None)
        if ax is None:
            ax = plt.gca()

        el = self.bottom_left
        eh = self.top_right
        pathdata = [
            (Path.MOVETO, (el[0], el[1])),
            (Path.LINETO, (eh[0], el[1])),
            (Path.LINETO, (eh[0], eh[1])),
            (Path.LINETO, (el[0], eh[1])),
            (Path.CLOSEPOLY, (el[0], el[1])),
            ]

        codes, verts = zip(*pathdata, strict=True)
        path = Path(verts, codes)
        patch = PathPatch(path, **kwargs)
        ax.add_patch(patch)

        if self.buckets:
            assert self.all_buckets is not None
            for bucket in self.all_buckets:
                bucket.plot(ax=ax, **kwargs)
