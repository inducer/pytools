from typing import IO, Any, Callable, Iterator, List, Optional, Sequence, Tuple

from pytools import Record


__doc__ = """
An in-memory relational database table
======================================

.. autoclass:: DataTable
"""


# type-ignore-reason: Record is untyped
class Row(Record):  # type: ignore[misc]
    pass


class DataTable:
    """An in-memory relational database table.

    .. automethod:: __init__
    .. automethod:: copy
    .. automethod:: deep_copy
    .. automethod:: join
    """

    def __init__(self, column_names: Sequence[str],
                 column_data: Optional[List[Any]] = None) -> None:
        """Construct a new table, with the given C{column_names}.

        :arg column_names: An indexable of column name strings.
        :arg column_data: None or a list of tuples of the same length as
             *column_names* indicating an initial set of data.
        """
        if column_data is None:
            self.data = []
        else:
            self.data = column_data

        self.column_names = column_names
        self.column_indices = {
                colname: i for i, colname in enumerate(column_names)}

        if len(self.column_indices) != len(self.column_names):
            raise RuntimeError("non-unique column names encountered")

    def __bool__(self) -> bool:
        return bool(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[List[Any]]:
        return self.data.__iter__()

    def __str__(self) -> str:
        """Return a pretty-printed version of the table."""

        def col_width(i: int) -> int:
            width = len(self.column_names[i])
            if self:
                width = max(width, max(len(str(row[i])) for row in self.data))
            return width
        col_widths = [col_width(i) for i in range(len(self.column_names))]

        def format_row(row: Sequence[str]) -> str:
            return "|".join([str(cell).ljust(col_width)
                      for cell, col_width in zip(row, col_widths)])

        lines = [format_row(self.column_names),
                "+".join("-"*col_width for col_width in col_widths)] + \
                [format_row(row) for row in self.data]
        return "\n".join(lines)

    def insert(self, **kwargs: Any) -> None:
        values = [None for i in range(len(self.column_names))]

        for key, val in kwargs.items():
            values[self.column_indices[key]] = val

        self.insert_row(tuple(values))

    def insert_row(self, values: Tuple[Any, ...]) -> None:
        assert isinstance(values, tuple)
        assert len(values) == len(self.column_names)
        self.data.append(values)

    def insert_rows(self, rows: Sequence[Tuple[Any, ...]]) -> None:
        for row in rows:
            self.insert_row(row)

    def filtered(self, **kwargs: Any) -> "DataTable":
        if not kwargs:
            return self

        criteria = tuple(
                (self.column_indices[key], value)
                for key, value in kwargs.items())

        result_data = []

        for row in self.data:
            satisfied = True
            for idx, val in criteria:
                if row[idx] != val:
                    satisfied = False
                    break

            if satisfied:
                result_data.append(row)

        return DataTable(self.column_names, result_data)

    def get(self, **kwargs: Any) -> Row:
        filtered = self.filtered(**kwargs)
        if not filtered:
            raise RuntimeError("no matching entry for get()")
        if len(filtered) > 1:
            raise RuntimeError("more than one matching entry for get()")

        return Row(dict(list(zip(self.column_names, filtered.data[0]))))

    def clear(self) -> None:
        del self.data[:]

    def copy(self) -> "DataTable":
        """Make a copy of the instance, but leave individual rows untouched.

        If the rows are modified later, they will also be modified in the copy.
        """
        return DataTable(self.column_names, self.data[:])

    def deep_copy(self) -> "DataTable":
        """Make a copy of the instance down to the row level.

        The copy's rows may be modified independently from the original.
        """
        return DataTable(self.column_names, [row[:] for row in self.data])

    def sort(self, columns: Sequence[str], reverse: bool = False) -> None:
        col_indices = [self.column_indices[col] for col in columns]

        def mykey(row: Sequence[Any]) -> Tuple[Any, ...]:
            return tuple(
                    row[col_index]
                    for col_index in col_indices)

        self.data.sort(reverse=reverse, key=mykey)

    def aggregated(self, groupby: Sequence[str], agg_column: str,
                   aggregate_func: Callable[[Sequence[Any]], Any]) -> "DataTable":
        gb_indices = [self.column_indices[col] for col in groupby]
        agg_index = self.column_indices[agg_column]

        first = True

        result_data = []

        # to pacify pyflakes:
        last_values: Tuple[Any, ...] = ()
        agg_values: List[Row] = []

        for row in self.data:
            this_values = tuple(row[i] for i in gb_indices)
            if first or this_values != last_values:
                if not first:
                    result_data.append(last_values + (aggregate_func(agg_values),))

                agg_values = [row[agg_index]]
                last_values = this_values
                first = False
            else:
                agg_values.append(row[agg_index])

        if not first and agg_values:
            result_data.append(this_values + (aggregate_func(agg_values),))

        return DataTable(
                [self.column_names[i] for i in gb_indices] + [agg_column],
                result_data)

    def join(self, column: str, other_column: str, other_table: "DataTable",
             outer: bool = False) -> "DataTable":
        """Return a table joining this and the C{other_table} on C{column}.

        The new table has the following columns:
        - C{column}, titled the same as in this table.
        - the columns of this table, minus C{column}.
        - the columns of C{other_table}, minus C{other_column}.

        Assumes both tables are sorted ascendingly by the column
        by which they are joined.
        """  # pylint:disable=too-many-locals,too-many-branches

        def without(indexable: Tuple[str, ...], idx: int) -> Tuple[str, ...]:
            return indexable[:idx] + indexable[idx+1:]

        this_key_idx = self.column_indices[column]
        other_key_idx = other_table.column_indices[other_column]

        this_iter = self.data.__iter__()
        other_iter = other_table.data.__iter__()

        result_columns = tuple(self.column_names[this_key_idx]) + \
                without(tuple(self.column_names), this_key_idx) + \
                without(tuple(other_table.column_names), other_key_idx)

        result_data = []

        this_row = next(this_iter)
        other_row = next(other_iter)

        this_over = False
        other_over = False

        while True:
            this_batch = []
            other_batch = []

            if this_over:
                run_other = True
            elif other_over:
                run_this = True
            else:
                this_key = this_row[this_key_idx]
                other_key = other_row[other_key_idx]

                run_this = this_key < other_key
                run_other = this_key > other_key
                if this_key == other_key:
                    run_this = run_other = True

            if run_this and not this_over:
                key = this_key
                while this_row[this_key_idx] == this_key:
                    this_batch.append(this_row)
                    try:
                        this_row = next(this_iter)
                    except StopIteration:
                        this_over = True
                        break
            else:
                if outer:
                    this_batch = [(None,) * len(self.column_names)]

            if run_other and not other_over:
                key = other_key
                while other_row[other_key_idx] == other_key:
                    other_batch.append(other_row)
                    try:
                        other_row = next(other_iter)
                    except StopIteration:
                        other_over = True
                        break
            else:
                if outer:
                    other_batch = [(None,) * len(other_table.column_names)]

            for this_batch_row in this_batch:
                for other_batch_row in other_batch:
                    result_data.append((key,)
                            + without(this_batch_row, this_key_idx)
                            + without(other_batch_row, other_key_idx))

            if outer:
                if this_over and other_over:
                    break
            else:
                if this_over or other_over:
                    break

        return DataTable(result_columns, result_data)

    def restricted(self, columns: Sequence[str]) -> "DataTable":
        col_indices = [self.column_indices[col] for col in columns]

        return DataTable(columns,
                [[row[i] for i in col_indices] for row in self.data])

    def column_data(self, column: str) -> List[Tuple[Any, ...]]:
        col_index = self.column_indices[column]
        return [row[col_index] for row in self.data]

    def write_csv(self, filelike: IO[Any], **kwargs: Any) -> None:
        from csv import writer
        csvwriter = writer(filelike, **kwargs)
        csvwriter.writerow(self.column_names)
        csvwriter.writerows(self.data)
