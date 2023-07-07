import collections
import heapq
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import json

from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data.aggregate import AggregateFn
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockMetadata,
    KeyType,
    U,
)
from ray.data.context import DataContext
from ray.data.row import TableRow
from ray.data._internal.table_block import (
    TableBlockAccessor,
    TableBlockBuilder,
)
from ray.data.aggregate import AggregateFn
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.progress_bar import ProgressBar


if TYPE_CHECKING:
    import pandas
    import pyarrow

    from ray.data._internal.sort import SortKeyT

T = TypeVar("T")

_pandas = None


def lazy_import_pandas():
    global _pandas
    if _pandas is None:
        import pandas

        _pandas = pandas
    return _pandas


class PandasRow(TableRow):
    """
    Row of a tabular Dataset backed by a Pandas DataFrame block.
    """

    def __getitem__(self, key: str) -> Any:
        from ray.data.extensions import TensorArrayElement

        col = self._row[key]
        if len(col) == 0:
            return None
        item = col.iloc[0]
        if isinstance(item, TensorArrayElement):
            # Getting an item in a Pandas tensor column may return a TensorArrayElement,
            # which we have to convert to an ndarray.
            item = item.to_numpy()
        try:
            # Try to interpret this as a numpy-type value.
            # See https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types.  # noqa: E501
            return item.item()
        except (AttributeError, ValueError):
            # Fallback to the original form.
            return item

    def __iter__(self) -> Iterator:
        for k in self._row.columns:
            yield k

    def __len__(self):
        return self._row.shape[1]


class PandasBlockBuilder(TableBlockBuilder):
    def __init__(self):
        pandas = lazy_import_pandas()
        super().__init__(pandas.DataFrame)

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> "pandas.DataFrame":
        pandas = lazy_import_pandas()
        for key, value in columns.items():
            if key == TENSOR_COLUMN_NAME or isinstance(
                next(iter(value), None), np.ndarray
            ):
                from ray.data.extensions.tensor_extension import TensorArray

                columns[key] = TensorArray(value)
        return pandas.DataFrame(columns)

    @staticmethod
    def _concat_tables(tables: List["pandas.DataFrame"]) -> "pandas.DataFrame":
        pandas = lazy_import_pandas()
        from ray.air.util.data_batch_conversion import (
            _cast_ndarray_columns_to_tensor_extension,
        )

        if len(tables) > 1:
            df = pandas.concat(tables, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = tables[0]
        ctx = DataContext.get_current()
        if ctx.enable_tensor_extension_casting:
            df = _cast_ndarray_columns_to_tensor_extension(df)
        return df

    @staticmethod
    def _concat_would_copy() -> bool:
        return True

    @staticmethod
    def _empty_table() -> "pandas.DataFrame":
        pandas = lazy_import_pandas()
        return pandas.DataFrame()


# This is to be compatible with pyarrow.lib.schema
# TODO (kfstorm): We need a format-independent way to represent schema.
PandasBlockSchema = collections.namedtuple("PandasBlockSchema", ["names", "types"])


class PandasBlockAccessor(TableBlockAccessor):
    ROW_TYPE = PandasRow

    def __init__(self, table: "pandas.DataFrame"):
        super().__init__(table)

    def column_names(self) -> List[str]:
        return self._table.columns.tolist()

    @staticmethod
    def _build_tensor_row(row: PandasRow) -> np.ndarray:
        from ray.data.extensions import TensorArrayElement

        tensor = row[TENSOR_COLUMN_NAME].iloc[0]
        if isinstance(tensor, TensorArrayElement):
            # Getting an item in a Pandas tensor column may return a TensorArrayElement,
            # which we have to convert to an ndarray.
            tensor = tensor.to_numpy()
        return tensor

    def slice(self, start: int, end: int, copy: bool = False) -> "pandas.DataFrame":
        view = self._table[start:end]
        view.reset_index(drop=True, inplace=True)
        if copy:
            view = view.copy(deep=True)
        return view

    def take(self, indices: List[int]) -> "pandas.DataFrame":
        table = self._table.take(indices)
        table.reset_index(drop=True, inplace=True)
        return table

    def select(self, columns: List[str]) -> "pandas.DataFrame":
        if not all(isinstance(col, str) for col in columns):
            raise ValueError(
                "Columns must be a list of column name strings when aggregating on "
                f"Pandas blocks, but got: {columns}."
            )
        return self._table[columns]

    def random_shuffle(self, random_seed: Optional[int]) -> "pandas.DataFrame":
        table = self._table.sample(frac=1, random_state=random_seed)
        table.reset_index(drop=True, inplace=True)
        return table

    def schema(self) -> PandasBlockSchema:
        dtypes = self._table.dtypes
        schema = PandasBlockSchema(
            names=dtypes.index.tolist(), types=dtypes.values.tolist()
        )
        # Column names with non-str types of a pandas DataFrame is not
        # supported by Ray Dataset.
        if any(not isinstance(name, str) for name in schema.names):
            raise ValueError(
                "A Pandas DataFrame with column names of non-str types"
                " is not supported by Ray Dataset. Column names of this"
                f" DataFrame: {schema.names!r}."
            )
        return schema

    def to_pandas(self) -> "pandas.DataFrame":
        from ray.air.util.data_batch_conversion import _cast_tensor_columns_to_ndarrays

        ctx = DataContext.get_current()
        table = self._table
        if ctx.enable_tensor_extension_casting:
            table = _cast_tensor_columns_to_ndarrays(table)
        return table

    def to_numpy(
        self, columns: Optional[Union[str, List[str]]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if columns is None:
            columns = self._table.columns.tolist()
            should_be_single_ndarray = False
        elif isinstance(columns, list):
            should_be_single_ndarray = False
        else:
            columns = [columns]
            should_be_single_ndarray = True

        for column in columns:
            if column not in self._table.columns:
                raise ValueError(
                    f"Cannot find column {column}, available columns: "
                    f"{self._table.columns.tolist()}"
                )

        arrays = []
        for column in columns:
            arrays.append(self._table[column].to_numpy())

        if should_be_single_ndarray:
            assert len(columns) == 1
        #     arrays = arrays[0]
        # else:
        arrays = dict(zip(columns, arrays))
        return arrays

    def to_arrow(self) -> "pyarrow.Table":
        import pyarrow

        return pyarrow.table(self._table)

    def num_rows(self) -> int:
        return self._table.shape[0]

    def size_bytes(self) -> int:
        return self._table.memory_usage(index=True, deep=True).sum()

    def _zip(self, acc: BlockAccessor) -> "pandas.DataFrame":
        r = self.to_pandas().copy(deep=False)
        s = acc.to_pandas()
        for col_name in s.columns:
            col = s[col_name]
            column_names = list(r.columns)
            # Ensure the column names are unique after zip.
            if col_name in column_names:
                i = 1
                new_name = col_name
                while new_name in column_names:
                    new_name = "{}_{}".format(col_name, i)
                    i += 1
                col_name = new_name
            r[col_name] = col
        return r

    @staticmethod
    def builder() -> PandasBlockBuilder:
        return PandasBlockBuilder()

    @staticmethod
    def _empty_table() -> "pandas.DataFrame":
        return PandasBlockBuilder._empty_table()

    def _sample(self, n_samples: int, key: "SortKeyT") -> "pandas.DataFrame":
        return self._table[[k[0] for k in key]].sample(n_samples, ignore_index=True)

    def _apply_agg(
        self, agg_fn: Callable[["pandas.Series", bool], U], on: str
    ) -> Optional[U]:
        """Helper providing null handling around applying an aggregation to a column."""
        pd = lazy_import_pandas()
        if on is not None and not isinstance(on, str):
            raise ValueError(
                "on must be a string or None when aggregating on Pandas blocks, but "
                f"got: {type(on)}."
            )

        if self.num_rows() == 0:
            return None

        col = self._table[on]
        try:
            val = agg_fn(col)
        except TypeError as e:
            # Converting an all-null column in an Arrow Table to a Pandas DataFrame
            # column will result in an all-None column of object type, which will raise
            # a type error when attempting to do most binary operations. We explicitly
            # check for this type failure here so we can properly propagate a null.
            if np.issubdtype(col.dtype, np.object_) and col.isnull().all():
                return None
            raise e from None
        if pd.isnull(val):
            return None
        return val

    def count(self, on: str) -> Optional[U]:
        return self._apply_agg(lambda col: col.count(), on)

    def sum(self, on: str, ignore_nulls: bool) -> Optional[U]:
        pd = lazy_import_pandas()
        if on is not None and not isinstance(on, str):
            raise ValueError(
                "on must be a string or None when aggregating on Pandas blocks, but "
                f"got: {type(on)}."
            )

        if self.num_rows() == 0:
            return None

        col = self._table[on]
        if col.isnull().all():
            # Short-circuit on an all-null column, returning None. This is required for
            # sum() since it will otherwise return 0 when summing on an all-null column,
            # which is not what we want.
            return None
        val = col.sum(skipna=ignore_nulls)
        if pd.isnull(val):
            return None
        return val

    def min(self, on: str, ignore_nulls: bool) -> Optional[U]:
        return self._apply_agg(lambda col: col.min(skipna=ignore_nulls), on)

    def max(self, on: str, ignore_nulls: bool) -> Optional[U]:
        return self._apply_agg(lambda col: col.max(skipna=ignore_nulls), on)

    def mean(self, on: str, ignore_nulls: bool) -> Optional[U]:
        return self._apply_agg(lambda col: col.mean(skipna=ignore_nulls), on)

    def sum_of_squared_diffs_from_mean(
        self,
        on: str,
        ignore_nulls: bool,
        mean: Optional[U] = None,
    ) -> Optional[U]:
        if mean is None:
            mean = self.mean(on, ignore_nulls)
        return self._apply_agg(
            lambda col: ((col - mean) ** 2).sum(skipna=ignore_nulls),
            on,
        )

    def sort_and_partition(
        self, boundaries: List[T], key: "SortKeyT", descending: bool
    ) -> List[Block]:
        # if len(key) > 1:
        #     raise NotImplementedError(
        #         "sorting by multiple columns is not supported yet"
        #     )

        if self._table.shape[0] == 0:
            # If the pyarrow table is empty we may not have schema
            # so calling sort_indices() will raise an error.
            return [self._empty_table() for _ in range(len(boundaries) + 1)]

        cols, order = [], []
        for k in key:
            cols.append(k[0])
            if k[1] == "ascending":
                order.append(True)
                continue
            order.append(False)
        table = self._table.sort_values(by=cols, ascending=order)
        if len(boundaries) == 0:
            return [table]

        partitions = []
        # For each boundary value, count the number of items that are less
        # than it. Since the block is sorted, these counts partition the items
        # such that boundaries[i] <= x < boundaries[i + 1] for each x in
        # partition[i]. If `descending` is true, `boundaries` would also be
        # in descending order and we only need to count the number of items
        # *greater than* the boundary value instead.
        # if descending:
        #     num_rows = len(table[col])
        #     bounds = num_rows - table[col].searchsorted(
        #         boundaries, sorter=np.arange(num_rows - 1, -1, -1)
        #     )
        # else:
        #     bounds = table[col].searchsorted(boundaries)
        
        bounds = searchsorted(table, boundaries, key, descending)
        last_idx = 0
        for idx in bounds:
            partitions.append(table[last_idx:idx])
            last_idx = idx
        partitions.append(table[last_idx:])
        return partitions

    def combine(self, key: str, aggs: Tuple[AggregateFn]) -> "pandas.DataFrame":
        """Combine rows with the same key into an accumulator.

        This assumes the block is already sorted by key in ascending order.

        Args:
            key: The column name of key or None for global aggregation.
            aggs: The aggregations to do.

        Returns:
            A sorted block of [k, v_1, ..., v_n] columns where k is the groupby
            key and v_i is the partially combined accumulator for the ith given
            aggregation.
            If key is None then the k column is omitted.
        """
        if key is not None and not isinstance(key, str) and not isinstance(key, list):
            raise ValueError(
                "key must be a string, None or List when aggregating on Pandas blocks, but "
                f"got: {type(key)}."
            )
        
        key_cols = []
        if isinstance(key, list):
            for k in key:
                key_cols.append(k[0])
        else:
            key_cols.append(key)

        def extract_key(row) -> Union[Any, List[Any]]:
            currVals = []
            for k in key_cols:
                currVals.append(row[k])
            return currVals
        
        def pretty_grouping(k: Union[Any, List[Any]]) -> str:
            if isinstance(k, str):
                return json.dumps({key_cols[0]: k})
            elif isinstance(k, list):
                keyDict = {key_cols[0]: k[0]}
                for i in range(1, len(k)):
                    keyDict[key_cols[i]] = k[i]
                return json.dumps(keyDict)
            else:
                return json.dumps({key_cols[0]: k})

        def serializedKey() -> str:
            if isinstance(key, str):
                return key
            serialized = k[0]
            for i in range(1, len(key)):
                serialized += "," + key[i]
            return serialized

        def iter_groups() -> Iterator[Tuple[KeyType, Block]]:
            """Creates an iterator over zero-copy group views."""
            if key is None:
                # Global aggregation consists of a single "group", so we short-circuit.
                yield None, self.to_block()
                return

            start = end = 0
            iter = self.iter_rows(public_row_format=False)
            next_row = None
            while True:
                try:
                    if next_row is None:
                        next_row = next(iter)
                    next_key = extract_key(next_row)
                    while extract_key(next_row) == next_key:
                        end += 1
                        try:
                            next_row = next(iter)
                        except StopIteration:
                            next_row = None
                            break
                    yield next_key, self.slice(start, end, copy=False)
                    start = end
                except StopIteration:
                    break

        builder = PandasBlockBuilder()
        for group_key, group_view in iter_groups():
            # Aggregate.
            accumulators = [agg.init(group_key) for agg in aggs]
            for i in range(len(aggs)):
                accumulators[i] = aggs[i].accumulate_block(accumulators[i], group_view)

            # Build the row.
            row = {}
            if key is not None:
                for i in range(len(key_cols)):
                    row[key_cols[i]] = group_key[i]

            count = collections.defaultdict(int)
            for agg, accumulator in zip(aggs, accumulators):
                name = agg.name
                # Check for conflicts with existing aggregation name.
                if count[name] > 0:
                    name = self._munge_conflict(name, count[name])
                count[name] += 1
                row[name] = accumulator

            print("INNNNNNIIIITTTROW", row)

            builder.add(row)

        return builder.build()

    @staticmethod
    def merge_sorted_blocks(
        blocks: List[Block], key: "SortKeyT", _descending: bool
    ) -> Tuple["pandas.DataFrame", BlockMetadata]:
        pd = lazy_import_pandas()
        stats = BlockExecStats.builder()
        blocks = [b for b in blocks if b.shape[0] > 0]
        if len(blocks) == 0:
            ret = PandasBlockAccessor._empty_table()
        else:
            cols, order = [], []
            for k in key:
                cols.append(k[0])
                if k[1] == "ascending":
                    order.append(True)
                    continue
                order.append(False)
            ret = pd.concat(blocks, ignore_index=True)
            ret = ret.sort_values(by=cols, ascending=order)
        return ret, PandasBlockAccessor(ret).get_metadata(
            None, exec_stats=stats.build()
        )
    
    # def demangle_row(block):

    #     """The grouping keys were serialized for the intermediate computations.
    #     Once the final reduce phases have been completed, we need to build the 
    #     final rows by extracting the row keys and building their respective columns"""

    #     key_fn = (lambda r: r[r._row.columns[0]])

    #     get_aggs = (lambda r: r[r._row.columns[1:]])

    #     iterator = PandasBlockAccessor(block).iter_rows(public_row_format=False)
    #     next_row = None
    #     while True:
    #         try:
    #             row = {}
    #             if next_row is None:
    #                 next_row = next(iterator)
    #             groupingkey = key_fn(next_row)
    #             toDict = json.loads(groupingkey)
    #             for k, v in toDict.items():
    #                 row[k] = v

    #             aggs = get_aggs(next_row)
    #             for c in aggs:
    #                 row[c] = next_row[c]
                
    #             yield row
                
    #         except StopIteration:
    #             break


    @staticmethod
    def aggregate_combined_blocks(
        blocks: List["pandas.DataFrame"],
        key: str,
        aggs: Tuple[AggregateFn],
        finalize: bool,
    ) -> Tuple["pandas.DataFrame", BlockMetadata]:
        """Aggregate sorted, partially combined blocks with the same key range.

        This assumes blocks are already sorted by key in ascending order,
        so we can do merge sort to get all the rows with the same key.

        Args:
            blocks: A list of partially combined and sorted blocks.
            key: The column name of key or None for global aggregation.
            aggs: The aggregations to do.
            finalize: Whether to finalize the aggregation. This is used as an
                optimization for cases where we repeatedly combine partially
                aggregated groups.

        Returns:
            A block of [k, v_1, ..., v_n] columns and its metadata where k is
            the groupby key and v_i is the corresponding aggregation result for
            the ith given aggregation.
            If key is None then the k column is omitted.
        """

        stats = BlockExecStats.builder()
        key_fn = (lambda r: tuple(r[r._row.columns[i]] for i in range(len(key)))) if key is not None else (lambda r: 0)

        iter = heapq.merge(
            *[
                PandasBlockAccessor(block).iter_rows(public_row_format=False)
                for block in blocks
            ],
            key=key_fn,
        )
        next_row = None
        builder = PandasBlockBuilder()
        while True:
            try:
                if next_row is None:
                    next_row = next(iter)
                
                next_key = key_fn(next_row)
                if not isinstance(next_key, str):
                    print("AfterROOOOOOOOOOO", next_row)
                next_key_name = tuple(next_row._row.columns[i] for i in range(len(key))) if key is not None else None

                def gen():
                    nonlocal iter
                    nonlocal next_row
                    while key_fn(next_row) == next_key:
                        yield next_row
                        try:
                            next_row = next(iter)
                        except StopIteration:
                            next_row = None
                            break

                # Merge.
                first = True
                accumulators = [None] * len(aggs)
                resolved_agg_names = [None] * len(aggs)
                for r in gen():
                    if first:
                        count = collections.defaultdict(int)
                        for i in range(len(aggs)):
                            name = aggs[i].name
                            # Check for conflicts with existing aggregation
                            # name.
                            if count[name] > 0:
                                name = PandasBlockAccessor._munge_conflict(
                                    name, count[name]
                                )
                            count[name] += 1
                            resolved_agg_names[i] = name
                            accumulators[i] = r[name]
                        first = False
                    else:
                        for i in range(len(aggs)):
                            accumulators[i] = aggs[i].merge(
                                accumulators[i], r[resolved_agg_names[i]]
                            )
                # Build the row.
                row = {}
                if key is not None:
                    # row[next_key_name] = next_key
                    for i in range(len(next_key_name)):
                        row[next_key_name[i]] = next_key[i]

                for agg, agg_name, accumulator in zip(
                    aggs, resolved_agg_names, accumulators
                ):
                    if finalize:
                        row[agg_name] = agg.finalize(accumulator)
                    else:
                        row[agg_name] = accumulator

                builder.add(row)
            except StopIteration:
                break

        ret = builder.build()
        return ret, PandasBlockAccessor(ret).get_metadata(
            None, exec_stats=stats.build())
    
    def _sorted_boundaries(self, key: "SortKeyT", descending: bool) -> "pandas.DataFrame":
        pandas = lazy_import_pandas()
        cols, orders = [], []
        for k in key:
            cols.append(k[0])
            if k[1] == "ascending":
                orders.append(True)
                continue
            orders.append(False)

        # if k[1] == "ascending":
        #     keys.append((k[0], "descending" if descending else "ascending"))
        # else:
        #     keys.append(k[0], "ascending" if descending else "descending")

        # mergeddf = pandas.concat(self._table)
        return self._table.sort_values(by=cols, ascending=orders)
    

# def pandas_searchsorted(table: "pandas.DataFrame", boundaries: List[int], key: "SortKeyT", descending: bool) -> List[int]:
#     """Pandas implementation for dataframe searchsorted as it only exists for index and series in the builin library"""

#     cols, orders = [], []
#     for k in key:
#         cols.append(k[0])
#         if k[1] == "ascending":
#             orders.append(True)
#             continue
#         orders.append(False)
#     # sortedTable = table.sort_values(by=cols, ascending=orders)
#     sortedindices = list(table.index.values)
#     partitionIdx = cached_remote_fn(pandas_localIdxbound)
#     bound_results = [partitionIdx.remote(table, i, key, descending) for i in boundaries]
#     bounds_bar = ProgressBar("Sort and Partition", len(bound_results))
#     bounds = bounds_bar.fetch_until_complete(bound_results)
#     return bounds

# def pandas_localIdxbound(table: "pandas.DataFrame", idx: int, key: "SortKeyT", descending: bool) -> int:

#     """
#     This function takes a sorted table and a given index and finds either the 
#     leftmost or rightmost occurence of that row with respect to the columns of the provided sortkey
#     """

#     cols, orders = [], []
#     for k in key:
#         cols.append(k[0])
#         if k[1] == "ascending":
#             orders.append(True)
#             continue
#         orders.append(False)
#     row = table.loc[[idx]]
#     rowinfo = row.select[cols]

#     numrows = len(table.index)
#     iterator = -1

#     if descending:
#         if idx > 0:
#             iterator = idx - 1
        
#         if iterator == -1:
#             return idx
        
#         while iterator >= 0:
#             comparerow = table.loc[[iterator]]
#             compareInfo = comparerow.select[cols]
#             if rowinfo != compareInfo:
#                 return iterator + 1
            
#             iterator -= 1

#         return 0
        
#     if idx < numrows - 1:
#         iterator = idx + 1

#     if iterator == -1:
#         return idx
    
#     while iterator < numrows:
#         comparerow = table.take([iterator])
#         compareInfo = comparerow.select(cols)
#         if rowinfo != compareInfo:
#             return iterator - 1
        
#         iterator += 1

#     return numrows - 1


def searchsorted(table: "pandas.DataFrame", boundaries: List[int], key: "SortKeyT", descending: bool) -> List[int]:
    """
    This method finds the index to place a row containing a set of columnar values to 
    maintain ordering of the sorted table. This is currently an open issue for the pandas
    framework, see here https://github.com/pandas-dev/pandas/issues/42872 and this is a
    currently workaround/implementation that utilizes numpy to essentially zone in 
    on the correct row index that maintains the orders with respect to an arbitrary key.
    """
    partitionIdx = cached_remote_fn(find_partitionIdx)
    bound_results = [partitionIdx.remote(table, [i] if not isinstance(i, np.ndarray) else i, key, descending) for i in boundaries]
    bounds_bar = ProgressBar("Sort and Partition", len(bound_results))
    bounds = bounds_bar.fetch_until_complete(bound_results)
    return bounds


def find_partitionIdx(table: "pandas.DataFrame", desired: List[Any], key:"SortKeyT", descending: bool) -> int:

    """
    This function is an implementation of np.searchsorted for pyarrow tables. It also
    extends the existing functionality of the numpy version as well as similar 
    implementation by allowing the user to pass in multi columnar keys with their ordering
    info to find the left or right most index at which the row could be placed into the table
    to maintain the current ordering. Note that the function assumes that the table 
    passed to it is already sorted with the desired columns and respective orders and the 
    order key passed to the function should be the one used to compute the table ordering.
    The implementation uses np.searchsorted as its foundation to take bounds for the i-th
    key based on the results of the previous i-1 keys.
    """
    
    assert len(desired) == len(key)

    left, right = 0, len(table.index)
    for i in range(len(desired)):
        colName = key[i][0]
        if key[i][1] == "ascending":
            dir = True
        else:
            dir = False
        colVals = table[colName].to_numpy()[left:right]
        desiredVal = desired[i]
        prevleft = left

        if not dir:
            left = prevleft + (len(colVals) - np.searchsorted(colVals, desiredVal, side="right", sorter=np.arange(len(colVals) - 1, -1, -1)))
            right = prevleft + (len(colVals) - np.searchsorted(colVals, desiredVal, side="left", sorter=np.arange(len(colVals) - 1, -1, -1)))
        else:
            left = prevleft + np.searchsorted(colVals, desiredVal, side="left")
            right = prevleft + np.searchsorted(colVals, desiredVal, side="right")
    
    return right