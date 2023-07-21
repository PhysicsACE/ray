from typing import TYPE_CHECKING, List, Any
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.progress_bar import ProgressBar
import numpy as np
from ray.data._internal.sort_key import SortKey

try:
    import pyarrow
except ImportError:
    pyarrow = None


if TYPE_CHECKING:
    from ray.data._internal.sort import SortKeyT

pl = None


def check_polars_installed():
    try:
        global pl
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars not installed. Install with `pip install polars` or set "
            "`DataContext.use_polars = False` to fall back to pyarrow"
        )


def sort(table: "pyarrow.Table", key: "SortKeyT", descending: bool) -> "pyarrow.Table":
    check_polars_installed()
    col, order = key.to_polars_sort_args()
    df = pl.from_arrow(table)
    return df.sort(col, reverse=order).to_arrow()


def sort_indices(table: "pyarrow.Table", key: "SortKeyT", descending: bool) -> "pyarrow.Table":
    check_polars_installed()
    cols, order = key.to_polars_sort_args()
    df = pl.from_arrow(table)
    return df.arg_sort_by(cols, descending=order)


def searchsorted(table: "pyarrow.Table", boundaries: List[int], key: "SortKeyT", descending: bool) -> List[int]:
    """
    This method finds the index to place a row containing a set of columnar values to 
    maintain ordering of the sorted table. 
    """
    check_polars_installed()
    # partitionIdx = cached_remote_fn(find_partitionIdx)
    df = pl.from_arrow(table)
    # bound_results = [partitionIdx.remote(df, [i] if not isinstance(i, np.ndarray) else i, key, descending) for i in boundaries]
    # bounds_bar = ProgressBar("Sort and Partition", len(bound_results))
    # bounds = bounds_bar.fetch_until_complete(bound_results)
    bounds = [find_partitionIdx(df, [i] if not isinstance(i, np.ndarray) else i, key, descending) for i in boundaries]

    return bounds


def find_partitionIdx(table: Any, desired: List[Any], key:"SortKeyT", descending: bool) -> int:

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
    
    normalizedkey = key.normalized_key()

    if len(normalizedkey) == 0:
        for name in table.column_names:
            normalizedkey.append((name, "ascending"))

    left, right = 0, table.height
    for i in range(len(desired)):

        if left == right:
            return right

        colName = normalizedkey[i][0]
        if normalizedkey[i][1] == "ascending":
            dir = True 
        else:
            dir = False
        colVals = table.get_column(colName).to_numpy()[left:right]
        desiredVal = desired[i]
        prevleft = left

        if not dir:
            left = prevleft + (len(colVals) - np.searchsorted(colVals, desiredVal, side="right", sorter=np.arange(len(colVals) - 1, -1, -1)))
            right = prevleft + (len(colVals) - np.searchsorted(colVals, desiredVal, side="left", sorter=np.arange(len(colVals) - 1, -1, -1)))
        else:
            left = prevleft + np.searchsorted(colVals, desiredVal, side="left")
            right = prevleft + np.searchsorted(colVals, desiredVal, side="right")
    
    return right


def concat_and_sort(
    blocks: List["pyarrow.Table"], key: "SortKeyT", descending: bool
) -> "pyarrow.Table":
    check_polars_installed()
    col, order = key.to_polars_sort_args()

    blocks = [pl.from_arrow(block) for block in blocks]
    df = pl.concat(blocks).sort(col, reverse=order)
    return df.to_arrow()
