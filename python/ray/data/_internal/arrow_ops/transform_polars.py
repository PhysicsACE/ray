from typing import TYPE_CHECKING, List, Any
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.progress_bar import ProgressBar
import numpy as np

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
    col, _ = key[0]
    df = pl.from_arrow(table)
    return df.sort(col, reverse=descending).to_arrow()


def sort_indices(table: "pyarrow.Table", key: "SortKeyT", descending: bool) -> "pyarrow.Table":
    check_polars_installed()
    cols, order = [], []
    for c in key:
        cols.append(c[0])
        if c[1] == "ascending":
            order.append(True)
            continue
        order.append(False)

    df = pl.from_arrow(table)
    return df.arg_sort_by(cols, descending=order)


def searchsorted(table: "pyarrow.Table", boundaries: List[int], key: "SortKeyT", descending: bool) -> List[int]:
    """
    This method finds the index to place a row containing a set of columnar values to 
    maintain ordering of the sorted table. 
    """
    check_polars_installed()
    partitionIdx = cached_remote_fn(find_partitionIdx)
    df = pl.from_arrow(table)
    bound_results = [partitionIdx.remote(df, [i] if isinstance(i, int) else i, key, descending) for i in boundaries]
    bounds_bar = ProgressBar("Sort and Partition", len(bound_results))
    bounds = bounds_bar.fetch_until_complete(bound_results)
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
    
    assert len(desired) == len(key)

    left, right = 0, table.height
    for i in range(len(desired)):
        colName = key[i][0]
        if key[i][1] == "ascending":
            dir = True if (not descending) else False
        else:
            dir = descending
        colVals = table.get_column(colName).to_numpy()[left:right]
        desiredVal = desired[i]
        prevleft = left

        if not dir:
            left = prevleft + np.searchsorted(colVals, desiredVal, side="right", sorter=np.arange(len(colVals) - 1, -1, -1))
            right = prevleft + np.searchsorted(colVals, desiredVal, side="left", sorter=np.arange(len(colVals) - 1, -1, -1))
        else:
            left = prevleft + np.searchsorted(colVals, desiredVal, side="left")
            right = prevleft + np.searchsorted(colVals, desiredVal, side="right")
    
    if descending:
        return left
    return right


def concat_and_sort(
    blocks: List["pyarrow.Table"], key: "SortKeyT", descending: bool
) -> "pyarrow.Table":
    check_polars_installed()
    col, order = [], []
    for k in key:
        col.append(k[0])
        if k[1] == "ascending":
            order.append(False)
            continue
        order.append(True)

    blocks = [pl.from_arrow(block) for block in blocks]
    df = pl.concat(blocks).sort(col, descending=descending)
    return df.to_arrow()
