"""Multi-raster merge strategies."""
from __future__ import annotations

import numpy as np


_STRATEGIES = ('first', 'last', 'mean', 'max', 'min')


def _validate_strategy(strategy):
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"strategy must be one of {list(_STRATEGIES)}, "
            f"got {strategy!r}"
        )


def _merge_arrays_numpy(arrays, nodata, strategy):
    """Merge a list of aligned 2-D numpy arrays using the given strategy.

    Parameters
    ----------
    arrays : list of ndarray
        Each array has shape (H, W). Nodata regions are marked with *nodata*.
    nodata : float
        Nodata sentinel value.
    strategy : str
        One of 'first', 'last', 'mean', 'max', 'min'.

    Returns
    -------
    ndarray (H, W)
    """
    if not arrays:
        raise ValueError("No arrays to merge")

    shape = arrays[0].shape
    use_nan = np.isnan(nodata)

    def _is_nodata(arr):
        if use_nan:
            return np.isnan(arr)
        return arr == nodata

    if strategy == 'first':
        result = np.full(shape, nodata, dtype=np.float64)
        mask = np.ones(shape, dtype=bool)  # True = still empty
        for arr in arrays:
            valid = ~_is_nodata(arr) & mask
            result[valid] = arr[valid]
            mask[valid] = False
        return result

    if strategy == 'last':
        result = np.full(shape, nodata, dtype=np.float64)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            result[valid] = arr[valid]
        return result

    if strategy == 'mean':
        accum = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.int32)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            accum[valid] += arr[valid]
            count[valid] += 1
        result = np.full(shape, nodata, dtype=np.float64)
        has_data = count > 0
        result[has_data] = accum[has_data] / count[has_data]
        return result

    if strategy == 'max':
        result = np.full(shape, nodata, dtype=np.float64)
        initialized = np.zeros(shape, dtype=bool)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            new = valid & ~initialized
            update = valid & initialized & (arr > result)
            result[new] = arr[new]
            result[update] = arr[update]
            initialized[valid] = True
        return result

    if strategy == 'min':
        result = np.full(shape, nodata, dtype=np.float64)
        initialized = np.zeros(shape, dtype=bool)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            new = valid & ~initialized
            update = valid & initialized & (arr < result)
            result[new] = arr[new]
            result[update] = arr[update]
            initialized[valid] = True
        return result


def _merge_arrays_cupy(arrays, nodata, strategy):
    """CuPy equivalent of ``_merge_arrays_numpy``."""
    import cupy as cp

    if not arrays:
        raise ValueError("No arrays to merge")

    shape = arrays[0].shape
    use_nan = np.isnan(nodata)

    def _is_nodata(arr):
        if use_nan:
            return cp.isnan(arr)
        return arr == nodata

    if strategy == 'first':
        result = cp.full(shape, nodata, dtype=cp.float64)
        mask = cp.ones(shape, dtype=cp.bool_)
        for arr in arrays:
            valid = ~_is_nodata(arr) & mask
            result[valid] = arr[valid]
            mask[valid] = False
        return result

    if strategy == 'last':
        result = cp.full(shape, nodata, dtype=cp.float64)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            result[valid] = arr[valid]
        return result

    if strategy == 'mean':
        accum = cp.zeros(shape, dtype=cp.float64)
        count = cp.zeros(shape, dtype=cp.int32)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            accum[valid] += arr[valid]
            count[valid] += 1
        result = cp.full(shape, nodata, dtype=cp.float64)
        has_data = count > 0
        result[has_data] = accum[has_data] / count[has_data]
        return result

    if strategy == 'max':
        result = cp.full(shape, nodata, dtype=cp.float64)
        initialized = cp.zeros(shape, dtype=cp.bool_)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            new = valid & ~initialized
            update = valid & initialized & (arr > result)
            result[new] = arr[new]
            result[update] = arr[update]
            initialized[valid] = True
        return result

    if strategy == 'min':
        result = cp.full(shape, nodata, dtype=cp.float64)
        initialized = cp.zeros(shape, dtype=cp.bool_)
        for arr in arrays:
            valid = ~_is_nodata(arr)
            new = valid & ~initialized
            update = valid & initialized & (arr < result)
            result[new] = arr[new]
            result[update] = arr[update]
            initialized[valid] = True
        return result
