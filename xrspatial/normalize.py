"""Numeric normalization utilities for raster data.

``rescale`` performs min-max normalization to a target range.
``standardize`` performs z-score normalization (subtract mean, divide by std).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    ngjit,
    not_implemented_func,
)


# ---------------------------------------------------------------------------
# rescale -- min-max normalization
# ---------------------------------------------------------------------------

@ngjit
def _cpu_rescale(data, data_min, data_range, new_min, new_range):
    out = np.empty(data.shape, dtype=np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(rows):
        for x in range(cols):
            val = data[y, x]
            if np.isfinite(val):
                if data_range == 0.0:
                    out[y, x] = new_min
                else:
                    out[y, x] = (val - data_min) / data_range * new_range + new_min
    return out


def _run_numpy_rescale(data, new_min, new_max):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.full(data.shape, np.nan, dtype=np.float64)
    data_min = float(finite.min())
    data_max = float(finite.max())
    data_range = data_max - data_min
    new_range = new_max - new_min
    return _cpu_rescale(data, data_min, data_range, new_min, new_range)


def _run_dask_numpy_rescale(data, new_min, new_max):
    # Compute global stats first (returns scalars), then map element-wise.
    finite_mask = da.isfinite(data)
    finite_vals = data[finite_mask]
    data_min = finite_vals.min()
    data_max = finite_vals.max()

    new_range = new_max - new_min
    data_range = data_max - data_min

    out = da.where(
        finite_mask,
        da.where(
            data_range == 0,
            new_min,
            (data - data_min) / data_range * new_range + new_min,
        ),
        np.nan,
    )
    return out


def _run_cupy_rescale(data, new_min, new_max):
    finite_mask = cupy.isfinite(data)
    finite_vals = data[finite_mask]
    if finite_vals.size == 0:
        return cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    data_min = float(finite_vals.min())
    data_max = float(finite_vals.max())
    data_range = data_max - data_min
    new_range = new_max - new_min

    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    if data_range == 0.0:
        out[finite_mask] = new_min
    else:
        out[finite_mask] = (
            (data[finite_mask] - data_min) / data_range * new_range + new_min
        )
    return out


def _run_dask_cupy_rescale(data, new_min, new_max):
    # Same lazy approach as dask+numpy; dask dispatches to cupy chunks.
    finite_mask = da.isfinite(data)
    finite_vals = data[finite_mask]
    data_min = finite_vals.min()
    data_max = finite_vals.max()

    new_range = new_max - new_min
    data_range = data_max - data_min

    out = da.where(
        finite_mask,
        da.where(
            data_range == 0,
            new_min,
            (data - data_min) / data_range * new_range + new_min,
        ),
        np.nan,
    )
    return out


@supports_dataset
def rescale(agg, new_min=0.0, new_max=1.0, name='rescale'):
    """Min-max normalization of a raster to a target range.

    Linearly maps finite values from ``[data_min, data_max]`` to
    ``[new_min, new_max]``.  NaN and infinite values pass through
    unchanged.  If all finite values are equal the output is filled
    with *new_min*.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array.
    new_min : float, default=0.0
        Lower bound of the output range.
    new_max : float, default=1.0
        Upper bound of the output range.
    name : str, default='rescale'
        Name of the output DataArray.

    Returns
    -------
    rescaled : xr.DataArray or xr.Dataset
        Rescaled raster with the same shape, dims, coords, and attrs.
        If *agg* is a Dataset, each variable is rescaled independently.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.normalize import rescale

        >>> data = np.array([
                [np.nan, 0., 5.],
                [10.,   20., 30.],
            ], dtype=np.float64)
        >>> agg = xr.DataArray(data)
        >>> rescale(agg)
        <xarray.DataArray 'rescale' (dim_0: 2, dim_1: 3)>
        array([[       nan, 0.        , 0.16666667],
               [0.33333333, 0.66666667, 1.        ]])
        Dimensions without coordinates: dim_0, dim_1
    """
    _validate_raster(agg, func_name='rescale', name='agg', ndim=None)
    _validate_scalar(new_min, func_name='rescale', name='new_min')
    _validate_scalar(new_max, func_name='rescale', name='new_max')

    if new_min > new_max:
        raise ValueError(
            f"new_min ({new_min}) must be <= new_max ({new_max})"
        )

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy_rescale,
        dask_func=_run_dask_numpy_rescale,
        cupy_func=_run_cupy_rescale,
        dask_cupy_func=_run_dask_cupy_rescale,
    )
    out = mapper(agg)(agg.data, new_min, new_max)
    return xr.DataArray(
        out, name=name, dims=agg.dims, coords=agg.coords, attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# standardize -- z-score normalization
# ---------------------------------------------------------------------------

@ngjit
def _cpu_standardize(data, mean, std):
    out = np.empty(data.shape, dtype=np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(rows):
        for x in range(cols):
            val = data[y, x]
            if np.isfinite(val):
                if std == 0.0:
                    out[y, x] = 0.0
                else:
                    out[y, x] = (val - mean) / std
    return out


def _run_numpy_standardize(data, ddof):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.full(data.shape, np.nan, dtype=np.float64)
    mean = float(finite.mean())
    std = float(finite.std(ddof=ddof))
    return _cpu_standardize(data, mean, std)


def _run_dask_numpy_standardize(data, ddof):
    finite_mask = da.isfinite(data)
    finite_vals = data[finite_mask]
    mean = finite_vals.mean()
    std = finite_vals.std(ddof=ddof)

    out = da.where(
        finite_mask,
        da.where(std == 0, 0.0, (data - mean) / std),
        np.nan,
    )
    return out


def _run_cupy_standardize(data, ddof):
    finite_mask = cupy.isfinite(data)
    finite_vals = data[finite_mask]
    if finite_vals.size == 0:
        return cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    mean = float(finite_vals.mean())
    std = float(finite_vals.std(ddof=ddof))

    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    if std == 0.0:
        out[finite_mask] = 0.0
    else:
        out[finite_mask] = (data[finite_mask] - mean) / std
    return out


def _run_dask_cupy_standardize(data, ddof):
    finite_mask = da.isfinite(data)
    finite_vals = data[finite_mask]
    mean = finite_vals.mean()
    std = finite_vals.std(ddof=ddof)

    out = da.where(
        finite_mask,
        da.where(std == 0, 0.0, (data - mean) / std),
        np.nan,
    )
    return out


@supports_dataset
def standardize(agg, ddof=0, name='standardize'):
    """Z-score normalization of a raster.

    Subtracts the mean and divides by the standard deviation of finite
    values.  NaN and infinite values pass through unchanged.  If all
    finite values are identical (std == 0), the output is filled with
    0.0 for those cells.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask array.
    ddof : int, default=0
        Delta degrees of freedom for standard deviation.  Use 0 for
        population std, 1 for sample std.
    name : str, default='standardize'
        Name of the output DataArray.

    Returns
    -------
    standardized : xr.DataArray or xr.Dataset
        Z-score normalized raster with the same shape, dims, coords,
        and attrs.  If *agg* is a Dataset, each variable is
        standardized independently.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.normalize import standardize

        >>> data = np.array([
                [np.nan, 1., 2.],
                [3.,    4.,  5.],
            ], dtype=np.float64)
        >>> agg = xr.DataArray(data)
        >>> standardize(agg)
        <xarray.DataArray 'standardize' (dim_0: 2, dim_1: 3)>
        array([[        nan, -1.41421356, -0.70710678],
               [ 0.        ,  0.70710678,  1.41421356]])
        Dimensions without coordinates: dim_0, dim_1
    """
    _validate_raster(agg, func_name='standardize', name='agg', ndim=None)
    _validate_scalar(ddof, func_name='standardize', name='ddof', dtype=int, min_val=0)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy_standardize,
        dask_func=_run_dask_numpy_standardize,
        cupy_func=_run_cupy_standardize,
        dask_cupy_func=_run_dask_cupy_standardize,
    )
    out = mapper(agg)(agg.data, ddof)
    return xr.DataArray(
        out, name=name, dims=agg.dims, coords=agg.coords, attrs=agg.attrs,
    )
