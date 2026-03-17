"""Topographic Wetness Index (TWI).

Computes ``ln(a / tan(β))`` where *a* is the specific catchment area
(flow accumulation × cell size) and *β* is slope in radians.

Flat areas (slope ≈ 0) are clamped to ``tan(0.001°)`` to avoid
division by zero and infinite TWI values.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    _validate_raster,
    get_dataarray_resolution,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
)

# Minimum tan(slope) clamp: tan(0.001°)
_TAN_MIN = np.tan(np.radians(0.001))


def twi_d8(flow_accum: xr.DataArray,
        slope_agg: xr.DataArray,
        name: str = 'twi') -> xr.DataArray:
    """Compute the Topographic Wetness Index.

    TWI = ln(a / tan(β)), where a = flow_accum × cellsize (specific
    catchment area) and β = slope in radians.  Slope input is expected
    in degrees (matching the output of ``slope()``).

    Parameters
    ----------
    flow_accum : xarray.DataArray
        2D flow accumulation grid (cell counts).
    slope_agg : xarray.DataArray
        2D slope grid in degrees.
    name : str, default 'twi'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 TWI grid.  NaN where either input is NaN.
    """
    _validate_raster(flow_accum, func_name='twi', name='flow_accum')
    _validate_raster(slope_agg, func_name='twi', name='slope_agg')

    cellsize_x, cellsize_y = get_dataarray_resolution(flow_accum)
    cellsize = np.sqrt(abs(cellsize_x * cellsize_y))

    fa_data = flow_accum.data
    sl_data = slope_agg.data

    if has_cuda_and_cupy() and is_dask_cupy(flow_accum):
        out = _twi_dask_cupy(fa_data, sl_data, cellsize)

    elif has_cuda_and_cupy() and is_cupy_array(fa_data):
        out = _twi_cupy(fa_data, sl_data, cellsize)

    elif da is not None and isinstance(fa_data, da.Array):
        out = _twi_dask(fa_data, sl_data, cellsize)

    elif isinstance(fa_data, np.ndarray):
        out = _twi_numpy(fa_data, sl_data, cellsize)

    else:
        raise TypeError(f"Unsupported array type: {type(fa_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_accum.coords,
                        dims=flow_accum.dims,
                        attrs=flow_accum.attrs)


def _twi_numpy(fa, sl, cellsize):
    fa = np.asarray(fa, dtype=np.float64)
    sl = np.asarray(sl, dtype=np.float64)
    sca = fa * cellsize
    tan_slope = np.tan(np.radians(sl))
    tan_slope = np.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    return np.log(sca / tan_slope)


def _twi_cupy(fa, sl, cellsize):
    import cupy as cp
    fa = cp.asarray(fa, dtype=cp.float64)
    sl = cp.asarray(sl, dtype=cp.float64)
    sca = fa * cellsize
    tan_slope = cp.tan(cp.radians(sl))
    tan_slope = cp.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    return cp.log(sca / tan_slope)


def _twi_dask(fa, sl, cellsize):
    import dask.array as _da
    sca = fa * cellsize
    tan_slope = _da.tan(_da.radians(sl))
    tan_slope = _da.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    return _da.log(sca / tan_slope)


def _twi_dask_cupy(fa, sl, cellsize):
    import cupy as cp
    fa_np = fa.map_blocks(
        lambda b: b.get(), dtype=fa.dtype,
        meta=np.array((), dtype=fa.dtype),
    )
    sl_np = sl.map_blocks(
        lambda b: b.get(), dtype=sl.dtype,
        meta=np.array((), dtype=sl.dtype),
    )
    result = _twi_dask(fa_np, sl_np, cellsize)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )
