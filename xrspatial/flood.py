"""Flood prediction and management tools.

Provides per-cell flood analysis functions that build on the existing
hydrology module (HAND, flow accumulation, flow length, etc.):

- ``flood_depth``: water depth above terrain given a HAND raster and
  water level
- ``inundation``: binary flood/no-flood mask
- ``curve_number_runoff``: SCS/NRCS curve number runoff estimation
- ``travel_time``: overland flow travel time via simplified Manning's
  equation
"""

from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
)

# Minimum tan(slope) clamp: tan(0.001 deg), same as TWI
_TAN_MIN = np.tan(np.radians(0.001))


# ---------------------------------------------------------------------------
# flood_depth
# ---------------------------------------------------------------------------

def flood_depth(
    hand_agg: xr.DataArray,
    water_level: float,
    name: str = 'flood_depth',
) -> xr.DataArray:
    """Compute flood depth from a HAND raster and a water level.

    ``depth = water_level - HAND`` where ``HAND <= water_level``,
    ``NaN`` elsewhere.

    Parameters
    ----------
    hand_agg : xarray.DataArray
        2D Height Above Nearest Drainage raster (output of ``hand()``).
    water_level : float
        Water surface elevation above the drainage network (>= 0).
    name : str, default 'flood_depth'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 grid of flood depths.  ``NaN`` where not inundated
        or where the input is ``NaN``.
    """
    _validate_raster(hand_agg, func_name='flood_depth', name='hand_agg')
    if not isinstance(water_level, (int, float)):
        raise TypeError("water_level must be numeric")
    if water_level < 0:
        raise ValueError("water_level must be >= 0")

    data = hand_agg.data

    if has_cuda_and_cupy() and is_dask_cupy(hand_agg):
        out = _flood_depth_dask_cupy(data, water_level)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _flood_depth_cupy(data, water_level)
    elif da is not None and isinstance(data, da.Array):
        out = _flood_depth_dask(data, water_level)
    elif isinstance(data, np.ndarray):
        out = _flood_depth_numpy(data, water_level)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out, name=name, coords=hand_agg.coords,
                        dims=hand_agg.dims, attrs=hand_agg.attrs)


def _flood_depth_numpy(hand, wl):
    hand = np.asarray(hand, dtype=np.float64)
    depth = np.where(hand <= wl, wl - hand, np.nan)
    # preserve input NaN
    depth = np.where(np.isnan(hand), np.nan, depth)
    return depth


def _flood_depth_cupy(hand, wl):
    import cupy as cp
    hand = cp.asarray(hand, dtype=cp.float64)
    depth = cp.where(hand <= wl, wl - hand, cp.nan)
    depth = cp.where(cp.isnan(hand), cp.nan, depth)
    return depth


def _flood_depth_dask(hand, wl):
    import dask.array as _da
    depth = _da.where(hand <= wl, wl - hand, np.nan)
    depth = _da.where(_da.isnan(hand), np.nan, depth)
    return depth


def _flood_depth_dask_cupy(hand, wl):
    import cupy as cp
    hand_np = hand.map_blocks(
        lambda b: b.get(), dtype=hand.dtype,
        meta=np.array((), dtype=hand.dtype),
    )
    result = _flood_depth_dask(hand_np, wl)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# inundation
# ---------------------------------------------------------------------------

def inundation(
    hand_agg: xr.DataArray,
    water_level: float,
    name: str = 'inundation',
) -> xr.DataArray:
    """Create a binary inundation mask from a HAND raster.

    Returns ``1.0`` where ``HAND <= water_level``, ``0.0`` elsewhere.
    ``NaN`` cells in the input remain ``NaN``.

    Parameters
    ----------
    hand_agg : xarray.DataArray
        2D HAND raster (output of ``hand()``).
    water_level : float
        Water surface elevation above the drainage network (>= 0).
    name : str, default 'inundation'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 binary mask.
    """
    _validate_raster(hand_agg, func_name='inundation', name='hand_agg')
    if not isinstance(water_level, (int, float)):
        raise TypeError("water_level must be numeric")
    if water_level < 0:
        raise ValueError("water_level must be >= 0")

    data = hand_agg.data

    if has_cuda_and_cupy() and is_dask_cupy(hand_agg):
        out = _inundation_dask_cupy(data, water_level)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _inundation_cupy(data, water_level)
    elif da is not None and isinstance(data, da.Array):
        out = _inundation_dask(data, water_level)
    elif isinstance(data, np.ndarray):
        out = _inundation_numpy(data, water_level)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out, name=name, coords=hand_agg.coords,
                        dims=hand_agg.dims, attrs=hand_agg.attrs)


def _inundation_numpy(hand, wl):
    hand = np.asarray(hand, dtype=np.float64)
    mask = np.where(hand <= wl, 1.0, 0.0)
    mask = np.where(np.isnan(hand), np.nan, mask)
    return mask


def _inundation_cupy(hand, wl):
    import cupy as cp
    hand = cp.asarray(hand, dtype=cp.float64)
    mask = cp.where(hand <= wl, 1.0, 0.0)
    mask = cp.where(cp.isnan(hand), cp.nan, mask)
    return mask


def _inundation_dask(hand, wl):
    import dask.array as _da
    mask = _da.where(hand <= wl, 1.0, 0.0)
    mask = _da.where(_da.isnan(hand), np.nan, mask)
    return mask


def _inundation_dask_cupy(hand, wl):
    import cupy as cp
    hand_np = hand.map_blocks(
        lambda b: b.get(), dtype=hand.dtype,
        meta=np.array((), dtype=hand.dtype),
    )
    result = _inundation_dask(hand_np, wl)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# curve_number_runoff
# ---------------------------------------------------------------------------

def curve_number_runoff(
    rainfall: xr.DataArray,
    curve_number: Union[float, xr.DataArray],
    name: str = 'curve_number_runoff',
) -> xr.DataArray:
    """Estimate runoff depth using the SCS/NRCS curve number method.

    Computes::

        S  = (25400 / CN) - 254
        Ia = 0.2 * S
        Q  = (P - Ia)^2 / (P + 0.8 * S)   where P > Ia
        Q  = 0                               where P <= Ia

    Parameters
    ----------
    rainfall : xarray.DataArray
        2D rainfall depth raster in millimetres (>= 0).
    curve_number : float or xarray.DataArray
        SCS curve number, range (0, 100].  A scalar applies uniformly;
        a DataArray allows spatially varying CN.
    name : str, default 'curve_number_runoff'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 runoff depth in millimetres.
    """
    _validate_raster(rainfall, func_name='curve_number_runoff',
                     name='rainfall')

    if isinstance(curve_number, xr.DataArray):
        cn_data = curve_number.data
    elif isinstance(curve_number, (int, float)):
        if curve_number <= 0 or curve_number > 100:
            raise ValueError("curve_number must be in (0, 100]")
        cn_data = float(curve_number)
    else:
        raise TypeError("curve_number must be numeric or an xarray.DataArray")

    data = rainfall.data

    if has_cuda_and_cupy() and is_dask_cupy(rainfall):
        out = _cn_runoff_dask_cupy(data, cn_data)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _cn_runoff_cupy(data, cn_data)
    elif da is not None and isinstance(data, da.Array):
        out = _cn_runoff_dask(data, cn_data)
    elif isinstance(data, np.ndarray):
        out = _cn_runoff_numpy(data, cn_data)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out, name=name, coords=rainfall.coords,
                        dims=rainfall.dims, attrs=rainfall.attrs)


def _cn_runoff_numpy(p, cn):
    p = np.asarray(p, dtype=np.float64)
    cn = np.asarray(cn, dtype=np.float64)
    s = (25400.0 / cn) - 254.0
    ia = 0.2 * s
    q = np.where(p > ia, (p - ia) ** 2 / (p + 0.8 * s), 0.0)
    # propagate NaN from rainfall
    q = np.where(np.isnan(p), np.nan, q)
    return q


def _cn_runoff_cupy(p, cn):
    import cupy as cp
    p = cp.asarray(p, dtype=cp.float64)
    cn = cp.asarray(cn, dtype=cp.float64)
    s = (25400.0 / cn) - 254.0
    ia = 0.2 * s
    q = cp.where(p > ia, (p - ia) ** 2 / (p + 0.8 * s), 0.0)
    q = cp.where(cp.isnan(p), cp.nan, q)
    return q


def _cn_runoff_dask(p, cn):
    import dask.array as _da
    s = (25400.0 / cn) - 254.0
    ia = 0.2 * s
    q = _da.where(p > ia, (p - ia) ** 2 / (p + 0.8 * s), 0.0)
    q = _da.where(_da.isnan(p), np.nan, q)
    return q


def _cn_runoff_dask_cupy(p, cn):
    import cupy as cp
    p_np = p.map_blocks(
        lambda b: b.get(), dtype=p.dtype,
        meta=np.array((), dtype=p.dtype),
    )
    if hasattr(cn, 'map_blocks'):
        cn_np = cn.map_blocks(
            lambda b: b.get(), dtype=cn.dtype,
            meta=np.array((), dtype=cn.dtype),
        )
    else:
        cn_np = cn
    result = _cn_runoff_dask(p_np, cn_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# travel_time
# ---------------------------------------------------------------------------

def travel_time(
    flow_length_agg: xr.DataArray,
    slope_agg: xr.DataArray,
    mannings_n: Union[float, xr.DataArray],
    name: str = 'travel_time',
) -> xr.DataArray:
    """Estimate overland flow travel time via simplified Manning's equation.

    Velocity is estimated as ``v = (1/n) * sqrt(tan(slope))`` and
    travel time as ``flow_length / v``.  Near-zero slopes are clamped
    to ``tan(0.001 deg)`` (same as TWI) to avoid division by zero.

    The *time of concentration* for a catchment is simply
    ``travel_time(...).max()``.

    Parameters
    ----------
    flow_length_agg : xarray.DataArray
        2D flow length raster (output of ``flow_length()``).
    slope_agg : xarray.DataArray
        2D slope raster in degrees.
    mannings_n : float or xarray.DataArray
        Manning's roughness coefficient (> 0).  A scalar applies
        uniformly; a DataArray allows spatially varying roughness.
    name : str, default 'travel_time'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 travel time grid (same time units as flow_length
        distance units and velocity units, typically seconds when
        flow_length is in metres).
    """
    _validate_raster(flow_length_agg, func_name='travel_time',
                     name='flow_length_agg')
    _validate_raster(slope_agg, func_name='travel_time', name='slope_agg')

    if isinstance(mannings_n, xr.DataArray):
        n_data = mannings_n.data
    elif isinstance(mannings_n, (int, float)):
        if mannings_n <= 0:
            raise ValueError("mannings_n must be > 0")
        n_data = float(mannings_n)
    else:
        raise TypeError(
            "mannings_n must be numeric or an xarray.DataArray")

    fl_data = flow_length_agg.data
    sl_data = slope_agg.data

    if has_cuda_and_cupy() and is_dask_cupy(flow_length_agg):
        out = _travel_time_dask_cupy(fl_data, sl_data, n_data)
    elif has_cuda_and_cupy() and is_cupy_array(fl_data):
        out = _travel_time_cupy(fl_data, sl_data, n_data)
    elif da is not None and isinstance(fl_data, da.Array):
        out = _travel_time_dask(fl_data, sl_data, n_data)
    elif isinstance(fl_data, np.ndarray):
        out = _travel_time_numpy(fl_data, sl_data, n_data)
    else:
        raise TypeError(f"Unsupported array type: {type(fl_data)}")

    return xr.DataArray(out, name=name, coords=flow_length_agg.coords,
                        dims=flow_length_agg.dims,
                        attrs=flow_length_agg.attrs)


def _travel_time_numpy(fl, sl, n):
    fl = np.asarray(fl, dtype=np.float64)
    sl = np.asarray(sl, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    tan_slope = np.tan(np.radians(sl))
    tan_slope = np.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    velocity = (1.0 / n) * np.sqrt(tan_slope)
    return fl / velocity


def _travel_time_cupy(fl, sl, n):
    import cupy as cp
    fl = cp.asarray(fl, dtype=cp.float64)
    sl = cp.asarray(sl, dtype=cp.float64)
    n = cp.asarray(n, dtype=cp.float64)
    tan_slope = cp.tan(cp.radians(sl))
    tan_slope = cp.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    velocity = (1.0 / n) * cp.sqrt(tan_slope)
    return fl / velocity


def _travel_time_dask(fl, sl, n):
    import dask.array as _da
    tan_slope = _da.tan(_da.radians(sl))
    tan_slope = _da.where(tan_slope < _TAN_MIN, _TAN_MIN, tan_slope)
    velocity = (1.0 / n) * _da.sqrt(tan_slope)
    return fl / velocity


def _travel_time_dask_cupy(fl, sl, n):
    import cupy as cp
    fl_np = fl.map_blocks(
        lambda b: b.get(), dtype=fl.dtype,
        meta=np.array((), dtype=fl.dtype),
    )
    sl_np = sl.map_blocks(
        lambda b: b.get(), dtype=sl.dtype,
        meta=np.array((), dtype=sl.dtype),
    )
    if hasattr(n, 'map_blocks'):
        n_np = n.map_blocks(
            lambda b: b.get(), dtype=n.dtype,
            meta=np.array((), dtype=n.dtype),
        )
    else:
        n_np = n
    result = _travel_time_dask(fl_np, sl_np, n_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )
