"""Flood prediction and management tools.

Provides per-cell flood analysis functions that build on the existing
hydrology module (HAND, flow accumulation, flow length, etc.):

- ``flood_depth``: water depth above terrain given a HAND raster and
  water level
- ``inundation``: binary flood/no-flood mask
- ``curve_number_runoff``: SCS/NRCS curve number runoff estimation
- ``travel_time``: overland flow travel time via simplified Manning's
  equation
- ``vegetation_roughness``: Manning's n from land cover or NDVI
- ``vegetation_curve_number``: SCS curve number from land cover and
  hydrologic soil group
- ``flood_depth_vegetation``: vegetation-adjusted flood depth via
  Manning's normal depth equation
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
# NLCD-to-Manning's n lookup (Chow 1959; Arcement & Schneider 1989)
# ---------------------------------------------------------------------------
NLCD_MANNINGS_N = {
    11: 0.025,   # Open Water
    21: 0.040,   # Developed, Open Space
    22: 0.100,   # Developed, Low Intensity
    23: 0.080,   # Developed, Medium Intensity
    24: 0.150,   # Developed, High Intensity
    31: 0.028,   # Barren Land
    41: 0.100,   # Deciduous Forest
    42: 0.120,   # Evergreen Forest
    43: 0.100,   # Mixed Forest
    52: 0.070,   # Shrub/Scrub
    71: 0.035,   # Grassland/Herbaceous
    81: 0.033,   # Pasture/Hay
    82: 0.038,   # Cultivated Crops
    90: 0.100,   # Woody Wetlands
    95: 0.070,   # Emergent Herbaceous Wetlands
}

# ---------------------------------------------------------------------------
# NLCD + Hydrologic Soil Group -> SCS Curve Number (TR-55, HEC-HMS)
# Soil groups: 1=A, 2=B, 3=C, 4=D.  "Good" hydrologic condition assumed.
# ---------------------------------------------------------------------------
NLCD_CURVE_NUMBER = {
    (11, 1): 100, (11, 2): 100, (11, 3): 100, (11, 4): 100,  # Open Water
    (21, 1):  39, (21, 2):  61, (21, 3):  74, (21, 4):  80,   # Dev Open
    (22, 1):  57, (22, 2):  72, (22, 3):  81, (22, 4):  86,   # Dev Low
    (23, 1):  77, (23, 2):  85, (23, 3):  90, (23, 4):  92,   # Dev Med
    (24, 1):  89, (24, 2):  92, (24, 3):  94, (24, 4):  95,   # Dev High
    (31, 1):  77, (31, 2):  86, (31, 3):  91, (31, 4):  94,   # Barren
    (41, 1):  30, (41, 2):  55, (41, 3):  70, (41, 4):  77,   # Deciduous
    (42, 1):  30, (42, 2):  55, (42, 3):  70, (42, 4):  77,   # Evergreen
    (43, 1):  30, (43, 2):  55, (43, 3):  70, (43, 4):  77,   # Mixed Forest
    (52, 1):  35, (52, 2):  56, (52, 3):  70, (52, 4):  77,   # Shrub
    (71, 1):  39, (71, 2):  61, (71, 3):  74, (71, 4):  80,   # Grassland
    (81, 1):  39, (81, 2):  61, (81, 3):  74, (81, 4):  80,   # Pasture
    (82, 1):  67, (82, 2):  78, (82, 3):  85, (82, 4):  89,   # Crops
    (90, 1):  30, (90, 2):  58, (90, 3):  71, (90, 4):  78,   # Woody Wetland
    (95, 1):  30, (95, 2):  58, (95, 3):  71, (95, 4):  78,   # Herb Wetland
}

# NDVI-to-Manning's n interpolation breakpoints
_NDVI_BREAKPOINTS = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
_NDVI_MANNINGS_N = np.array([0.02, 0.03, 0.05, 0.10, 0.16])


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


# ---------------------------------------------------------------------------
# vegetation_roughness
# ---------------------------------------------------------------------------

def vegetation_roughness(
    vegetation_agg: xr.DataArray,
    mode: str = 'nlcd',
    lookup: dict | None = None,
    name: str = 'mannings_n',
) -> xr.DataArray:
    """Derive Manning's n roughness from land cover or NDVI.

    Parameters
    ----------
    vegetation_agg : xarray.DataArray
        2D raster.  For ``mode='nlcd'`` this should contain integer NLCD
        land cover codes.  For ``mode='ndvi'`` this should contain
        continuous NDVI values (typically -1 to 1).
    mode : {'nlcd', 'ndvi'}, default 'nlcd'
        How to convert vegetation data to roughness.

        * ``'nlcd'``: categorical lookup from NLCD codes to Manning's n.
        * ``'ndvi'``: piecewise linear interpolation of NDVI to Manning's n.
    lookup : dict, optional
        Override the default NLCD-to-n mapping.  Keys are integer NLCD
        codes, values are Manning's n floats.  Only used when
        ``mode='nlcd'``.
    name : str, default 'mannings_n'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 Manning's n raster.  Unrecognized NLCD codes and
        NaN inputs map to NaN.
    """
    _validate_raster(vegetation_agg, func_name='vegetation_roughness',
                     name='vegetation_agg')
    if mode not in ('nlcd', 'ndvi'):
        raise ValueError(f"mode must be 'nlcd' or 'ndvi', got {mode!r}")

    data = vegetation_agg.data

    if mode == 'nlcd':
        lut_dict = dict(NLCD_MANNINGS_N)
        if lookup is not None:
            lut_dict.update(lookup)
        max_code = max(lut_dict)
        lut = np.full(max_code + 1, np.nan, dtype=np.float64)
        for code, val in lut_dict.items():
            lut[code] = val

        if has_cuda_and_cupy() and is_dask_cupy(vegetation_agg):
            out = _veg_roughness_nlcd_dask_cupy(data, lut)
        elif has_cuda_and_cupy() and is_cupy_array(data):
            out = _veg_roughness_nlcd_cupy(data, lut)
        elif da is not None and isinstance(data, da.Array):
            out = _veg_roughness_nlcd_dask(data, lut)
        elif isinstance(data, np.ndarray):
            out = _veg_roughness_nlcd_numpy(data, lut)
        else:
            raise TypeError(f"Unsupported array type: {type(data)}")
    else:
        if has_cuda_and_cupy() and is_dask_cupy(vegetation_agg):
            out = _veg_roughness_ndvi_dask_cupy(data)
        elif has_cuda_and_cupy() and is_cupy_array(data):
            out = _veg_roughness_ndvi_cupy(data)
        elif da is not None and isinstance(data, da.Array):
            out = _veg_roughness_ndvi_dask(data)
        elif isinstance(data, np.ndarray):
            out = _veg_roughness_ndvi_numpy(data)
        else:
            raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out, name=name, coords=vegetation_agg.coords,
                        dims=vegetation_agg.dims,
                        attrs=vegetation_agg.attrs)


def _veg_roughness_nlcd_numpy(data, lut):
    data = np.asarray(data)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    mask = (data >= 0) & (data < len(lut)) & ~np.isnan(data.astype(np.float64))
    idx = data[mask].astype(np.intp)
    out[mask] = lut[idx]
    return out


def _veg_roughness_nlcd_cupy(data, lut):
    import cupy as cp
    lut_gpu = cp.asarray(lut)
    data = cp.asarray(data)
    out = cp.full(data.shape, cp.nan, dtype=cp.float64)
    data_f = data.astype(cp.float64)
    mask = (data >= 0) & (data < len(lut_gpu)) & ~cp.isnan(data_f)
    idx = data[mask].astype(cp.intp)
    out[mask] = lut_gpu[idx]
    return out


def _veg_roughness_nlcd_dask(data, lut):
    import dask.array as _da

    def _apply_lut(block, lut=lut):
        return _veg_roughness_nlcd_numpy(block, lut)

    return _da.map_blocks(
        _apply_lut, data, dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _veg_roughness_nlcd_dask_cupy(data, lut):
    import cupy as cp
    data_np = data.map_blocks(
        lambda b: b.get(), dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    result = _veg_roughness_nlcd_dask(data_np, lut)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


def _veg_roughness_ndvi_numpy(data):
    data = np.asarray(data, dtype=np.float64)
    out = np.interp(data, _NDVI_BREAKPOINTS, _NDVI_MANNINGS_N)
    out = np.where(np.isnan(data), np.nan, out)
    return out


def _veg_roughness_ndvi_cupy(data):
    import cupy as cp
    data = cp.asarray(data, dtype=cp.float64)
    bp = cp.asarray(_NDVI_BREAKPOINTS)
    mn = cp.asarray(_NDVI_MANNINGS_N)
    out = cp.interp(data.ravel(), bp, mn).reshape(data.shape)
    out = cp.where(cp.isnan(data), cp.nan, out)
    return out


def _veg_roughness_ndvi_dask(data):
    import dask.array as _da

    def _apply(block):
        return _veg_roughness_ndvi_numpy(block)

    return _da.map_blocks(
        _apply, data, dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _veg_roughness_ndvi_dask_cupy(data):
    import cupy as cp
    data_np = data.map_blocks(
        lambda b: b.get(), dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    result = _veg_roughness_ndvi_dask(data_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# vegetation_curve_number
# ---------------------------------------------------------------------------

def vegetation_curve_number(
    landcover_agg: xr.DataArray,
    soil_group_agg: xr.DataArray,
    lookup: dict | None = None,
    name: str = 'curve_number',
) -> xr.DataArray:
    """Derive SCS curve number from NLCD land cover and hydrologic soil group.

    Parameters
    ----------
    landcover_agg : xarray.DataArray
        2D integer raster of NLCD land cover codes.
    soil_group_agg : xarray.DataArray
        2D integer raster of hydrologic soil groups (1=A, 2=B, 3=C, 4=D).
    lookup : dict, optional
        Override the default ``(nlcd_code, soil_group) -> CN`` mapping.
    name : str, default 'curve_number'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 curve number raster.  Unknown ``(code, group)``
        pairs and NaN inputs map to NaN.
    """
    _validate_raster(landcover_agg, func_name='vegetation_curve_number',
                     name='landcover_agg')
    _validate_raster(soil_group_agg, func_name='vegetation_curve_number',
                     name='soil_group_agg')

    lut_dict = dict(NLCD_CURVE_NUMBER)
    if lookup is not None:
        lut_dict.update(lookup)

    max_code = max(k[0] for k in lut_dict)
    max_sg = max(k[1] for k in lut_dict)
    lut = np.full((max_code + 1, max_sg + 1), np.nan, dtype=np.float64)
    for (code, sg), cn in lut_dict.items():
        lut[code, sg] = cn

    lc_data = landcover_agg.data
    sg_data = soil_group_agg.data

    if has_cuda_and_cupy() and is_dask_cupy(landcover_agg):
        out = _veg_cn_dask_cupy(lc_data, sg_data, lut)
    elif has_cuda_and_cupy() and is_cupy_array(lc_data):
        out = _veg_cn_cupy(lc_data, sg_data, lut)
    elif da is not None and isinstance(lc_data, da.Array):
        out = _veg_cn_dask(lc_data, sg_data, lut)
    elif isinstance(lc_data, np.ndarray):
        out = _veg_cn_numpy(lc_data, sg_data, lut)
    else:
        raise TypeError(f"Unsupported array type: {type(lc_data)}")

    return xr.DataArray(out, name=name, coords=landcover_agg.coords,
                        dims=landcover_agg.dims,
                        attrs=landcover_agg.attrs)


def _veg_cn_numpy(lc, sg, lut):
    lc = np.asarray(lc)
    sg = np.asarray(sg)
    max_code, max_sg = lut.shape
    out = np.full(lc.shape, np.nan, dtype=np.float64)
    lc_f = lc.astype(np.float64)
    sg_f = sg.astype(np.float64)
    mask = (
        (lc >= 0) & (lc < max_code)
        & (sg >= 0) & (sg < max_sg)
        & ~np.isnan(lc_f) & ~np.isnan(sg_f)
    )
    lc_idx = lc[mask].astype(np.intp)
    sg_idx = sg[mask].astype(np.intp)
    out[mask] = lut[lc_idx, sg_idx]
    return out


def _veg_cn_cupy(lc, sg, lut):
    import cupy as cp
    lut_gpu = cp.asarray(lut)
    lc = cp.asarray(lc)
    sg = cp.asarray(sg)
    max_code, max_sg = lut_gpu.shape
    out = cp.full(lc.shape, cp.nan, dtype=cp.float64)
    lc_f = lc.astype(cp.float64)
    sg_f = sg.astype(cp.float64)
    mask = (
        (lc >= 0) & (lc < max_code)
        & (sg >= 0) & (sg < max_sg)
        & ~cp.isnan(lc_f) & ~cp.isnan(sg_f)
    )
    lc_idx = lc[mask].astype(cp.intp)
    sg_idx = sg[mask].astype(cp.intp)
    out[mask] = lut_gpu[lc_idx, sg_idx]
    return out


def _veg_cn_dask(lc, sg, lut):
    import dask.array as _da

    def _apply(lc_block, sg_block, lut=lut):
        return _veg_cn_numpy(lc_block, sg_block, lut)

    return _da.map_blocks(
        _apply, lc, sg, dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _veg_cn_dask_cupy(lc, sg, lut):
    import cupy as cp
    lc_np = lc.map_blocks(
        lambda b: b.get(), dtype=lc.dtype,
        meta=np.array((), dtype=lc.dtype),
    )
    sg_np = sg.map_blocks(
        lambda b: b.get(), dtype=sg.dtype,
        meta=np.array((), dtype=sg.dtype),
    )
    result = _veg_cn_dask(lc_np, sg_np, lut)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# flood_depth_vegetation
# ---------------------------------------------------------------------------

def flood_depth_vegetation(
    hand_agg: xr.DataArray,
    slope_agg: xr.DataArray,
    mannings_n: Union[float, xr.DataArray],
    unit_discharge: float,
    name: str = 'flood_depth_vegetation',
) -> xr.DataArray:
    """Compute vegetation-adjusted flood depth via Manning's normal depth.

    For each cell, the equilibrium (normal) depth for steady uniform flow
    is::

        h = (q * n / sqrt(tan(slope)))^(3/5)

    where *q* is unit discharge (m^2/s), *n* is Manning's roughness, and
    *slope* is terrain slope in degrees.  The flood depth is
    ``h - HAND`` where ``h > HAND``, and NaN elsewhere.

    Higher roughness (denser vegetation) produces deeper, slower water.

    Parameters
    ----------
    hand_agg : xarray.DataArray
        2D HAND raster (output of ``hand()``).
    slope_agg : xarray.DataArray
        2D slope raster in degrees.
    mannings_n : float or xarray.DataArray
        Manning's roughness coefficient.  A scalar applies uniformly;
        a DataArray (e.g. output of ``vegetation_roughness()``) allows
        spatially varying roughness.
    unit_discharge : float
        Unit discharge in m^2/s (discharge per unit width, > 0).
    name : str, default 'flood_depth_vegetation'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 flood depth grid.  NaN where not inundated or
        where inputs are NaN.
    """
    _validate_raster(hand_agg, func_name='flood_depth_vegetation',
                     name='hand_agg')
    _validate_raster(slope_agg, func_name='flood_depth_vegetation',
                     name='slope_agg')
    if not isinstance(unit_discharge, (int, float)):
        raise TypeError("unit_discharge must be numeric")
    if unit_discharge <= 0:
        raise ValueError("unit_discharge must be > 0")

    if isinstance(mannings_n, xr.DataArray):
        n_data = mannings_n.data
    elif isinstance(mannings_n, (int, float)):
        if mannings_n <= 0:
            raise ValueError("mannings_n must be > 0")
        n_data = float(mannings_n)
    else:
        raise TypeError(
            "mannings_n must be numeric or an xarray.DataArray")

    h_data = hand_agg.data
    sl_data = slope_agg.data
    q = float(unit_discharge)

    if has_cuda_and_cupy() and is_dask_cupy(hand_agg):
        out = _fdv_dask_cupy(h_data, sl_data, n_data, q)
    elif has_cuda_and_cupy() and is_cupy_array(h_data):
        out = _fdv_cupy(h_data, sl_data, n_data, q)
    elif da is not None and isinstance(h_data, da.Array):
        out = _fdv_dask(h_data, sl_data, n_data, q)
    elif isinstance(h_data, np.ndarray):
        out = _fdv_numpy(h_data, sl_data, n_data, q)
    else:
        raise TypeError(f"Unsupported array type: {type(h_data)}")

    return xr.DataArray(out, name=name, coords=hand_agg.coords,
                        dims=hand_agg.dims, attrs=hand_agg.attrs)


def _fdv_numpy(hand, sl, n, q):
    hand = np.asarray(hand, dtype=np.float64)
    sl = np.asarray(sl, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    tan_s = np.tan(np.radians(sl))
    tan_s = np.where(tan_s < _TAN_MIN, _TAN_MIN, tan_s)
    h_normal = (q * n / np.sqrt(tan_s)) ** 0.6
    depth = np.where(h_normal > hand, h_normal - hand, np.nan)
    # propagate NaN from any input
    any_nan = np.isnan(hand) | np.isnan(sl) | np.isnan(n)
    depth = np.where(any_nan, np.nan, depth)
    return depth


def _fdv_cupy(hand, sl, n, q):
    import cupy as cp
    hand = cp.asarray(hand, dtype=cp.float64)
    sl = cp.asarray(sl, dtype=cp.float64)
    n = cp.asarray(n, dtype=cp.float64)
    tan_s = cp.tan(cp.radians(sl))
    tan_s = cp.where(tan_s < _TAN_MIN, _TAN_MIN, tan_s)
    h_normal = (q * n / cp.sqrt(tan_s)) ** 0.6
    depth = cp.where(h_normal > hand, h_normal - hand, cp.nan)
    any_nan = cp.isnan(hand) | cp.isnan(sl) | cp.isnan(n)
    depth = cp.where(any_nan, cp.nan, depth)
    return depth


def _fdv_dask(hand, sl, n, q):
    import dask.array as _da
    tan_s = _da.tan(_da.radians(sl))
    tan_s = _da.where(tan_s < _TAN_MIN, _TAN_MIN, tan_s)
    h_normal = (q * n / _da.sqrt(tan_s)) ** 0.6
    depth = _da.where(h_normal > hand, h_normal - hand, np.nan)
    any_nan = _da.isnan(hand) | _da.isnan(sl) | _da.isnan(n)
    depth = _da.where(any_nan, np.nan, depth)
    return depth


def _fdv_dask_cupy(hand, sl, n, q):
    import cupy as cp
    hand_np = hand.map_blocks(
        lambda b: b.get(), dtype=hand.dtype,
        meta=np.array((), dtype=hand.dtype),
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
    result = _fdv_dask(hand_np, sl_np, n_np, q)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )
