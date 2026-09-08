from __future__ import annotations

# std lib
from functools import partial
from math import atan, isnan, nan
from typing import Optional, Union

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr
from numba import cuda

# local modules
from xrspatial.dataset_support import supports_dataset
from xrspatial.geodesic import (INV_2R, WGS84_A2, WGS84_B2, _check_geodesic_memory_backend_aware,
                                _cpu_geodesic_slope, _run_gpu_geodesic_slope)
from xrspatial.utils import (Z_UNITS, ArrayTypeFunctionMapping, _boundary_to_dask,
                             _extract_latlon_coords, _pad_array, _validate_boundary,
                             _validate_raster, cuda_args, get_dataarray_resolution, ngjit,
                             warn_if_unit_mismatch)
from xrspatial.utils import _dask_task_name_kwargs


def _geodesic_cuda_dims(shape):
    """Smaller thread block for register-heavy geodesic kernels."""
    tpb = (16, 16)
    bpg = (
        (shape[0] + tpb[0] - 1) // tpb[0],
        (shape[1] + tpb[1] - 1) // tpb[1],
    )
    return bpg, tpb


# =====================================================================
# Planar backend functions
# =====================================================================

@ngjit
def _cpu(data, cellsize_x, cellsize_y):
    data = data.astype(np.float32)
    out = np.empty(data.shape, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            # No early-out on a NaN centre: the branch is unpredictable on
            # speckled nodata and costs more than the arithmetic it skips.
            # Neighbour NaN already propagates through the stencil; the centre
            # is folded back in with a select so out[y, x] is NaN whenever
            # data[y, x] is.
            a = data[y + 1, x - 1]
            b = data[y + 1, x]
            c = data[y + 1, x + 1]
            d = data[y, x - 1]
            f = data[y, x + 1]
            g = data[y - 1, x - 1]
            h = data[y - 1, x]
            i = data[y - 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
            r = np.arctan(p) * 57.29578
            ctr = data[y, x]
            out[y, x] = r if ctr == ctr else np.nan
    return out


def _run_numpy(data: np.ndarray,
               cellsize_x: Union[int, float],
               cellsize_y: Union[int, float],
               boundary: str = 'nan') -> np.ndarray:
    if boundary == 'nan':
        return _cpu(data, cellsize_x, cellsize_y)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded, cellsize_x, cellsize_y)
    return result[1:-1, 1:-1]


def _run_dask_numpy(data: da.Array,
                    cellsize_x: Union[int, float],
                    cellsize_y: Union[int, float],
                    boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float32)
    _func = partial(_cpu,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary),
                           meta=np.array((), dtype=np.float32),
                           **_dask_task_name_kwargs('xrspatial.slope'))
    return out


def _run_dask_cupy(data: da.Array,
                   cellsize_x: Union[int, float],
                   cellsize_y: Union[int, float],
                   boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float32)
    _func = partial(_run_cupy,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary, is_cupy=True),
                           meta=cupy.array((), dtype=cupy.float32),
                           **_dask_task_name_kwargs('xrspatial.slope'))
    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize_x, cellsize_y):
    if isnan(arr[1, 1]):
        return nan
    a = arr[2, 0]
    b = arr[2, 1]
    c = arr[2, 2]
    d = arr[1, 0]
    f = arr[1, 2]
    g = arr[0, 0]
    h = arr[0, 1]
    i = arr[0, 2]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x[0])
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y[0])
    p = (dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5
    return atan(p) * 57.29578


@cuda.jit
def _run_gpu(arr, cellsize_x_arr, cellsize_y_arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di < out.shape[0] and
            j - dj >= 0 and j + dj < out.shape[1]):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1],
                         cellsize_x_arr,
                         cellsize_y_arr)


def _run_cupy(data: cupy.ndarray,
              cellsize_x: Union[int, float],
              cellsize_y: Union[int, float],
              boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded, cellsize_x, cellsize_y)
        return result[1:-1, 1:-1]

    cellsize_x_arr = cupy.array([float(cellsize_x)], dtype='f4')
    cellsize_y_arr = cupy.array([float(cellsize_y)], dtype='f4')
    data = data.astype(cupy.float32)

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data,
                                cellsize_x_arr,
                                cellsize_y_arr,
                                out)
    return out


# =====================================================================
# Geodesic backend functions
# =====================================================================

def _run_numpy_geodesic(data, lat_2d, lon_2d, a2, b2, z_factor, boundary='nan'):
    if boundary != 'nan':
        data_p = _pad_array(data.astype(np.float64), 1, boundary)
        lat_p = _pad_array(lat_2d, 1, boundary)
        lon_p = _pad_array(lon_2d, 1, boundary)
        stacked = np.stack([data_p, lat_p, lon_p], axis=0)
        result = _cpu_geodesic_slope(stacked, a2, b2, z_factor)
        return result[1:-1, 1:-1]
    stacked = np.stack([
        data.astype(np.float64),
        lat_2d,
        lon_2d,
    ], axis=0)
    return _cpu_geodesic_slope(stacked, a2, b2, z_factor)


def _run_cupy_geodesic(data, lat_2d, lon_2d, a2, b2, z_factor, boundary='nan'):
    if boundary != 'nan':
        data_p = _pad_array(data.astype(cupy.float64), 1, boundary)
        lat_p = _pad_array(cupy.asarray(lat_2d, dtype=cupy.float64), 1, boundary)
        lon_p = _pad_array(cupy.asarray(lon_2d, dtype=cupy.float64), 1, boundary)
        stacked = cupy.stack([data_p, lat_p, lon_p], axis=0)
        H, W = data_p.shape
        out = cupy.full((H, W), cupy.nan, dtype=cupy.float32)
        a2_arr = cupy.array([a2], dtype=cupy.float64)
        b2_arr = cupy.array([b2], dtype=cupy.float64)
        zf_arr = cupy.array([z_factor], dtype=cupy.float64)
        inv_2r_arr = cupy.array([INV_2R], dtype=cupy.float64)
        griddim, blockdim = _geodesic_cuda_dims((H, W))
        _run_gpu_geodesic_slope[griddim, blockdim](stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out)
        return out[1:-1, 1:-1]

    lat_2d_gpu = cupy.asarray(lat_2d, dtype=cupy.float64)
    lon_2d_gpu = cupy.asarray(lon_2d, dtype=cupy.float64)
    stacked = cupy.stack([
        data.astype(cupy.float64),
        lat_2d_gpu,
        lon_2d_gpu,
    ], axis=0)

    H, W = data.shape
    out = cupy.full((H, W), cupy.nan, dtype=cupy.float32)

    a2_arr = cupy.array([a2], dtype=cupy.float64)
    b2_arr = cupy.array([b2], dtype=cupy.float64)
    zf_arr = cupy.array([z_factor], dtype=cupy.float64)
    inv_2r_arr = cupy.array([INV_2R], dtype=cupy.float64)

    griddim, blockdim = _geodesic_cuda_dims((H, W))
    _run_gpu_geodesic_slope[griddim, blockdim](stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out)
    return out


def _dask_geodesic_slope_chunk(stacked_chunk, a2, b2, z_factor):
    """Process a single chunk for dask map_overlap. stacked_chunk shape = (3, h, w).
    Returns (3, h, w) to preserve shape for map_overlap."""
    result_2d = _cpu_geodesic_slope(stacked_chunk, a2, b2, z_factor)
    out = np.empty_like(stacked_chunk, dtype=np.float32)
    out[0] = result_2d
    out[1] = 0.0
    out[2] = 0.0
    return out


def _dask_geodesic_slope_chunk_cupy(stacked_chunk, a2, b2, z_factor):
    """Process a single chunk for dask+cupy map_overlap."""
    H, W = stacked_chunk.shape[1], stacked_chunk.shape[2]
    result_2d = cupy.full((H, W), cupy.nan, dtype=cupy.float32)

    a2_arr = cupy.array([a2], dtype=cupy.float64)
    b2_arr = cupy.array([b2], dtype=cupy.float64)
    zf_arr = cupy.array([z_factor], dtype=cupy.float64)
    inv_2r_arr = cupy.array([INV_2R], dtype=cupy.float64)

    griddim, blockdim = _geodesic_cuda_dims((H, W))
    _run_gpu_geodesic_slope[griddim, blockdim](
        stacked_chunk, a2_arr, b2_arr, zf_arr, inv_2r_arr, result_2d
    )

    out = cupy.zeros_like(stacked_chunk, dtype=cupy.float32)
    out[0] = result_2d
    return out


def _run_dask_numpy_geodesic(data, lat_2d, lon_2d, a2, b2, z_factor, boundary='nan'):
    lat_dask = da.from_array(lat_2d, chunks=data.chunksize)
    lon_dask = da.from_array(lon_2d, chunks=data.chunksize)
    stacked = da.stack([
        data.astype(np.float64),
        lat_dask,
        lon_dask,
    ], axis=0).rechunk({0: 3})  # all 3 channels in one chunk

    _func = partial(_dask_geodesic_slope_chunk, a2=a2, b2=b2, z_factor=z_factor)
    dask_bnd = _boundary_to_dask(boundary)
    out = stacked.map_overlap(
        _func,
        depth=(0, 1, 1),
        boundary={0: np.nan, 1: dask_bnd, 2: dask_bnd},
        meta=np.array((), dtype=np.float32),
        **_dask_task_name_kwargs('xrspatial.slope'),
    )
    return out[0]


def _to_cupy_f64(block):
    # Only reached from the dask+cupy path, so `cupy` is the real module here,
    # never the import-time fallback class.
    return cupy.asarray(block, dtype=cupy.float64)


def _run_dask_cupy_geodesic(data, lat_2d, lon_2d, a2, b2, z_factor, boundary='nan'):
    # Keep lat/lon as dask-of-numpy on the (zero-stride) broadcast views, then
    # convert each block to cupy lazily. Converting up front with
    # cupy.asarray(lat_2d) would densify the full (H, W) grid onto a single GPU
    # at graph-construction time and OOM on large rasters.
    lat_dask = da.from_array(lat_2d, chunks=data.chunksize).map_blocks(
        _to_cupy_f64, dtype=np.float64)
    lon_dask = da.from_array(lon_2d, chunks=data.chunksize).map_blocks(
        _to_cupy_f64, dtype=np.float64)
    stacked = da.stack([
        data.astype(cupy.float64),
        lat_dask,
        lon_dask,
    ], axis=0).rechunk({0: 3})  # all 3 channels in one chunk

    _func = partial(_dask_geodesic_slope_chunk_cupy, a2=a2, b2=b2, z_factor=z_factor)
    dask_bnd = _boundary_to_dask(boundary, is_cupy=True)
    out = stacked.map_overlap(
        _func,
        depth=(0, 1, 1),
        boundary={0: cupy.nan, 1: dask_bnd, 2: dask_bnd},
        meta=cupy.array((), dtype=cupy.float32),
        **_dask_task_name_kwargs('xrspatial.slope'),
    )
    return out[0]


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def slope(agg: xr.DataArray,
          name: Optional[str] = 'slope',
          method: str = 'planar',
          z_unit: str = 'meter',
          boundary: str = 'nan') -> xr.DataArray:
    """
    Returns slope of input aggregate in degrees.

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        2D array of elevation data.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='slope'
        Name of output DataArray.
    method : str, default='planar'
        ``'planar'`` uses the classic Horn algorithm with uniform cell size.
        ``'geodesic'`` converts cells to Earth-Centered Earth-Fixed (ECEF)
        coordinates and fits a 3D plane, yielding accurate results for
        geographic (lat/lon) coordinate systems.
    z_unit : str, default='meter'
        Unit of the elevation values.  Only used when ``method='geodesic'``.
        Accepted values: ``'meter'``, ``'foot'``, ``'kilometer'``, ``'mile'``
        (and common aliases).
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     — fill missing neighbours with NaN (default).
        ``'nearest'`` — repeat edge values.
        ``'reflect'`` — mirror at boundary.
        ``'wrap'``    — periodic / toroidal.

    Returns
    -------
    slope_agg : xr.DataArray or xr.Dataset
        If `agg` is a DataArray, returns a DataArray of the same type.
        If `agg` is a Dataset, returns a Dataset with slope computed
        for each data variable.
        2D array of slope values in degrees. The output ``attrs['units']``
        is set to ``'degrees'``; all other input attributes are preserved.

    Notes
    -----
    The ``'planar'`` method uses the coordinate spacing directly as the cell
    size. If the coordinates are in degrees (lat/lon) but the elevation values
    are in meters, the result is wrong by orders of magnitude. When this
    mismatch is detected, a ``UserWarning`` is emitted suggesting you reproject
    to a projected CRS or use ``method='geodesic'``. The ``'planar'`` method also
    raises a ``ValueError`` if the cell size on either axis is zero, negative, or
    non-finite, since those values cannot produce a valid slope.

    References
    ----------
        - arcgis: http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import slope
        >>> data = np.array([
        ...     [0, 0, 0, 0, 0],
        ...     [0, 0, 0, -1, 2],
        ...     [0, 0, 0, 0, 1],
        ...     [0, 0, 0, 5, 0]])
        >>> agg = xr.DataArray(data)
        >>> slope_agg = slope(agg)
        >>> slope_agg
        <xarray.DataArray 'slope' (dim_0: 4, dim_1: 5)>
        array([[      nan,       nan,       nan,       nan,       nan],
               [      nan,  0.      , 14.036243, 32.512516,       nan],
               [      nan,  0.      , 42.031113, 53.395725,       nan],
               [      nan,       nan,       nan,       nan,       nan]],
              dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """

    _validate_raster(agg, func_name='slope', name='agg')

    if method not in ('planar', 'geodesic'):
        raise ValueError(
            f"method must be 'planar' or 'geodesic', got {method!r}"
        )
    _validate_boundary(boundary)

    if method == 'planar':
        warn_if_unit_mismatch(agg)
        cellsize_x, cellsize_y = get_dataarray_resolution(agg)
        # Reject negatives too (curvature() only checks == 0): a negative cell
        # size would silently flip the slope sign rather than error out.
        if (not np.isfinite(cellsize_x) or not np.isfinite(cellsize_y)
                or cellsize_x <= 0 or cellsize_y <= 0):
            raise ValueError(
                "slope(method='planar') requires a positive, finite cell size "
                f"on both axes; got cellsize_x={cellsize_x!r}, "
                f"cellsize_y={cellsize_y!r}. "
                "Set agg.attrs['res'] to a (x, y) tuple of positive floats, "
                "or attach numeric x/y coordinates to the DataArray."
            )
        mapper = ArrayTypeFunctionMapping(
            numpy_func=_run_numpy,
            cupy_func=_run_cupy,
            dask_func=_run_dask_numpy,
            dask_cupy_func=_run_dask_cupy,
        )
        out = mapper(agg)(agg.data, cellsize_x, cellsize_y, boundary)

    else:  # geodesic
        if z_unit not in Z_UNITS:
            raise ValueError(
                f"z_unit must be one of {sorted(Z_UNITS)}, "
                f"got {z_unit!r}"
            )
        z_factor = Z_UNITS[z_unit]

        _check_geodesic_memory_backend_aware(agg, func_name='slope')

        lat_2d, lon_2d = _extract_latlon_coords(agg)

        mapper = ArrayTypeFunctionMapping(
            numpy_func=_run_numpy_geodesic,
            cupy_func=_run_cupy_geodesic,
            dask_func=_run_dask_numpy_geodesic,
            dask_cupy_func=_run_dask_cupy_geodesic,
        )
        out = mapper(agg)(agg.data, lat_2d, lon_2d, WGS84_A2, WGS84_B2, z_factor, boundary)

    # slope is reported in degrees, so override the input elevation 'units'
    # (e.g. 'm') with the angle unit the result actually carries. Other input
    # attrs are preserved.
    attrs = dict(agg.attrs)
    attrs['units'] = 'degrees'

    result = xr.DataArray(out,
                          name=name,
                          coords=agg.coords,
                          dims=agg.dims,
                          attrs=attrs)
    # On dask backends, xr.DataArray keeps the dask array's internal graph
    # token as .name when name=None, so reset it post-construction to match
    # the numpy/cupy backends. (Same fix as zonal #2611, focal #2733.)
    result.name = name
    return result
