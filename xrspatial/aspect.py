from __future__ import annotations

from functools import partial
from math import atan2
from typing import Optional

try:
    import dask.array as da
except ImportError:
    da = None


import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import Z_UNITS
from xrspatial.utils import _boundary_to_dask
from xrspatial.utils import _extract_latlon_coords
from xrspatial.utils import _pad_array
from xrspatial.utils import _validate_boundary
from xrspatial.utils import _validate_raster
from xrspatial.utils import cuda_args
from xrspatial.utils import ngjit
from xrspatial.dataset_support import supports_dataset


def _geodesic_cuda_dims(shape):
    """Smaller thread block for register-heavy geodesic kernels."""
    tpb = (16, 16)
    bpg = (
        (shape[0] + tpb[0] - 1) // tpb[0],
        (shape[1] + tpb[1] - 1) // tpb[1],
    )
    return bpg, tpb

from xrspatial.geodesic import (
    INV_2R,
    WGS84_A2,
    WGS84_B2,
    _cpu_geodesic_aspect,
    _run_gpu_geodesic_aspect,
)

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

RADIAN = 180 / np.pi


# =====================================================================
# Planar backend functions (unchanged)
# =====================================================================

@ngjit
def _cpu(data: np.ndarray):
    data = data.astype(np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            a = data[y-1, x-1]
            b = data[y-1, x]
            c = data[y-1, x+1]
            d = data[y, x-1]
            f = data[y, x+1]
            g = data[y+1, x-1]
            h = data[y+1, x]
            i = data[y+1, x+1]

            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            if dz_dx == 0 and dz_dy == 0:
                # flat surface, slope = 0, thus invalid aspect
                out[y, x] = -1.
            else:
                _aspect = np.arctan2(dz_dy, -dz_dx) * RADIAN
                # convert to compass direction values (0-360 degrees)
                if _aspect < 0:
                    out[y, x] = 90.0 - _aspect
                elif _aspect > 90.0:
                    out[y, x] = 360.0 - _aspect + 90.0
                else:
                    out[y, x] = 90.0 - _aspect

    return out


def _run_numpy(data: np.ndarray, boundary: str = 'nan') -> np.ndarray:
    if boundary == 'nan':
        return _cpu(data)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded)
    return result[1:-1, 1:-1]


@cuda.jit(device=True)
def _gpu(arr):

    a = arr[0, 0]
    b = arr[0, 1]
    c = arr[0, 2]
    d = arr[1, 0]
    f = arr[1, 2]
    g = arr[2, 0]
    h = arr[2, 1]
    i = arr[2, 2]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

    if dz_dx == 0 and dz_dy == 0:
        # flat surface, slope = 0, thus invalid aspect
        _aspect = -1
    else:
        _aspect = atan2(dz_dy, -dz_dx) * 57.29578
        # convert to compass direction values (0-360 degrees)
        if _aspect < 0:
            _aspect = 90 - _aspect
        elif _aspect > 90:
            _aspect = 360 - _aspect + 90
        else:
            _aspect = 90 - _aspect

    if _aspect > 359.999:  # lame float equality check...
        return 0
    else:
        return _aspect


@cuda.jit
def _run_gpu(arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i-di >= 0 and
        i+di < out.shape[0] and
            j-dj >= 0 and
            j+dj < out.shape[1]):
        out[i, j] = _gpu(arr[i-di:i+di+1, j-dj:j+dj+1])


def _run_cupy(data: cupy.ndarray, boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded)
        return result[1:-1, 1:-1]

    data = data.astype(cupy.float32)
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    _run_gpu[griddim, blockdim](data, out)
    return out


def _run_dask_numpy(data: da.Array, boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float32)
    _func = partial(_cpu)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary),
                           meta=np.array(()))
    return out


def _run_dask_cupy(data: da.Array, boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float32)
    _func = partial(_run_cupy)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary, is_cupy=True),
                           meta=cupy.array(()))
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
        result = _cpu_geodesic_aspect(stacked, a2, b2, z_factor)
        return result[1:-1, 1:-1]
    stacked = np.stack([
        data.astype(np.float64),
        lat_2d,
        lon_2d,
    ], axis=0)
    return _cpu_geodesic_aspect(stacked, a2, b2, z_factor)


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
        _run_gpu_geodesic_aspect[griddim, blockdim](stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out)
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
    _run_gpu_geodesic_aspect[griddim, blockdim](stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out)
    return out


def _dask_geodesic_aspect_chunk(stacked_chunk, a2, b2, z_factor):
    """Returns (3, h, w) to preserve shape for map_overlap."""
    result_2d = _cpu_geodesic_aspect(stacked_chunk, a2, b2, z_factor)
    out = np.empty_like(stacked_chunk, dtype=np.float32)
    out[0] = result_2d
    out[1] = 0.0
    out[2] = 0.0
    return out


def _dask_geodesic_aspect_chunk_cupy(stacked_chunk, a2, b2, z_factor):
    H, W = stacked_chunk.shape[1], stacked_chunk.shape[2]
    result_2d = cupy.full((H, W), cupy.nan, dtype=cupy.float32)

    a2_arr = cupy.array([a2], dtype=cupy.float64)
    b2_arr = cupy.array([b2], dtype=cupy.float64)
    zf_arr = cupy.array([z_factor], dtype=cupy.float64)
    inv_2r_arr = cupy.array([INV_2R], dtype=cupy.float64)

    griddim, blockdim = _geodesic_cuda_dims((H, W))
    _run_gpu_geodesic_aspect[griddim, blockdim](stacked_chunk, a2_arr, b2_arr, zf_arr, inv_2r_arr, result_2d)

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
    ], axis=0).rechunk({0: 3})

    _func = partial(_dask_geodesic_aspect_chunk, a2=a2, b2=b2, z_factor=z_factor)
    dask_bnd = _boundary_to_dask(boundary)
    out = stacked.map_overlap(
        _func,
        depth=(0, 1, 1),
        boundary={0: np.nan, 1: dask_bnd, 2: dask_bnd},
        meta=np.array((), dtype=np.float32),
    )
    return out[0]


def _run_dask_cupy_geodesic(data, lat_2d, lon_2d, a2, b2, z_factor, boundary='nan'):
    lat_dask = da.from_array(cupy.asarray(lat_2d, dtype=cupy.float64),
                             chunks=data.chunksize)
    lon_dask = da.from_array(cupy.asarray(lon_2d, dtype=cupy.float64),
                             chunks=data.chunksize)
    stacked = da.stack([
        data.astype(cupy.float64),
        lat_dask,
        lon_dask,
    ], axis=0).rechunk({0: 3})

    _func = partial(_dask_geodesic_aspect_chunk_cupy, a2=a2, b2=b2, z_factor=z_factor)
    dask_bnd = _boundary_to_dask(boundary, is_cupy=True)
    out = stacked.map_overlap(
        _func,
        depth=(0, 1, 1),
        boundary={0: cupy.nan, 1: dask_bnd, 2: dask_bnd},
        meta=cupy.array((), dtype=cupy.float32),
    )
    return out[0]


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def aspect(agg: xr.DataArray,
           name: Optional[str] = 'aspect',
           method: str = 'planar',
           z_unit: str = 'meter',
           boundary: str = 'nan') -> xr.DataArray:
    """
    Calculates the aspect value of an elevation aggregate.

    Calculates, for all cells in the array, the downward slope direction
    of each cell based on the elevation of its neighbors in a 3x3 grid.
    The value is measured clockwise in degrees with 0 (due north), and 360
    (again due north). Values along the edges are not calculated.

    Direction of the aspect can be determined by its value:
    From 0     to 22.5:  North
    From 22.5  to 67.5:  Northeast
    From 67.5  to 112.5: East
    From 112.5 to 157.5: Southeast
    From 157.5 to 202.5: South
    From 202.5 to 247.5: West
    From 247.5 to 292.5: Northwest
    From 337.5 to 360:   North

    Note that values of -1 denote flat areas.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, or Dask with NumPy-backed xarray DataArray
        of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='aspect'
        Name of ouput DataArray.
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
    aspect_agg : xarray.DataArray or xr.Dataset
        If `agg` is a DataArray, returns a DataArray of the same type.
        If `agg` is a Dataset, returns a Dataset with aspect computed
        for each data variable.
        2D aggregate array of calculated aspect values.
        All other input attributes are preserved.

    References
    ----------


    Examples
    --------
    Aspect works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import aspect

        >>> data = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 2, 0],
            [1, 1, 1, 0, 0],
            [4, 4, 9, 2, 4],
            [1, 5, 0, 1, 4],
            [1, 5, 0, 5, 5]
        ], dtype=np.float32)
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> aspect_agg = aspect(raster)
    """

    _validate_raster(agg, func_name='aspect', name='agg')

    if method not in ('planar', 'geodesic'):
        raise ValueError(
            f"method must be 'planar' or 'geodesic', got {method!r}"
        )
    _validate_boundary(boundary)

    if method == 'planar':
        mapper = ArrayTypeFunctionMapping(
            numpy_func=_run_numpy,
            dask_func=_run_dask_numpy,
            cupy_func=_run_cupy,
            dask_cupy_func=_run_dask_cupy,
        )
        out = mapper(agg)(agg.data, boundary=boundary)

    else:  # geodesic
        if z_unit not in Z_UNITS:
            raise ValueError(
                f"z_unit must be one of {sorted(set(Z_UNITS.values()), key=str)}, "
                f"got {z_unit!r}"
            )
        z_factor = Z_UNITS[z_unit]

        lat_2d, lon_2d = _extract_latlon_coords(agg)

        mapper = ArrayTypeFunctionMapping(
            numpy_func=_run_numpy_geodesic,
            cupy_func=_run_cupy_geodesic,
            dask_func=_run_dask_numpy_geodesic,
            dask_cupy_func=_run_dask_cupy_geodesic,
        )
        out = mapper(agg)(agg.data, lat_2d, lon_2d, WGS84_A2, WGS84_B2, z_factor, boundary)

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
