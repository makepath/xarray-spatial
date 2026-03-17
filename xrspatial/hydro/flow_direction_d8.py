from __future__ import annotations

import math
from functools import partial
from typing import Union

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

from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import _boundary_to_dask
from xrspatial.utils import _pad_array
from xrspatial.utils import _validate_boundary
from xrspatial.utils import _validate_raster
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import ngjit
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _cpu(data, cellsize_x, cellsize_y):
    out = np.empty(data.shape, np.float64)
    out[:] = np.nan
    rows, cols = data.shape

    # 8 neighbor offsets: E, SE, S, SW, W, NW, N, NE
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    codes = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])

    diag = math.sqrt(cellsize_x * cellsize_x + cellsize_y * cellsize_y)
    # distances: E=cx, SE=diag, S=cy, SW=diag, W=cx, NW=diag, N=cy, NE=diag
    dists = np.array([cellsize_x, diag, cellsize_y, diag,
                      cellsize_x, diag, cellsize_y, diag])

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            if center != center:  # NaN check
                continue

            has_nan = False
            for k in range(8):
                v = data[y + dy[k], x + dx[k]]
                if v != v:
                    has_nan = True
                    break
            if has_nan:
                continue

            max_slope = -math.inf
            direction = 0.0
            for k in range(8):
                v = data[y + dy[k], x + dx[k]]
                grad = (center - v) / dists[k]
                if grad > max_slope:
                    max_slope = grad
                    direction = codes[k]

            if max_slope <= 0.0:
                out[y, x] = 0.0
            else:
                out[y, x] = direction

    return out


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit(device=True)
def _gpu(arr, cellsize_x, cellsize_y):
    center = arr[1, 1]
    if center != center:
        return center  # NaN

    cx = cellsize_x[0]
    cy = cellsize_y[0]
    diag = (cx * cx + cy * cy) ** 0.5

    max_slope = -1.0e308
    direction = 0.0

    # E: arr[1, 2], distance = cx, code = 1
    v = arr[1, 2]
    if v != v:
        return v
    grad = (center - v) / cx
    if grad > max_slope:
        max_slope = grad
        direction = 1.0

    # SE: arr[2, 2], distance = diag, code = 2
    v = arr[2, 2]
    if v != v:
        return v
    grad = (center - v) / diag
    if grad > max_slope:
        max_slope = grad
        direction = 2.0

    # S: arr[2, 1], distance = cy, code = 4
    v = arr[2, 1]
    if v != v:
        return v
    grad = (center - v) / cy
    if grad > max_slope:
        max_slope = grad
        direction = 4.0

    # SW: arr[2, 0], distance = diag, code = 8
    v = arr[2, 0]
    if v != v:
        return v
    grad = (center - v) / diag
    if grad > max_slope:
        max_slope = grad
        direction = 8.0

    # W: arr[1, 0], distance = cx, code = 16
    v = arr[1, 0]
    if v != v:
        return v
    grad = (center - v) / cx
    if grad > max_slope:
        max_slope = grad
        direction = 16.0

    # NW: arr[0, 0], distance = diag, code = 32
    v = arr[0, 0]
    if v != v:
        return v
    grad = (center - v) / diag
    if grad > max_slope:
        max_slope = grad
        direction = 32.0

    # N: arr[0, 1], distance = cy, code = 64
    v = arr[0, 1]
    if v != v:
        return v
    grad = (center - v) / cy
    if grad > max_slope:
        max_slope = grad
        direction = 64.0

    # NE: arr[0, 2], distance = diag, code = 128
    v = arr[0, 2]
    if v != v:
        return v
    grad = (center - v) / diag
    if grad > max_slope:
        max_slope = grad
        direction = 128.0

    if max_slope <= 0.0:
        return 0.0

    return direction


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


# =====================================================================
# Backend wrappers
# =====================================================================

def _run_numpy(data: np.ndarray,
               cellsize_x: Union[int, float],
               cellsize_y: Union[int, float],
               boundary: str = 'nan') -> np.ndarray:
    data = data.astype(np.float64)
    if boundary == 'nan':
        return _cpu(data, cellsize_x, cellsize_y)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded, cellsize_x, cellsize_y)
    return result[1:-1, 1:-1]


def _run_dask_numpy(data: da.Array,
                    cellsize_x: Union[int, float],
                    cellsize_y: Union[int, float],
                    boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float64)
    _func = partial(_cpu,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary),
                           meta=np.array(()))
    return out


def _run_cupy(data: cupy.ndarray,
              cellsize_x: Union[int, float],
              cellsize_y: Union[int, float],
              boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded, cellsize_x, cellsize_y)
        return result[1:-1, 1:-1]

    cellsize_x_arr = cupy.array([float(cellsize_x)], dtype='f8')
    cellsize_y_arr = cupy.array([float(cellsize_y)], dtype='f8')
    data = data.astype(cupy.float64)

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f8')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data,
                                cellsize_x_arr,
                                cellsize_y_arr,
                                out)
    return out


def _run_dask_cupy(data: da.Array,
                   cellsize_x: Union[int, float],
                   cellsize_y: Union[int, float],
                   boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float64)
    _func = partial(_run_cupy,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary, is_cupy=True),
                           meta=cupy.array(()))
    return out


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_direction_d8(agg: xr.DataArray,
                   name: str = 'flow_direction',
                   boundary: str = 'nan') -> xr.DataArray:
    """Compute D8 flow direction for each cell.

    Determines which of the 8 neighbors has the steepest downhill
    gradient from the center cell.  Uses the standard power-of-two D8
    direction encoding::

         32  64  128
         16   0    1
          8   4    2

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='flow_direction'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     - fill missing neighbours with NaN (default).
        ``'nearest'`` - repeat edge values.
        ``'reflect'`` - mirror at boundary.
        ``'wrap'``    - periodic / toroidal.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of D8 flow direction codes.  Valid values are
        ``{0, 1, 2, 4, 8, 16, 32, 64, 128}`` for interior cells
        (0 indicates a pit or flat with no downhill neighbor).
        Edge cells and cells with NaN in their 3x3 window are NaN.

    References
    ----------
    Jenson, S.K. and Domingue, J.O. (1988). Extracting Topographic
    Structure from Digital Elevation Data for Geographic Information
    System Analysis. Photogrammetric Engineering and Remote Sensing,
    54(11), 1593-1600.
    """
    _validate_raster(agg, func_name='flow_direction', name='agg')
    _validate_boundary(boundary)

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy,
    )
    out = mapper(agg)(agg.data, cellsize_x, cellsize_y, boundary)

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
