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

    cx = cellsize_x
    cy = cellsize_y
    diag = math.sqrt(cx * cx + cy * cy)
    pi_over_4 = math.pi / 4.0
    two_pi = 2.0 * math.pi

    # 8 neighbors counterclockwise from E:
    #   E, NE, N, NW, W, SW, S, SE
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    # d1 for each facet (distance from center to e1)
    d1 = np.array([cx, diag, cy, diag, cx, diag, cy, diag])
    # d2 for each facet (distance from e1 to e2)
    d2 = np.array([cy, cx, cx, cy, cy, cx, cx, cy])

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            if center != center:  # NaN check
                continue

            # Check all 8 neighbors for NaN
            has_nan = False
            for k in range(8):
                v = data[y + nb_dy[k], x + nb_dx[k]]
                if v != v:
                    has_nan = True
                    break
            if has_nan:
                continue

            max_slope = -1.0e308
            best_angle = -1.0

            for k in range(8):
                k2 = (k + 1) % 8
                e1 = data[y + nb_dy[k], x + nb_dx[k]]
                e2 = data[y + nb_dy[k2], x + nb_dx[k2]]

                s1 = (center - e1) / d1[k]
                s2 = (e1 - e2) / d2[k]
                r = math.atan2(s2, s1)

                if r < 0.0:
                    r = 0.0
                    s = s1
                elif r > pi_over_4:
                    r = pi_over_4
                    s = (center - e2) / math.sqrt(d1[k] * d1[k] + d2[k] * d2[k])
                else:
                    s = math.sqrt(s1 * s1 + s2 * s2)

                if s > max_slope:
                    max_slope = s
                    best_angle = k * pi_over_4 + r

            if max_slope <= 0.0:
                out[y, x] = -1.0
            else:
                # Wrap 2*pi -> 0
                if best_angle >= two_pi:
                    best_angle = 0.0
                out[y, x] = best_angle

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
    pi_over_4 = 0.7853981633974483  # pi/4
    two_pi = 6.283185307179586      # 2*pi

    # Read 8 neighbors: E, NE, N, NW, W, SW, S, SE
    e = arr[1, 2]
    ne = arr[0, 2]
    n = arr[0, 1]
    nw = arr[0, 0]
    w = arr[1, 0]
    sw = arr[2, 0]
    s = arr[2, 1]
    se = arr[2, 2]

    # NaN check on all neighbors
    if (e != e or ne != ne or n != n or nw != nw or
            w != w or sw != sw or s != s or se != se):
        return e * 0.0 / 0.0  # NaN

    max_slope = -1.0e308
    best_angle = -1.0

    # Facet 0: e1=E, e2=NE, d1=cx, d2=cy, start=0
    s1 = (center - e) / cx
    s2 = (e - ne) / cy
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - ne) / diag
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = r  # 0 + r

    # Facet 1: e1=NE, e2=N, d1=diag, d2=cx, start=pi/4
    s1 = (center - ne) / diag
    s2 = (ne - n) / cx
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - n) / (diag * diag + cx * cx) ** 0.5
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = pi_over_4 + r

    # Facet 2: e1=N, e2=NW, d1=cy, d2=cx, start=pi/2
    s1 = (center - n) / cy
    s2 = (n - nw) / cx
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - nw) / diag
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 2.0 * pi_over_4 + r

    # Facet 3: e1=NW, e2=W, d1=diag, d2=cy, start=3*pi/4
    s1 = (center - nw) / diag
    s2 = (nw - w) / cy
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - w) / (diag * diag + cy * cy) ** 0.5
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 3.0 * pi_over_4 + r

    # Facet 4: e1=W, e2=SW, d1=cx, d2=cy, start=pi
    s1 = (center - w) / cx
    s2 = (w - sw) / cy
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - sw) / diag
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 4.0 * pi_over_4 + r

    # Facet 5: e1=SW, e2=S, d1=diag, d2=cx, start=5*pi/4
    s1 = (center - sw) / diag
    s2 = (sw - s) / cx
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - s) / (diag * diag + cx * cx) ** 0.5
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 5.0 * pi_over_4 + r

    # Facet 6: e1=S, e2=SE, d1=cy, d2=cx, start=3*pi/2
    s1 = (center - s) / cy
    s2 = (s - se) / cx
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - se) / diag
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 6.0 * pi_over_4 + r

    # Facet 7: e1=SE, e2=E, d1=diag, d2=cy, start=7*pi/4
    s1 = (center - se) / diag
    s2 = (se - e) / cy
    r = math.atan2(s2, s1)
    if r < 0.0:
        r = 0.0
        slope_val = s1
    elif r > pi_over_4:
        r = pi_over_4
        slope_val = (center - e) / (diag * diag + cy * cy) ** 0.5
    else:
        slope_val = (s1 * s1 + s2 * s2) ** 0.5
    if slope_val > max_slope:
        max_slope = slope_val
        best_angle = 7.0 * pi_over_4 + r

    if max_slope <= 0.0:
        return -1.0

    # Wrap 2*pi -> 0
    if best_angle >= two_pi:
        best_angle = 0.0

    return best_angle


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
def flow_direction_dinf(agg: xr.DataArray,
                        name: str = 'flow_direction_dinf',
                        boundary: str = 'nan') -> xr.DataArray:
    """Compute D-infinity flow direction for each cell.

    Determines flow direction as a continuous angle toward the steepest
    downslope facet, following the Tarboton (1997) algorithm.  The 3x3
    neighborhood is divided into 8 triangular facets; the steepest
    downslope plane gives the flow angle.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='flow_direction_dinf'
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
        2D array of continuous flow direction angles in radians.
        Valid values are in the range ``[0, 2*pi)``.
        ``-1.0`` indicates a pit or flat with no downslope neighbor.
        Edge cells and cells with NaN in their 3x3 window are NaN.

    References
    ----------
    Tarboton, D.G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models.
    Water Resources Research, 33(2), 309-319.
    """
    _validate_raster(agg, func_name='flow_direction_dinf', name='agg')
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
