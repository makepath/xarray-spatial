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

import numba
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

# Neighbor order: E, SE, S, SW, W, NW, N, NE
NEIGHBOR_NAMES = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']

_SQRT2_INV = 1.0 / math.sqrt(2.0)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _cpu(data, cellsize_x, cellsize_y, p_fixed):
    """Compute MFD flow fractions.

    Returns (8, rows, cols) float64 array.  p_fixed <= 0 triggers the
    adaptive exponent from Qin et al. (2007).
    """
    rows, cols = data.shape
    out = np.full((8, rows, cols), np.nan, dtype=np.float64)

    cx = cellsize_x
    cy = cellsize_y
    diag = math.sqrt(cx * cx + cy * cy)
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    # 8 neighbor offsets: E, SE, S, SW, W, NW, N, NE
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    dists = np.array([cx, diag, cy, diag, cx, diag, cy, diag])
    contour = np.array([1.0, sqrt2_inv, 1.0, sqrt2_inv,
                        1.0, sqrt2_inv, 1.0, sqrt2_inv])

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            if center != center:  # NaN
                continue

            has_nan = False
            for k in range(8):
                v = data[y + dy[k], x + dx[k]]
                if v != v:
                    has_nan = True
                    break
            if has_nan:
                continue

            # Compute slopes to downslope neighbors
            slopes = np.zeros(8, np.float64)
            n_down = 0
            sum_slope = 0.0
            max_slope = 0.0

            for k in range(8):
                v = data[y + dy[k], x + dx[k]]
                s = (center - v) / dists[k]
                if s > 0.0:
                    slopes[k] = s
                    sum_slope += s
                    n_down += 1
                    if s > max_slope:
                        max_slope = s

            if n_down == 0:
                # Pit or flat
                for k in range(8):
                    out[k, y, x] = 0.0
                continue

            # Determine exponent
            if p_fixed > 0.0:
                p = p_fixed
            else:
                # Adaptive: Qin et al. (2007)
                mean_slope = sum_slope / n_down
                p = max_slope / mean_slope

            # Weighted fractions
            total = 0.0
            weights = np.zeros(8, np.float64)
            for k in range(8):
                if slopes[k] > 0.0:
                    w = (slopes[k] * contour[k]) ** p
                    weights[k] = w
                    total += w

            for k in range(8):
                if total > 0.0 and weights[k] > 0.0:
                    out[k, y, x] = weights[k] / total
                else:
                    out[k, y, x] = 0.0

    return out


# =====================================================================
# GPU kernel
# =====================================================================

@cuda.jit
def _run_gpu(arr, cellsize_x_arr, cellsize_y_arr, p_arr, out):
    i, j = cuda.grid(2)
    if i < 1 or i >= out.shape[1] - 1 or j < 1 or j >= out.shape[2] - 1:
        return

    center = arr[i, j]
    if center != center:
        for k in range(8):
            out[k, i, j] = center  # NaN
        return

    cx = cellsize_x_arr[0]
    cy = cellsize_y_arr[0]
    diag = (cx * cx + cy * cy) ** 0.5
    sqrt2_inv = 0.7071067811865475

    # Read 8 neighbors: E, SE, S, SW, W, NW, N, NE
    nb = cuda.local.array(8, numba.float64)
    nb[0] = arr[i, j + 1]      # E
    nb[1] = arr[i + 1, j + 1]  # SE
    nb[2] = arr[i + 1, j]      # S
    nb[3] = arr[i + 1, j - 1]  # SW
    nb[4] = arr[i, j - 1]      # W
    nb[5] = arr[i - 1, j - 1]  # NW
    nb[6] = arr[i - 1, j]      # N
    nb[7] = arr[i - 1, j + 1]  # NE

    # NaN check
    for k in range(8):
        if nb[k] != nb[k]:
            nan_val = nb[k]
            for m in range(8):
                out[m, i, j] = nan_val
            return

    dist = cuda.local.array(8, numba.float64)
    dist[0] = cx;   dist[1] = diag; dist[2] = cy;   dist[3] = diag
    dist[4] = cx;   dist[5] = diag; dist[6] = cy;   dist[7] = diag

    cont = cuda.local.array(8, numba.float64)
    cont[0] = 1.0;        cont[1] = sqrt2_inv
    cont[2] = 1.0;        cont[3] = sqrt2_inv
    cont[4] = 1.0;        cont[5] = sqrt2_inv
    cont[6] = 1.0;        cont[7] = sqrt2_inv

    slopes = cuda.local.array(8, numba.float64)
    n_down = 0
    sum_slope = 0.0
    max_slope = 0.0

    for k in range(8):
        s = (center - nb[k]) / dist[k]
        if s > 0.0:
            slopes[k] = s
            sum_slope += s
            n_down += 1
            if s > max_slope:
                max_slope = s
        else:
            slopes[k] = 0.0

    if n_down == 0:
        for k in range(8):
            out[k, i, j] = 0.0
        return

    # Exponent
    p_val = p_arr[0]
    if p_val <= 0.0:
        mean_slope = sum_slope / n_down
        p_val = max_slope / mean_slope

    # Weights
    total = 0.0
    weights = cuda.local.array(8, numba.float64)
    for k in range(8):
        if slopes[k] > 0.0:
            w = (slopes[k] * cont[k]) ** p_val
            weights[k] = w
            total += w
        else:
            weights[k] = 0.0

    for k in range(8):
        if total > 0.0 and weights[k] > 0.0:
            out[k, i, j] = weights[k] / total
        else:
            out[k, i, j] = 0.0


# =====================================================================
# Backend wrappers
# =====================================================================

def _run_numpy(data: np.ndarray,
               cellsize_x: Union[int, float],
               cellsize_y: Union[int, float],
               p_fixed: float,
               boundary: str = 'nan') -> np.ndarray:
    data = data.astype(np.float64)
    if boundary == 'nan':
        return _cpu(data, cellsize_x, cellsize_y, p_fixed)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded, cellsize_x, cellsize_y, p_fixed)
    return result[:, 1:-1, 1:-1]


def _run_dask_numpy(data: da.Array,
                    cellsize_x: Union[int, float],
                    cellsize_y: Union[int, float],
                    p_fixed: float,
                    boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float64)
    bnd = _boundary_to_dask(boundary)

    # map_overlap requires same-shape output, so compute one band at a
    # time and stack.  The Numba-JIT kernel is fast enough that the
    # redundant per-band calls are cheap relative to dask overhead.
    bands = []
    for band_idx in range(8):
        def _band_k(chunk, cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y, p_fixed=p_fixed,
                    k=band_idx):
            result = _cpu(chunk, cellsize_x, cellsize_y, p_fixed)
            return result[k]

        band = data.map_overlap(_band_k,
                                depth=(1, 1),
                                boundary=bnd,
                                meta=np.array(()))
        bands.append(band)

    return da.stack(bands, axis=0)


def _run_cupy(data: cupy.ndarray,
              cellsize_x: Union[int, float],
              cellsize_y: Union[int, float],
              p_fixed: float,
              boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded, cellsize_x, cellsize_y, p_fixed)
        return result[:, 1:-1, 1:-1]

    cellsize_x_arr = cupy.array([float(cellsize_x)], dtype='f8')
    cellsize_y_arr = cupy.array([float(cellsize_y)], dtype='f8')
    p_arr = cupy.array([float(p_fixed)], dtype='f8')
    data = data.astype(cupy.float64)

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.full((8,) + data.shape, cupy.nan, dtype='f8')

    _run_gpu[griddim, blockdim](data,
                                cellsize_x_arr,
                                cellsize_y_arr,
                                p_arr,
                                out)
    return out


def _run_dask_cupy(data: da.Array,
                   cellsize_x: Union[int, float],
                   cellsize_y: Union[int, float],
                   p_fixed: float,
                   boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float64)
    bnd = _boundary_to_dask(boundary, is_cupy=True)

    bands = []
    for band_idx in range(8):
        def _band_k(chunk, cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y, p_fixed=p_fixed,
                    k=band_idx):
            result = _run_cupy(chunk, cellsize_x, cellsize_y, p_fixed)
            return result[k]

        band = data.map_overlap(_band_k,
                                depth=(1, 1),
                                boundary=bnd,
                                meta=cupy.array(()))
        bands.append(band)

    return da.stack(bands, axis=0)


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_direction_mfd(agg: xr.DataArray,
                       p: float = None,
                       name: str = 'flow_direction_mfd',
                       boundary: str = 'nan') -> xr.DataArray:
    """Compute multiple flow direction fractions for each cell.

    Partitions flow from each cell to all downslope neighbors using
    the MFD algorithm.  An adaptive flow-partition exponent (Qin et al.
    2007) adjusts automatically based on local terrain so that steep
    convergent areas concentrate flow while gentle slopes spread it out.

    The output is a 3-D array of shape ``(8, H, W)`` where each band
    holds the fraction of flow directed to one of the 8 neighbors
    (E, SE, S, SW, W, NW, N, NE).  Fractions sum to 1.0 at each cell.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2-D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    p : float or None, default=None
        Flow-partition exponent.  ``None`` uses the adaptive exponent
        from Qin et al. (2007): ``p = max_slope / mean_slope`` across
        downslope neighbors.  A positive float sets a fixed exponent
        (e.g. ``p=1.0`` for Quinn et al. 1991, ``p=1.1`` for
        Holmgren 1994).
    name : str, default='flow_direction_mfd'
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
        3-D array with dimensions ``('neighbor', <ydim>, <xdim>)``
        and a ``neighbor`` coordinate labelled
        ``['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']``.
        Values are flow fractions in ``[0, 1]`` that sum to 1.0 at
        each cell (0.0 everywhere for pits/flats, NaN at edges or
        where any neighbor is NaN).

    References
    ----------
    Qin, C., Zhu, A.X., Pei, T., Li, B., Zhou, C., and Yang, L.
    (2007). An adaptive approach to selecting a flow-partition
    exponent for a multiple-flow-direction algorithm. International
    Journal of Geographical Information Science, 21(4), 443-458.

    Quinn, P., Beven, K., Chevallier, P., and Planchon, O. (1991).
    The prediction of hillslope flow paths for distributed
    hydrological modelling using digital terrain models.
    Hydrological Processes, 5(1), 59-79.
    """
    _validate_raster(agg, func_name='flow_direction_mfd', name='agg')
    _validate_boundary(boundary)

    if p is not None:
        if p <= 0:
            raise ValueError("p must be a positive number, got %s" % p)
        p_fixed = float(p)
    else:
        p_fixed = -1.0  # sentinel for adaptive mode

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy,
    )
    out = mapper(agg)(agg.data, cellsize_x, cellsize_y, p_fixed, boundary)

    dims = ('neighbor',) + agg.dims
    coords = dict(agg.coords)
    coords['neighbor'] = NEIGHBOR_NAMES

    return xr.DataArray(out,
                        name=name,
                        coords=coords,
                        dims=dims,
                        attrs=agg.attrs)
