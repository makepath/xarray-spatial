"""Inverse Distance Weighting (IDW) interpolation."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    ngjit,
)

from ._validation import extract_grid_coords, validate_points

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None


# ---------------------------------------------------------------------------
# CPU all-points kernel (numba JIT)
# ---------------------------------------------------------------------------

@ngjit
def _idw_cpu_allpoints(x_pts, y_pts, z_pts, n_pts,
                       x_grid, y_grid, power, fill_value):
    ny = y_grid.shape[0]
    nx = x_grid.shape[0]
    out = np.empty((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            gx = x_grid[j]
            gy = y_grid[i]
            w_sum = 0.0
            wz_sum = 0.0
            exact = False
            exact_val = 0.0

            for p in range(n_pts):
                dx = gx - x_pts[p]
                dy = gy - y_pts[p]
                d2 = dx * dx + dy * dy
                if d2 == 0.0:
                    exact = True
                    exact_val = z_pts[p]
                    break
                w = 1.0 / (d2 ** (power * 0.5))
                w_sum += w
                wz_sum += w * z_pts[p]

            if exact:
                out[i, j] = exact_val
            elif w_sum > 0.0:
                out[i, j] = wz_sum / w_sum
            else:
                out[i, j] = fill_value

    return out


# ---------------------------------------------------------------------------
# CPU k-nearest (scipy cKDTree)
# ---------------------------------------------------------------------------

def _idw_knearest_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                        power, k, fill_value):
    from scipy.spatial import cKDTree

    pts = np.column_stack([x_pts, y_pts])
    tree = cKDTree(pts)

    gx, gy = np.meshgrid(x_grid, y_grid)
    query_pts = np.column_stack([gx.ravel(), gy.ravel()])
    dists, indices = tree.query(query_pts, k=k)

    if k == 1:
        dists = dists[:, np.newaxis]
        indices = indices[:, np.newaxis]

    exact = dists == 0.0
    dists_safe = np.where(exact, 1.0, dists)
    weights = np.where(exact, 1.0, 1.0 / (dists_safe ** power))

    has_exact = np.any(exact, axis=1)
    weights[has_exact] = np.where(exact[has_exact], 1.0, 0.0)

    z_vals = z_pts[indices]
    wz = np.sum(weights * z_vals, axis=1)
    w_total = np.sum(weights, axis=1)
    result = np.where(w_total > 0, wz / w_total, fill_value)
    return result.reshape(len(y_grid), len(x_grid))


# ---------------------------------------------------------------------------
# Numpy backend
# ---------------------------------------------------------------------------

def _idw_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
               power, k, fill_value, template_data):
    if k is not None:
        return _idw_knearest_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                                   power, k, fill_value)
    return _idw_cpu_allpoints(x_pts, y_pts, z_pts, len(x_pts),
                              x_grid, y_grid, power, fill_value)


# ---------------------------------------------------------------------------
# CUDA kernel (all-points only)
# ---------------------------------------------------------------------------

@cuda.jit
def _idw_cuda_kernel(x_pts, y_pts, z_pts, n_pts,
                     x_grid, y_grid, power, fill_value, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        gx = x_grid[j]
        gy = y_grid[i]
        w_sum = 0.0
        wz_sum = 0.0
        exact = False
        exact_val = 0.0

        for p in range(n_pts):
            dx = gx - x_pts[p]
            dy = gy - y_pts[p]
            d2 = dx * dx + dy * dy
            if d2 == 0.0:
                exact = True
                exact_val = z_pts[p]
                break
            w = 1.0 / (d2 ** (power * 0.5))
            w_sum += w
            wz_sum += w * z_pts[p]

        if exact:
            out[i, j] = exact_val
        elif w_sum > 0.0:
            out[i, j] = wz_sum / w_sum
        else:
            out[i, j] = fill_value


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------

def _idw_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
              power, k, fill_value, template_data):
    if k is not None:
        raise NotImplementedError(
            "idw(): k-nearest mode is not supported on GPU. "
            "Use k=None for all-points IDW on GPU, or use a "
            "numpy/dask+numpy backend."
        )

    x_gpu = cupy.asarray(x_pts)
    y_gpu = cupy.asarray(y_pts)
    z_gpu = cupy.asarray(z_pts)
    xg_gpu = cupy.asarray(x_grid)
    yg_gpu = cupy.asarray(y_grid)

    ny, nx = len(y_grid), len(x_grid)
    out = cupy.full((ny, nx), fill_value, dtype=np.float64)

    griddim, blockdim = cuda_args((ny, nx))
    _idw_cuda_kernel[griddim, blockdim](
        x_gpu, y_gpu, z_gpu, len(x_pts),
        xg_gpu, yg_gpu, power, fill_value, out,
    )
    return out


# ---------------------------------------------------------------------------
# Dask + numpy backend
# ---------------------------------------------------------------------------

def _idw_dask_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                    power, k, fill_value, template_data):

    def _chunk(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        return _idw_numpy(x_pts, y_pts, z_pts, x_sl, y_sl,
                          power, k, fill_value, None)

    return da.map_blocks(_chunk, template_data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dask + cupy backend
# ---------------------------------------------------------------------------

def _idw_dask_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                   power, k, fill_value, template_data):
    if k is not None:
        raise NotImplementedError(
            "idw(): k-nearest mode is not supported on GPU."
        )

    def _chunk(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        return _idw_cupy(x_pts, y_pts, z_pts, x_sl, y_sl,
                         power, None, fill_value, None)

    return da.map_blocks(
        _chunk, template_data, dtype=np.float64,
        meta=cupy.array((), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def idw(x, y, z, template, power=2.0, k=None,
        fill_value=np.nan, name='idw'):
    """Inverse Distance Weighting interpolation.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates and values of scattered input points.
    template : xr.DataArray
        2-D DataArray whose grid defines the output raster.
    power : float, default 2.0
        Distance weighting exponent.
    k : int or None, default None
        Number of nearest neighbours.  ``None`` uses all points
        (numba JIT); an integer uses ``scipy.spatial.cKDTree``
        (CPU only).
    fill_value : float, default np.nan
        Value for pixels with zero total weight.
    name : str, default 'idw'
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
    """
    _validate_raster(template, func_name='idw', name='template')
    x_arr, y_arr, z_arr = validate_points(x, y, z, func_name='idw')
    _validate_scalar(power, func_name='idw', name='power',
                     min_val=0.0, min_exclusive=True)

    if k is not None:
        _validate_scalar(k, func_name='idw', name='k',
                         dtype=int, min_val=1)
        k = min(k, len(x_arr))

    x_grid, y_grid = extract_grid_coords(template, func_name='idw')

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_idw_numpy,
        cupy_func=_idw_cupy,
        dask_func=_idw_dask_numpy,
        dask_cupy_func=_idw_dask_cupy,
    )

    out = mapper(template)(
        x_arr, y_arr, z_arr, x_grid, y_grid,
        power, k, fill_value, template.data,
    )

    return xr.DataArray(
        out, name=name,
        coords=template.coords,
        dims=template.dims,
        attrs=template.attrs,
    )
