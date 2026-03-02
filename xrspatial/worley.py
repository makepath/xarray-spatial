from __future__ import annotations

import math
from functools import partial

import numpy as np
import xarray as xr

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

import numba as nb
from numba import cuda, jit

from xrspatial.perlin import _make_perm_table
from xrspatial.utils import (ArrayTypeFunctionMapping, _validate_raster,
                             cuda_args, not_implemented_func)


@jit(nopython=True, nogil=True)
def _hash2d(p, ix, iy, component):
    """Hash 2D cell coordinates to a pseudo-random float in [0,1).

    component: 0 for x offset, 1 for y offset of the feature point.
    """
    h = p[(p[ix & 255] + iy) & 255]
    # mix in component to get independent x/y values
    h = p[(h + component) & 255]
    return h / 255.0


@jit(nopython=True, nogil=True)
def _worley_cpu(p, x, y, out):
    """Compute Worley noise for 2D coordinate arrays.

    For each pixel, searches the 3x3 cell neighborhood for the nearest
    feature point and returns the distance.
    """
    height, width = out.shape
    for i in range(height):
        for j in range(width):
            px = x[i, j]
            py = y[i, j]
            cell_x = int(math.floor(px))
            cell_y = int(math.floor(py))
            min_dist = 1e10

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cx = cell_x + dx
                    cy = cell_y + dy
                    # feature point position within cell
                    fx = cx + _hash2d(p, cx, cy, 0)
                    fy = cy + _hash2d(p, cx, cy, 1)
                    ddx = px - fx
                    ddy = py - fy
                    dist = ddx * ddx + ddy * ddy
                    if dist < min_dist:
                        min_dist = dist

            out[i, j] = min_dist ** 0.5


def _worley_numpy(data, freq, seed):
    p = _make_perm_table(seed)
    height, width = data.shape
    linx = np.linspace(0, freq, width, endpoint=False, dtype=np.float32)
    liny = np.linspace(0, freq, height, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(linx, liny)
    out = np.empty_like(data)
    _worley_cpu(p, x, y, out)
    return out


def _worley_numpy_xy(p, x, y):
    """Worley noise from pre-built coordinate arrays."""
    out = np.empty(x.shape, dtype=np.float32)
    _worley_cpu(p, x, y, out)
    return out


@cuda.jit(device=True)
def _hash2d_gpu(sp, ix, iy, component):
    h = sp[(sp[ix & 255] + iy) & 255]
    h = sp[(h + component) & 255]
    return h / 255.0


@cuda.jit(fastmath=True, opt=True)
def _worley_gpu(p, x0, x1, y0, y1, out):
    sp = cuda.shared.array(512, nb.int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx
    block_size = bw * bh
    for k in range(tid, 512, block_size):
        sp[k] = p[k]
    cuda.syncthreads()

    i, j = nb.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        px = x0 + j * (x1 - x0) / out.shape[1]
        py = y0 + i * (y1 - y0) / out.shape[0]
        cell_x = int(math.floor(px))
        cell_y = int(math.floor(py))
        min_dist = 1e10

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx = cell_x + dx
                cy = cell_y + dy
                fx = cx + _hash2d_gpu(sp, cx, cy, 0)
                fy = cy + _hash2d_gpu(sp, cx, cy, 1)
                ddx = px - fx
                ddy = py - fy
                dist = ddx * ddx + ddy * ddy
                if dist < min_dist:
                    min_dist = dist

        out[i, j] = min_dist ** 0.5


@cuda.jit(fastmath=True, opt=True)
def _worley_gpu_xy(p, x_arr, y_arr, out):
    """Like _worley_gpu but takes 2D coordinate arrays."""
    sp = cuda.shared.array(512, nb.int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx
    block_size = bw * bh
    for k in range(tid, 512, block_size):
        sp[k] = p[k]
    cuda.syncthreads()

    i, j = nb.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        px = x_arr[i, j]
        py = y_arr[i, j]
        cell_x = int(math.floor(px))
        cell_y = int(math.floor(py))
        min_dist = 1e10

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx = cell_x + dx
                cy = cell_y + dy
                fx = cx + _hash2d_gpu(sp, cx, cy, 0)
                fy = cy + _hash2d_gpu(sp, cx, cy, 1)
                ddx = px - fx
                ddy = py - fy
                dist = ddx * ddx + ddy * ddy
                if dist < min_dist:
                    min_dist = dist

        out[i, j] = min_dist ** 0.5


def _worley_cupy(data, freq, seed):
    p = cupy.asarray(_make_perm_table(seed))
    out = cupy.empty(data.shape, dtype=cupy.float32)
    griddim, blockdim = cuda_args(data.shape)
    _worley_gpu[griddim, blockdim](p, 0, freq, 0, freq, out)
    return out


def _worley_dask_numpy(data, freq, seed):
    p = _make_perm_table(seed)
    height, width = data.shape
    linx = da.linspace(0, freq, width, endpoint=False, dtype=np.float32,
                       chunks=data.chunks[1][0])
    liny = da.linspace(0, freq, height, endpoint=False, dtype=np.float32,
                       chunks=data.chunks[0][0])
    x, y = da.meshgrid(linx, liny)
    _func = partial(_worley_numpy_xy, p)
    out = da.map_blocks(_func, x, y, meta=np.array((), dtype=np.float32))
    return out


def _worley_dask_cupy(data, freq, seed):
    p = cupy.asarray(_make_perm_table(seed))
    height, width = data.shape

    def _chunk_worley(block, block_info=None):
        info = block_info[0]
        y_start, y_end = info['array-location'][0]
        x_start, x_end = info['array-location'][1]
        x0 = freq * x_start / width
        x1 = freq * x_end / width
        y0 = freq * y_start / height
        y1 = freq * y_end / height
        out = cupy.empty(block.shape, dtype=cupy.float32)
        griddim, blockdim = cuda_args(block.shape)
        _worley_gpu[griddim, blockdim](p, x0, x1, y0, y1, out)
        return out

    out = da.map_blocks(_chunk_worley, data, dtype=cupy.float32,
                        meta=cupy.array((), dtype=cupy.float32))
    return out


def worley(agg, freq=4, seed=5, name='worley'):
    """Generate Worley (cellular) noise.

    Parameters
    ----------
    agg : xr.DataArray
        2D template array.
    freq : float
        Frequency (number of cells per unit).
    seed : int
        Seed for the permutation table.
    name : str
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        2D array of Worley noise values (distance to nearest feature point).
    """
    _validate_raster(agg, func_name='worley', name='agg')

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_worley_numpy,
        cupy_func=_worley_cupy,
        dask_func=_worley_dask_numpy,
        dask_cupy_func=_worley_dask_cupy,
    )
    out = mapper(agg)(agg.data, freq, seed)
    return xr.DataArray(out, dims=agg.dims, attrs=agg.attrs, name=name)
