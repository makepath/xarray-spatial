# std lib
from functools import partial
from math import atan
from typing import Union

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da

import numba as nb
from numba import cuda, stencil, vectorize

import numpy as np
import xarray as xr

# local modules
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.utils import is_cupy_backed


@stencil
def kernel_D(arr):
    return (arr[0, -1] + arr[0, 1]) / 2 - arr[0, 0]


@stencil
def kernel_E(arr):
    return (arr[-1, 0] + arr[1, 0]) / 2 - arr[0, 0]


@vectorize(["float64(float64, float64)"], nopython=True, target="parallel")
def _cpu(matrix_D, matrix_E):
    curv = -2 * (matrix_D + matrix_E) * 100
    return curv


def _run_numpy(data: np.ndarray,
               cellsize: Union[int, float]) -> np.ndarray:
    matrix_D = kernel_D(data)
    matrix_E = kernel_E(data)

    out = _cpu(matrix_D, matrix_E)

    # TODO: handle border edge effect
    # currently, set borders to np.nan
    out[0, :] = np.nan
    out[-1, :] = np.nan
    out[:, 0] = np.nan
    out[:, -1] = np.nan
    out = out / (cellsize * cellsize)

    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize):
    d = (arr[1, 0] + arr[1, 2]) / 2 - arr[1, 1]
    e = (arr[0, 1] + arr[2, 1]) / 2 - arr[1, 1]
    curv = -2 * (d + e) * 100 / (cellsize[0] * cellsize[0])
    return curv


@cuda.jit
def _run_gpu(arr, cellsize, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1], cellsize)


def _run_cupy(data: cupy.ndarray,
              cellsize: Union[int, float]) -> cupy.ndarray:

    cellsize_arr = cupy.array([float(cellsize)], dtype='f4')

    # TODO: add padding
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data, cellsize_arr, out)

    return out


def curvature(agg, name='curvature'):
    """Compute the curvature (second derivatives) of a agg surface.

    Parameters
    ----------
    agg: xarray.xr.DataArray
        2D input agg image with shape=(height, width)

    Returns
    -------
    curvature: xarray.xr.DataArray
        Curvature image with shape=(height, width)
    """

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    cellsize = (cellsize_x + cellsize_y) / 2

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, cellsize)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy(agg.data, cellsize)

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
