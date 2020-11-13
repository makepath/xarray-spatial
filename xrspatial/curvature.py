import numpy as np
import xarray as xr
from numba import stencil, vectorize

from numba import cuda

from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args


@stencil
def kernel_D(arr):
    return (arr[0, -1] + arr[0, 1]) / 2 - arr[0, 0]


@stencil
def kernel_E(arr):
    return (arr[-1, 0] + arr[1, 0]) / 2 - arr[0, 0]


@vectorize(["float64(float64, float64)"], nopython=True, target="parallel")
def _horn_curvature(matrix_D, matrix_E):
    curv = -2 * (matrix_D + matrix_E) * 100
    return curv


@cuda.jit(device=True)
def _gpu_curvature(arr):
    d = (arr[1, 0] + arr[1, 2]) / 2 - arr[1, 1]
    e = (arr[0, 1] + arr[2, 1]) / 2 - arr[1, 1]
    curv = -2 * (d + e) * 100
    return curv


@cuda.jit
def _horn_curvature_cuda(arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _gpu_curvature(arr[i - di:i + di + 1, j - dj:j + dj + 1])


def curvature(agg, name='curvature', use_cuda=True, use_cupy=True):
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

    if not isinstance(agg, xr.DataArray):
        raise TypeError("`agg` must be instance of xr.DataArray")

    if agg.ndim != 2:
        raise ValueError("`agg` must be 2D")

    if not (issubclass(agg.values.dtype.type, np.integer) or
            issubclass(agg.values.dtype.type, np.floating)):
        raise ValueError(
            "`agg` must be an array of integers or float")

    agg_values = agg.values

    cellsize_x = 1
    cellsize_y = 1

    # calculate cell size from input `agg`
    for dim in agg.dims:
        if (dim.lower().count('x')) > 0 or (dim.lower().count('lon')) > 0:
            # dimension of x-coordinates
            if len(agg[dim]) > 1:
                cellsize_x = agg[dim].values[1] - agg[dim].values[0]
        elif (dim.lower().count('y')) > 0 or (dim.lower().count('lat')) > 0:
            # dimension of y-coordinates
            if len(agg[dim]) > 1:
                cellsize_y = agg[dim].values[1] - agg[dim].values[0]

    cellsize = (cellsize_x + cellsize_y) / 2

    if has_cuda() and use_cuda:
        # TODO: add padding
        # padding is not desired, set pads to 0
        pad_width = 0

        curv_data = np.pad(agg.data, pad_width=pad_width, mode="reflect")

        griddim, blockdim = cuda_args(curv_data.shape)
        curv_agg = np.empty(curv_data.shape, dtype='f4')
        curv_agg[:] = np.nan

        if use_cupy:
            import cupy
            curv_agg = cupy.asarray(curv_agg)

        _horn_curvature_cuda[griddim, blockdim](curv_data, curv_agg)

    else:
        matrix_D = kernel_D(agg_values)
        matrix_E = kernel_E(agg_values)

        curv_agg = _horn_curvature(matrix_D, matrix_E)
        # TODO: handle border edge effect
        # currently, set borders to np.nan
        curv_agg[0, :] = np.nan
        curv_agg[-1, :] = np.nan
        curv_agg[:, 0] = np.nan
        curv_agg[:, -1] = np.nan

    curv_agg = curv_agg / (cellsize * cellsize)

    result = xr.DataArray(curv_agg,
                          name=name,
                          coords=agg.coords,
                          dims=agg.dims,
                          attrs=agg.attrs)

    return result
