'''Focal Related Utilities'''
from functools import partial
import warnings

from math import isnan

from numba import prange
import numpy as np
from xarray import DataArray
import dask.array as da

from numba import cuda

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.utils import cuda_args
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.convolution import convolve_2d, custom_kernel

warnings.simplefilter('default')

# TODO: Make convolution more generic with numba first-class functions.


@ngjit
def _equal_numpy(x, y):
    if x == y or (np.isnan(x) and np.isnan(y)):
        return True
    return False


@ngjit
def _mean_numpy(data, excludes):
    # TODO: exclude nans, what if nans in the kernel?
    out = np.zeros_like(data)
    out[:, :] = np.nan
    rows, cols = data.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            exclude = False
            for ex in excludes:
                if _equal_numpy(data[y, x], ex):
                    exclude = True
                    break

            if not exclude:
                a, b, c, d, e, f, g, h, i = [data[y - 1, x - 1],
                                             data[y, x - 1],
                                             data[y + 1, x - 1],
                                             data[y - 1, x], data[y, x],
                                             data[y + 1, x],
                                             data[y - 1, x + 1],
                                             data[y, x + 1],
                                             data[y + 1, x + 1]]
                out[y, x] = (a + b + c + d + e + f + g + h + i) / 9
            else:
                out[y, x] = data[y, x]
    return out


@cuda.jit(device=True)
def _kernel_mean_gpu(data):
    return (data[-1, -1] + data[-1, 0] + data[-1, 1]
            + data[0, -1] + data[0, 0] + data[0, 1]
            + data[1, -1] + data[1, 0] + data[1, 1]) / 9


@cuda.jit
def _mean_gpu(data, excludes, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1

    for ex in excludes:
        if (data[i, j] == ex) or (isnan(data[i, j]) and isnan(ex)):
            out[i, j] = data[i, j]
            return

    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _kernel_mean_gpu(data[i-1:i+2, j-1:j+2])


def _mean_cupy(data, excludes):
    out = cupy.zeros_like(data)
    out[:, :] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _mean_gpu[griddim, blockdim](data, cupy.asarray(excludes), out)

    return out


def _mean_dask_numpy(data, excludes):
    _func = partial(_mean_numpy, excludes=excludes)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


def _mean(data, excludes):
    # numpy case
    if isinstance(data, np.ndarray):
        out = _mean_numpy(data.astype(np.float), excludes)

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        out = _mean_cupy(data, excludes)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _mean_dask_numpy(data.astype(np.float), excludes)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(data)))

    return out


def mean(agg, passes=1, excludes=[np.nan], name='mean'):
    """
    Returns Mean filtered array using a 3x3 window.
    Default behaviour to 'mean' is to pad the borders with nans

    Parameters
    ----------
    agg : DataArray
    passes : int
      number of times to run mean
    name : str
      output xr.DataArray.name property

    Returns
    -------
    data: DataArray
    """
    out = agg.data
    for i in range(passes):
        out = _mean(out, tuple(excludes))

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def calc_mean(array):
    return np.nanmean(array)


@ngjit
def calc_sum(array):
    return np.nansum(array)


@ngjit
def upper_bound_p_value(zscore):
    if abs(zscore) >= 2.33:
        return 0.0099
    if abs(zscore) >= 1.65:
        return 0.0495
    if abs(zscore) >= 1.29:
        return 0.0985
    return 1


@ngjit
def _hot_cold(zscore):
    if zscore > 0:
        return 1
    if zscore < 0:
        return -1
    return 0


@ngjit
def _confidence(zscore):
    p_value = upper_bound_p_value(zscore)
    if abs(zscore) > 2.58 and p_value < 0.01:
        return 99
    if abs(zscore) > 1.96 and p_value < 0.05:
        return 95
    if abs(zscore) > 1.65 and p_value < 0.1:
        return 90
    return 0


@ngjit
def _apply(data, kernel_array, func):
    out = np.zeros_like(data)
    rows, cols = data.shape
    krows, kcols = kernel_array.shape
    hrows, hcols = int(krows / 2), int(kcols / 2)
    kernel_values = np.zeros_like(kernel_array, dtype=data.dtype)

    for y in prange(rows):
        for x in prange(cols):
            # kernel values are all nans at the beginning of each step
            kernel_values.fill(np.nan)
            for ky in range(y - hrows, y + hrows + 1):
                for kx in range(x - hcols, x + hcols + 1):
                    if ky >= 0 and kx >= 0:
                        if ky >= 0 and ky < rows and kx >= 0 and kx < cols:
                            kyidx, kxidx = ky - (y - hrows), kx - (x - hcols)
                            if kernel_array[kyidx, kxidx] == 1:
                                kernel_values[kyidx, kxidx] = data[ky, kx]
            out[y, x] = func(kernel_values)
    return out


def apply(raster, kernel, x='x', y='y', func=calc_mean):
    """
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.floating)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates:"
                         "(%s, %s)".format(y, x))

    # Validate the kernel
    kernel = custom_kernel(kernel)

    # apply kernel to raster values
    out = _apply(raster.values.astype(float), kernel, func)

    result = DataArray(out,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result


@ngjit
def _hotspots(z_array):
    out = np.zeros_like(z_array, dtype=np.int8)
    rows, cols = z_array.shape
    for y in prange(rows):
        for x in prange(cols):
            out[y, x] = _hot_cold(z_array[y, x]) * _confidence(z_array[y, x])
    return out


def hotspots(raster, kernel, x='x', y='y'):
    """Identify statistically significant hot spots and cold spots in an input
    raster. To be a statistically significant hot spot, a feature will have a
    high value and be surrounded by other features with high values as well.
    Neighborhood of a feature defined by the input kernel, which currently
    support a shape of circle, annulus, or custom kernel.

    The result should be a raster with the following 7 values:
    90 for 90% confidence high value cluster
    95 for 95% confidence high value cluster
    99 for 99% confidence high value cluster
    -90 for 90% confidence low value cluster
    -95 for 95% confidence low value cluster
    -99 for 99% confidence low value cluster
    0 for no significance

    Parameters
    ----------
    raster: xarray.DataArray
        Input raster image with shape=(height, width)
    kernel: Kernel

    Returns
    -------
    hotspots: xarray.DataArray
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.floating)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates: (%s, %s)".format(y, x))

    # apply kernel to raster values
    mean_array = convolve_2d(raster.values, kernel / kernel.sum())

    # calculate z-scores
    global_mean = np.nanmean(raster.values)
    global_std = np.nanstd(raster.values)
    if global_std == 0:
        raise ZeroDivisionError("Standard deviation of the input raster values is 0.")
    z_array = (mean_array - global_mean) / global_std

    out = _hotspots(z_array)

    result = DataArray(out,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result
