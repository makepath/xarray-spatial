'''Focal Related Utilities'''
from functools import partial

from math import isnan

from numba import prange
import numpy as np
from xarray import DataArray
import dask.array as da

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from numba import cuda
from xrspatial.utils import cuda_args
from xrspatial.utils import has_cuda
from xrspatial.utils import is_cupy_backed
from xrspatial.utils import ngjit
from xrspatial.convolution import convolve_2d, custom_kernel

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


def _mean_dask_numpy(data, excludes):
    _func = partial(_mean_numpy, excludes=excludes)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
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


def _mean_dask_cupy(data, excludes):
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)


def _mean(data, excludes):
    # numpy case
    if isinstance(data, np.ndarray):
        out = _mean_numpy(data.astype(np.float), excludes)

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        out = _mean_cupy(data.astype(cupy.float), excludes)

    # dask + cupy case
    elif has_cuda() and isinstance(data, da.Array) and \
            type(data._meta).__module__.split('.')[0] == 'cupy':
        out = _mean_dask_cupy(data.astype(cupy.float), excludes)

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

    Parameters:
    ----------
    agg : xarray.DataArray
        2D array of input values to be filtered.
    passes : int (default = 1)
        Number of times to run mean.
    name : str, optional (default = 'mean')
        output xr.DataArray.name property

    Returns:
    ----------
    data: xarray.DataArray
        2D array of filtered values.

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import focal

    Create Data Array
    >>> np.random.seed(0)
    >>> agg = xr.DataArray(np.random.rand(4,4), dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon

    Calculate Mean
    >>> focal.mean(agg)
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
           [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
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
def _apply_numpy(data, kernel, func):
    out = np.zeros_like(data)
    rows, cols = data.shape
    krows, kcols = kernel.shape
    hrows, hcols = int(krows / 2), int(kcols / 2)
    kernel_values = np.zeros_like(kernel, dtype=data.dtype)

    for y in prange(rows):
        for x in prange(cols):
            # kernel values are all nans at the beginning of each step
            kernel_values.fill(np.nan)
            for ky in range(y - hrows, y + hrows + 1):
                for kx in range(x - hcols, x + hcols + 1):
                    if ky >= 0 and ky < rows and kx >= 0 and kx < cols:
                        kyidx, kxidx = ky - (y - hrows), kx - (x - hcols)
                        if kernel[kyidx, kxidx] == 1:
                            kernel_values[kyidx, kxidx] = data[ky, kx]
            out[y, x] = func(kernel_values)
    return out


def _apply_dask_numpy(data, kernel, func):
    _func = partial(_apply_numpy, kernel=kernel, func=func)

    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    out = data.map_overlap(_func,
                           depth=(pad_h, pad_w),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


def _apply_cupy(data, kernel, func):

    out = cupy.zeros(data.shape, dtype=data.dtype)
    out[:] = cupy.nan

    rows, cols = data.shape
    krows, kcols = kernel.shape
    hrows, hcols = int(krows / 2), int(kcols / 2)
    kernel_values = cupy.zeros_like(kernel, dtype=data.dtype)

    for y in prange(rows):
        for x in prange(cols):
            # kernel values are all nans at the beginning of each step
            kernel_values.fill(np.nan)
            for ky in range(y - hrows, y + hrows + 1):
                for kx in range(x - hcols, x + hcols + 1):
                    if ky >= 0 and ky < rows and kx >= 0 and kx < cols:
                        kyidx, kxidx = ky - (y - hrows), kx - (x - hcols)
                        if kernel[kyidx, kxidx] == 1:
                            kernel_values[kyidx, kxidx] = data[ky, kx]
            out[y, x] = func(kernel_values)

    return out


def _apply_dask_cupy(data, kernel, func):
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)


def _apply(data, kernel, func):
    # numpy case
    if isinstance(data, np.ndarray):
        if not (issubclass(data.dtype.type, np.integer) or
                issubclass(data.dtype.type, np.floating)):
            raise ValueError("data type must be integer or float")

        out = _apply_numpy(data, kernel, func)

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        if not (issubclass(data.dtype.type, cupy.integer) or
                issubclass(data.dtype.type, cupy.floating)):
            raise ValueError("data type must be integer or float")
        out = _apply_cupy(data, cupy.asarray(kernel), func)

    # dask + cupy case
    elif has_cuda() and isinstance(data, da.Array) and \
            type(data._meta).__module__.split('.')[0] == 'cupy':
        out = _apply_dask_cupy(data, cupy.asarray(kernel), func)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _apply_dask_numpy(data, kernel, func)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(data)))

    return out


def apply(raster, kernel, func=calc_mean):
    """
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    # Validate the kernel
    kernel = custom_kernel(kernel)

    # apply kernel to raster values
    # if raster is a numpy or dask with numpy backed data array,
    # the function func must be a @ngjit
    out = _apply(raster.data.astype(float), kernel, func)

    result = DataArray(out,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result


@ngjit
def _calc_hotspots_numpy(z_array):
    out = np.zeros_like(z_array, dtype=np.int8)
    rows, cols = z_array.shape

    for y in prange(rows):
        for x in prange(cols):
            zscore = z_array[y, x]

            # find p value
            p_value = 1.0
            if abs(zscore) >= 2.33:
                p_value = 0.0099
            elif abs(zscore) >= 1.65:
                p_value = 0.0495
            elif abs(zscore) >= 1.29:
                p_value = 0.0985

            # confidence
            confidence = 0
            if abs(zscore) > 2.58 and p_value < 0.01:
                confidence = 99
            elif abs(zscore) > 1.96 and p_value < 0.05:
                confidence = 95
            elif abs(zscore) > 1.65 and p_value < 0.1:
                confidence = 90

            hot_cold = 0
            if zscore > 0:
                hot_cold = 1
            elif zscore < 0:
                hot_cold = -1

            out[y, x] = hot_cold * confidence
    return out


def _hotspots_numpy(raster, kernel):
    if not (issubclass(raster.data.dtype.type, np.integer) or
            issubclass(raster.data.dtype.type, np.floating)):
        raise ValueError("data type must be integer or float")

    # apply kernel to raster values
    mean_array = convolve_2d(raster.data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = np.nanmean(raster.data)
    global_std = np.nanstd(raster.data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )
    z_array = (mean_array - global_mean) / global_std

    out = _calc_hotspots_numpy(z_array)
    return out


def _hotspots_dask_numpy(raster, kernel):

    # apply kernel to raster values
    mean_array = convolve_2d(raster.data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = da.nanmean(raster.data)
    global_std = da.nanstd(raster.data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )
    z_array = (mean_array - global_mean) / global_std

    _func = partial(_calc_hotspots_numpy)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    out = z_array.map_overlap(_func,
                              depth=(pad_h, pad_w),
                              boundary=np.nan,
                              meta=np.array(()))
    return out


def _calc_hotspots_cupy(z_array):
    out = cupy.zeros_like(z_array, dtype=cupy.int8)
    rows, cols = z_array.shape

    for y in prange(rows):
        for x in prange(cols):
            zscore = z_array[y, x]

            # find p value
            p_value = 1.0
            if abs(zscore) >= 2.33:
                p_value = 0.0099
            elif abs(zscore) >= 1.65:
                p_value = 0.0495
            elif abs(zscore) >= 1.29:
                p_value = 0.0985

            # confidence
            confidence = 0
            if abs(zscore) > 2.58 and p_value < 0.01:
                confidence = 99
            elif abs(zscore) > 1.96 and p_value < 0.05:
                confidence = 95
            elif abs(zscore) > 1.65 and p_value < 0.1:
                confidence = 90

            hot_cold = 0
            if zscore > 0:
                hot_cold = 1
            elif zscore < 0:
                hot_cold = -1

            out[y, x] = hot_cold * confidence
    return out


def _hotspots_cupy(raster, kernel):
    if not (issubclass(raster.data.dtype.type, cupy.integer) or
            issubclass(raster.data.dtype.type, cupy.floating)):
        raise ValueError("data type must be integer or float")

    # apply kernel to raster values
    mean_array = convolve_2d(raster.data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = cupy.nanmean(raster.data)
    global_std = cupy.nanstd(raster.data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )
    z_array = (mean_array - global_mean) / global_std

    out = _calc_hotspots_cupy(z_array)
    return out


def hotspots(raster, kernel):
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

    Parameters:
    ----------
    raster: xarray.DataArray
        2D Input raster image with shape = (height, width).
    kernel: Numpy Array
        2D array where values of 1 indicate the kernel.

    Returns:
    ----------
    xarray.DataArray
        2D array of hotspots with values indicating confidence level.

    Examples:
    ----------
        Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import focal

    Create Data Array
    >>> agg = xr.DataArray(np.array([[0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 10, 10, 10, 0, 0],
    >>>                              [0, 0, 10, 10, 10, 0, 0],
    >>>                              [0, 0, 10, 10, 10, 0, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0]]),
    >>>                              dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat

        Create Kernel
    >>> kernel = focal.circle_kernel(1, 1, 1)

        Create Hotspot Data Array
    >>> focal.hotspots(agg, kernel, x = 'lon', y = 'lat')
    <xarray.DataArray (lat: 7, lon: 7)>
    array([[ 0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0, 95,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0]], dtype=int8)
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    # numpy case
    if isinstance(raster.data, np.ndarray):
        out = _hotspots_numpy(raster, kernel)

    # cupy case
    elif has_cuda() and isinstance(raster.data, cupy.ndarray):
        out = _hotspots_cupy(raster, kernel)

    # dask + cupy case
    elif has_cuda() and isinstance(raster.data, da.Array) and \
            is_cupy_backed(raster):
        raise NotImplementedError()

    # dask + numpy case
    elif isinstance(raster.data, da.Array):
        out = _hotspots_dask_numpy(raster, kernel)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(raster.data)))

    return DataArray(out,
                     coords=raster.coords,
                     dims=raster.dims,
                     attrs=raster.attrs)
