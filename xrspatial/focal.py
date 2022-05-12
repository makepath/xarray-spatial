import copy
from functools import partial
from math import isnan, sqrt

import dask.array as da
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
from numba import cuda, prange
from xarray import DataArray

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.convolution import convolve_2d, custom_kernel
from xrspatial.utils import ArrayTypeFunctionMapping, cuda_args, ngjit, not_implemented_func

# TODO: Make convolution more generic with numba first-class functions.


@ngjit
def _equal_numpy(x, y):
    if x == y or (np.isnan(x) and np.isnan(y)):
        return True
    return False


@ngjit
def _mean_numpy(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape

    for y in range(rows):
        for x in range(cols):

            exclude = False
            for ex in excludes:
                if _equal_numpy(data[y, x], ex):
                    exclude = True
                    break

            if not exclude:
                left = max(x-1, 0)
                right = min(x+2, cols)
                bottom = max(y-1, 0)
                top = min(y+2, rows)
                kernel_data = data[bottom:top, left:right]
                out[y, x] = np.nanmean(kernel_data)
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


@cuda.jit
def _mean_gpu(data, excludes, out):
    i, j = cuda.grid(2)

    for ex in excludes:
        if (data[i, j] == ex) or (isnan(data[i, j]) and isnan(ex)):
            out[i, j] = data[i, j]
            return

    rows, cols = out.shape
    if 0 <= i < rows and 0 <= j < cols:
        left = max(j - 1, 0)
        right = min(j + 2, cols)
        bottom = max(i - 1, 0)
        top = min(i + 2, rows)

        sum = 0
        num = 0
        for y in range(bottom, top):
            for x in range(left, right):
                if not isnan(data[y, x]):
                    sum += data[y, x]
                    num += 1
        if num > 0:
            out[i, j] = sum / num


def _mean_cupy(data, excludes):
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    _mean_gpu[griddim, blockdim](data, cupy.asarray(excludes), out)
    return out


def _mean(data, excludes):
    agg = xr.DataArray(data)
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_mean_numpy,
        cupy_func=_mean_cupy,
        dask_func=_mean_dask_numpy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='mean() does not support dask with cupy backed DataArray.'),  # noqa
    )
    out = mapper(agg)(agg.data, excludes)
    return out


def mean(agg, passes=1, excludes=[np.nan], name='mean'):
    """
    Returns Mean filtered array using a 3x3 window.
    Default behaviour to 'mean' is to exclude NaNs from calculations.

    Parameters
    ----------
    agg : xarray.DataArray
        2D array of input values to be filtered.
    passes : int, default=1
        Number of times to run mean.
    name : str, default='mean'
        Output xr.DataArray.name property.

    Returns
    -------
    mean_agg : xarray.DataArray of same type as `agg`
        2D aggregate array of filtered values.

    Examples
    --------
    Focal mean works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.focal import mean
        >>> data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 9., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
        >>> raster = xr.DataArray(data)
        >>> mean_agg = mean(raster)
        >>> print(mean_agg)
        <xarray.DataArray 'mean' (dim_0: 5, dim_1: 5)>
        array([[0., 0., 0., 0., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 0., 0., 0., 0.]])
        Dimensions without coordinates: dim_0, dim_1

    Focal mean works with Dask with NumPy backed xarray DataArray.
    Increase number of runs by setting a specific value for parameter `passes`
    .. sourcecode:: python

        >>> import dask.array as da
        >>> data_da = da.from_array(data, chunks=(3, 3))
        >>> raster_da = xr.DataArray(data_da, dims=['y', 'x'], name='raster_da')  # noqa
        >>> print(raster_da)
        <xarray.DataArray 'raster_da' (y: 5, x: 5)>
        dask.array<array, shape=(5, 5), dtype=int64, chunksize=(3, 3), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: y, x
        >>> mean_da = mean(raster_da, passes=2)
        >>> print(mean_da)
        <xarray.DataArray 'mean' (y: 5, x: 5)>
        dask.array<_trim, shape=(5, 5), dtype=float64, chunksize=(3, 3), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: y, x
        >>> print(mean_da.compute())
        <xarray.DataArray 'mean' (y: 5, x: 5)>
        array([[0.25      , 0.33333333, 0.5       , 0.33333333, 0.25      ],
               [0.33333333, 0.44444444, 0.66666667, 0.44444444, 0.33333333],
               [0.5       , 0.66666667, 1.        , 0.66666667, 0.5       ],
               [0.33333333, 0.44444444, 0.66666667, 0.44444444, 0.33333333],
               [0.25      , 0.33333333, 0.5       , 0.33333333, 0.25      ]])
        Dimensions without coordinates: y, x

    Focal mean works with CuPy backed xarray DataArray.
    In this example, we set `passes` to the number of elements of the array,
    we'll get a mean array where every element has the same value.
    .. sourcecode:: python

        >>> import cupy
        >>> raster_cupy = xr.DataArray(cupy.asarray(data), name='raster_cupy')
        >>> mean_cupy = mean(raster_cupy, passes=25)
        >>> print(type(mean_cupy.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(mean_cupy)
        <xarray.DataArray 'mean' (dim_0: 5, dim_1: 5)>
        array([[0.47928995, 0.47928995, 0.47928995, 0.47928995, 0.47928995],
               [0.47928995, 0.47928995, 0.47928995, 0.47928995, 0.47928995],
               [0.47928995, 0.47928995, 0.47928995, 0.47928995, 0.47928995],
               [0.47928995, 0.47928995, 0.47928995, 0.47928995, 0.47928995],
               [0.47928995, 0.47928995, 0.47928995, 0.47928995, 0.47928995]])
        Dimensions without coordinates: dim_0, dim_1
    """

    out = agg.data.astype(float)
    for i in range(passes):
        out = _mean(out, tuple(excludes))

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _calc_mean(array):
    return np.nanmean(array)


@ngjit
def _calc_sum(array):
    return np.nansum(array)


@ngjit
def _calc_min(array):
    return np.nanmin(array)


@ngjit
def _calc_max(array):
    return np.nanmax(array)


@ngjit
def _calc_std(array):
    return np.nanstd(array)


@ngjit
def _calc_range(array):
    value_min = _calc_min(array)
    value_max = _calc_max(array)
    return value_max - value_min


@ngjit
def _calc_var(array):
    return np.nanvar(array)


@ngjit
def _apply_numpy(data, kernel, func):
    data = data.astype(np.float32)

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
    data = data.astype(np.float32)
    _func = partial(_apply_numpy, kernel=kernel, func=func)

    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    out = data.map_overlap(_func,
                           depth=(pad_h, pad_w),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


def apply(raster, kernel, func=_calc_mean, name='focal_apply'):
    """
    Returns custom function applied array using a user-created window.

    Parameters
    ----------
    raster : xarray.DataArray
        2D array of input values to be filtered. Can be a NumPy backed,
        or Dask with NumPy backed DataArray.
    kernel : numpy.ndarray
        2D array where values of 1 indicate the kernel.
    func : callable, default=xrspatial.focal._calc_mean
        Function which takes an input array and returns an array.

    Returns
    -------
    agg : xarray.DataArray of same type as `raster`
        2D aggregate array of filtered values.

    Examples
    --------
    Focal apply works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.convolution import circle_kernel
        >>> from xrspatial.focal import apply
        >>> data = np.arange(20, dtype=np.float64).reshape(4, 5)
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> print(raster)
        <xarray.DataArray 'raster' (y: 4, x: 5)>
        array([[ 0.,  1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.,  9.],
               [10., 11., 12., 13., 14.],
               [15., 16., 17., 18., 19.]])
        Dimensions without coordinates: y, x
        >>> kernel = circle_kernel(2, 2, 3)
        >>> kernel
        array([[0., 1., 0.],
               [1., 1., 1.],
               [0., 1., 0.]])
        >>> # apply kernel mean by default
        >>> apply_mean_agg = apply(raster, kernel)
        >>> apply_mean_agg
        <xarray.DataArray 'focal_apply' (y: 4, x: 5)>
        array([[ 2.        ,  2.25   ,  3.25      ,  4.25      ,  5.33333333],
               [ 5.25      ,  6.     ,  7.        ,  8.        ,  8.75      ],
               [10.25      , 11.     , 12.        , 13.        , 13.75      ],
               [13.66666667, 14.75   , 15.75      , 16.75      , 17.        ]])
        Dimensions without coordinates: y, x

    Focal apply works with Dask with NumPy backed xarray DataArray.
    Note that if input raster is a numpy or dask with numpy backed data array,
    the applied function must be decorated with ``numba.jit``
    xrspatial already provides ``ngjit`` decorator, where:
    ``ngjit = numba.jit(nopython=True, nogil=True)``

    .. sourcecode:: python

    >>> from xrspatial.utils import ngjit
    >>> from xrspatial.convolution import custom_kernel
    >>> kernel = custom_kernel(np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]))
    >>> weight = np.array([
        [0, 0.5, 0],
        [0, 1, 0.5],
        [0, 0.5, 0],
    ])
    >>> @ngjit
    >>> def func(kernel_data):
    ...     weight = np.array([
    ...         [0, 0.5, 0],
    ...         [0, 1, 0.5],
    ...         [0, 0.5, 0],
    ...     ])
    ...     return np.nansum(kernel_data * weight)

    >>> import dask.array as da
    >>> data_da = da.from_array(np.ones((6, 4), dtype=np.float64), chunks=(3, 2))
    >>> raster_da = xr.DataArray(data_da, dims=['y', 'x'], name='raster_da')
    >>> print(raster_da)
    <xarray.DataArray 'raster_da' (y: 6, x: 4)>
    dask.array<array, shape=(6, 4), dtype=float64, chunksize=(3, 2), chunktype=numpy.ndarray>  # noqa
    Dimensions without coordinates: y, x
    >>> apply_func_agg = apply(raster_da, kernel, func)
    >>> print(apply_func_agg)
    <xarray.DataArray 'focal_apply' (y: 6, x: 4)>
    dask.array<_trim, shape=(6, 4), dtype=float64, chunksize=(3, 2), chunktype=numpy.ndarray>  # noqa
    Dimensions without coordinates: y, x
    >>> print(apply_func_agg.compute())
    <xarray.DataArray 'focal_apply' (y: 6, x: 4)>
    array([[2. , 2. , 2. , 1.5],
           [2.5, 2.5, 2.5, 2. ],
           [2.5, 2.5, 2.5, 2. ],
           [2.5, 2.5, 2.5, 2. ],
           [2.5, 2.5, 2.5, 2. ],
           [2. , 2. , 2. , 1.5]])
    Dimensions without coordinates: y, x
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
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_apply_numpy,
        cupy_func=lambda *args: not_implemented_func(
            *args, messages='apply() does not support cupy backed DataArray.'),
        dask_func=_apply_dask_numpy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='apply() does not support dask with cupy backed DataArray.'),
    )
    out = mapper(raster)(raster.data, kernel, func)
    result = DataArray(out,
                       name=name,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)
    return result


@cuda.jit
def _focal_min_cuda(data, kernel, out):
    i, j = cuda.grid(2)

    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape

    if i < delta_rows or i >= data_rows - delta_rows or \
            j < delta_cols or j >= data_cols - delta_cols:
        return

    s = data[i, j]
    for k in range(kernel.shape[0]):
        for h in range(kernel.shape[1]):
            i_k = i + k - delta_rows
            j_h = j + h - delta_cols
            if (i_k >= 0) and (i_k < data_rows) and (j_h >= 0) and (j_h < data_cols):
                if (kernel[k, h] != 0) and s > data[i_k, j_h]:
                    s = data[i_k, j_h]
    out[i, j] = s


@cuda.jit
def _focal_max_cuda(data, kernel, out):
    i, j = cuda.grid(2)

    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape

    if i < delta_rows or i >= data_rows - delta_rows or \
            j < delta_cols or j >= data_cols - delta_cols:
        return

    s = data[i, j]
    for k in range(kernel.shape[0]):
        for h in range(kernel.shape[1]):
            i_k = i + k - delta_rows
            j_h = j + h - delta_cols
            if (i_k >= 0) and (i_k < data_rows) and (j_h >= 0) and (j_h < data_cols):
                if (kernel[k, h] != 0) and s < data[i_k, j_h]:
                    s = data[i_k, j_h]
    out[i, j] = s


def _focal_range_cupy(data, kernel):
    focal_min = _focal_stats_func_cupy(data, kernel, _focal_min_cuda)
    focal_max = _focal_stats_func_cupy(data, kernel, _focal_max_cuda)
    out = focal_max - focal_min
    return out


@cuda.jit
def _focal_std_cuda(data, kernel, out):
    i, j = cuda.grid(2)

    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape

    if i < delta_rows or i >= data_rows - delta_rows or \
            j < delta_cols or j >= data_cols - delta_cols:
        return

    sum_squares = 0
    sum = 0
    count = 0
    for k in range(kernel.shape[0]):
        for h in range(kernel.shape[1]):
            i_k = i + k - delta_rows
            j_h = j + h - delta_cols
            if (i_k >= 0) and (i_k < data_rows) and (j_h >= 0) and (j_h < data_cols):
                sum_squares += (kernel[k, h]*data[i_k, j_h])**2
                sum += kernel[k, h]*data[i_k, j_h]
                count += kernel[k, h]
    squared_sum = sum**2
    out[i, j] = sqrt((sum_squares - squared_sum/count) / count)


@cuda.jit
def _focal_var_cuda(data, kernel, out):
    i, j = cuda.grid(2)

    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape

    if i < delta_rows or i >= data_rows - delta_rows or \
            j < delta_cols or j >= data_cols - delta_cols:
        return

    sum_squares = 0
    sum = 0
    count = 0
    for k in range(kernel.shape[0]):
        for h in range(kernel.shape[1]):
            i_k = i + k - delta_rows
            j_h = j + h - delta_cols
            if (i_k >= 0) and (i_k < data_rows) and (j_h >= 0) and (j_h < data_cols):
                sum_squares += (kernel[k, h]*data[i_k, j_h])**2
                sum += kernel[k, h]*data[i_k, j_h]
                count += kernel[k, h]
    squared_sum = sum**2
    out[i, j] = (sum_squares - squared_sum/count) / count


def _focal_mean_cupy(data, kernel):
    out = convolve_2d(data, kernel / kernel.sum())
    return out


def _focal_sum_cupy(data, kernel):
    out = convolve_2d(data, kernel)
    return out


def _focal_stats_func_cupy(data, kernel, func=_focal_max_cuda):
    out = cupy.empty(data.shape, dtype='f4')
    out[:, :] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    func[griddim, blockdim](data, kernel, cupy.asarray(out))
    return out


def _focal_stats_cupy(agg, kernel, stats_funcs):
    _stats_cupy_mapper = dict(
        mean=_focal_mean_cupy,
        sum=_focal_sum_cupy,
        range=_focal_range_cupy,
        max=lambda *args: _focal_stats_func_cupy(*args, func=_focal_max_cuda),
        min=lambda *args: _focal_stats_func_cupy(*args, func=_focal_min_cuda),
        std=lambda *args: _focal_stats_func_cupy(*args, func=_focal_std_cuda),
        var=lambda *args: _focal_stats_func_cupy(*args, func=_focal_var_cuda),
    )
    stats_aggs = []
    for stats in stats_funcs:
        data = agg.data.astype(cupy.float32)
        stats_data = _stats_cupy_mapper[stats](data, kernel)
        stats_agg = xr.DataArray(
            stats_data,
            dims=agg.dims,
            coords=agg.coords,
            attrs=agg.attrs
          )
        stats_aggs.append(stats_agg)
    stats = xr.concat(stats_aggs, pd.Index(stats_funcs, name='stats'))
    return stats


def _focal_stats_cpu(agg, kernel, stats_funcs):
    _function_mapping = {
        'mean': _calc_mean,
        'max': _calc_max,
        'min': _calc_min,
        'range': _calc_range,
        'std': _calc_std,
        'var': _calc_var,
        'sum': _calc_sum
    }
    stats_aggs = []
    for stats in stats_funcs:
        stats_agg = apply(agg, kernel, func=_function_mapping[stats])
        stats_aggs.append(stats_agg)
    stats = xr.concat(stats_aggs, pd.Index(stats_funcs, name='stats'))
    return stats


def focal_stats(agg,
                kernel,
                stats_funcs=[
                    'mean', 'max', 'min', 'range', 'std', 'var', 'sum'
                ]):
    """
    Calculates statistics of the values within a specified focal neighborhood
    for each pixel in an input raster. The statistics types are Mean, Maximum,
    Minimum, Range, Standard deviation, Variation and Sum.

    Parameters
    ----------
    agg : xarray.DataArray
        2D array of input values to be analysed. Can be a NumPy backed,
        Cupy backed, or Dask with NumPy backed DataArray.
    kernel : numpy.array
        2D array where values of 1 indicate the kernel.
    stats_funcs: list of string
        List of statistics types to be calculated.
        Default set to ['mean', 'max', 'min', 'range', 'std', 'var', 'sum'].

    Returns
    -------
    stats_agg : xarray.DataArray of same type as `agg`
        3D array with dimensions of `(stat, y, x)` and with values
        indicating the focal stats.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.convolution import circle_kernel
        >>> kernel = circle_kernel(1, 1, 1)
        >>> kernel
        array([[0., 1., 0.],
               [1., 1., 1.],
               [0., 1., 0.]])
        >>> data = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 2, 2, 1, 1],
            [2, 2, 1, 1, 2, 2],
            [3, 3, 0, 0, 3, 3],
        ])
        >>> from xrspatial.focal import focal_stats
        >>> focal_stats(xr.DataArray(data), kernel, stats_funcs=['min', 'sum'])
        <xarray.DataArray 'focal_apply' (stats: 2, dim_0: 4, dim_1: 6)>
        array([[[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 1., 1.],
                [2., 0., 0., 0., 0., 2.]],
               [[1., 1., 2., 2., 1., 1.],
                [4., 6., 6., 6., 6., 4.],
                [8., 9., 6., 6., 9., 8.],
                [8., 8., 4., 4., 8., 8.]]])
        Coordinates:
          * stats    (stats) object 'min' 'sum'
        Dimensions without coordinates: dim_0, dim_1
    """
    # validate raster
    if not isinstance(agg, DataArray):
        raise TypeError("`agg` must be instance of DataArray")

    if agg.ndim != 2:
        raise ValueError("`agg` must be 2D")

    # Validate the kernel
    kernel = custom_kernel(kernel)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_focal_stats_cpu,
        cupy_func=_focal_stats_cupy,
        dask_func=_focal_stats_cpu,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='focal_stats() does not support dask with cupy backed DataArray.'),
    )
    result = mapper(agg)(agg, kernel, stats_funcs)
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

    data = raster.data.astype(np.float32)
    # apply kernel to raster values
    mean_array = convolve_2d(data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = np.nanmean(data)
    global_std = np.nanstd(data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )
    z_array = (mean_array - global_mean) / global_std

    out = _calc_hotspots_numpy(z_array)
    return out


def _hotspots_dask_numpy(raster, kernel):
    data = raster.data.astype(np.float32)

    # apply kernel to raster values
    mean_array = convolve_2d(data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = da.nanmean(data)
    global_std = da.nanstd(data)

    # commented out to avoid early compute to check if global_std is zero
    # if global_std == 0:
    #     raise ZeroDivisionError(
    #         "Standard deviation of the input raster values is 0."
    #     )

    z_array = (mean_array - global_mean) / global_std

    _func = partial(_calc_hotspots_numpy)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    out = z_array.map_overlap(_func,
                              depth=(pad_h, pad_w),
                              boundary=np.nan,
                              meta=np.array(()))
    return out


@nb.cuda.jit(device=True)
def _gpu_hotspots(data):
    zscore = data[0, 0]

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

    return hot_cold * confidence


@nb.cuda.jit
def _run_gpu_hotspots(data, out):
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]:
        out[i, j] = _gpu_hotspots(data[i:i + 1, j:j + 1])


def _hotspots_cupy(raster, kernel):
    if not (issubclass(raster.data.dtype.type, cupy.integer) or
            issubclass(raster.data.dtype.type, cupy.floating)):
        raise ValueError("data type must be integer or float")

    data = raster.data.astype(cupy.float32)

    # apply kernel to raster values
    mean_array = convolve_2d(data, kernel / kernel.sum())

    # calculate z-scores
    global_mean = cupy.nanmean(data)
    global_std = cupy.nanstd(data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )
    z_array = (mean_array - global_mean) / global_std

    out = cupy.zeros_like(z_array, dtype=cupy.int8)
    griddim, blockdim = cuda_args(z_array.shape)
    _run_gpu_hotspots[griddim, blockdim](z_array, out)
    return out


def hotspots(raster, kernel):
    """
    Identify statistically significant hot spots and cold spots in an
    input raster. To be a statistically significant hot spot, a feature
    will have a high value and be surrounded by other features with
    high values as well.
    Neighborhood of a feature defined by the input kernel, which
    currently support a shape of circle, annulus, or custom kernel.

    The result should be a raster with the following 7 values:
        - 90 for 90% confidence high value cluster
        - 95 for 95% confidence high value cluster
        - 99 for 99% confidence high value cluster
        - 90 for 90% confidence low value cluster
        - 95 for 95% confidence low value cluster
        - 99 for 99% confidence low value cluster
        - 0 for no significance

    Parameters
    ----------
    raster : xarray.DataArray
        2D Input raster image with `raster.shape` = (height, width).
        Can be a NumPy backed, CuPy backed, or Dask with NumPy backed DataArray
    kernel : Numpy Array
        2D array where values of 1 indicate the kernel.

    Returns
    -------
    hotspots_agg : xarray.DataArray of same type as `raster`
        2D array of hotspots with values indicating confidence level.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.convolution import custom_kernel
        >>> kernel = custom_kernel(np.array([[1, 1, 0]]))
        >>> data = np.array([
        ...    [0, 1000, 1000, 0, 0, 0],
        ...    [0, 0, 0, -1000, -1000, 0],
        ...    [0, -900, -900, 0, 0, 0],
        ...    [0, 100, 1000, 0, 0, 0]])
        >>> from xrspatial.focal import hotspots
        >>> hotspots(xr.DataArray(data), kernel)
        array([[  0,   0,  95,   0,   0,   0],
               [  0,   0,   0,   0, -90,   0],
               [  0,   0, -90,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0]], dtype=int8)
        Dimensions without coordinates: dim_0, dim_1
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_hotspots_numpy,
        cupy_func=_hotspots_cupy,
        dask_func=_hotspots_dask_numpy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='hotspots() does not support dask with cupy backed DataArray.'),  # noqa
    )
    out = mapper(raster)(raster, kernel)

    attrs = copy.deepcopy(raster.attrs)
    attrs['unit'] = '%'

    return DataArray(out,
                     coords=raster.coords,
                     dims=raster.dims,
                     attrs=attrs)
