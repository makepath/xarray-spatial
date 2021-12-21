import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from functools import partial
from math import isnan
from numba import prange, cuda
from xarray import DataArray

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.utils import cuda_args
from xrspatial.utils import ngjit
from xrspatial.utils import not_implemented_func
from xrspatial.utils import ArrayTypeFunctionMapping

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
    out[:] = np.nan
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
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    _mean_gpu[griddim, blockdim](data, cupy.asarray(excludes), out)
    return out


def _mean(data, excludes):
    agg = xr.DataArray(data.astype(float))
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_mean_numpy,
        cupy_func=_mean_cupy,
        dask_func=_mean_dask_numpy,
        dask_cupy_func=not_implemented_func
    )
    out = mapper(agg)(agg.data, excludes)
    return out


def mean(agg, passes=1, excludes=[np.nan], name='mean'):
    """
    Returns Mean filtered array using a 3x3 window.
    Default behaviour to 'mean' is to pad the borders with nans

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
        >>> data = np.zeros((5, 5), dtype=np.float64)
        >>> data[2, 2] = 9
        >>> raster = xr.DataArray(data)
        >>> mean_agg = mean(raster, passes=1)
        >>> print(mean_agg)
        <xarray.DataArray 'mean' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan,  1.,  1.,  1., nan],
               [nan,  1.,  1.,  1., nan],
               [nan,  1.,  1.,  1., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: dim_0, dim_1

    Focal mean works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python
        >>> import dask.array as da
        >>> data_da = np.arange(25).reshape(5, 5)
        >>> print(data_da)
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        >>> data_da = da.from_array(data_da, chunks=(3, 3))
        >>> raster_da = xr.DataArray(data_da, dims=['y', 'x'], name='raster_da')  # noqa
        >>> print(raster_da)
        <xarray.DataArray 'raster_da' (y: 5, x: 5)>
        dask.array<array, shape=(5, 5), dtype=int64, chunksize=(3, 3), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: y, x
        >>> mean_da = mean(raster_da)
        >>> print(mean_da)
        <xarray.DataArray 'mean' (y: 5, x: 5)>
        dask.array<_trim, shape=(5, 5), dtype=float64, chunksize=(3, 3), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: y, x
        >>> print(mean_da.compute())
        <xarray.DataArray 'mean' (y: 5, x: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan,  6.,  7.,  8., nan],
               [nan, 11., 12., 13., nan],
               [nan, 16., 17., 18., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: y, x

    Focal mean works with CuPy backed xarray DataArray.
    .. sourcecode:: python
        >>> import cupy
        >>> data_cupy = cupy.asarray([
            [0, 1, 1, 1, 1, 2],
            [0, 0, 1, 1, 2, 2],
            [0, -1, 0, 2, 2, 2],
            [-2, -2, -1, 0, 1, 1],
        ])
        >>> raster_cupy = xr.DataArray(data_cupy, dims=['y', 'x'])
        >>> mean_cupy = mean(raster_cupy)
        >>> print(type(mean_cupy.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(mean_cupy)
        <xarray.DataArray 'mean' (y: 4, x: 6)>
        array([[nan,         nan,         nan,         nan,         nan, nan],
               [nan,  0.22222222,  0.6666667 ,  1.2222222 ,  1.6666666 , nan],
               [nan, -0.5555556 ,  0.        ,  0.8888889 ,  1.4444444 , nan],
               [nan,         nan,         nan,         nan,         nan, nan]], dtype=float32)
        Dimensions without coordinates: y, x
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
    kernel = cupy.asarray(kernel)

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


def apply(raster, kernel, func=_calc_mean, name='focal_apply'):
    """
    Returns custom function applied array using a user-created window.

    Parameters
    ----------
    raster : xarray.DataArray
        2D array of input values to be filtered.
    kernel : numpy.array
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
    >>>     weight = np.array([
                [0, 0.5, 0],
                [0, 1, 0.5],
                [0, 0.5, 0],
            ])
    >>>    return np.nansum(kernel_data * weight)

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
        cupy_func=_apply_cupy,
        dask_func=_apply_dask_numpy,
        dask_cupy_func=not_implemented_func,
    )
    out = mapper(raster)(raster.data.astype(float), kernel, func)
    result = DataArray(out,
                       name=name,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)
    return result


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
        2D array of input values to be analysed.
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
    """

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
    kernel : Numpy Array
        2D array where values of 1 indicate the kernel.

    Returns
    -------
    hotspots_agg : xarray.DataArray of same type as `raster`
        2D array of hotspots with values indicating confidence level.

    Examples
    --------
    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain, aspect
        from xrspatial.convolution import circle_kernel
        from xrspatial.focal import hotspots


        # Generate Example Terrain
        W = 500
        H = 300

        template_terrain = xr.DataArray(np.zeros((H, W)))
        x_range=(-20e6, 20e6)
        y_range=(-20e6, 20e6)

        terrain_agg = generate_terrain(
            template_terrain, x_range=x_range, y_range=y_range
        )

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )

        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Kernel
        kernel = circle_kernel(10, 10, 100)

        # Create Hotspots Aggregate array
        hotspots_agg = hotspots(raster = terrain_agg,
                                kernel = kernel)

        # Edit Attributes
        hotspots_agg = hotspots_agg.rename('Significance')
        hotspots_agg = hotspots_agg.assign_attrs(
            {
                'Description': 'Example Hotspots',
                'units': '%',
            }
        )

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Hotspots
        hotspots_agg.plot(aspect = 2, size = 4)
        plt.title("Hotspots")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02296597, 1261.947921  ],
               [1285.37105519, 1282.48079719],
               [1306.02339636, 1303.4069579 ]])
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(hotspots_agg[200:203, 200:202])
        <xarray.DataArray 'Significance' (lat: 3, lon: 2)>
        array([[0, 0],
               [0, 0],
               [0, 0]], dtype=int8)
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
            Description:    Example Hotspots
            units:          %
            Max Elevation:  4000
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
        dask_cupy_func=not_implemented_func,
    )
    out = mapper(raster)(raster, kernel)

    return DataArray(out,
                     coords=raster.coords,
                     dims=raster.dims,
                     attrs=raster.attrs)
