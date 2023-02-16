import warnings
from functools import partial
from typing import List, Optional

import cmath

import xarray as xr

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da
import numba as nb
import numpy as np

from xrspatial.utils import ArrayTypeFunctionMapping, cuda_args, ngjit, not_implemented_func


@ngjit
def _cpu_binary(data, values):
    out = np.zeros_like(data)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if np.any(values == data[y, x]):
                out[y, x] = 1
            elif np.isfinite(data[y, x]):
                out[y, x] = 0
    return out


def _run_numpy_binary(data, values):
    values = np.asarray(values)
    out = _cpu_binary(data, values)
    return out


def _run_dask_numpy_binary(data, values):
    _func = partial(_run_numpy_binary, values=values)
    out = data.map_blocks(_func)
    return out


@nb.cuda.jit(device=True)
def _gpu_binary(val, values):
    for v in values:
        if val == v:
            return 1
    return 0


@nb.cuda.jit
def _run_gpu_binary(data, values, out):
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]:
        if cmath.isfinite(data[i, j]):
            out[i, j] = _gpu_binary(data[i, j], values)


def _run_cupy_binary(data, values):
    values_cupy = cupy.asarray(values)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _run_gpu_binary[griddim, blockdim](data, values_cupy, out)
    return out


def _run_dask_cupy_binary(data, values_cupy):
    out = data.map_blocks(lambda da: _run_cupy_binary(da, values_cupy), meta=cupy.array(()))
    return out


def binary(agg, values, name='binary'):
    """
    Binarize a data array based on a set of values. Data that equals to a value in the set will be
    set to 1. In contrast, data that does not equal to any value in the set will be set to 0.
    Note that NaNs and infinite values will be set to NaNs.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    values : array-like object
        Values to keep in the binarized array.
    name : str, default='binary'
        Name of output aggregate array.

    Returns
    -------
    binarized_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array of binarized data array.
        All other input attributes are preserved.

    Examples
    --------
    Binary works with NumPy backed xarray DataArray

    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import binary

        >>> data = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., np.inf],
        ], dtype=np.float32)
        >>> agg = xr.DataArray(data)
        >>> values = [1, 2, 3]
        >>> agg_binary = binary(agg, values)
        >>> print(agg_binary)
        <xarray.DataArray 'binary' (dim_0: 4, dim_1: 5)>
        array([[np.nan,  1.,  1.,  1.,  0.],
               [0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  np.nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """

    mapper = ArrayTypeFunctionMapping(numpy_func=_run_numpy_binary,
                                      dask_func=_run_dask_numpy_binary,
                                      cupy_func=_run_cupy_binary,
                                      dask_cupy_func=_run_dask_cupy_binary)
    out = mapper(agg)(agg.data, values)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


@ngjit
def _cpu_bin(data, bins, new_values):
    out = np.zeros(data.shape, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    nbins = len(bins)
    for y in range(0, rows):
        for x in range(0, cols):
            val = data[y, x]
            val_bin = -1

            # find bin
            if np.isfinite(val):
                if val <= bins[0]:
                    val_bin = 0
                elif val <= bins[nbins - 1]:
                    start = 0
                    end = nbins - 1
                    mid = (end + start) // 2
                    while start <= end:
                        if bins[mid] < val:
                            start = mid + 1
                        elif val > bins[mid - 1]:
                            break
                        else:
                            end = mid - 1
                        mid = (end + start) // 2

                    val_bin = mid

            if val_bin > -1:
                out[y, x] = new_values[val_bin]
            else:
                out[y, x] = np.nan

    return out


def _run_numpy_bin(data, bins, new_values):
    bins = np.asarray(bins)
    new_values = np.asarray(new_values)
    out = _cpu_bin(data, bins, new_values)
    return out


def _run_dask_numpy_bin(data, bins, new_values):
    _func = partial(_run_numpy_bin,
                    bins=bins,
                    new_values=new_values)

    out = data.map_blocks(_func)
    return out


@nb.cuda.jit(device=True)
def _gpu_bin(data, bins, new_values):
    nbins = len(bins)
    val = data[0, 0]
    val_bin = -1

    # find bin
    for b in range(0, nbins):

        # first bin
        if b == 0:
            if val <= bins[b]:
                val_bin = b
                break
        else:
            if val > bins[b - 1] and val <= bins[b]:
                val_bin = b
                break

    if val_bin > -1:
        out = new_values[val_bin]
    else:
        out = np.nan

    return out


@nb.cuda.jit
def _run_gpu_bin(data, bins, new_values, out):
    i, j = nb.cuda.grid(2)
    if (i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]):
        out[i, j] = _gpu_bin(data[i:i+1, j:j+1], bins, new_values)


def _run_cupy_bin(data, bins, new_values):
    # replace inf by nan to avoid classify these values as we want to treat them as outliers
    data = cupy.where(data == cupy.inf, cupy.nan, data)
    data = cupy.where(data == -cupy.inf, cupy.nan, data)

    bins_cupy = cupy.asarray(bins)
    new_values_cupy = cupy.asarray(new_values)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _run_gpu_bin[griddim, blockdim](data,
                                    bins_cupy,
                                    new_values_cupy,
                                    out)
    return out


def _run_dask_cupy_bin(data, bins_cupy, new_values_cupy):
    out = data.map_blocks(lambda da:
                          _run_cupy_bin(da, bins_cupy, new_values_cupy),
                          meta=cupy.array(()))
    return out


def _bin(agg, bins, new_values):
    mapper = ArrayTypeFunctionMapping(numpy_func=_run_numpy_bin,
                                      dask_func=_run_dask_numpy_bin,
                                      cupy_func=_run_cupy_bin,
                                      dask_cupy_func=_run_dask_cupy_bin)

    out = mapper(agg)(agg.data, bins, new_values)
    return out


def reclassify(agg: xr.DataArray,
               bins: List[int],
               new_values: List[int],
               name: Optional[str] = 'reclassify') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on user
    defined bins.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    bins : array-like object
        Values or ranges of values to be changed.
    new_values : array-like object
        New values for each bin.
    name : str, default='reclassify'
        Name of output aggregate array.

    Returns
    -------
    reclass_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array of reclassified allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html # noqa

    Examples
    --------
    Reclassify works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import reclassify

        >>> data = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., np.inf],
        ])
        >>> agg = xr.DataArray(data)
        >>> print(agg)
        <xarray.DataArray (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.,  9.],
               [10., 11., 12., 13., 14.],
               [15., 16., 17., 18., inf]])
        Dimensions without coordinates: dim_0, dim_1
        >>> bins=[10, 15, np.inf]
        >>> new_values=[1, 2, 3]
        >>> agg_reclassify = reclassify(agg, bins=bins, new_values=new_values)
        >>> print(agg_reclassify)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import dask.array as da
        >>> data_da = da.from_array(data, chunks=(3, 3))
        >>> agg_da = xr.DataArray(data_da, name='agg_da')
        >>> print(agg_da)
        <xarray.DataArray 'agg_da' (dim_0: 4, dim_1: 5)>
        dask.array<array, shape=(4, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        >>> agg_reclassify_da = reclassify(agg_da, bins=bins, new_values=new_values)  # noqa
        >>> print(agg_reclassify_da)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        dask.array<_run_numpy_bin, shape=(4, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        >>> print(agg_reclassify_da.compute())  # print the computed the results
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with CuPy backed xarray DataArray.
    Make sure you have a GPU and CuPy installed to run this example.
    .. sourcecode:: python

        >>> import cupy
        >>> data_cupy = cupy.asarray(data)
        >>> agg_cupy = xr.DataArray(data_cupy)
        >>> agg_reclassify_cupy = reclassify(agg_cupy, bins, new_values)
        >>> print(type(agg_reclassify_cupy.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(agg_reclassify_cupy)
        <xarray.DataArray 'reclassify' (dim_0: 4, dim_1: 5)>
        array([[nan,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 2.,  3.,  3.,  3.,  3.]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    Reclassify works with Dask with CuPy backed xarray DataArray.
    """

    if len(bins) != len(new_values):
        raise ValueError(
            'bins and new_values mismatch. Should have same length.'
        )
    out = _bin(agg, bins, new_values)
    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


def _run_quantile(data, k, module):
    w = 100.0 / k
    p = module.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = module.percentile(data[module.isfinite(data)], p)
    q = module.unique(q)
    return q


def _run_dask_cupy_quantile(data, k):
    msg = 'Currently percentile calculation has not' \
          'been supported for Dask array backed by CuPy.' \
          'See issue at https://github.com/dask/dask/issues/6942'
    raise NotImplementedError(msg)


def _quantile(agg, k):
    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_quantile(*args, module=np),
        dask_func=lambda *args: _run_quantile(*args, module=da),
        cupy_func=lambda *args: _run_quantile(*args, module=cupy),
        dask_cupy_func=_run_dask_cupy_quantile
    )
    out = mapper(agg)(agg.data, k)
    return out


def quantile(agg: xr.DataArray,
             k: int = 4,
             name: Optional[str] = 'quantile') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on quantile
    groups of equal size.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=4
        Number of quantiles to be produced.
    name : str, default='quantile'
        Name of the output aggregate array.

    Returns
    -------
    quantile_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array, of quantile allocations.
        All other input attributes are preserved.

    Notes
    -----
        - Dask's percentile algorithm is approximate, while numpy's is exact.
        - This may cause some differences between results of vanilla numpy
    and dask version of the input agg. (https://github.com/dask/dask/issues/3099) # noqa

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Quantiles # noqa

    Examples
    --------
    Quantile work with numpy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import quantile

        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_quantile = quantile(agg_numpy, k=5)
        >>> print(numpy_quantile)
        <xarray.DataArray 'quantile' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  1.,  1.,  1.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)
    """

    q = _quantile(agg, k)
    k_q = q.shape[0]
    if k_q < k:
        print("Quantile Warning: Not enough unique values"
              "for k classes (using {} bins)".format(k_q))
        k = k_q

    out = _bin(agg, bins=q, new_values=np.arange(k))

    return xr.DataArray(out,
                        name=name,
                        dims=agg.dims,
                        coords=agg.coords,
                        attrs=agg.attrs)


@nb.jit(nopython=True)
def _run_numpy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros(
        (n_data + 1, n_classes + 1), dtype=np.float32
    )
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = np.zeros(
        (n_data + 1, n_classes + 1), dtype=np.float32
    )
    var_combinations[2:n_data + 1, 1:n_classes + 1] = np.inf

    variance = 0.0

    for l in range(2, n_data + 1): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(l):
            # `III` originally
            lower_class_limit = l - m
            i4 = lower_class_limit - 1

            val = np.float32(data[i4])

            # here we're estimating variance for each potential classing
            # of the data, for each potential number of classes. `w`
            # is the number of data points considered so far.
            w += 1.0

            # increase the current sum and sum-of-squares
            sum += val
            sum_squares += val * val

            # the variance at this point in the sequence is the difference
            # between the sum of squares and the total x 2, over the number
            # of samples.
            variance = sum_squares - (sum * sum) / w

            if i4 == 0:
                continue
            for j in range(2, n_classes + 1):
                # if adding this element to an existing class
                # will increase its variance beyond the limit, break
                # the class at this point, setting the lower_class_limit
                # at this point.
                new_variance = variance + var_combinations[i4, j-1]
                if var_combinations[l, j] >= new_variance:
                    lower_class_limits[l, j] = lower_class_limit
                    var_combinations[l, j] = new_variance

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


def _run_jenks(data, n_classes):
    # ported from existing cython implementation:
    # https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    data.sort()
    lower_class_limits, _ = _run_numpy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = np.zeros(n_classes + 1, dtype=np.float32)
    kclass[0] = data[0]
    kclass[-1] = data[-1]
    count_num = n_classes

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _run_natural_break(agg, num_sample, k):
    data = agg.data
    num_data = data.size
    max_data = np.max(data[np.isfinite(data)])

    if num_sample is not None and num_sample < num_data:
        # randomly select sample from the whole dataset
        # create a pseudo random number generator
        # Note: cupy and nupy generate different random numbers
        # use numpy.random to ensure the same result
        generator = np.random.RandomState(1234567890)
        idx = np.linspace(
            0, data.size, data.size, endpoint=False, dtype=np.uint32
        )
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
        sample_data = data.flatten()[sample_idx]
    else:
        sample_data = data.flatten()

    # warning if number of total data points to fit the model bigger than 40k
    if sample_data.size >= 40000:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Natural break '
                          'classification (Jenks) has a complexity of O(n^2), '
                          'your classification with {} data points may take '
                          'a long time.'.format(sample_data.size),
                          Warning)

    sample_data = np.asarray(sample_data)

    # only include finite values
    sample_data = sample_data[np.isfinite(sample_data)]
    uv = np.unique(sample_data)
    uvk = len(uv)

    if uvk < k:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn('natural_breaks Warning: Not enough unique values '
                          'in data array for {} classes. '
                          'n_samples={} should be >= n_clusters={}. '
                          'Using k={} instead.'.format(k, uvk, k, uvk),
                          Warning)
        uv.sort()
        bins = uv
    else:
        centroids = _run_jenks(sample_data, k)
        bins = np.array(centroids[1:])
        bins[-1] = max_data

    out = _bin(agg, bins, np.arange(uvk))
    return out


def natural_breaks(agg: xr.DataArray,
                   num_sample: Optional[int] = 20000,
                   name: Optional[str] = 'natural_breaks',
                   k: int = 5) -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on Natural
    Breaks or K-Means clustering method. Values are grouped so that
    similar values are placed in the same group and space between
    groups is maximized.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy DataArray of values to be reclassified.
    num_sample : int, default=20000
        Number of sample data points used to fit the model.
        Natural Breaks (Jenks) classification is indeed O(n²) complexity,
        where n is the total number of data points, i.e: `agg.size`
        When n is large, we should fit the model on a small sub-sample
        of the data instead of using the whole dataset.
    k : int, default=5
        Number of classes to be produced.
    name : str, default='natural_breaks'
        Name of output aggregate.

    Returns
    -------
    natural_breaks_agg : xarray.DataArray of the same type as `agg`
        2D aggregate array of natural break allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#NaturalBreaks # noqa
        - jenks: https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    Examples
    -------
    natural_breaks() works with numpy backed xarray DataArray.
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import natural_breaks

        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_natural_breaks = natural_breaks(agg_numpy, k=5)
        >>> print(numpy_natural_breaks)
        <xarray.DataArray 'natural_breaks' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  2.],
               [ 2.,  2.,  2.,  2.,  3.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)

    natural_breaks() works with cupy backed xarray DataArray.
    .. sourcecode:: python

        >>> import cupy
        >>> agg_cupy = xr.DataArray(cupy.asarray(elevation))
        >>> cupy_natural_breaks = natural_breaks(agg_cupy)
        >>> print(type(cupy_natural_breaks))
        <class 'xarray.core.dataarray.DataArray'>
        >>> print(cupy_natural_breaks)
        <xarray.DataArray 'natural_breaks' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  2.],
               [ 2.,  2.,  2.,  2.,  3.],
               [ 3.,  3.,  3.,  3.,  4.],
               [ 4.,  4.,  4.,  4., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_natural_break(*args),
        dask_func=lambda *args: not_implemented_func(
            *args, messages='natural_breaks() does not support dask with numpy backed DataArray.'),  # noqa
        cupy_func=lambda *args: not_implemented_func(
            *args, messages='natural_breaks() does not support cupy backed DataArray.'),  # noqa
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='natural_breaks() does not support dask with cupy backed DataArray.'),  # noqa
    )
    out = mapper(agg)(agg, num_sample, k)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)


def _run_equal_interval(agg, k, module):
    data = agg.data.ravel()
    if module == cupy:
        nan = cupy.nan
        inf = cupy.inf
    else:
        nan = np.nan
        inf = np.inf

    data = module.where(data == inf, nan, data)
    data = module.where(data == -inf, nan, data)

    max_data = module.nanmax(data)
    min_data = module.nanmin(data)
    if module == cupy:
        min_data = min_data.get()
        max_data = max_data.get()

    width = (max_data - min_data) * 1.0 / k
    cuts = module.arange(min_data + width, max_data + width, width)
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]

    if module == da:
        # work around to assign cuts[-1] = max_data
        bins = da.concatenate([cuts[:k-1], [max_data]])
        out = _bin(agg, bins, np.arange(l_cuts))
    else:
        cuts[-1] = max_data
        out = _bin(agg, cuts, np.arange(l_cuts))

    return out


def equal_interval(agg: xr.DataArray,
                   k: int = 5,
                   name: Optional[str] = 'equal_interval') -> xr.DataArray:
    """
    Reclassifies data for array `agg` into new values based on intervals
    of equal width.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of values to be reclassified.
    k : int, default=5
        Number of classes to be produced.
    name : str, default='equal_interval'
        Name of output aggregate.

    Returns
    -------
    equal_interval_agg : xarray.DataArray of the same type as `agg`
        2D aggregate array of equal interval allocations.
        All other input attributes are preserved.

    References
    ----------
        - PySAL: https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#EqualInterval # noqa
        - scikit-learn: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.classify import equal_interval
        >>> elevation = np.array([
            [np.nan,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., np.inf]
        ])
        >>> agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
        >>> numpy_equal_interval = equal_interval(agg_numpy, k=5)
        >>> print(numpy_equal_interval)
        <xarray.DataArray 'equal_interval' (dim_0: 5, dim_1: 5)>
        array([[nan,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2., nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10.0, 10.0)
    """

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _run_equal_interval(*args, module=np),
        dask_func=lambda *args: _run_equal_interval(*args, module=da),
        cupy_func=lambda *args: _run_equal_interval(*args, module=cupy),
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='equal_interval() does support dask with cupy backed DataArray.'),  # noqa
    )
    out = mapper(agg)(agg, k)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
