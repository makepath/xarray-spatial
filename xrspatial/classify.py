from functools import partial
import xarray as xr

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import datashader.transfer_functions as tf
import numpy as np
from datashader.colors import rgb
from xarray import DataArray

from numba import cuda
import dask.array as da

from numpy.random import RandomState

from xrspatial.utils import cuda_args
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.utils import is_cupy_backed

from typing import List, Optional


import warnings


def color_values(agg, color_key, alpha=255):
    def _convert_color(c):
        r, g, b = rgb(c)
        return np.array([r, g, b, alpha]).astype(np.uint8).view(np.uint32)[0]

    _converted_colors = {k: _convert_color(v) for k, v in color_key.items()}
    f = np.vectorize(lambda v: _converted_colors.get(v, 0))
    return tf.Image(f(agg.data))


@ngjit
def _binary(data, values):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if np.any(values == data[y, x]):
                out[y, x] = 1
            else:
                out[y, x] = 0
    return out


def binary(agg, values, name='binary'):

    if isinstance(values, (list, tuple)):
        vals = np.array(values)
    else:
        vals = values

    return DataArray(_binary(agg.data, vals),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _cpu_bin(data, bins, new_values):
    out = np.zeros(data.shape, dtype=np.float32)
    out[:, :] = np.nan
    rows, cols = data.shape
    nbins = len(bins)
    for y in range(0, rows):
        for x in range(0, cols):
            val = data[y, x]
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
                out[y, x] = new_values[val_bin]
            else:
                out[y, x] = np.nan

    return out


def _run_numpy_bin(data, bins, new_values):
    out = _cpu_bin(data, bins, new_values)
    return out


def _run_dask_numpy_bin(data, bins, new_values):
    _func = partial(_run_numpy_bin,
                    bins=bins,
                    new_values=new_values)

    out = data.map_blocks(_func)
    return out


@cuda.jit(device=True)
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


@cuda.jit
def _run_gpu_bin(data, bins, new_values, out):
    i, j = cuda.grid(2)
    if (i >= 0 and i < out.shape[0] and j >= 0 and j < out.shape[1]):
        out[i, j] = _gpu_bin(data[i:i+1, j:j+1], bins, new_values)


def _run_cupy_bin(data, bins_cupy, new_values_cupy):
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


def _bin(data, bins, new_values):
    # numpy case
    if isinstance(data, np.ndarray):
        out = _run_numpy_bin(data, np.asarray(bins), np.asarray(new_values))

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        bins_cupy = cupy.asarray(bins, dtype='f4')
        new_values_cupy = cupy.asarray(new_values, dtype='f4')
        out = _run_cupy_bin(data, bins_cupy, new_values_cupy)

    # dask + cupy case
    elif has_cuda() and isinstance(data, da.Array) and \
            type(data._meta).__module__.split('.')[0] == 'cupy':
        bins_cupy = cupy.asarray(bins, dtype='f4')
        new_values_cupy = cupy.asarray(new_values, dtype='f4')
        out = _run_dask_cupy_bin(data, bins_cupy, new_values_cupy)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _run_dask_numpy_bin(data, np.asarray(bins),
                                  np.asarray(new_values))

    return out


def reclassify(agg: xr.DataArray,
               bins: List[int],
               new_values: List[int],
               name: Optional[str] = 'reclassify') -> xr.DataArray:
    """
    Reclassifies data for array (agg) into new values based on bins.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of values to be reclassified.
        NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array.
    bins: array-like object
        Values or ranges of values to be changed.
    new_values: array-like object
        New values for each bin.
    name: str, optional (default = "reclassify")
        Name of output aggregate.

    Returns:
    ----------
    xarray.DataArray, reclassified aggregate.
        2D array of new values. All input attributes are preserved.

    Notes:
    ----------
    Adapted from PySal:
        - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial.classify import reclassify

    Create Initial DataArray
    >>> np.random.seed(1)
    >>> agg = xr.DataArray(np.random.randint(2, 8, (4, 4)),
    >>>                    dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat
    >>> print(agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[7, 5, 6, 2],
           [3, 5, 7, 2],
           [2, 3, 6, 7],
           [6, 3, 4, 6]])
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0

    Reclassify
    >>> bins = list(range(2, 8))
    >>> new_val = list(range(20, 80, 10))
    >>> reclassify_agg = reclassify(agg, bins, new_val)
    >>> print(reclassify_agg)
    <xarray.DataArray 'reclassify' (lat: 4, lon: 4)>
    array([[70., 50., 60., 20.],
           [30., 50., 70., 20.],
           [20., 30., 60., 70.],
           [60., 30., 40., 60.]], dtype=float32)
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0
    """

    if len(bins) != len(new_values):
        raise ValueError('bins and new_values mismatch.'
                         'Should have same length.')
    out = _bin(agg.data, bins, new_values)
    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


def _run_cpu_quantile(data, k):
    w = 100.0 / k
    p = np.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = np.percentile(data, p)
    q = np.unique(q)
    return q


def _run_dask_numpy_quantile(data, k):
    w = 100.0 / k
    p = da.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = da.percentile(data.flatten(), p)
    q = da.unique(q)
    return q


def _run_cupy_quantile(data, k):
    w = 100.0 / k
    p = cupy.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    q = cupy.percentile(data, p)
    q = cupy.unique(q)
    return q


def _run_dask_cupy_quantile(data, k):
    msg = 'Currently percentile calculation has not' \
          'been supported for Dask array backed by CuPy.' \
          'See issue at https://github.com/dask/dask/issues/6942'
    raise NotImplementedError(msg)


def _quantile(agg, k):
    # numpy case
    if isinstance(agg.data, np.ndarray):
        q = _run_cpu_quantile(agg.data, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        q = _run_cupy_quantile(agg.data, k)

    # dask + cupy case
    elif has_cuda() and \
        isinstance(agg.data, cupy.ndarray) and \
            is_cupy_backed(agg):
        q = _run_dask_cupy_quantile(agg.data, k)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        q = _run_dask_numpy_quantile(agg.data, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return q


def quantile(agg: xr.DataArray,
             k: int = 4,
             name: Optional[str] = 'quantile') -> xr.DataArray:
    """
    Groups data for array (agg) into quantiles by distributing
    the values into groups that contain an equal number of values.
    The number of quantiles produced is based on (k) with a default value
    of 4. The result is an xarray.DataArray.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of values to bin:
        NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array.
    k: int
        Number of quantiles to be produced, default = 4.
    name: str, optional (default = "quantile")
        Name of the output aggregate array.

    Returns:
    ----------
    xarray.DataArray, quantiled aggregate
        2D array, of the same type as the input, of quantile allocations.
        All other input attributes are preserved.

    Notes:
    ----------
    Adapted from PySAL:
    - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Quantiles # noqa

    Note that dask's percentile algorithm is approximate,
    while numpy's is exact. This may cause some differences
    between results of vanilla numpy and dask version of the input agg.
    - https://github.com/dask/dask/issues/3099

    Examples:
    ----------
        Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial.classify import quantile

        Create DataArray
    >>> np.random.seed(0)
    >>> agg = xr.DataArray(np.random.rand(4,4),
                                    dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> agg["lat"] = _lat
    >>> agg["lon"] = _lon
    >>> print(agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
            [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
    Coordinates:
    * lon      (lon) float64 0.0 1.0 2.0 3.0
    * lat      (lat) float64 0.0 1.0 2.0 3.0

    Create Quantile Aggregate
    >>> quantile_agg = quantile(agg)
    >>> print(quantile_agg)
    <xarray.DataArray 'quantile' (lat: 4, lon: 4)>
    array([[1., 2., 2., 1.],
           [0., 2., 1., 3.],
           [3., 0., 3., 1.],
           [2., 3., 0., 0.]], dtype=float32)
    Coordinates:
    * lon      (lon) float64 0.0 1.0 2.0 3.0
    * lat      (lat) float64 0.0 1.0 2.0 3.0

    With k quantiles
    >>> quantile_agg = quantile(agg, k = 6, name = "Six Quantiles")
    >>> print(quantile_agg)
    <xarray.DataArray 'Six Quantiles' (lat: 4, lon: 4)>
    array([[2., 4., 3., 2.],
           [1., 3., 1., 5.],
           [5., 0., 4., 1.],
           [3., 5., 0., 0.]], dtype=float32)
    Coordinates:
    * lon      (lon) float64 0.0 1.0 2.0 3.0
    * lat      (lat) float64 0.0 1.0 2.0 3.0
    """

    q = _quantile(agg, k)
    k_q = q.shape[0]
    if k_q < k:
        print("Quantile Warning: Not enough unique values"
              "for k classes (using {} bins)".format(k_q))
        k = k_q

    out = _bin(agg.data, bins=q, new_values=np.arange(k))

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _run_numpy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros((n_data + 1, n_classes + 1),
                                  dtype=np.float64)
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = np.zeros((n_data + 1, n_classes + 1), dtype=np.float64)
    var_combinations[2:n_data + 1, 1:n_classes + 1] = np.inf

    nl = data.shape[0] + 1
    variance = 0.0

    for l in range(2, nl): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(1, l + 1):
            # `III` originally
            lower_class_limit = l - m + 1
            i4 = lower_class_limit - 1

            val = data[i4]

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

            if i4 != 0:
                for j in range(2, n_classes + 1):
                    jm1 = j - 1
                    if var_combinations[l, j] >= \
                            (variance + var_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        var_combinations[l, j] = variance + \
                            var_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


@ngjit
def _run_numpy_jenks(data, n_classes):
    # ported from existing cython implementation:
    # https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    data.sort()

    lower_class_limits, _ = _run_numpy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = [0.] * (n_classes + 1)
    count_num = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _run_numpy_natural_break(data, num_sample, k):
    num_data = data.size

    if num_sample is not None and num_sample < num_data:
        # randomly select sample from the whole dataset
        # create a pseudo random number generator
        generator = RandomState(1234567890)
        idx = [i for i in range(0, data.size)]
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
        centroids = _run_numpy_jenks(sample_data, k)
        bins = np.array(centroids[1:])

    out = _bin(data, bins, np.arange(uvk))
    return out


def _run_cupy_jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = cupy.zeros((n_data + 1, n_classes + 1), dtype='f4')
    lower_class_limits[1, 1:n_classes + 1] = 1.0

    var_combinations = cupy.zeros((n_data + 1, n_classes + 1), dtype='f4')
    var_combinations[2:n_data + 1, 1:n_classes + 1] = cupy.inf

    nl = data.shape[0] + 1
    variance = 0.0

    for l in range(2, nl): # noqa
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(1, l + 1):
            # `III` originally
            lower_class_limit = l - m + 1
            i4 = lower_class_limit - 1

            val = data[i4]

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

            if i4 != 0:
                for j in range(2, n_classes + 1):
                    jm1 = j - 1
                    if var_combinations[l, j] >= \
                            (variance + var_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        var_combinations[l, j] = variance + \
                            var_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


def _run_cupy_jenks(data, n_classes):
    data.sort()

    lower_class_limits, _ = _run_cupy_jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = [0.] * (n_classes + 1)
    count_num = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def _run_cupy_natural_break(data, num_sample, k):
    num_data = data.size

    if num_sample is not None and num_sample < num_data:
        generator = cupy.random.RandomState(1234567890)
        idx = [i for i in range(0, data.size)]
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

    uv = cupy.unique(sample_data)
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
        centroids = _run_cupy_jenks(sample_data, k)
        bins = cupy.array(centroids[1:])

    out = _bin(data, bins, cupy.arange(uvk))
    return out


def natural_breaks(agg: xr.DataArray,
                   num_sample: Optional[int] = None,
                   name: Optional[str] = 'natural_breaks',
                   k: int = 5) -> xr.DataArray:
    """
    Groups data for array (agg) by distributing
    values using the Jenks Natural Breaks or k-means
    clustering method. Values are grouped so that
    similar values are placed in the same group and
    space between groups is maximized.
    The result is an xarray.DataArray.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of values to bin.
        NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
    num_sample: int (optional)
        Number of sample data points used to fit the model.
        Natural Breaks (Jenks) classification is indeed O(n²) complexity,
        where n is the total number of data points, i.e: agg.size
        When n is large, we should fit the model on a small sub-sample
        of the data instead of using the whole dataset.
    k: int (default = 5)
        Number of classes to be produced.
    name: str, optional (default = "natural_breaks")
        Name of output aggregate.

    Returns:
    ----------
    natural_breaks_agg: xarray.DataArray
        2D array, of the same type as the input, of class allocations.

    Algorithm References:
    ----------
    Map Classify:
    - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#NaturalBreaks # noqa
    perrygeo:
    - https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial.classify import natural_breaks

    Create DataArray
    >>> np.random.seed(0)
    >>> agg = xr.DataArray(np.random.rand(4,4),
                                    dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> agg["lat"] = _lat
    >>> agg["lon"] = _lon
    >>> print(agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
           [0.96366276, 0.38344152, 0.79172504, 0.52889492],
            [0.56804456, 0.92559664, 0.07103606, 0.0871293 ]])
    Coordinates:
    * lon      (lon) float64 0.0 1.0 2.0 3.0
    * lat      (lat) float64 0.0 1.0 2.0 3.0

    Create Natural Breaks Aggregate
    >>> natural_breaks_agg = natural_breaks(agg, k = 5)
    >>> print(natural_breaks_agg)
    <xarray.DataArray 'natural_breaks' (lat: 4, lon: 4)>
    array([[2., 3., 2., 2.],
           [1., 2., 1., 4.],
           [4., 1., 3., 2.],
           [2., 4., 0., 0.]], dtype=float32)
    Coordinates:
      * lat      (lat) float64 0.0 1.0 2.0 3.0
      * lon      (lon) float64 0.0 1.0 2.0 3.0
    """

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy_natural_break(agg.data, num_sample, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy_natural_break(agg.data, num_sample, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return DataArray(out,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)


def _run_numpy_equal_interval(data, k):
    max_data = np.nanmax(data)
    min_data = np.nanmin(data)
    rg = max_data - min_data
    width = rg * 1.0 / k
    cuts = np.arange(min_data + width, max_data + width, width)
    l_cuts = len(cuts)
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    cuts[-1] = max_data
    out = _run_numpy_bin(data, cuts, np.arange(l_cuts))
    return out


def _run_dask_numpy_equal_interval(data, k):
    max_data = da.nanmax(data)
    min_data = da.nanmin(data)
    width = (max_data - min_data) / k
    cuts = da.arange(min_data + width, max_data + width, width)
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    # work around to assign cuts[-1] = max_data
    bins = da.concatenate([cuts[:k-1], [max_data]])
    out = _bin(data, bins, np.arange(l_cuts))
    return out


def _run_cupy_equal_interval(data, k):
    max_data = cupy.nanmax(data)
    min_data = cupy.nanmin(data)
    width = (max_data - min_data) / k
    cuts = cupy.arange(min_data.get() +
                       width.get(), max_data.get() +
                       width.get(), width.get())
    l_cuts = cuts.shape[0]
    if l_cuts > k:
        # handle overshooting
        cuts = cuts[0:k]
    cuts[-1] = max_data
    out = _bin(data, cuts, cupy.arange(l_cuts))
    return out


def _run_dask_cupy_equal_interval(data, k):
    msg = 'Not yet supported.'
    raise NotImplementedError(msg)


def equal_interval(agg: xr.DataArray,
                   k: int = 5,
                   name: Optional[str] = 'equal_interval') -> xr.DataArray:
    """
    Groups data for array (agg) by distributing values into at equal intervals.
    The result is an xarray.DataArray.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of values to bin.
        NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
    k: int
        Number of classes to be produced.
    name: str, optional (default = "equal_interval")
        Name of output aggregate.

    Returns:
    ----------
    equal_interval_agg: xarray.DataArray
        2D array, of the same type as the input, of class allocations.

    Notes:
    ----------
    Intervals defined to have equal width:

    Algorithm References:
    ----------
    PySal:
    - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#EqualInterval # noqa
    SciKit:
    - https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py # noqa

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial.classify import equal_interval, natural_breaks

        Create Initial DataArray
    >>> np.random.seed(1)
    >>> agg = xr.DataArray(np.random.randint(2, 8, (4, 4)),
    >>>                    dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat
    >>> print(agg)
    <xarray.DataArray (lat: 4, lon: 4)>
    array([[7, 5, 6, 2],
           [3, 5, 7, 2],
           [2, 3, 6, 7],
           [6, 3, 4, 6]])
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0

    Create Equal Interval DataArray
    >>> equal_interval_agg = equal_interval(agg, k = 5)
    >>> print(equal_interval_agg)
    <xarray.DataArray 'equal_interval' (lat: 4, lon: 4)>
    array([[4., 2., 3., 0.],
           [0., 2., 4., 0.],
           [0., 0., 3., 4.],
           [3., 0., 1., 3.]], dtype=float32)
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0
    """

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy_equal_interval(agg.data, k)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy_equal_interval(agg.data, k)

    # dask + cupy case
    elif has_cuda() and \
            isinstance(agg.data, cupy.ndarray) and \
            is_cupy_backed(agg):
        out = _run_dask_cupy_equal_interval(agg.data, k)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy_equal_interval(agg.data, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return DataArray(out,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
