from functools import partial
from typing import Union

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

import warnings
warnings.simplefilter('default')


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
    out = data.map_blocks(lambda da: _run_cupy_bin(da, bins_cupy, new_values_cupy),
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
    elif has_cuda() and isinstance(data, da.Array) and type(data._meta).__module__.split('.')[0] == 'cupy':
        bins_cupy = cupy.asarray(bins, dtype='f4')
        new_values_cupy = cupy.asarray(new_values, dtype='f4')
        out = _run_dask_cupy_bin(data, bins_cupy, new_values_cupy)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _run_dask_numpy_bin(data, np.asarray(bins), np.asarray(new_values))

    return out


def reclassify(agg, bins, new_values, name='reclassify'):
    """
    Reclassify xr.DataArray to new values based on bins

    Adapted from PySAL:
    https://pysal.org/pysal/_modules/pysal/viz/mapclassify/classifiers.html#Quantiles

    Parameters
    ----------
    agg: xr.DataArray
        xarray.DataArray of value to classify
    bins: array-like object
        values to bin
    new_values: array-like object
    name : str
        name of output aggregate

    Returns
    -------
    reclassified_agg : xr.DataArray

    Examples
    --------
    >>> from xrspatial.classify import reclassify
    >>> reclassify_agg = reclassify(my_agg)
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
    msg = 'Currently percentile calculation has not been supported for Dask array backed by CuPy.' \
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
    elif has_cuda() and isinstance(agg.data, cupy.ndarray) and is_cupy_backed(agg):
        q = _run_dask_cupy_quantile(agg.data, k)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        q = _run_dask_numpy_quantile(agg.data, k)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return q


def quantile(agg, k=4, name='quantile'):
    """
    Calculates the quantiles for an array

    Adapted from PySAL:
    https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#Quantiles

    Note that dask's percentile algorithm is approximate, while numpy's is exact.
    This may cause some differences between results of vanilla numpy and
    dask version of the input agg.
    https://github.com/dask/dask/issues/3099

    Parameters
    ----------
    agg : xr.DataArray
        xarray.DataArray of value to classify
    k : int
        number of quantiles
    name : str
        name of output aggregate

    Returns
    -------
    quantiled_agg : xr.DataArray

    Examples
    --------
    >>> from xrspatial.classify import quantile
    >>> quantile_agg = quantile(my_agg)
    """

    q = _quantile(agg, k)
    k_q = q.shape[0]
    if k_q < k:
        print("Quantile Warning: Not enough unique values for k classes (using {} bins)".format(k_q))
        k = k_q

    out = _bin(agg.data, bins=q, new_values=np.arange(k))

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _jenks_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros((n_data+1, n_classes+1), dtype=np.float64)
    lower_class_limits[1, 1:n_classes+1] = 1.0

    var_combinations = np.zeros((n_data+1, n_classes+1), dtype=np.float64)
    var_combinations[2:n_data+1, 1:n_classes+1] = np.inf

    nl = data.shape[0] + 1
    variance = 0.0

    for l in range(2, nl):
        sum = 0.0
        sum_squares = 0.0
        w = 0.0

        for m in range(1, l+1):
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
                for j in range(2, n_classes+1):
                    jm1 = j - 1
                    if var_combinations[l, j] >= (variance + var_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        var_combinations[l, j] = variance + var_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        var_combinations[l, 1] = variance

    return lower_class_limits, var_combinations


@ngjit
def _jenks(data, n_classes):
    # ported from existing cython implementation:
    # https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    data.sort()

    lower_class_limits, _ = _jenks_matrices(data, n_classes)

    k = data.shape[0]
    kclass = [0.] * (n_classes+1)
    count_num = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while count_num > 1:
        elt = int(lower_class_limits[k][count_num] - 2)
        kclass[count_num - 1] = data[elt]
        k = int(lower_class_limits[k][count_num] - 1)
        count_num -= 1

    return kclass


def natural_breaks(agg, num_sample=None, name='natural_breaks', k=5):
    """
    Calculate Jenks natural breaks (a.k.a kmeans in one dimension)
    for an input raster xarray.

    Parameters
    ----------
    agg : xarray.DataArray
        xarray.DataArray of values to bin
    num_sample: int (optional)
        Number of sample data points used to fit the model.
        Natural Breaks (Jenks) classification is indeed O(n²) complexity,
        where n is the total number of data points, i.e: agg.size
        When n is large, we should fit the model on a small sub-sample
        of the data instead of using the whole dataset.
    k: int
        Number of classes
    name : str
        name of output aggregate

    Returns
    -------
    natural_breaks_agg: xarray.DataArray

    Algorithm References:
     - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#NaturalBreaks
     - https://github.com/perrygeo/jenks/blob/master/jenks.pyx

    Examples
    --------
    >>> n, m = 4, 3
    >>> agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['y', 'x'])
    >>> agg['y'] = np.linspace(0, n, n)
    >>> agg['x'] = np.linspace(0, m, m)
    >>> agg.data
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])
    >>> k = 5
    >>> natural_breaks_agg = natural_breaks(agg, k=5)
    >>> natural_breaks_agg.data
    array([[0., 0., 1.],
           [1., 2., 2.],
           [3., 3., 3.],
           [4., 4., 4.]]
    """

    num_data = agg.size

    if num_sample is not None and num_sample < num_data:
        # randomly select sample from the whole dataset
        # create a pseudo random number generator
        generator = RandomState(1234567890)
        idx = [i for i in range(0, agg.size)]
        generator.shuffle(idx)
        sample_idx = idx[:num_sample]
        sample_data = agg.data.flatten()[sample_idx]
    else:
        sample_data = agg.data.flatten()

    # warning if number of total data points to fit the model bigger than 40k
    if sample_data.size >= 40000:
        warnings.warn('natural_breaks Warning: Natural break classification '
                      '(Jenks) has a complexity of O(n^2), '
                      'your classification with {} data points may take '
                      'a long time.'.format(sample_data.size),
                      Warning)

    uv = np.unique(sample_data)
    uvk = len(uv)

    if uvk < k:
        warnings.warn('natural_breaks Warning: Not enough unique values '
                      'in data array for {} classes. '
                      'n_samples={} should be >= n_clusters={}. '
                      'Using k={} instead.'.format(k, uvk, k, uvk),
                      Warning)
        uv.sort()
        bins = uv
    else:
        centroids = _jenks(sample_data, k)
        bins = np.array(centroids[1:])

    return DataArray(_run_numpy_bin(agg.data, bins, np.arange(uvk)),
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)


def equal_interval(agg, k=5, name='equal_interval'):
    """
    Equal Interval Classification

    Parameters
    ----------
    agg     : xr.DataArray
             xarray.DataArray of value to classify
    k       : int
              number of classes required
    name : str
        name of output aggregate

    Returns
        equal_interval_agg : xr.DataArray

    Notes:
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`

    Algorithm References:
     - https://pysal.org/mapclassify/_modules/mapclassify/classifiers.html#EqualInterval
     - https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    Examples
    --------

    >>> In []: ei = np.array([1, 1, 0, 2,4,5,6])

    >>> In []: ei_array =xarray.DataArray(ei)

    >>> In []: xrspatial.equal_interval(ei_array)
    >>> Out[]:
    <xarray.DataArray 'equal_interval' (dim_0: 4)>
    array([1.5, 3. , 4.5, 6. ])
    """

    max_agg = np.nanmax(agg.data)
    min_agg = np.nanmin(agg.data)
    rg = max_agg - min_agg
    width = rg * 1.0 / k
    cuts = np.arange(min_agg + width, max_agg + width, width)
    l_cuts = len(cuts)
    if l_cuts > k:
        print('EqualInterv Warning: Not enough unique values in array for {} classes'.format(
            l_cuts)),  # handle overshooting
        cuts = cuts[0:k]
    cuts[-1] = max_agg
    bins = cuts.copy()
    return DataArray(_run_numpy_bin(agg.data, bins, np.arange(l_cuts)),
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
