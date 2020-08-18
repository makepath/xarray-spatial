import datashader.transfer_functions as tf
import numpy as np
import scipy.stats as stats
from datashader.colors import rgb
from datashader.utils import ngjit
from xarray import DataArray

from numpy.random import RandomState

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
def _bin(data, bins, new_values, nodata=np.nan, dtype=np.float32):
    out = np.zeros(data.shape, dtype=dtype)
    rows, cols = data.shape
    nbins = len(bins)
    val = None
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
                out[y, x] = nodata

    return out


def reclassify(agg, bins, new_values, name='reclassify',
               nodata=np.nan, dtype=np.float32):
    """
    Reclassify xr.DataArray to new values based on bins

    Adapted from PySAL:
    https://pysal.org/pysal/_modules/pysal/viz/mapclassify/classifiers.html#Quantiles

    Parameters
    ----------
    agg : xr.DataArray
        xarray.DataArray of value to classify
    k : int
        number of quantiles
    name : str
        name of data dim in output xr.DataArray

    Returns
    -------
    quantiled_agg : xr.DataArray

    Examples
    --------
    >>> from xrspatial.classify import quantile
    >>> quantile_agg = quantile(my_agg)
    """

    if len(bins) != len(new_values):
        raise ValueError('bins and new_values mismatch')

    return DataArray(_bin(agg.data, bins, new_values),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


def quantile(agg, k=4, name='quantile', ignore_vals=tuple()):
    """
    Calculates the quantiles for an array

    Adapted from PySAL:
    https://pysal.org/pysal/_modules/pysal/viz/mapclassify/classifiers.html#Quantiles

    Parameters
    ----------
    agg : xr.DataArray
        xarray.DataArray of value to classify
    k : int
        number of quantiles
    name : str
        name of data dim in output xr.DataArray

    Returns
    -------
    quantiled_agg : xr.DataArray

    Examples
    --------
    >>> from xrspatial.classify import quantile
    >>> quantile_agg = quantile(my_agg)
    """

    w = 100.0 / k
    p = np.arange(w, 100 + w, w)

    if p[-1] > 100.0:
        p[-1] = 100.0

    data = agg.data[~np.isnan(agg.data) & ~np.isin(agg.data, ignore_vals)]

    q = np.array([stats.scoreatpercentile(data, pct) for pct in p])
    q = np.unique(q)
    k_q = len(q)

    if k_q < k:
        print("Quantile Warning: Not enough unique values for k classes (using {} bins)".format(k_q))

    return DataArray(_bin(agg.data, q, np.arange(k_q)),
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

    return DataArray(_bin(agg.data, bins, np.arange(uvk)),
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


    Returns
        -------
        equal_interval_agg : xr.DataArray

    Notes:
    ------
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`

    Algorithm References:
     - https://pysal.org/pysal/_modules/pysal/viz/mapclassify/classifiers.html#EqualInterval
     - https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    Examples
    --------

    >>> In []: ei = np.array([1, 1, 0, 2,4,5,6])                                                                                                                        

    >>> In []: ei_array =xarray.DataArray(ei)                                                                                                                         

    >>> In []: xrspatial.equal_interval(ei_array)                                                                                                                     
    >>> Out[]: 
    <xarray.DataArray 'equal_interval' (dim_0: 4)>
    array([1.5, 3. , 4.5, 6. ])


    Notes
    -----
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`
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
    return DataArray(_bin(agg.data, bins, np.arange(l_cuts)),
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
