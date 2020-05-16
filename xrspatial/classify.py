import datashader.transfer_functions as tf
import numpy as np
import scipy.stats as stats
from datashader.colors import rgb
from datashader.utils import ngjit
from sklearn.cluster import KMeans as KMEANS
from xarray import DataArray


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
    for x in range(0, rows):
        for y in range(0, cols):
            if data[y, x] in values:
                out[y, x] = True
            else:
                out[y, x] = False
    return out


def binary(agg, values, name='binary'):
    return DataArray(_binary(agg.data, values),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


@ngjit
def _bin(data, bins, new_values):
    out = np.zeros(data.shape, dtype=new_values.dtype)
    rows, cols = data.shape
    nbins = len(bins)
    val = None
    for x in range(0, rows):
        for y in range(0, cols):
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


def quantile(agg, k=4, name='quantile'):
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

    q = np.array([stats.scoreatpercentile(agg.data, pct) for pct in p])
    q = np.unique(q)
    k_q = len(q)

    if k_q < k:
        print("Quantile Warning: Not enough unique values for k classes (using {} bins)".format(k_q))

    return DataArray(_bin(agg.data, q, np.arange(k_q)),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)


def _kmeans(agg, k=5, n_init=10):
    """
    Helper function to do k-means in one dimension

    Parameters
    ----------

    agg     : xr.DataArray
             xarray.DataArray of value to classify
    k       : int
              number of classes to form

    n_init : int, default: 10
              number of initial  solutions. Best of initial results is returned.
    """

    agg.data = agg.data * 1.0  # KMEANS needs float or double dtype
    agg.data.shape = (-1, 1)
    result = KMEANS(n_clusters=k, init="k-means++", n_init=n_init).fit(agg.data)
    class_ids = result.labels_
    centroids = result.cluster_centers_
    binning = []
    for c in range(k):
        values = agg.data[class_ids == c]
        binning.append([values.max(), len(values)])
    binning = np.array(binning)
    binning = binning[binning[:, 0].argsort()]
    cuts = binning[:, 0]

    y_cent = np.zeros_like(agg.data)
    for c in range(k):
        y_cent[class_ids == c] = centroids[c]
    diffs = agg.data - y_cent
    diffs *= diffs

    return class_ids, cuts, diffs.sum(), centroids


def natural_breaks_helper(agg, number_classes=5, init=10):
    """
    natural breaks helper function
    Jenks natural breaks is kmeans in one dimension

    Parameters
    ----------

    agg : xr.DataArray

        xarray.DataArray of values to bin
    number_classes : int
        Number of classes
    init: int, default:10
        Number of different solutions to obtain using different centroids. Best solution is returned.


    Algorithm References:
     - https://pysal.org/pysal/_modules/pysal/viz/mapclassify/classifiers.html#NaturalBreaks
     - https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    Examples
    --------
    >>> from xrspatial.classify import natural_breaks
    >>> natural_agg = natural_breaks(my_agg)
    >>> values = np.array([1, 1, 0, 2,4,5,6])
    >>> val1 =xarray.DataArray(values)
    >>> In []: xrspatial.natural_breaks(val1)
    >>> Out[]:
    >>> <xarray.DataArray 'natural_breaks' (dim_0: 5)>
    >>> array([0., 1., 2., 4., 6.])
    """
    dr_values = np.array(agg.data)
    agg_dr = DataArray(dr_values,
                       dims=agg.dims,
                       coords=agg.coords,
                       attrs=agg.attrs)

    unique_values = np.unique(dr_values)
    unique_num_classes = len(unique_values)
    print(unique_num_classes, number_classes)
    if unique_num_classes < number_classes:
        print('NBreaks Warning: Not enough unique values in array for {} classes'.format(unique_num_classes))
        number_classes = unique_num_classes

    kres = _kmeans(agg_dr, number_classes)
    sids = kres[-1]  # centroids

    fit = kres[-2]

    class_ids = kres[0]

    cuts = kres[1]
    return sids, class_ids, fit, cuts


def natural_breaks(agg, name='natural_breaks', k=5, init=10):
    agg_copy = agg.copy()

    values = np.array(agg_copy.data)
    uv = np.unique(values)
    uvk = len(uv)

    if uvk < k:
        print('NBreaks Warning: Not enough unique values in array for {} classes'.format(uvk))
        k = uvk
        uv.sort()
        bins = uv
        k = k
    else:
        res0 = natural_breaks_helper(agg_copy, k, init=init)
        bins = np.array(res0[-1])
    return DataArray(_bin(agg.data, bins, np.arange(uvk)),
                    name=name,
                    coords=agg.coords,
                    dims=agg.dims,
                    attrs=agg.attrs)


def equal_interval(agg, k=4, name='equal_interval'):
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
    max_agg = np.max(agg.data)
    min_agg = np.min(agg.data)
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
