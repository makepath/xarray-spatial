import numpy as np

from datashader.colors import rgb

import datashader.transfer_functions as tf
from datashader.utils import ngjit
from xarray import DataArray

import scipy.stats as stats


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
                    if val > bins[b-1] and val <= bins[b]:
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

def natural_breaks(values, k=5, init=10):
    """
    natural breaks helper function

    Jenks natural breaks is kmeans in one dimension

    Parameters
    ----------

    values : array
             (n, 1) values to bin

    k : int
        Number of classes

    init: int, default:10
        Number of different solutions to obtain using different centroids. Best solution is returned.


    """
    values = np.array(values)
    uv = np.unique(values)
    uvk = len(uv)
    if uvk < k:
        print(
            "Nbreaks Warning: Not enough unique values for k classes (using {} bins)".format(uvk))"
        k = uvk
    kres = _kmeans(values, k, n_init=init)
    sids = kres[-1]  # centroids
    fit = kres[-2]
    class_ids = kres[0]
    cuts = kres[1]
    return DataArray(_bin(agg.data, sids, class_ids,fit,cuts, np.arange(uvk)),
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)



class EqualInterval(MapClassifier):
    """
    Equal Interval Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0
              otherwise
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> ei = mc.EqualInterval(cal, k = 5)
    >>> ei.k
    5
    >>> ei.counts
    array([57,  0,  0,  0,  1])
    >>> ei.bins
    array([ 822.394, 1644.658, 2466.922, 3289.186, 4111.45 ])

    Notes
    -----
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`
    """

    def __init__(self, y, k=K):
        """
        see class docstring

        """

        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "Equal Interval"


    def _set_bins(self):
        y = self.y
        k = self.k
        max_y = max(y)
        min_y = min(y)
        rg = max_y - min_y
        width = rg * 1.0 / k
        cuts = np.arange(min_y + width, max_y + width, width)
        if len(cuts) > self.k:  # handle overshooting
            cuts = cuts[0:k]
        cuts[-1] = max_y
        bins = cuts.copy()
        self.bins = bins



msg = _dep_message("Equal_Interval", "EqualInterval")
Equal_Interval = DeprecationHelper(EqualInterval, message=msg)
