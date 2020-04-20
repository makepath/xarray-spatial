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
