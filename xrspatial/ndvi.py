import numpy as np

from xarray import DataArray

from xrspatial.utils import ngjit


@ngjit
def _ndvi(nir_data, red_data):
    out = np.zeros_like(nir_data)
    rows, cols = nir_data.shape
    for y in range(0, rows):
        for x in range(0, cols):
            nir = nir_data[y, x]
            red = red_data[y, x]

            if nir == red:  # cover zero divison case
                continue

            soma = nir + red
            out[y, x] = (nir - red) / soma
    return out


# TODO: add optional name parameter `name='ndvi'`
def ndvi(nir_agg, red_agg):
    """Returns Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    nir_agg : DataArray
        near-infrared band data
    red_agg : DataArray
        red band data

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://ceholden.github.io/open-geo-tutorial/python/chapter_2_indices.html
    """

    if not isinstance(nir_agg, DataArray):
        raise TypeError("nir_agg must be instance of DataArray")

    if not isinstance(red_agg, DataArray):
        raise TypeError("red_agg must be instance of DataArray")

    if not red_agg.shape == nir_agg.shape:
        raise ValueError("red_agg and nir_agg expected to have equal shapes")

    return DataArray(_ndvi(nir_agg.data, red_agg.data),
                     name='ndvi',
                     coords=nir_agg.coords,
                     dims=nir_agg.dims,
                     attrs=nir_agg.attrs)
