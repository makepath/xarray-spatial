import numpy as np

from xarray import DataArray

from xrspatial.utils import ngjit


@ngjit
def _horn_slope(data, cellsize):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            a = data[y+1, x-1]
            b = data[y+1, x]
            c = data[y+1, x+1]
            d = data[y, x-1]
            f = data[y, x+1]
            g = data[y-1, x-1]
            h = data[y-1, x]
            i = data[y-1, x+1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
            out[y, x] = np.arctan(p) * 57.29578
    return out


# TODO: add optional name parameter `name='slope'`
def slope(agg):
    """Returns slope of input aggregate in degrees.

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
     - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical Information Systems (Oxford University Press, New York), pp 406
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    if not agg.attrs.get('res'):
        #TODO: maybe monkey-patch a "res" attribute valueing unity is reasonable
        raise ValueError('input xarray must have numeric `res` attr.')

    slope_agg = _horn_slope(agg.data, agg.attrs['res'])

    return DataArray(slope_agg,
                     name='slope',
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)

