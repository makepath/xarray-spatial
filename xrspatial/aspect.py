import numpy as np

from xarray import DataArray

from xrspatial.utils import ngjit


@ngjit
def _horn_aspect(data):
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

            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            aspect = np.arctan2(dz_dy, -dz_dx) * 57.29578  # (180 / pi)

            if aspect < 0:
                out[y, x] = 90.0 - aspect
            elif aspect > 90.0:
                out[y, x] = 360.0 - aspect + 90.0
            else:
                out[y, x] = 90.0 - aspect

    return out


# TODO: add optional name parameter `name='aspect'`
def aspect(agg):
    """Returns downward slope direction in compass degrees (0 - 360) with 0 at 12 o'clock.

    Parameters
    ----------
    agg : DataArray

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm#ESRI_SECTION1_4198691F8852475A9F4BC71246579FAA
     - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical Information Systems (Oxford University Press, New York), pp 406
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    return DataArray(_horn_aspect(agg.data),
                     name='aspect',
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)
