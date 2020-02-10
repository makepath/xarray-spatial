import numpy as np

from xarray import DataArray

from xrspatial.utils import ngjit


#TODO: Make convolution more generic with numba first-class functions.


@ngjit
def _mean(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            exclude = False
            for ex in excludes:
                if data[y,x] == ex:
                    exclude = True
                    break

            if not exclude:
                a,b,c,d,e,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                     data[y-1, x],   data[y, x],   data[y+1, x],
                                     data[y-1, x+1], data[y, x+1], data[y+1, x+1]]
                out[y, x] = (a+b+c+d+e+f+g+h+i) / 9
            else:
                out[y, x] = data[y, x]
    return out

# TODO: add optional name parameter `name='mean'`
def mean(agg, passes=1, excludes=[np.nan]):
    """
    Returns Mean filtered array using a 3x3 window

    Parameters
    ----------
    agg : DataArray
    passes : int, number of times to run mean

    Returns
    -------
    data: DataArray
    """
    out = None
    for i in range(passes):
        if out is None:
            out = _mean(agg.data, tuple(excludes))
        else:
            out = _mean(out, tuple(excludes))

    return DataArray(out, name='mean',
                     dims=agg.dims, coords=agg.coords, attrs=agg.attrs)
