from math import atan
import numpy as np

from numba import cuda

from xarray import DataArray

from xrspatial.utils import ngjit
from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args


@ngjit
def _horn_slope(data, cellsize_x, cellsize_y):
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
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
            out[y, x] = np.arctan(p) * 57.29578
    return out


@cuda.jit(device=True)
def _gpu_slope(arr, cellsize_x, cellsize_y):
    a = arr[2, 0]
    b = arr[2, 1]
    c = arr[2, 2]
    d = arr[1, 0]
    f = arr[1, 2]
    g = arr[0, 0]
    h = arr[0, 1]
    i = arr[0, 2]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x[0])
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y[0])
    p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
    return atan(p) * 57.29578


@cuda.jit
def _horn_slope_cuda(arr, cellsize_x_arr, cellsize_y_arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i-di >= 0 and i+di < out.shape[0] and 
        j-dj >= 0 and j+dj < out.shape[1]):
        out[i, j] = _gpu_slope(arr[i-di:i+di+1, j-dj:j+dj+1],
                               cellsize_x_arr,
                               cellsize_y_arr)
    else:
        out[i, j] = np.nan


def slope(agg, name='slope', use_cuda=True):
    """Returns slope of input aggregate in degrees.
    Parameters
    ----------
    agg : DataArray
    name : str - name property of output xr.DataArray
    use_cuda : bool - use CUDA device if available

    Returns
    -------
    data: DataArray

    Notes:
    ------
    Algorithm References:
     - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
     - Burrough, P. A., and McDonell, R. A., 1998.
      Principles of Geographical Information Systems
      (Oxford University Press, New York), pp 406
    """

    if not isinstance(agg, DataArray):
        raise TypeError("agg must be instance of DataArray")

    if not agg.attrs.get('res'):
        raise ValueError('input xarray must have `res` attr.')

    # get cellsize out from 'res' attribute
    cellsize = agg.attrs.get('res')
    if isinstance(cellsize, tuple) and len(cellsize) == 2 \
            and isinstance(cellsize[0], (int, float)) \
            and isinstance(cellsize[1], (int, float)):
        cellsize_x, cellsize_y = cellsize
    elif isinstance(cellsize, (int, float)):
        cellsize_x = cellsize
        cellsize_y = cellsize
    else:
        raise ValueError('`res` attr of input xarray must be a numeric'
                         ' or a tuple of numeric values.')
    
    if has_cuda() and use_cuda:
        cellsize_x_arr =  np.array([float(cellsize_x)], dtype='f8')
        cellsize_y_arr =  np.array([float(cellsize_y)], dtype='f8')

        griddim, blockdim = cuda_args(agg.data.shape)
        slope_agg = np.zeros(agg.data.shape, dtype='f8')
        _horn_slope_cuda[griddim, blockdim](agg.data,
                                            cellsize_x_arr,
                                            cellsize_y_arr,
                                            slope_agg)
        pass
    else:
        slope_agg = _horn_slope(agg.data, cellsize_x, cellsize_y)

    return DataArray(slope_agg,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
