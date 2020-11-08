from math import atan2
from math import pi
import numpy as np
import numba as nb

from numba import cuda

from xarray import DataArray

from xrspatial.utils import ngjit
from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args

RADIAN = 180 / np.pi


@ngjit
def _horn_aspect(data):
    out = np.zeros_like(data, dtype=np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            a = data[y-1, x-1]
            b = data[y-1, x]
            c = data[y-1, x+1]
            d = data[y, x-1]
            f = data[y, x+1]
            g = data[y+1, x-1]
            h = data[y+1, x]
            i = data[y+1, x+1]

            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            if dz_dx == 0 and dz_dy == 0:
                # flat surface, slope = 0, thus invalid aspect
                out[y, x] = -1.
            else:
                aspect = np.arctan2(dz_dy, -dz_dx) * RADIAN
                # convert to compass direction values (0-360 degrees)
                if aspect < 0:
                    out[y, x] = 90.0 - aspect
                elif aspect > 90.0:
                    out[y, x] = 360.0 - aspect + 90.0
                else:
                    out[y, x] = 90.0 - aspect

    return out


@cuda.jit(device=True)
def _gpu_aspect(arr):

    a = arr[0, 0]
    b = arr[0, 1]
    c = arr[0, 2]
    d = arr[1, 0]
    f = arr[1, 2]
    g = arr[2, 0]
    h = arr[2, 1]
    i = arr[2, 2]

    two = nb.int32(2.)  # reducing size to int8 causes wrong results
    eight = nb.int32(8.)  # reducing size to int8 causes wrong results
    ninety = nb.float32(90.)

    dz_dx = ((c + two * f + i) - (a + two * d + g)) / eight
    dz_dy = ((g + two * h + i) - (a + two * b + c)) / eight

    if dz_dx == 0 and dz_dy == 0:
        # flat surface, slope = 0, thus invalid aspect
        aspect = nb.float32(-1.)  # TODO: return null instead
    else:
        aspect = atan2(dz_dy, -dz_dx) * nb.float32(57.29578)
        # convert to compass direction values (0-360 degrees)
        if aspect < nb.float32(0.):
            aspect = ninety - aspect
        elif aspect > ninety:
            aspect = nb.float32(360.0) - aspect + ninety
        else:
            aspect = ninety - aspect
    
    if aspect > nb.float32(359.999):  # lame float equality check...
        return nb.float32(0.)
    else:
        return aspect


@cuda.jit
def _horn_aspect_cuda(arr, out):
    minus_one = nb.float32(-1.)
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i-di >= 1 and i+di < out.shape[0] - 1 and 
        j-dj >= 1 and j+dj < out.shape[1] - 1):
        out[i, j] = _gpu_aspect(arr[i-di:i+di+1, j-dj:j+dj+1])



def aspect(agg, name='aspect', use_cuda=True, pad=True, use_cupy=True):
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

    if has_cuda() and use_cuda:

        if pad:
            pad_rows = 3 // 2
            pad_cols = 3 // 2
            pad_width = ((pad_rows, pad_rows),
                        (pad_cols, pad_cols))
        else:
            # If padding is not desired, set pads to 0
            pad_rows = 0
            pad_cols = 0
            pad_width = 0

        data = np.pad(agg.data, pad_width=pad_width, mode="reflect")

        griddim, blockdim = cuda_args(data.shape)
        out = np.empty(data.shape, dtype='f4')
        out[:] = np.nan

        if use_cupy:
            import cupy
            out = cupy.asarray(out)

        _horn_aspect_cuda[griddim, blockdim](data, out)
        if pad:
            out = out[pad_rows:-pad_rows, pad_cols:-pad_cols]
        
    else:
        out = _horn_aspect(agg.data)

    return DataArray(out,
                     name=name,
                     dims=agg.dims,
                     coords=agg.coords,
                     attrs=agg.attrs)
