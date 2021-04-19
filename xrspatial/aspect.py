from math import atan2
import numpy as np
import numba as nb

from functools import partial

import dask.array as da

from numba import cuda

import xarray as xr

from xrspatial.utils import ngjit
from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args
from xrspatial.utils import is_cupy_backed

from typing import Optional

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

RADIAN = 180 / np.pi


@ngjit
def _cpu(data):
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
def _gpu(arr):

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
def _run_gpu(arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i-di >= 0 and
        i+di < out.shape[0] and
            j-dj >= 0 and
            j+dj < out.shape[1]):
        out[i, j] = _gpu(arr[i-di:i+di+1, j-dj:j+dj+1])


def _run_cupy(data: cupy.ndarray) -> cupy.ndarray:
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    _run_gpu[griddim, blockdim](data, out)
    return out


def _run_dask_cupy(data: da.Array) -> da.Array:
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)

    # add any func args
    # TODO: probably needs cellsize args
    _func = partial(_run_cupy)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=cupy.nan,
                           dtype=cupy.float32,
                           meta=cupy.array(()))
    return out


def _run_numpy(data: np.ndarray) -> np.ndarray:
    out = _cpu(data)
    return out


def _run_dask_numpy(data: da.Array) -> da.Array:
    _func = partial(_cpu)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


def aspect(agg: xr.DataArray,
           name: Optional[str] = 'aspect') -> xr.DataArray:
    """
    Calculates, for all cells in the array,
    the downward slope direction of each cell
    based on the elevation of its neighbors in a 3x3 grid.
    The value is measured clockwise in degrees with 0 and 360 at due north.
    Flat areas are given a value of -1.
    Values along the edges are not calculated.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of elevation values. NumPy, CuPy, NumPy-backed Dask,
        or Cupy-backed Dask array.
    name: str, optional (default = "aspect")
        Name of ouput DataArray.

    Returns:
    ----------
    xarray.DataArray
        2D array, of the same type as the input, of calculated aspect values.
        All other input attributes are preserved.

    Notes:
    ----------
    Algorithm References:
    - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm#ESRI_SECTION1_4198691F8852475A9F4BC71246579FAA # noqa
    - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical
    Information Systems (Oxford University Press, New York), pp 406

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Elevation DataArray
    >>> agg = xr.DataArray(np.array([[0, 1, 0, 0],
    >>>                              [1, 1, 0, 0],
    >>>                              [0, 1, 2, 2],
    >>>                              [1, 0, 2, 0],
    >>>                              [0, 2, 2, 2]]),
    >>>                    dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat

    Create Aspect DataArray
    >>> aspect = xrspatial.aspect(agg)
    >>> print(aspect)
    <xarray.DataArray 'aspect' (lat: 5, lon: 4)>
    array([[nan,  nan,  nan,  nan],
           [nan,   0.,  18.43494882,  nan],
           [nan, 270., 341.56505118,  nan],
           [nan, 288.43494882, 315.,  nan],
           [nan,  nan,  nan,  nan]])
    Coordinates:
    * lon      (lon) float64 0.0 1.0 2.0 3.0
    * lat      (lat) float64 0.0 1.0 2.0 3.0 4.0

    Terrain Example:
    - https://makepath.github.io/xarray-spatial/assets/examples/user-guide.html
    """

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy(agg.data)

    # dask + cupy case
    elif has_cuda() and isinstance(agg.data, da.Array) and is_cupy_backed(agg):
        out = _run_dask_cupy(agg.data)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy(agg.data)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
