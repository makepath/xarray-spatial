# std lib
from functools import partial
from math import atan
from typing import Union

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da

from numba import cuda

import numpy as np
import xarray as xr

# local modules
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.utils import is_dask_cupy


@ngjit
def _cpu(data, cellsize_x, cellsize_y):
    out = np.zeros_like(data, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            a = data[y + 1, x - 1]
            b = data[y + 1, x]
            c = data[y + 1, x + 1]
            d = data[y, x - 1]
            f = data[y, x + 1]
            g = data[y - 1, x - 1]
            h = data[y - 1, x]
            i = data[y - 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** .5
            out[y, x] = np.arctan(p) * 57.29578
    return out


def _run_numpy(data: np.ndarray,
               cellsize_x: Union[int, float],
               cellsize_y: Union[int, float]) -> np.ndarray:
    out = _cpu(data, cellsize_x, cellsize_y)
    return out


def _run_dask_numpy(data: da.Array,
                    cellsize_x: Union[int, float],
                    cellsize_y: Union[int, float]) -> da.Array:
    _func = partial(_cpu,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize_x, cellsize_y):
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
    p = (dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5
    return atan(p) * 57.29578


@cuda.jit
def _run_gpu(arr, cellsize_x_arr, cellsize_y_arr, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di < out.shape[0] and
            j - dj >= 0 and j + dj < out.shape[1]):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1],
                         cellsize_x_arr,
                         cellsize_y_arr)


def _run_cupy(data: cupy.ndarray,
              cellsize_x: Union[int, float],
              cellsize_y: Union[int, float]) -> cupy.ndarray:
    cellsize_x_arr = cupy.array([float(cellsize_x)], dtype='f4')
    cellsize_y_arr = cupy.array([float(cellsize_y)], dtype='f4')

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data,
                                cellsize_x_arr,
                                cellsize_y_arr,
                                out)
    return out


def _run_dask_cupy(data: da.Array,
                   cellsize_x: Union[int, float],
                   cellsize_y: Union[int, float]) -> da.Array:
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)

    _func = partial(_run_cupy,
                    cellsize_x=cellsize_x,
                    cellsize_y=cellsize_y)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=cupy.nan,
                           dtype=cupy.float32,
                           meta=cupy.array(()))
    return out


def slope(agg: xr.DataArray,
          name: str = 'slope') -> xr.DataArray:
    """
    Returns slope of input aggregate in degrees.

    Parameters:
    -----------
    agg: xarray.DataArray
        2D array of elevation band data.
    name: str, optional (default = 'slope')
        name property of output xarray.DataArray

    Returns:
    ---------
    xarray.DataArray
        2D array, of the same type as the input, of calculated slope values.
        All other input attributes are preserved.

    Notes:
    ------
    Algorithm References:
    - http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm # noqa
    - Burrough, P. A., and McDonell, R. A., 1998. Principles of Geographical
    Information Systems (Oxford University Press, New York), pp 406

    Examples:
    ---------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import slope

    Create Data Array
    >>> agg = xr.DataArray(np.array([[0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 2, 4, 0, 8, 0],
    >>>                              [0, 2, 2, 4, 6, 8, 0],
    >>>                              [0, 4, 4, 4, 6, 8, 0],
    >>>                              [0, 6, 6, 6, 6, 8, 0],
    >>>                              [0, 8, 8, 8, 8, 8, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0]]),
    >>>                     dims = ["lat", "lon"],
    >>>                     attrs = dict(res = 1))
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat

    Create Slope Data Array
    >>> print(slope(agg))
    <xarray.DataArray 'slope' (lat: 7, lon: 7)>
    array([[ 0,  0,  0,  0,  0,  0,  0],
           [ 0, 46, 60, 63, 73, 70,  0],
           [ 0, 60, 54, 54, 68, 67,  0],
           [ 0, 68, 60, 54, 60, 71,  0],
           [ 0, 73, 63, 60, 54, 72,  0],
           [ 0, 74, 71, 71, 72, 75,  0],
           [ 0,  0,  0,  0,  0,  0,  0]])
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0
    Attributes:
        res:      1
    """

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, cellsize_x, cellsize_y)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy(agg.data, cellsize_x, cellsize_y)

    # dask + cupy case
    elif has_cuda() and is_dask_cupy(agg):
        out = _run_dask_cupy(agg.data, cellsize_x, cellsize_y)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy(agg.data, cellsize_x, cellsize_y)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
