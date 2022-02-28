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
import numpy as np
import xarray as xr
from numba import cuda

# local modules
from xrspatial.utils import (ArrayTypeFunctionMapping, cuda_args, get_dataarray_resolution, ngjit,
                             not_implemented_func)


@ngjit
def _cpu(data, cellsize_x, cellsize_y):
    data = data.astype(np.float32)
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
    data = data.astype(np.float32)
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
    data = data.astype(cupy.float32)

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data,
                                cellsize_x_arr,
                                cellsize_y_arr,
                                out)
    return out


def slope(agg: xr.DataArray,
          name: str = 'slope') -> xr.DataArray:
    """
    Returns slope of input aggregate in degrees.

    Parameters
    ----------
    agg : xr.DataArray
        2D array of elevation data.
    name : str, default='slope'
        Name of output DataArray.

    Returns
    -------
    slope_agg : xr.DataArray of same type as `agg`
        2D array of slope values.
        All other input attributes are preserved.

    References
    ----------
        - arcgis: http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import slope
        >>> data = np.array([
        ...     [0, 0, 0, 0, 0],
        ...     [0, 0, 0, -1, 2],
        ...     [0, 0, 0, 0, 1],
        ...     [0, 0, 0, 5, 0]])
        >>> agg = xr.DataArray(data)
        >>> slope_agg = slope(agg)
        >>> slope_agg
        <xarray.DataArray 'slope' (dim_0: 4, dim_1: 5)>
        array([[      nan,       nan,       nan,       nan,       nan],
               [      nan,  0.      , 14.036243, 32.512516,       nan],
               [      nan,  0.      , 42.031113, 53.395725,       nan],
               [      nan,       nan,       nan,       nan,       nan]],
              dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='slope() does not support dask with cupy backed DataArray'  # noqa
        ),
    )
    out = mapper(agg)(agg.data, cellsize_x, cellsize_y)

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
