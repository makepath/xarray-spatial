from functools import partial
from math import atan2
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.utils import ArrayTypeFunctionMapping, cuda_args, ngjit, not_implemented_func

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

RADIAN = 180 / np.pi


@ngjit
def _run_numpy(data: np.ndarray):
    data = data.astype(np.float32)
    out = np.zeros_like(data, dtype=np.float32)
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
                _aspect = np.arctan2(dz_dy, -dz_dx) * RADIAN
                # convert to compass direction values (0-360 degrees)
                if _aspect < 0:
                    out[y, x] = 90.0 - _aspect
                elif _aspect > 90.0:
                    out[y, x] = 360.0 - _aspect + 90.0
                else:
                    out[y, x] = 90.0 - _aspect

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

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

    if dz_dx == 0 and dz_dy == 0:
        # flat surface, slope = 0, thus invalid aspect
        _aspect = -1
    else:
        _aspect = atan2(dz_dy, -dz_dx) * 57.29578
        # convert to compass direction values (0-360 degrees)
        if _aspect < 0:
            _aspect = 90 - _aspect
        elif _aspect > 90:
            _aspect = 360 - _aspect + 90
        else:
            _aspect = 90 - _aspect

    if _aspect > 359.999:  # lame float equality check...
        return 0
    else:
        return _aspect


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
    data = data.astype(cupy.float32)
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan
    _run_gpu[griddim, blockdim](data, out)
    return out


def _run_dask_numpy(data: da.Array) -> da.Array:
    data = data.astype(np.float32)
    _func = partial(_run_numpy)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


def aspect(agg: xr.DataArray,
           name: Optional[str] = 'aspect') -> xr.DataArray:
    """
    Calculates the aspect value of an elevation aggregate.

    Calculates, for all cells in the array, the downward slope direction
    of each cell based on the elevation of its neighbors in a 3x3 grid.
    The value is measured clockwise in degrees with 0 (due north), and 360
    (again due north). Values along the edges are not calculated.

    Direction of the aspect can be determined by its value:
    From 0     to 22.5:  North
    From 22.5  to 67.5:  Northeast
    From 67.5  to 112.5: East
    From 112.5 to 157.5: Southeast
    From 157.5 to 202.5: South
    From 202.5 to 247.5: West
    From 247.5 to 292.5: Northwest
    From 337.5 to 360:   North

    Note that values of -1 denote flat areas.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, or Dask with NumPy-backed xarray DataArray
        of elevation values.
    name : str, default='aspect'
        Name of ouput DataArray.

    Returns
    -------
    aspect_agg : xarray.DataArray of the same type as `agg`
        2D aggregate array of calculated aspect values.
        All other input attributes are preserved.

    References
    ----------
        - arcgis: http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm#ESRI_SECTION1_4198691F8852475A9F4BC71246579FAA # noqa

    Examples
    --------
    Aspect works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import aspect

        >>> data = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 2, 0],
            [1, 1, 1, 0, 0],
            [4, 4, 9, 2, 4],
            [1, 5, 0, 1, 4],
            [1, 5, 0, 5, 5]
        ], dtype=np.float32)
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> print(raster)
        <xarray.DataArray 'raster' (y: 6, x: 5)>
        array([[1., 1., 1., 1., 1.],
               [1., 1., 1., 2., 0.],
               [1., 1., 1., 0., 0.],
               [4., 4., 9., 2., 4.],
               [1., 5., 0., 1., 4.],
               [1., 5., 0., 5., 5.]])
        Dimensions without coordinates: y, x
        >>> aspect_agg = aspect(raster)
        >>> print(aspect_agg)
        <xarray.DataArray 'aspect' (y: 6, x: 5)>
        array([[ nan,  nan        ,   nan       ,   nan       , nan],
               [ nan,  -1.        ,   225.      ,   135.      , nan],
               [ nan, 343.61045967,   8.97262661,  33.69006753, nan],
               [ nan, 307.87498365,  71.56505118,  54.46232221, nan],
               [ nan, 191.30993247, 144.46232221, 255.96375653, nan],
               [ nan,  nan        ,   nan       ,   nan       , nan]])
        Dimensions without coordinates: y, x

    Aspect works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import dask.array as da
        >>> data_da = da.from_array(data, chunks=(3, 3))
        >>> raster_da = xr.DataArray(data_da, dims=['y', 'x'], name='raster_da')
        >>> print(raster_da)
        <xarray.DataArray 'raster' (y: 6, x: 5)>
        dask.array<array, shape=(6, 5), dtype=int64, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: y, x
        >>> aspect_da = aspect(raster_da)
        >>> print(aspect_da)
        <xarray.DataArray 'aspect' (y: 6, x: 5)>
        dask.array<_trim, shape=(6, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: y, x
        >>> print(aspect_da.compute())  # compute the results
        <xarray.DataArray 'aspect' (y: 6, x: 5)>
        array([[ nan,  nan        ,   nan       ,   nan       , nan],
               [ nan,  -1.        ,   225.      ,   135.      , nan],
               [ nan, 343.61045967,   8.97262661,  33.69006753, nan],
               [ nan, 307.87498365,  71.56505118,  54.46232221, nan],
               [ nan, 191.30993247, 144.46232221, 255.96375653, nan],
               [ nan,  nan        ,   nan       ,   nan       , nan]])
        Dimensions without coordinates: y, x

    Aspect works with CuPy backed xarray DataArray.
    Make sure you have a GPU and CuPy installed to run this example.
    .. sourcecode:: python

        >>> import cupy
        >>> data_cupy = cupy.asarray(data)
        >>> raster_cupy = xr.DataArray(data_cupy, dims=['y', 'x'])
        >>> aspect_cupy = aspect(raster_cupy)
        >>> print(type(aspect_cupy.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(aspect_cupy)
        <xarray.DataArray 'aspect' (y: 6, x: 5)>
        array([[       nan,       nan,        nan,        nan,        nan],
               [       nan,       -1.,       225.,       135.,        nan],
               [       nan, 343.61047,   8.972626,  33.690067,        nan],
               [       nan, 307.87497,  71.56505 ,  54.462322,        nan],
               [       nan, 191.30994, 144.46233 ,  255.96376,        nan],
               [       nan,       nan,        nan,        nan,        nan]],
              dtype=float32)
        Dimensions without coordinates: y, x
    """
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        dask_func=_run_dask_numpy,
        cupy_func=_run_cupy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='aspect() does not support dask with cupy backed DataArray')  # noqa
    )

    out = mapper(agg)(agg.data)

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
