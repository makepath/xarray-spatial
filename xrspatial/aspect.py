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
    Calculates the aspect value of an elevation aggregate.

    Calculates, for all cells in the array, the downward slope direction
    of each cell based on the elevation of its neighbors in a 3x3 grid.
    The value is measured clockwise in degrees with 0 and 360 at due
    north. Flat areas are given a value of -1. Values along the edges
    are not calculated.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
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
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain, aspect

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )
        
        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Create Aspect Aggregate Array
        aspect_agg = aspect(agg = terrain_agg, name = 'Aspect')

        # Edit Attributes
        aspect_agg = aspect_agg.assign_attrs(
            {
                'Description': 'Example Aspect',
                'units': 'deg',
            }
        )

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Aspect
        aspect_agg.plot(aspect = 2, size = 4)
        plt.title("Aspect")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
               [1285.37061171, 1282.48046696],
               [1306.02305679, 1303.40657515]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

    .. sourcecode:: python

        >>> print(aspect_agg[200:203, 200:202])
        <xarray.DataArray 'Aspect' (lat: 3, lon: 2)>
        array([[ 8.18582638,  8.04675084],
               [ 5.49302641,  9.86625477],
               [12.04270534, 16.87079619]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Aspect
            units:          deg
            Max Elevation:  4000
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
