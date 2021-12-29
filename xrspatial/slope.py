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
from xrspatial.utils import ngjit
from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import not_implemented_func


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
    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain, slope


        # Generate Example Terrain
        W = 500
        H = 300

        template_terrain = xr.DataArray(np.zeros((H, W)))
        x_range=(-20e6, 20e6)
        y_range=(-20e6, 20e6)

        terrain_agg = generate_terrain(
            template_terrain, x_range=x_range, y_range=y_range
        )

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

        # Create Slope Aggregate Array
        slope_agg = slope(agg = terrain_agg, name = 'Slope')

        # Edit Attributes
        slope_agg = slope_agg.assign_attrs({'Description': 'Example Slope',
                                            'units': 'deg'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Slope
        slope_agg.plot(aspect = 2, size = 4)
        plt.title("Slope")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02296597, 1261.947921  ],
               [1285.37105519, 1282.48079719],
               [1306.02339636, 1303.4069579 ]])
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(slope_agg[200:203, 200:202])
        <xarray.DataArray 'Slope' (lat: 3, lon: 2)>
        array([[0.00757718, 0.00726441],
               [0.00893266, 0.00916095],
               [0.00773291, 0.00699103]], dtype=float32)
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
            Description:    Example Slope
            units:          deg
            Max Elevation:  4000
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
