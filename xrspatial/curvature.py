# std lib
from functools import partial
from typing import Union
from typing import Optional

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
from xrspatial.utils import not_implemented_func
from xrspatial.utils import ArrayTypeFunctionMapping


@ngjit
def _cpu(data, cellsize):
    out = np.empty(data.shape, np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            d = (data[y + 1, x] + data[y - 1, x]) / 2 - data[y, x]
            e = (data[y, x + 1] + data[y, x - 1]) / 2 - data[y, x]
            out[y, x] = -2 * (d + e) * 100 / (cellsize * cellsize)
    return out


def _run_numpy(data: np.ndarray,
               cellsize: Union[int, float]) -> np.ndarray:
    # TODO: handle border edge effect
    out = _cpu(data, cellsize)
    return out


def _run_dask_numpy(data: da.Array,
                    cellsize: Union[int, float]) -> da.Array:
    _func = partial(_cpu,
                    cellsize=cellsize)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize):
    d = (arr[1, 0] + arr[1, 2]) / 2 - arr[1, 1]
    e = (arr[0, 1] + arr[2, 1]) / 2 - arr[1, 1]
    curv = -2 * (d + e) * 100 / (cellsize[0] * cellsize[0])
    return curv


@cuda.jit
def _run_gpu(arr, cellsize, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1], cellsize)


def _run_cupy(data: cupy.ndarray,
              cellsize: Union[int, float]) -> cupy.ndarray:

    cellsize_arr = cupy.array([float(cellsize)], dtype='f4')

    # TODO: add padding
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data, cellsize_arr, out)

    return out


def curvature(agg: xr.DataArray,
              name: Optional[str] = 'curvature') -> xr.DataArray:
    """
    Calculates, for all cells in the array, the curvature (second
    derivative) of each cell based on the elevation of its neighbors
    in a 3x3 grid. A positive curvature indicates the surface is
    upwardly convex. A negative value indicates it is upwardly
    concave. A value of 0 indicates a flat surface.

    Units of the curvature output raster are one hundredth (1/100)
    of a z-unit.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of elevation values.
        Must contain `res` attribute.
    name : str, default='curvature'
        Name of output DataArray.

    Returns
    -------
    curvature_agg : xarray.DataArray, of the same type as `agg`
        2D aggregate array of curvature values.
        All other input attributes are preserved.

    References
    ----------
        - arcgis: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-curvature-works.htm # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain, curvature


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

        # Create Curvature Aggregate Array
        curvature_agg = curvature(agg = terrain_agg, name = 'Curvature')

        # Edit Attributes
        curvature_agg = curvature_agg.assign_attrs(
            {
                'Description': 'Curvature',
                'units': 'rad',
            }
        )

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Curvature
        curvature_agg.plot(aspect = 2, size = 4)
        plt.title("Curvature")
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

    .. sourcecode:: python

        >>> print(curvature_agg[200:203, 200:202])
        <xarray.DataArray 'Curvature' (lat: 3, lon: 2)>
        array([[-8.14280993e-08, -5.46873377e-08],
               [ 2.85253552e-08,  1.24471207e-08],
               [ 8.91172878e-08,  1.58492666e-07]])
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
            Description:    Curvature
            units:          rad
            Max Elevation:  4000
    """
    cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    cellsize = (cellsize_x + cellsize_y) / 2

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=not_implemented_func,
    )
    out = mapper(agg)(agg.data, cellsize)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
