# std lib
from functools import partial
from typing import Union
from typing import Optional

# 3rd-party
import numpy as np
import pandas as pd
import datashader as ds
import xarray as xr

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da

from numba import cuda
from numba import jit
import numba as nb


# local modules
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import has_cuda
from xrspatial.utils import ngjit
from xrspatial.utils import is_dask_cupy

from .fast_perlin import _perlin
from .fast_perlin import _perlin_gpu


def _scale(value, old_range, new_range):
    d = (value - old_range[0]) / (old_range[1] - old_range[0])
    return d * (new_range[1] - new_range[0]) + new_range[0]


def _gen_terrain(height_map, seed, x_range=(0, 1), y_range=(0, 1)):
    height, width = height_map.shape
    # multiplier, (xfreq, yfreq)
    NOISE_LAYERS = ((1 / 2**i, (2**i, 2**i)) for i in range(16))

    linx = np.linspace(x_range[0], x_range[1], width,
                       endpoint=False, dtype=np.float32)
    liny = np.linspace(y_range[0], y_range[1], height,
                       endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(linx, liny)
    nrange = np.arange(2**20, dtype=int)
    for i, (m, (xfreq, yfreq)) in enumerate(NOISE_LAYERS):
        np.random.seed(seed+i)
        p = np.random.permutation(nrange)
        #p = np.arange(2**20, dtype=int)
        p = np.append(p, p)

        noise = _perlin(p, x * xfreq, y * yfreq) * m

        height_map += noise

    height_map /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
    height_map = height_map ** 3
    return height_map


def _run_numpy(data: np.ndarray,
               width: Union[int, float],
               height: Union[int, float],
               seed: int,
               x_range_scaled: tuple,
               y_range_scaled: tuple,
               zfactor: int) -> np.ndarray:

    # data.fill(0)
    data = data * 0
    data[:] = _gen_terrain(data, seed,
                           x_range=x_range_scaled,
                           y_range=y_range_scaled)

    data = (data - np.min(data))/np.ptp(data)
    data[data < 0.3] = 0  # create water
    data *= zfactor

    return data


def _run_dask_numpy(data: da.Array,
                    width: Union[int, float],
                    height: Union[int, float],
                    seed: int,
                    x_range_scaled: tuple,
                    y_range_scaled: tuple,
                    zfactor: int) -> da.Array:

    data = data * 0

    _func = partial(_gen_terrain, seed=seed,
                    x_range=x_range_scaled, y_range=y_range_scaled)
    data = da.map_blocks(_func, data, meta=np.array((), dtype=np.float32))

    data = (data - da.min(data))/da.ptp(data)
    data[data < 0.3] = 0  # create water
    data *= zfactor

    return data


def _gen_terrain_gpu(height_map, seed, x_range=(0, 1), y_range=(0, 1)):
    height, width = height_map.shape

    NOISE_LAYERS = ((1 / 2**i, (2**i, 2**i)) for i in range(16))

    noise = cupy.empty_like(height_map, dtype=np.float32)
    nrange = cupy.arange(2**20, dtype=int)

    blockdim = (24, 24)
    griddim = (10, 80)
    for i, (m, (xfreq, yfreq)) in enumerate(NOISE_LAYERS):
        cupy.random.seed(seed+i)
        p = cupy.random.permutation(nrange)
        p = cupy.append(p, p)

        _perlin_gpu[griddim, blockdim](p, x_range[0] * xfreq, x_range[1] * xfreq,
                                       y_range[0] * yfreq, y_range[1] * yfreq,
                                       m, noise)

        height_map += noise

    height_map /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
    height_map = height_map ** 3
    return height_map


def _run_cupy(data: cupy.ndarray,
              width: Union[int, float],
              height: Union[int, float],
              seed: int,
              x_range_scaled: tuple,
              y_range_scaled: tuple,
              zfactor: int) -> cupy.ndarray:

    data = data * 0

    data[:] = _gen_terrain_gpu(data, seed,
                               x_range=x_range_scaled,
                               y_range=y_range_scaled)
    minimum = cupy.amin(data)
    maximum = cupy.amax(data)

    data[:] = (data - minimum)/(maximum - minimum)
    data[data < 0.3] = 0  # create water
    data *= zfactor

    return data


def generate_fast_terrain(agg: xr.DataArray,
                          x_range: tuple = (0, 500),
                          y_range: tuple = (0, 500),
                          seed: int = 10,
                          zfactor: int = 4000,
                          full_extent: Optional[str] = None) -> xr.DataArray:
    """
    Generates a pseudo-random terrain which can be helpful for testing
    raster functions.

    Parameters
    ----------
    x_range : tuple, default=(0, 500)
        Range of x values.
    x_range : tuple, default=(0, 500)
        Range of y values.
    seed : int, default=10
        Seed for random number generator.
    zfactor : int, default=4000
        Multipler for z values.
    full_extent : str, default=None
        bbox<xmin, ymin, xmax, ymax>. Full extent of coordinate system.

    Returns
    -------
    terrain : xr.DataArray
        2D array of generated terrain values.

    References
    ----------
        - Michael McHugh: https://www.youtube.com/watch?v=O33YV4ooHSo
        - Red Blob Games: https://www.redblobgames.com/maps/terrain-from-noise/

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import xarray as xr
        import matplotlib.pyplot as plt
        from xrspatial import generate_fast_terrain, aspect

        # Create Canvas
        W = 500
        H = 300
        x_range = (-20e6, 20e6)
        y_range = (-20e6, 20e6)
        seed = 42
        zfactor = 4000

        # Generate Example Terrain
        data = xr.DataArray(np.zeros((H, W), dtype=np.float32),
                            name='terrain')
        terrain_agg = generate_fast_terrain(data, x_range, y_range, seed, zfactor)

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

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
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
    """

    height, width = agg.shape

    mercator_extent = (-np.pi * 6378137, -np.pi * 6378137,
                       np.pi * 6378137, np.pi * 6378137)
    crs_extents = {'3857': mercator_extent}

    if isinstance(full_extent, str):
        full_extent = crs_extents[full_extent]

    elif full_extent is None:
        full_extent = (x_range[0], y_range[0],
                       x_range[1], y_range[1])

    elif not isinstance(full_extent, (list, tuple)) and len(full_extent) != 4:
        raise TypeError('full_extent must be tuple(4) or str wkid')

    full_xrange = (full_extent[0], full_extent[2])
    full_yrange = (full_extent[1], full_extent[3])

    x_range_scaled = (_scale(x_range[0], full_xrange, (0.0, 1.0)),
                      _scale(x_range[1], full_xrange, (0.0, 1.0)))

    y_range_scaled = (_scale(y_range[0], full_yrange, (0.0, 1.0)),
                      _scale(y_range[1], full_yrange, (0.0, 1.0)))

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, width, height, seed,
                         x_range_scaled, y_range_scaled,
                         zfactor)
    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy(agg.data, width, height, seed,
                        x_range_scaled, y_range_scaled,
                        zfactor)
    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy(agg.data, width, height, seed,
                              x_range_scaled, y_range_scaled,
                              zfactor)
    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    hack_agg = ds.Canvas(plot_width=width, plot_height=height,
                         x_range=x_range, y_range=y_range).\
        points(pd.DataFrame({'x': [], 'y': []}), 'x', 'y')
    return xr.DataArray(out, name='terrain', coords=hack_agg.coords,
                        dims=hack_agg.dims, attrs={'res': 1})