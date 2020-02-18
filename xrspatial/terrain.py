import numpy as np
import pandas as pd

from xarray import DataArray
from .perlin import _perlin


# TODO: add optional name parameter `name='terrain'`
def generate_terrain(x_range=(0, 500), y_range=(0, 500),
                     width=25, height=30, canvas=None,
                     seed=10, zfactor=4000, full_extent=None):
    """
    Generates a pseudo-random terrain which can be helpful
    for testing raster functions

    Parameters
    ----------
    canvas : ds.Canvas instance for passing output dimensions / ranges

    seed : seed for random number generator

    zfactor : used as multipler for z values

    full_extent : optional string, bbox<xmin, ymin, xmax, ymax>
      full extent of coordinate system.

    Returns
    -------
    terrain: DataArray

    Notes:
    ------
    Algorithm References:
     - This was inspired by Michael McHugh's 2016 PyCon Canada talk:
       https://www.youtube.com/watch?v=O33YV4ooHSo
     - https://www.redblobgames.com/maps/terrain-from-noise/
    """

    def _gen_heights(bumps):
        out = np.zeros(len(bumps))
        for i, b in enumerate(bumps):
            x = b[0]
            y = b[1]
            val = agg.data[y, x]
            if val >= 0.33 and val <= 3:
                out[i] = 0.1
        return out

    def _scale(value, old_range, new_range):
        d = (value - old_range[0]) / (old_range[1] - old_range[0])
        return d * (new_range[1] - new_range[0]) + new_range[0]




    mercator_extent = (-np.pi * 6378137, -np.pi * 6378137, np.pi * 6378137, np.pi * 6378137)
    crs_extents = {'3857': mercator_extent}

    if isinstance(full_extent, str):
        full_extent = crs_extents[full_extent]

    elif full_extent is None:
        full_extent = (canvas.x_range[0], canvas.y_range[0],
                       canvas.x_range[1], canvas.y_range[1])

    elif not isinstance(full_extent, (list, tuple)) and len(full_extent) != 4:
        raise TypeError('full_extent must be tuple(4) or str wkid')

    full_xrange = (full_extent[0], full_extent[2])
    full_yrange = (full_extent[1], full_extent[3])

    x_range_scaled = (_scale(canvas.x_range[0], full_xrange, (0.0, 1.0)),
                      _scale(canvas.x_range[1], full_xrange, (0.0, 1.0)))

    y_range_scaled = (_scale(canvas.y_range[0], full_yrange, (0.0, 1.0)),
                      _scale(canvas.y_range[1], full_yrange, (0.0, 1.0)))

    data = _gen_terrain(canvas.plot_width, canvas.plot_height, seed,
                        x_range=x_range_scaled, y_range=y_range_scaled)

    data = (data - np.min(data))/np.ptp(data)
    data[data < 0.3] = 0  # create water
    data *= zfactor

    # DataArray coords were coming back different from cvs.points...
    hack_agg = canvas.points(pd.DataFrame({'x': [], 'y': []}), 'x', 'y')
    agg = DataArray(data,
                    name='terrain',
                    coords=hack_agg.coords,
                    dims=hack_agg.dims,
                    attrs={'res': 1})

    return agg


def _gen_terrain(width, height, seed, x_range=None, y_range=None):

    if not x_range:
        x_range = (0, 1)

    if not y_range:
        y_range = (0, 1)

    # multiplier, (xfreq, yfreq)
    NOISE_LAYERS = ((1 / 2**i, (2**i, 2**i)) for i in range(16))

    linx = np.linspace(x_range[0], x_range[1], width, endpoint=False)
    liny = np.linspace(y_range[0], y_range[1], height, endpoint=False)
    x, y = np.meshgrid(linx, liny)

    height_map = None
    for i, (m, (xfreq, yfreq)) in enumerate(NOISE_LAYERS):
        noise = _perlin(x * xfreq, y * yfreq, seed=seed + i) * m
        if height_map is None:
            height_map = noise
        else:
            height_map += noise

    height_map /= (1.00 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
    height_map = height_map ** 3
    return height_map
