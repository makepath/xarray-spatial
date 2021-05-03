import numpy as np
import pandas as pd
import datashader as ds
from typing import Optional
import xarray as xr

from xarray import DataArray
from .perlin import _perlin


# TODO: add optional name parameter `name='terrain'`
def generate_terrain(x_range: tuple = (0, 500),
                     y_range: tuple = (0, 500),
                     width: int = 25,
                     height: int = 30,
                     canvas: ds.Canvas = None,
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
    width : int, default=25
        Width of output data array in pixels.
    height : int, default=30
        Height of output data array in pixels.
    canvas : ds.Canvas, default=None
        Instance for passing output dimensions / ranges.
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

    mercator_extent = (-np.pi * 6378137, -np.pi * 6378137,
                       np.pi * 6378137, np.pi * 6378137)
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
