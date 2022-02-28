from typing import Optional

import numpy as np
import xarray as xr
from xarray import DataArray

from xrspatial.utils import ngjit

# TODO: change parameters to take agg instead of height / width


@ngjit
def _finish_bump(width, height, locs, heights, spread):
    out = np.zeros((height, width))
    rows, cols = out.shape
    s = spread ** 2  # removed sqrt for perf.
    for i in range(len(heights)):
        x = locs[i][0]
        y = locs[i][1]
        z = heights[i]
        out[y, x] = out[y, x] + z
        if s > 0:
            for nx in range(max(x - spread, 0), min(x + spread, width)):
                for ny in range(max(y - spread, 0), min(y + spread, height)):
                    d2 = (nx - x) * (nx - x) + (ny - y) * (ny - y)
                    if d2 <= s:
                        out[ny, nx] = out[ny, nx] + (out[y, x] * (d2 / s))
    return out


def bump(width: int,
         height: int,
         count: Optional[int] = None,
         height_func=None,
         spread: int = 1) -> xr.DataArray:
    """
    Generate a simple bump map to simulate the appearance of land
    features.

    Using a user-defined height function, determines at what elevation
    a specific bump height is acceptable. Bumps of number `count` are
    applied over the area `width` x `height`.

    Parameters
    ----------
    width : int
        Total width, in pixels, of the image.
    height : int
        Total height, in pixels, of the image.
    count : int
        Number of bumps to generate.
    height_func : function which takes x, y and returns a height value
        Function used to apply varying bump heights to different
        elevations.
    spread : int, default=1
        Number of pixels to spread on all sides.

    Returns
    -------
    bump_agg : xarray.DataArray
        2D aggregate array of calculated bump heights.

    References
    ----------
        - ICA: http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf # noqa

    Examples
    --------
    .. plot::
       :include-source:

        from functools import partial

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain, bump


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

        # Create Height Function
        def heights(locations, src, src_range, height = 20):
            num_bumps = locations.shape[0]
            out = np.zeros(num_bumps, dtype = np.uint16)
            for r in range(0, num_bumps):
                loc = locations[r]
                x = loc[0]
                y = loc[1]
                val = src[y, x]
                if val >= src_range[0] and val < src_range[1]:
                    out[r] = height
            return out

        # Create Bump Map Aggregate Array
        bump_count = 10000
        src = terrain_agg.data

        # Short Bumps from z = 1000 to z = 1300
        bump_agg = bump(width = W, height = H, count = bump_count,
                        height_func = partial(heights, src = src,
                                            src_range = (1000, 1300),
                                            height = 5))

        # Tall Bumps from z = 1300 to z = 1700
        bump_agg += bump(width = W, height = H, count = bump_count // 2,
                        height_func = partial(heights, src = src,
                                            src_range = (1300, 1700),
                                            height=20))

        # Short Bumps from z = 1700 to z = 2000
        bump_agg += bump(width = W, height = H, count = bump_count // 3,
                        height_func = partial(heights, src = src,
                                            src_range = (1700, 2000),
                                            height=5))
        # Edit Attributes
        bump_agg = bump_agg.assign_attrs({'Description': 'Example Bump Map',
                                          'units': 'km'})

        bump_agg = bump_agg.rename('Bump Height')

        # Rename Coordinates
        bump_agg = bump_agg.assign_coords({'x': terrain_agg.coords['lon'].data,
                                           'y': terrain_agg.coords['lat'].data})

        # Remove zeros
        bump_agg.data[bump_agg.data == 0] = np.nan

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Bump Map
        bump_agg.plot(cmap = 'summer', aspect = 2, size = 4)
        plt.title("Bump Map")
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

        >>> print(bump_agg[200:205, 200:206])
        <xarray.DataArray 'Bump Height' (y: 5, x: 6)>
        array([[nan, nan, nan, nan,  5.,  5.],
               [nan, nan, nan, nan, nan,  5.],
               [nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan]])
        Coordinates:
        * x        (x) float64 -3.96e+06 -3.88e+06 -3.8e+06 ... -3.64e+06 -3.56e+06
        * y        (y) float64 6.733e+06 6.867e+06 7e+06 7.133e+06 7.267e+06
        Attributes:
            res:          1
            Description:  Example Bump Map
            units:        km
    """
    linx = range(width)
    liny = range(height)

    if count is None:
        count = width * height // 10

    if height_func is None:
        height_func = lambda bumps: np.ones(len(bumps)) # noqa

    # create 2d array of random x, y for bump locations
    locs = np.empty((count, 2), dtype=np.uint16)
    locs[:, 0] = np.random.choice(linx, count)
    locs[:, 1] = np.random.choice(liny, count)

    heights = height_func(locs)

    bumps = _finish_bump(width, height, locs, heights, spread)
    return DataArray(bumps, dims=['y', 'x'], attrs=dict(res=1))
