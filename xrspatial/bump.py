import numpy as np

import xarray as xr
from xarray import DataArray

from xrspatial.utils import ngjit

from typing import Optional

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
        height_func = None,
        spread: int = 1) -> xr.DataArray:
    """Generate a simple bump map to simulate the appearance of land features.

    Using a user-defined height function, determines at what elevation a
    specific bump height is acceptable. Bumps of number "count" are applied
    over the area "width" x "height".

    Parameters
    ----------
    width : int
        Total width, in pixels, of the image.
    height : int
        Total height, in pixels, of the image.
    count : int, default: w * h / 10
        Number of bumps to generate.
    height_func : function which takes x, y and returns a height value
        Function used to apply varying bump heights to different elevations.
    spread : tuple boundaries, default = 1

    Returns
    -------
    bump_agg : xarray.DataArray
        2D aggregate array of calculated bump heights.

    Notes
    -----
    Algorithm References
        - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf
    Terrrain Example
        - https://makepath.github.io/xarray-spatial/assets/examples/user-guide.html

    Example
    -------
    >>>     import datashader as ds
    >>>     from xrspatial import generate_terrain, bump
    >>>     from datashader.transfer_functions import shade, stack, set_background
    >>>     from datashader.colors import Elevation
    >>>     from functools import partial
    >>>     import numpy as np

    >>>     # Create Canvas
    >>>     W = 500 
    >>>     H = 300
    >>>     cvs = ds.Canvas(plot_width = W,
    >>>                     plot_height = H,
    >>>                     x_range = (-20e6, 20e6),
    >>>                     y_range = (-20e6, 20e6))
    >>>     # Generate Example Terrain
    >>>     terrain_agg = generate_terrain(canvas = cvs)
    >>>     terrain_agg = terrain_agg.assign_attrs({'Description': 'Elevation',
    >>>                                             'Max Elevation': '3000',
    >>>                                             'units': 'meters'})
    >>>     terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
    >>>     terrain_agg = terrain_agg.rename('example_terrain')
    >>>     # Shade Terrain
    >>>     terrain_img = shade(agg = terrain_agg,
    >>>                         cmap = ['grey', 'white'],
    >>>                         how = 'linear')
    >>>     print(terrain_agg[200:203, 200:202])
    >>>     terrain_img
    ...     <xarray.DataArray 'example_terrain' (lat: 3, lon: 2)>
    ...     array([[1264.02249454, 1261.94748873],
    ...            [1285.37061171, 1282.48046696],
    ...            [1306.02305679, 1303.40657515]])
    ...     Coordinates:
    ...       * lon      (lon) float64 -3.96e+06 -3.88e+06
    ...       * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
    ...     Attributes:
    ...         res:            1
    ...         Description:    Elevation
    ...         Max Elevation:  3000
    ...         units:          meters

            .. image :: ./docs/source/_static/img/docstring/terrain_example_grey.png

    >>>      # Create Height Function
    >>>     def heights(locations, src, src_range, height = 20):
    >>>         num_bumps = locations.shape[0]
    >>>         out = np.zeros(num_bumps, dtype = np.uint16)
    >>>         for r in range(0, num_bumps):
    >>>             loc = locations[r]
    >>>             x = loc[0]
    >>>             y = loc[1]
    >>>             val = src[y, x]
    >>>             if val >= src_range[0] and val < src_range[1]:
    >>>                 out[r] = height
    >>>         return out

    >>>     # Create Bump Map Aggregate Array
    >>>     bump_count = 10000
    >>>     src = terrain_agg.data
    >>>     # Short Bumps from z = 1000 to z = 1300
    >>>     bump_agg = bump(width = W, height = H, count = bump_count,
    >>>                     height_func = partial(heights, src = src,
    >>>                                           src_range = (1000, 1300),
    >>>                                           height = 5))
    >>>     # Tall Bumps from z = 1300 to z = 1700
    >>>     bump_agg += bump(width = W, height = H, count = bump_count // 2,
    >>>                      height_func = partial(heights, src = src,
    >>>                                            src_range = (1300, 1700),
    >>>                                            height=20))
    >>>     # Short Bumps from z = 1700 to z = 2000
    >>>     bump_agg += bump(width = W, height = H, count = bump_count // 3,
    >>>                      height_func = partial(heights,  src = src,
    >>>                                            src_range = (1700, 2000),
    >>>                                            height=5))
    >>>     # Remove zeros
    >>>     bump_agg_color = bump_agg.copy()
    >>>     bump_agg_color.data[bump_agg_color.data == 0] = np.nan
    >>>     # Shade Image
    >>>     bump_img = shade(agg = bump_agg_color,
    >>>                      cmap = 'green',
    >>>                      how ='linear',
    >>>                      alpha = 255)
    >>>     print(bump_agg_color[200:205, 200:206])
    >>>     bump_img_background = set_background(bump_img, 'black')
    >>>     bump_img_background
    ...     <xarray.DataArray (y: 5, x: 6)>
    ...     array([[ 5.,  5.,  5.,  5.,  5., nan],
    ...            [nan,  5.,  5., nan, nan, nan],
    ...            [nan, nan, nan, nan, nan, nan],
    ...            [nan, nan, nan, nan, nan, nan],
    ...            [nan, nan, nan, nan, nan, nan]])
    ...     Dimensions without coordinates: y, x
    ...     Attributes:
    ...         res:      1

            .. image :: ./docs/source/_static/img/docstring/bump_example.png

    >>>     # Combine Images
    >>>     composite_img = stack(terrain_img, hillshade_img, bump_img)
    >>>     composite_img

            .. image :: ./docs/source/_static/img/docstring/bump_composite.png

    """

    linx = range(width)
    liny = range(height)

    if count is None:
        count = width * height // 10

    if height_func is None:
        height_func = lambda bumps: np.ones(len(bumps))

    # create 2d array of random x, y for bump locations
    locs = np.empty((count, 2), dtype=np.uint16)
    locs[:, 0] = np.random.choice(linx, count)
    locs[:, 1] = np.random.choice(liny, count)

    heights = height_func(locs)

    bumps = _finish_bump(width, height, locs, heights, spread)
    return DataArray(bumps, dims=['y', 'x'], attrs=dict(res=1))
