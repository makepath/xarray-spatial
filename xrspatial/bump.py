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


def bump(width: int, height: int, count: Optional[int] = None,
         height_func=None, spread: int = 1) -> xr.DataArray:
    """
    Generate a simple bump map to simulate the appearance of land features.
    Using a user-defined height function, determines at what elevation a
    specific bump height is acceptable.
    Bumps of number "count" are applied over the area "width" x "height".

    Parameters:
    ----------
    width: int
        Total width in pixels of the image.
    height: int
        Total height, in pixels, of the image.
    count: int (defaults: w * h / 10)
        Number of bumps to generate.
    height_func: function which takes x, y and returns a height value
        Function used to apply varying bump heights to different elevations.
    spread: tuple boundaries (default = 1)

    Returns:
    ----------
    xarray.DataArray, 2D DataArray of calculated bump heights.

    Notes:
    ----------
    Algorithm References:
        - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import datashader as ds
    >>> from datashader.transfer_functions import shade

    Generate Terrain
    >>> from xrspatial import generate_terrain

    >>> W = 800
    >>> H = 600

    >>> cvs = ds.Canvas(plot_width = W, plot_height = H,
                            x_range = (-20e6, 20e6), y_range = (-20e6, 20e6))
    >>> terrain = generate_terrain(canvas=cvs)

    Create Height Function
    >>> from functools import partial
    >>> from xrspatial import bump

    >>> def heights(locations, src, src_range, height=20):
    >>>     num_bumps = locations.shape[0]
    >>>     out = np.zeros(num_bumps, dtype=np.uint16)
    >>>     for r in range(0, num_bumps):
    >>>         loc = locations[r]
    >>>         x = loc[0]
    >>>         y = loc[1]
    >>>         val = src[y, x]
    >>>         if val >= src_range[0] and val < src_range[1]:
    >>>             out[r] = height
    >>>    return out

    Create Bump Map
    >>> bump_count = 10000
    >>> src = terrain.data
    >>> bumps = bump(W, H, count = bump_count,
                     height_func = partial(heights,
                                           src = src,
                                           src_range = (1000, 1300),
                                           height = 5))
    >>> bumps += bump(W, H, count = bump_count//2,
                     height_func = partial(heights,
                                           src = src,
                                           src_range = (1300, 1700),
                                           height = 20))
    >>> print(bumps)
    <xarray.DataArray (y: 600, x: 800)>
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    Dimensions without coordinates: y, x
    Attributes:
        res:      1

    Terrrain Example:
        - https://makepath.github.io/xarray-spatial/assets/examples/user-guide.html # noqa
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
