import numpy as np

from xarray import DataArray

from xrspatial.utils import ngjit


# TODO: change parameters to take agg instead of height / width
def bump(width, height, count=None, height_func=None, spread=1):
    """
    Generate a simple bump map

    Parameters
    ----------
    width : int
    height : int
    count : int (defaults: w * h / 10)
    height_func : function which takes x, y and returns a height value
    spread : tuple boundaries

    Returns
    -------
    bumpmap: DataArray

    Notes:
    ------
    Algorithm References:
     - http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf
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
                    d2 = (nx - x) * (nx - x) + (ny -  y) * (ny - y)
                    if d2 <= s:
                        out[ny, nx] = out[ny,nx] + (out[y, x] * (d2 / s))
    return out
