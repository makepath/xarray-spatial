"""
Multi-observer viewshed and line-of-sight profile tools.

Functions
---------
cumulative_viewshed
    Count how many observers can see each cell.
visibility_frequency
    Fraction of observers that can see each cell.
line_of_sight
    Elevation profile and visibility along a straight line between two points.
"""

import numpy as np

from .utils import has_cuda_and_cupy, has_dask_array, is_cupy_array


def _bresenham_line(r0, c0, r1, c1):
    """Return list of (row, col) cells along the line from (r0,c0) to (r1,c1).

    Uses Bresenham's line algorithm. Both endpoints are included.
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def _extract_transect(raster, cells):
    """Extract elevation, x-coords, and y-coords for a list of (row, col) cells.

    For dask or cupy-backed rasters the values are pulled to numpy.
    Returns (elevations, x_coords, y_coords) as 1-D numpy arrays.
    """
    rows = np.array([r for r, c in cells])
    cols = np.array([c for r, c in cells])

    x_coords = raster.coords['x'].values[cols]
    y_coords = raster.coords['y'].values[rows]

    data = raster.data
    if has_dask_array():
        import dask.array as da
        if isinstance(data, da.Array):
            data = data.compute()
    if has_cuda_and_cupy() and is_cupy_array(data):
        data = data.get()

    elevations = data[rows, cols].astype(np.float64)
    return elevations, x_coords, y_coords
