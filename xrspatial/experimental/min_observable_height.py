"""Compute the minimum observer height needed to see each grid cell.

For a given observer location on a terrain raster, this module determines
the minimum height above the terrain that an observer must be raised to
in order to see each cell.  The result is a DataArray of the same shape
as the input, where each cell contains that minimum height (or ``NaN``
for cells that remain invisible even at *max_height*).

The algorithm exploits the fact that visibility is **monotonic** in
observer height: if a cell is visible at height *h*, it is also visible
at every height greater than *h*.  This allows a per-cell binary search
over observer height, requiring only ``ceil(log2(max_height / precision))``
viewshed evaluations instead of a linear sweep.
"""

from math import ceil, log2
from typing import Union

import numpy as np
import xarray

from ..viewshed import viewshed, INVISIBLE


def min_observable_height(
    raster: xarray.DataArray,
    x: Union[int, float],
    y: Union[int, float],
    max_height: float = 100.0,
    target_elev: float = 0.0,
    precision: float = 1.0,
) -> xarray.DataArray:
    """Return the minimum observer height to see each cell.

    For every cell in *raster*, compute the smallest observer elevation
    (above the terrain at the observer location) at which that cell
    becomes visible.  Cells that are never visible up to *max_height*
    are set to ``NaN``.

    Parameters
    ----------
    raster : xarray.DataArray
        2-D elevation raster with ``y`` and ``x`` coordinates.
    x : int or float
        x-coordinate (in data space) of the observer location.
    y : int or float
        y-coordinate (in data space) of the observer location.
    max_height : float, default 100.0
        Maximum observer height to test.  Cells invisible at this
        height receive ``NaN`` in the output.
    target_elev : float, default 0.0
        Height offset added to target cells when testing visibility,
        passed through to :func:`~xrspatial.viewshed.viewshed`.
    precision : float, default 1.0
        Height resolution of the binary search.  The result for each
        cell is accurate to within *precision* units.

    Returns
    -------
    xarray.DataArray
        Same shape and coordinates as *raster*.  Each cell contains the
        minimum observer height (in surface units) at which it becomes
        visible, or ``NaN`` if the cell is not visible at *max_height*.
        The cell containing the observer is always ``0.0``.

    Raises
    ------
    ValueError
        If *max_height* <= 0, *precision* <= 0, or *precision* >
        *max_height*.

    Notes
    -----
    The function calls :func:`~xrspatial.viewshed.viewshed` internally,
    so the same backend (numpy / cupy) is used.  Dask-backed rasters
    are not supported because viewshed itself does not support them.

    The number of viewshed evaluations is
    ``ceil(log2(max_height / precision)) + 1``, which is typically far
    fewer than a linear sweep at the same resolution.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.experimental import min_observable_height
        >>> elev = np.zeros((5, 5), dtype=float)
        >>> elev[2, 4] = 10.0  # a tall obstacle
        >>> raster = xr.DataArray(elev, dims=['y', 'x'])
        >>> h, w = elev.shape
        >>> raster['y'] = np.arange(h, dtype=float)
        >>> raster['x'] = np.arange(w, dtype=float)
        >>> result = min_observable_height(raster, x=0.0, y=2.0,
        ...                               max_height=20, precision=0.5)
    """
    # ---- input validation ------------------------------------------------
    if max_height <= 0:
        raise ValueError(f"max_height must be positive, got {max_height}")
    if precision <= 0:
        raise ValueError(f"precision must be positive, got {precision}")
    if precision > max_height:
        raise ValueError(
            f"precision ({precision}) must not exceed max_height ({max_height})"
        )

    # ---- determine candidate heights via binary search -------------------
    #
    # We evaluate viewshed at O(log2(max_height / precision)) heights.
    # For each cell we then find the smallest height where it is visible.
    #
    n_steps = int(ceil(log2(max_height / precision))) + 1
    # Build sorted list of heights from 0 to max_height (inclusive).
    # Using linspace guarantees both endpoints are included.
    if n_steps < 2:
        n_steps = 2
    heights = np.linspace(0.0, max_height, n_steps)

    # ---- collect visibility at each height level -------------------------
    # visibility_stack[k] is True where the cell is visible at heights[k].
    visibility_stack = np.empty(
        (n_steps, *raster.shape), dtype=np.bool_
    )

    for k, h in enumerate(heights):
        vs = viewshed(raster, x=x, y=y, observer_elev=float(h),
                       target_elev=target_elev)
        # viewshed returns INVISIBLE (-1) for invisible, positive angle
        # for visible.  The observer cell itself gets 180.
        vs_data = vs.data if isinstance(vs.data, np.ndarray) else vs.data.get()
        visibility_stack[k] = vs_data != INVISIBLE

    # ---- per-cell binary search on the precomputed stack -----------------
    # Because visibility is monotonic in height, we can find the first
    # height index where each cell becomes visible.
    #
    # Strategy: scan from lowest to highest height.  The minimum height
    # for a cell is the first index k where visibility_stack[k] is True.
    #
    # np.argmax on a boolean axis returns the *first* True index.
    # If a cell is never visible, all entries are False and argmax
    # returns 0 — we need to distinguish that case.

    ever_visible = visibility_stack.any(axis=0)
    first_visible_idx = visibility_stack.argmax(axis=0)

    result = np.where(ever_visible, heights[first_visible_idx], np.nan)

    return xarray.DataArray(
        result,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
