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
import xarray

from .utils import _validate_raster, has_cuda_and_cupy, has_dask_array, is_cupy_array

SPEED_OF_LIGHT = 299_792_458.0  # m/s


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


def _fresnel_radius_1(d1, d2, freq_hz):
    """First Fresnel zone radius at a point d1 from transmitter, d2 from receiver."""
    D = d1 + d2
    if D == 0 or freq_hz == 0:
        return 0.0
    wavelength = SPEED_OF_LIGHT / freq_hz
    return np.sqrt(wavelength * d1 * d2 / D)


def line_of_sight(
    raster: xarray.DataArray,
    x0: float, y0: float,
    x1: float, y1: float,
    observer_elev: float = 0,
    target_elev: float = 0,
    frequency_mhz: float = None,
) -> xarray.Dataset:
    """Compute elevation profile and visibility along a straight line.

    Parameters
    ----------
    raster : xarray.DataArray
        Elevation raster.
    x0, y0 : float
        Observer location in data-space coordinates.
    x1, y1 : float
        Target location in data-space coordinates.
    observer_elev : float
        Height above terrain at the observer.
    target_elev : float
        Height above terrain at the target.
    frequency_mhz : float, optional
        Radio frequency in MHz. When set, first Fresnel zone clearance
        is computed at each sample point.

    Returns
    -------
    xarray.Dataset
        Dataset with dimension ``sample`` containing variables
        ``distance``, ``elevation``, ``los_height``, ``visible``,
        ``x``, ``y``, and optionally ``fresnel_radius`` and
        ``fresnel_clear``.
    """
    _validate_raster(raster, func_name='line_of_sight', name='raster')

    x_coords = raster.coords['x'].values
    y_coords = raster.coords['y'].values

    # snap to nearest grid cell
    c0 = int(np.argmin(np.abs(x_coords - x0)))
    r0 = int(np.argmin(np.abs(y_coords - y0)))
    c1 = int(np.argmin(np.abs(x_coords - x1)))
    r1 = int(np.argmin(np.abs(y_coords - y1)))

    cells = _bresenham_line(r0, c0, r1, c1)
    elevations, xs, ys = _extract_transect(raster, cells)

    n = len(cells)

    # cumulative distance along the transect
    distance = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        distance[i] = distance[i - 1] + np.sqrt(dx * dx + dy * dy)

    total_dist = distance[-1] if n > 1 else 0.0

    # LOS height: linear interpolation from observer to target
    obs_h = elevations[0] + observer_elev
    tgt_h = elevations[-1] + target_elev if n > 1 else obs_h
    if total_dist > 0:
        los_height = obs_h + (tgt_h - obs_h) * (distance / total_dist)
    else:
        los_height = np.array([obs_h])

    # visibility: track max elevation angle from observer
    visible = np.ones(n, dtype=bool)
    max_angle = -np.inf
    for i in range(1, n):
        if distance[i] == 0:
            continue
        angle = (elevations[i] - obs_h) / distance[i]
        if angle >= max_angle:
            max_angle = angle
        else:
            visible[i] = False

    data_vars = {
        'distance': ('sample', distance),
        'elevation': ('sample', elevations),
        'los_height': ('sample', los_height),
        'visible': ('sample', visible),
        'x': ('sample', xs),
        'y': ('sample', ys),
    }

    if frequency_mhz is not None:
        freq_hz = frequency_mhz * 1e6
        fresnel = np.zeros(n, dtype=np.float64)
        fresnel_clear = np.ones(n, dtype=bool)
        for i in range(n):
            d1 = distance[i]
            d2 = total_dist - d1
            fresnel[i] = _fresnel_radius_1(d1, d2, freq_hz)
            clearance = los_height[i] - elevations[i]
            if clearance < fresnel[i]:
                fresnel_clear[i] = False
        data_vars['fresnel_radius'] = ('sample', fresnel)
        data_vars['fresnel_clear'] = ('sample', fresnel_clear)

    return xarray.Dataset(data_vars)
