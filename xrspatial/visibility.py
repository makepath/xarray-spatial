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
            # Only compute the needed cells, not the entire array
            elevations = data.vindex[rows, cols].compute().astype(np.float64)
            return elevations, x_coords, y_coords
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


def cumulative_viewshed(
    raster: xarray.DataArray,
    observers: list,
    target_elev: float = 0,
    max_distance: float = None,
) -> xarray.DataArray:
    """Count how many observers can see each cell.

    Parameters
    ----------
    raster : xarray.DataArray
        Elevation raster (numpy, cupy, or dask-backed).
    observers : list of dict
        Each dict must have ``x`` and ``y`` keys (data-space coords).
        Optional keys: ``observer_elev`` (default 0), ``target_elev``
        (overrides function-level default), ``max_distance`` (per-observer
        analysis radius).
    target_elev : float
        Default target elevation for observers that don't specify one.
    max_distance : float, optional
        Default maximum analysis radius.

    Returns
    -------
    xarray.DataArray
        Integer raster (int32) with the count of observers that have
        line-of-sight to each cell.
    """
    from .viewshed import viewshed, INVISIBLE

    _validate_raster(raster, func_name='cumulative_viewshed', name='raster')
    if not observers:
        raise ValueError("observers list must not be empty")

    # Detect dask backend to keep accumulation lazy
    _is_dask = False
    if has_dask_array():
        import dask.array as da
        _is_dask = isinstance(raster.data, da.Array)

    if _is_dask:
        count = da.zeros(raster.shape, dtype=np.int32, chunks=raster.data.chunks)
    else:
        count = np.zeros(raster.shape, dtype=np.int32)

    for obs in observers:
        ox = obs['x']
        oy = obs['y']
        oe = obs.get('observer_elev', 0)
        te = obs.get('target_elev', target_elev)
        md = obs.get('max_distance', max_distance)

        vs = viewshed(raster, x=ox, y=oy, observer_elev=oe,
                      target_elev=te, max_distance=md)

        vs_data = vs.data
        if _is_dask and not isinstance(vs_data, da.Array):
            vs_data = da.from_array(vs_data, chunks=raster.data.chunks)
        count = count + (vs_data != INVISIBLE).astype(np.int32)

    result = xarray.DataArray(count, coords=raster.coords,
                              dims=raster.dims, attrs=raster.attrs)
    if _is_dask:
        chunk_dict = dict(zip(raster.dims, raster.data.chunks))
        result = result.chunk(chunk_dict)
    return result


def visibility_frequency(
    raster: xarray.DataArray,
    observers: list,
    target_elev: float = 0,
    max_distance: float = None,
) -> xarray.DataArray:
    """Fraction of observers that can see each cell.

    Parameters are the same as :func:`cumulative_viewshed`.

    Returns
    -------
    xarray.DataArray
        Float64 raster with values in [0, 1].
    """
    cum = cumulative_viewshed(raster, observers, target_elev, max_distance)
    freq = cum.astype(np.float64) / len(observers)
    return freq
