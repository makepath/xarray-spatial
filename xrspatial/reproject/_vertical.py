"""Vertical datum transformations: ellipsoidal height <-> orthometric height.

Provides geoid undulation lookup from vendored EGM96 (2.6MB, 15-arcmin
global grid) for converting between:

- **Ellipsoidal height** (height above the WGS84 ellipsoid, what GPS gives)
- **Orthometric height** (height above mean sea level / geoid, what maps show)
- **Depth below chart datum** (bathymetric convention, positive downward)

The relationship is:
    h_ellipsoidal = H_orthometric + N_geoid

where N is the geoid undulation (can be positive or negative, ranges
from -107m to +85m globally for EGM96).

Usage
-----
>>> from xrspatial.reproject import geoid_height, ellipsoidal_to_orthometric
>>> N = geoid_height(-74.0, 40.7)                    # New York: ~-33m
>>> H = ellipsoidal_to_orthometric(h_gps, lon, lat)  # GPS -> map height
>>> h = orthometric_to_ellipsoidal(H_map, lon, lat)  # map height -> GPS
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Geoid grid loading
# ---------------------------------------------------------------------------

_VENDORED_DIR = os.path.join(os.path.dirname(__file__), 'grids')
_PROJ_CDN = "https://cdn.proj.org"

_GEOID_MODELS = {
    'EGM96': (
        'us_nga_egm96_15.tif',
        f'{_PROJ_CDN}/us_nga_egm96_15.tif',
    ),
    'EGM2008': (
        'us_nga_egm08_25.tif',
        f'{_PROJ_CDN}/us_nga_egm08_25.tif',
    ),
}

_loaded_geoids = {}


def _find_file(filename, cdn_url=None):
    """Find a file: vendored dir, user cache, then download."""
    vendored = os.path.join(_VENDORED_DIR, filename)
    if os.path.exists(vendored):
        return vendored

    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'xrspatial', 'proj_grids')
    cached = os.path.join(cache_dir, filename)
    if os.path.exists(cached):
        return cached

    if cdn_url:
        os.makedirs(cache_dir, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cdn_url, cached)
        return cached
    return None


def _load_geoid(model='EGM96'):
    """Load a geoid model, returning (data, left, top, res_x, res_y, h, w)."""
    if model in _loaded_geoids:
        return _loaded_geoids[model]

    if model not in _GEOID_MODELS:
        raise ValueError(f"Unknown geoid model: {model!r}. "
                         f"Available: {list(_GEOID_MODELS)}")

    filename, cdn_url = _GEOID_MODELS[model]
    path = _find_file(filename, cdn_url)
    if path is None:
        raise FileNotFoundError(
            f"Geoid model {model} not found. File: {filename}")

    try:
        import rasterio
        with rasterio.open(path) as ds:
            data = ds.read(1).astype(np.float64)
            b = ds.bounds
            h, w = ds.height, ds.width
            res_x = (b.right - b.left) / w
            res_y = (b.top - b.bottom) / h
            result = (np.ascontiguousarray(data), b.left, b.top, res_x, res_y, h, w)
    except ImportError:
        from xrspatial.geotiff import read_geotiff
        da = read_geotiff(path)
        vals = da.values.astype(np.float64)
        if vals.ndim == 3:
            vals = vals[0] if vals.shape[0] == 1 else vals[:, :, 0]
        y = da.coords['y'].values
        x = da.coords['x'].values
        h, w = vals.shape
        res_x = abs(float(x[1] - x[0])) if len(x) > 1 else 0.25
        res_y = abs(float(y[1] - y[0])) if len(y) > 1 else 0.25
        left = float(x[0]) - res_x / 2
        top = float(y[0]) + res_y / 2
        result = (np.ascontiguousarray(vals), left, top, res_x, res_y, h, w)

    _loaded_geoids[model] = result
    return result


# ---------------------------------------------------------------------------
# Numba interpolation
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _interp_geoid_point(lon, lat, data, left, top, res_x, res_y, h, w):
    """Bilinear interpolation of geoid undulation at a single point."""
    # Wrap longitude to [-180, 180)
    lon_w = lon
    while lon_w < -180.0:
        lon_w += 360.0
    while lon_w >= 180.0:
        lon_w -= 360.0

    col_f = (lon_w - left) / res_x
    row_f = (top - lat) / res_y

    if row_f < 0 or row_f > h - 1:
        return 0.0  # outside latitude range

    # Wrap column for global grids
    c0 = int(col_f) % w
    c1 = (c0 + 1) % w
    r0 = int(row_f)
    if r0 >= h - 1:
        r0 = h - 2
    r1 = r0 + 1

    dc = col_f - int(col_f)
    dr = row_f - r0

    N = (data[r0, c0] * (1.0 - dr) * (1.0 - dc) +
         data[r0, c1] * (1.0 - dr) * dc +
         data[r1, c0] * dr * (1.0 - dc) +
         data[r1, c1] * dr * dc)
    return N


@njit(nogil=True, cache=True, parallel=True)
def _interp_geoid_batch(lons, lats, out, data, left, top, res_x, res_y, h, w):
    """Batch bilinear interpolation of geoid undulation."""
    for i in prange(lons.shape[0]):
        out[i] = _interp_geoid_point(lons[i], lats[i], data, left, top,
                                     res_x, res_y, h, w)


@njit(nogil=True, cache=True, parallel=True)
def _interp_geoid_2d(lons_2d, lats_2d, out_2d, data, left, top, res_x, res_y, h, w):
    """2D batch geoid interpolation for raster grids."""
    for i in prange(lons_2d.shape[0]):
        for j in range(lons_2d.shape[1]):
            out_2d[i, j] = _interp_geoid_point(
                lons_2d[i, j], lats_2d[i, j], data, left, top,
                res_x, res_y, h, w)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geoid_height(lon, lat, model='EGM96'):
    """Get the geoid undulation N at given coordinates.

    Parameters
    ----------
    lon, lat : float, array-like, or xr.DataArray
        Geographic coordinates in degrees (WGS84).
    model : str
        Geoid model: 'EGM96' (vendored, 2.6MB) or 'EGM2008' (77MB, downloaded on first use).

    Returns
    -------
    N : same type as input
        Geoid undulation in metres. Positive means the geoid is above
        the ellipsoid.

    Examples
    --------
    >>> geoid_height(-74.0, 40.7)           # New York: ~-33m
    >>> geoid_height(np.array([0, 90]), np.array([0, 0]))  # batch
    """
    data, left, top, res_x, res_y, h, w = _load_geoid(model)

    scalar = np.ndim(lon) == 0 and np.ndim(lat) == 0
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64)).ravel()
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64)).ravel()

    out = np.empty(lon_arr.shape[0], dtype=np.float64)
    _interp_geoid_batch(lon_arr, lat_arr, out, data, left, top,
                        res_x, res_y, h, w)

    return float(out[0]) if scalar else out.reshape(np.shape(lon))


def geoid_height_raster(raster, model='EGM96'):
    """Get geoid undulation for every pixel in a geographic raster.

    Parameters
    ----------
    raster : xr.DataArray
        Raster with y (latitude) and x (longitude) coordinates in degrees.
    model : str
        Geoid model name.

    Returns
    -------
    xr.DataArray
        Geoid undulation N in metres, same shape as input.
    """
    import xarray as xr

    data, left, top, res_x, res_y, h, w = _load_geoid(model)

    y = raster.coords[raster.dims[-2]].values.astype(np.float64)
    x = raster.coords[raster.dims[-1]].values.astype(np.float64)
    xx, yy = np.meshgrid(x, y)

    out = np.empty_like(xx)
    _interp_geoid_2d(xx, yy, out, data, left, top, res_x, res_y, h, w)

    return xr.DataArray(
        out, dims=raster.dims[-2:],
        coords={raster.dims[-2]: raster.coords[raster.dims[-2]],
                raster.dims[-1]: raster.coords[raster.dims[-1]]},
        name='geoid_undulation',
        attrs={'units': 'metres', 'model': model},
    )


def ellipsoidal_to_orthometric(height, lon, lat, model='EGM96'):
    """Convert ellipsoidal height to orthometric (mean-sea-level) height.

    H = h - N

    Parameters
    ----------
    height : float or array-like
        Ellipsoidal height in metres (e.g. from GPS).
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    model : str
        Geoid model name.

    Returns
    -------
    H : same type as height
        Orthometric height in metres.
    """
    N = geoid_height(lon, lat, model)
    return np.asarray(height) - N


def orthometric_to_ellipsoidal(height, lon, lat, model='EGM96'):
    """Convert orthometric (mean-sea-level) height to ellipsoidal height.

    h = H + N

    Parameters
    ----------
    height : float or array-like
        Orthometric height in metres.
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    model : str
        Geoid model name.

    Returns
    -------
    h : same type as height
        Ellipsoidal height in metres.
    """
    N = geoid_height(lon, lat, model)
    return np.asarray(height) + N


def depth_to_ellipsoidal(depth, lon, lat, model='EGM96'):
    """Convert depth below chart datum (positive downward) to ellipsoidal height.

    Assumes chart datum is approximately mean sea level (the geoid).

    h = -depth + N

    Parameters
    ----------
    depth : float or array-like
        Depth below chart datum in metres (positive downward).
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    model : str
        Geoid model name.

    Returns
    -------
    h : same type as depth
        Ellipsoidal height in metres (negative below ellipsoid).
    """
    N = geoid_height(lon, lat, model)
    return -np.asarray(depth) + N


def ellipsoidal_to_depth(height, lon, lat, model='EGM96'):
    """Convert ellipsoidal height to depth below chart datum (positive downward).

    Assumes chart datum is approximately mean sea level (the geoid).

    depth = -(h - N) = N - h

    Parameters
    ----------
    height : float or array-like
        Ellipsoidal height in metres.
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    model : str
        Geoid model name.

    Returns
    -------
    depth : same type as height
        Depth below chart datum in metres (positive downward).
    """
    N = geoid_height(lon, lat, model)
    return N - np.asarray(height)
