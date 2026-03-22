"""Datum shift grid loading and interpolation.

Downloads horizontal offset grids from the PROJ CDN, caches them locally,
and provides Numba JIT bilinear interpolation for per-pixel datum shifts.

Grid format: GeoTIFF with 2+ bands:
  Band 1: latitude offset (arc-seconds)
  Band 2: longitude offset (arc-seconds)
"""
from __future__ import annotations

import math
import os
import urllib.request

import numpy as np
from numba import njit, prange

_PROJ_CDN = "https://cdn.proj.org"

# Vendored grid directory (shipped with the package)
_VENDORED_DIR = os.path.join(os.path.dirname(__file__), 'grids')

# Grid registry: key -> (filename, coverage bounds, description, cdn_url)
# Bounds are (lon_min, lat_min, lon_max, lat_max).
GRID_REGISTRY = {
    # --- NAD27 -> NAD83 (US + territories) ---
    'NAD27_CONUS': (
        'us_noaa_conus.tif',
        (-131, 20, -63, 50),
        'NAD27->NAD83 CONUS (NADCON)',
        f'{_PROJ_CDN}/us_noaa_conus.tif',
    ),
    'NAD27_NADCON5_CONUS': (
        'us_noaa_nadcon5_nad27_nad83_1986_conus.tif',
        (-125, 24, -66, 50),
        'NAD27->NAD83 CONUS (NADCON5)',
        f'{_PROJ_CDN}/us_noaa_nadcon5_nad27_nad83_1986_conus.tif',
    ),
    'NAD27_ALASKA': (
        'us_noaa_alaska.tif',
        (-194, 50, -128, 72),
        'NAD27->NAD83 Alaska (NADCON)',
        f'{_PROJ_CDN}/us_noaa_alaska.tif',
    ),
    'NAD27_HAWAII': (
        'us_noaa_hawaii.tif',
        (-164, 17, -154, 23),
        'Old Hawaiian->NAD83 (NADCON)',
        f'{_PROJ_CDN}/us_noaa_hawaii.tif',
    ),
    'NAD27_PRVI': (
        'us_noaa_prvi.tif',
        (-68, 17, -64, 19),
        'NAD27->NAD83 Puerto Rico/Virgin Islands',
        f'{_PROJ_CDN}/us_noaa_prvi.tif',
    ),
    # --- OSGB36 -> ETRS89 (UK) ---
    'OSGB36_UK': (
        'uk_os_OSTN15_NTv2_OSGBtoETRS.tif',
        (-9, 49, 3, 61),
        'OSGB36->ETRS89 (Ordnance Survey OSTN15)',
        f'{_PROJ_CDN}/uk_os_OSTN15_NTv2_OSGBtoETRS.tif',
    ),
    # --- Australia (parent grid covers NT region only) ---
    'AGD66_GDA94': (
        'au_icsm_A66_National_13_09_01.tif',
        (104, -14, 129, -10),
        'AGD66->GDA94 (Australia, NT region)',
        f'{_PROJ_CDN}/au_icsm_A66_National_13_09_01.tif',
    ),
    # --- Europe ---
    'DHDN_ETRS89_DE': (
        'de_adv_BETA2007.tif',
        (5, 47, 16, 56),
        'DHDN->ETRS89 (Germany)',
        f'{_PROJ_CDN}/de_adv_BETA2007.tif',
    ),
    'MGI_ETRS89_AT': (
        'at_bev_AT_GIS_GRID.tif',
        (9, 46, 18, 50),
        'MGI->ETRS89 (Austria)',
        f'{_PROJ_CDN}/at_bev_AT_GIS_GRID.tif',
    ),
    'ED50_ETRS89_ES': (
        'es_ign_SPED2ETV2.tif',
        (1, 38, 5, 41),
        'ED50->ETRS89 (Spain, eastern coast/Balearics)',
        f'{_PROJ_CDN}/es_ign_SPED2ETV2.tif',
    ),
    'RD_ETRS89_NL': (
        'nl_nsgi_rdcorr2018.tif',
        (2, 50, 8, 56),
        'RD->ETRS89 (Netherlands)',
        f'{_PROJ_CDN}/nl_nsgi_rdcorr2018.tif',
    ),
    'BD72_ETRS89_BE': (
        'be_ign_bd72lb72_etrs89lb08.tif',
        (2, 49, 7, 52),
        'BD72->ETRS89 (Belgium)',
        f'{_PROJ_CDN}/be_ign_bd72lb72_etrs89lb08.tif',
    ),
    'CH1903_ETRS89_CH': (
        'ch_swisstopo_CHENyx06_ETRS.tif',
        (5, 45, 11, 48),
        'CH1903->ETRS89 (Switzerland)',
        f'{_PROJ_CDN}/ch_swisstopo_CHENyx06_ETRS.tif',
    ),
    'D73_ETRS89_PT': (
        'pt_dgt_D73_ETRS89_geo.tif',
        (-10, 36, -6, 43),
        'D73->ETRS89 (Portugal)',
        f'{_PROJ_CDN}/pt_dgt_D73_ETRS89_geo.tif',
    ),
}

# Cache directory for grids not vendored
_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'xrspatial', 'proj_grids')


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _find_grid_file(filename, cdn_url=None):
    """Find a grid file: check vendored dir first, then cache, then download."""
    # 1. Vendored (shipped with package)
    vendored = os.path.join(_VENDORED_DIR, filename)
    if os.path.exists(vendored):
        return vendored

    # 2. User cache
    cached = os.path.join(_CACHE_DIR, filename)
    if os.path.exists(cached):
        return cached

    # 3. Download from CDN
    if cdn_url:
        _ensure_cache_dir()
        urllib.request.urlretrieve(cdn_url, cached)
        return cached

    return None


def load_grid(grid_key):
    """Load a datum shift grid by registry key.

    Returns (dlat, dlon, bounds, resolution) where:
    - dlat, dlon: numpy float64 arrays (arc-seconds), shape (H, W)
    - bounds: (left, bottom, right, top) in degrees
    - resolution: (res_lon, res_lat) in degrees
    """
    if grid_key not in GRID_REGISTRY:
        return None

    filename, _, _, cdn_url = GRID_REGISTRY[grid_key]
    path = _find_grid_file(filename, cdn_url)
    if path is None:
        return None

    # Read with rasterio for correct multi-band handling
    try:
        import rasterio
        with rasterio.open(path) as ds:
            dlat = ds.read(1).astype(np.float64)  # arc-seconds
            dlon = ds.read(2).astype(np.float64)  # arc-seconds
            b = ds.bounds
            bounds = (b.left, b.bottom, b.right, b.top)
            h, w = ds.height, ds.width
            # Compute resolution from bounds and shape (avoids ds.res ordering ambiguity)
            res_x = (b.right - b.left) / w if w > 1 else 0.25
            res_y = (b.top - b.bottom) / h if h > 1 else 0.25
            return dlat, dlon, bounds, (res_x, res_y)
    except ImportError:
        pass

    # Fallback: read with our own reader (may need band axis handling)
    from xrspatial.geotiff import read_geotiff
    da = read_geotiff(path)
    data = da.values
    if data.ndim == 3:
        # (H, W, bands) or (bands, H, W)
        if data.shape[2] == 2:
            dlat = data[:, :, 0].astype(np.float64)
            dlon = data[:, :, 1].astype(np.float64)
        else:
            dlat = data[0].astype(np.float64)
            dlon = data[1].astype(np.float64)
    else:
        return None

    y_coords = da.coords['y'].values
    x_coords = da.coords['x'].values
    bounds = (float(x_coords[0]), float(y_coords[-1]),
              float(x_coords[-1]), float(y_coords[0]))
    res_x = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 0.25
    res_y = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 0.25
    return dlat, dlon, bounds, (res_x, res_y)


# ---------------------------------------------------------------------------
# Numba bilinear grid interpolation
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _grid_interp_point(lon, lat, dlat_grid, dlon_grid,
                       grid_left, grid_top, grid_res_x, grid_res_y,
                       grid_h, grid_w):
    """Bilinear interpolation of a single point in the shift grid.

    Returns (dlat_arcsec, dlon_arcsec) or (0, 0) if outside the grid.
    """
    col_f = (lon - grid_left) / grid_res_x
    row_f = (grid_top - lat) / grid_res_y

    if col_f < 0 or col_f > grid_w - 1 or row_f < 0 or row_f > grid_h - 1:
        return 0.0, 0.0

    c0 = int(col_f)
    r0 = int(row_f)
    if c0 >= grid_w - 1:
        c0 = grid_w - 2
    if r0 >= grid_h - 1:
        r0 = grid_h - 2

    dc = col_f - c0
    dr = row_f - r0

    w00 = (1.0 - dr) * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w10 = dr * (1.0 - dc)
    w11 = dr * dc

    dlat = (dlat_grid[r0, c0] * w00 + dlat_grid[r0, c0 + 1] * w01 +
            dlat_grid[r0 + 1, c0] * w10 + dlat_grid[r0 + 1, c0 + 1] * w11)
    dlon = (dlon_grid[r0, c0] * w00 + dlon_grid[r0, c0 + 1] * w01 +
            dlon_grid[r0 + 1, c0] * w10 + dlon_grid[r0 + 1, c0 + 1] * w11)

    return dlat, dlon


@njit(nogil=True, cache=True, parallel=True)
def apply_grid_shift_forward(lon_arr, lat_arr, dlat_grid, dlon_grid,
                             grid_left, grid_top, grid_res_x, grid_res_y,
                             grid_h, grid_w):
    """Apply grid-based datum shift: source -> target (add offsets)."""
    for i in prange(lon_arr.shape[0]):
        dlat, dlon = _grid_interp_point(
            lon_arr[i], lat_arr[i], dlat_grid, dlon_grid,
            grid_left, grid_top, grid_res_x, grid_res_y,
            grid_h, grid_w,
        )
        lat_arr[i] += dlat / 3600.0  # arc-seconds to degrees
        lon_arr[i] += dlon / 3600.0


@njit(nogil=True, cache=True, parallel=True)
def apply_grid_shift_inverse(lon_arr, lat_arr, dlat_grid, dlon_grid,
                             grid_left, grid_top, grid_res_x, grid_res_y,
                             grid_h, grid_w):
    """Apply inverse grid-based datum shift: target -> source (subtract offsets).

    Uses iterative approach: the grid is indexed by source coordinates,
    but we have target coordinates.  One iteration is usually sufficient
    since the shifts are small relative to the grid spacing.
    """
    for i in prange(lon_arr.shape[0]):
        # Initial estimate: subtract the shift at the target coords
        dlat, dlon = _grid_interp_point(
            lon_arr[i], lat_arr[i], dlat_grid, dlon_grid,
            grid_left, grid_top, grid_res_x, grid_res_y,
            grid_h, grid_w,
        )
        lon_est = lon_arr[i] - dlon / 3600.0
        lat_est = lat_arr[i] - dlat / 3600.0

        # Refine: re-interpolate at the estimated source coords
        dlat2, dlon2 = _grid_interp_point(
            lon_est, lat_est, dlat_grid, dlon_grid,
            grid_left, grid_top, grid_res_x, grid_res_y,
            grid_h, grid_w,
        )
        lon_arr[i] -= dlon2 / 3600.0
        lat_arr[i] -= dlat2 / 3600.0


# ---------------------------------------------------------------------------
# Grid cache (loaded grids, keyed by grid_key)
# ---------------------------------------------------------------------------

_loaded_grids = {}  # cleared on module reload


def get_grid(grid_key):
    """Get a loaded grid, downloading if necessary.

    Returns (dlat, dlon, left, top, res_x, res_y, h, w) or None.
    """
    if grid_key in _loaded_grids:
        return _loaded_grids[grid_key]

    result = load_grid(grid_key)
    if result is None:
        _loaded_grids[grid_key] = None
        return None

    dlat, dlon, bounds, (res_x, res_y) = result
    h, w = dlat.shape
    # Ensure contiguous float64 for Numba
    dlat = np.ascontiguousarray(dlat, dtype=np.float64)
    dlon = np.ascontiguousarray(dlon, dtype=np.float64)
    entry = (dlat, dlon, bounds[0], bounds[3], res_x, res_y, h, w)
    _loaded_grids[grid_key] = entry
    return entry


def find_grid_for_point(lon, lat, datum_key):
    """Find the best grid covering a given point.

    Returns the grid_key or None.
    """
    # Map datum/ellipsoid names to grid keys, ordered by preference.
    # Keys are matched against the 'datum' or 'ellps' field from CRS.to_dict().
    datum_grids = {
        'NAD27': ['NAD27_NADCON5_CONUS', 'NAD27_CONUS', 'NAD27_ALASKA',
                  'NAD27_HAWAII', 'NAD27_PRVI'],
        'clarke66': ['NAD27_NADCON5_CONUS', 'NAD27_CONUS', 'NAD27_ALASKA',
                     'NAD27_HAWAII', 'NAD27_PRVI'],
        'OSGB36': ['OSGB36_UK'],
        'airy': ['OSGB36_UK'],
        'AGD66': ['AGD66_GDA94'],
        'aust_SA': ['AGD66_GDA94'],
        'DHDN': ['DHDN_ETRS89_DE'],
        'bessel': ['DHDN_ETRS89_DE'],  # Bessel used by DHDN, MGI, etc.
        'MGI': ['MGI_ETRS89_AT'],
        'ED50': ['ED50_ETRS89_ES'],
        'intl': ['ED50_ETRS89_ES'],  # International 1924 ellipsoid
        'BD72': ['BD72_ETRS89_BE'],
        'CH1903': ['CH1903_ETRS89_CH'],
        'D73': ['D73_ETRS89_PT'],
    }

    candidates = datum_grids.get(datum_key, [])
    for grid_key in candidates:
        entry = GRID_REGISTRY.get(grid_key)
        if entry is None:
            continue
        _, coverage, _, _ = entry
        lon_min, lat_min, lon_max, lat_max = coverage
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return grid_key
    return None
