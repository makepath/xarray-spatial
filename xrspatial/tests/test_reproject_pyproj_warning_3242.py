"""Regression tests for issue #3242.

The projection parameter extractors in ``xrspatial.reproject._projections``
read parameters via ``pyproj.CRS.to_dict()``, which pyproj implements on
top of ``to_proj4()`` and therefore emits a "You will likely lose important
projection information" UserWarning. The extractors only read short names
and numeric parameters, never the lossy PROJ string, so the warning is
noise. All ``to_dict()`` calls must go through ``_crs_to_dict``, which
silences exactly that warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)

from xrspatial.reproject import reproject  # noqa: E402
from xrspatial.reproject import _projections  # noqa: E402

LOSSY_MSG = 'lose important projection information'

# One representative EPSG code per param extractor's projection family.
EXTRACTOR_CASES = [
    ('_lcc_params', 2154),     # Lambert Conformal Conic (RGF93 / Lambert-93)
    ('_aea_params', 5070),     # Albers Equal Area (NAD83 / CONUS Albers)
    ('_cea_params', 6933),     # Cylindrical Equal Area (EASE-Grid 2.0)
    ('_laea_params', 3035),    # Lambert Azimuthal Equal Area (ETRS89-LAEA)
    ('_stere_params', 3413),   # Polar Stereographic (NSIDC Sea Ice North)
    ('_sterea_params', 28992),  # Oblique Stereographic (RD New)
    ('_omerc_params', 3375),   # Oblique Mercator Hotine (RSO Peninsular)
    ('_tmerc_params', 32633),  # Transverse Mercator (UTM 33N)
    ('_is_wgs84_compatible_ellipsoid', 5070),
    ('_get_datum_params', 27700),  # OSGB36, non-WGS84 datum
]


def _lossy_warnings(record):
    return [w for w in record if LOSSY_MSG in str(w.message)]


@pytest.mark.parametrize('func_name,epsg', EXTRACTOR_CASES)
def test_param_extractors_no_lossy_proj_warning(func_name, epsg):
    func = getattr(_projections, func_name)
    crs = pyproj.CRS.from_epsg(epsg)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        func(crs)
    assert not _lossy_warnings(record)


def test_sinu_params_no_lossy_proj_warning():
    # Sinusoidal (MODIS) has no EPSG code; build from a PROJ string.
    crs = pyproj.CRS.from_proj4(
        '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m'
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        _projections._sinu_params(crs)
    assert not _lossy_warnings(record)


def test_reproject_pyproj_crs_no_lossy_proj_warning():
    # End-to-end repro of #3242: a projected raster whose CRS arrives as a
    # pyproj.CRS object (as it does when read from a GeoTIFF) reprojected
    # to geographic coordinates.
    ny, nx = 32, 32
    raster = xr.DataArray(
        np.random.RandomState(42).rand(ny, nx).astype(np.float32),
        dims=('y', 'x'),
        coords={
            'y': np.linspace(1_500_000, 1_400_000, ny),
            'x': np.linspace(-1_200_000, -1_100_000, nx),
        },
    )
    src = pyproj.CRS.from_epsg(5070)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        out = reproject(raster, pyproj.CRS.from_epsg(4326), source_crs=src)
    assert not _lossy_warnings(record)
    assert np.isfinite(out.data).any()


def test_reproject_dask_pyproj_crs_no_lossy_proj_warning():
    # Same as above but dask-backed: the param extractors also run per
    # chunk inside worker tasks, so a leak there would not show up in
    # the numpy-path test.
    pytest.importorskip('dask.array')
    ny, nx = 32, 32
    raster = xr.DataArray(
        np.random.RandomState(42).rand(ny, nx).astype(np.float32),
        dims=('y', 'x'),
        coords={
            'y': np.linspace(1_500_000, 1_400_000, ny),
            'x': np.linspace(-1_200_000, -1_100_000, nx),
        },
    ).chunk({'y': 16, 'x': 16})
    src = pyproj.CRS.from_epsg(5070)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        out = reproject(raster, pyproj.CRS.from_epsg(4326), source_crs=src)
        result = out.compute()
    assert not _lossy_warnings(record)
    assert np.isfinite(result.data).any()


def test_crs_to_dict_matches_raw_to_dict():
    crs = pyproj.CRS.from_epsg(5070)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = crs.to_dict()
    assert _projections._crs_to_dict(crs) == expected
