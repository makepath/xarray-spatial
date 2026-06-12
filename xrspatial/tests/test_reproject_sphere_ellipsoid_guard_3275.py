"""Regression tests for GH #3275.

The numba/CUDA fast-path dispatch applied WGS84 ellipsoid constants to
CRSes on other ellipsoids:

1. ``_is_wgs84_compatible_ellipsoid`` only inspected the ``ellps`` /
   ``datum`` keys of ``crs.to_dict()`` and treated empty values as
   WGS84. PROJ normalizes spherical definitions (``+a=R +b=R``) to a
   bare ``R`` key with no ``ellps``, so the standard MODIS sinusoidal
   grid passed the check and the WGS84 meridional-arc kernel ran
   against a sphere: ~18.9 km error at mid latitudes.
2. ``_aea_params`` and ``_cea_params`` had no ellipsoid check at all,
   so sphere-based AEA/CEA definitions ran the WGS84 kernels with up to
   ~24 km error.

Non-representable pairs must return None from the dispatchers and fall
back to pyproj, like the non-WGS84 datum pairs from GH #2651 already do.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from xrspatial.utils import has_cuda_and_cupy

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

HAS_CUPY = has_cuda_and_cupy()

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)

MODIS_SINU = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)
SPHERE_AEA = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
    "+a=6370997 +b=6370997 +units=m +no_defs"
)
SPHERE_CEA = "+proj=cea +lat_ts=30 +a=6371228 +b=6371228 +units=m +no_defs"


class TestEllipsoidCompatibilityHelper:
    def test_rejects_r_defined_sphere(self):
        from xrspatial.reproject._projections import _is_wgs84_compatible_ellipsoid

        # PROJ normalizes a=b to a bare R key.
        modis = pyproj.CRS(MODIS_SINU)
        assert 'R' in modis.to_dict()
        assert not _is_wgs84_compatible_ellipsoid(modis)

    def test_rejects_named_sphere(self):
        from xrspatial.reproject._projections import _is_wgs84_compatible_ellipsoid

        crs = pyproj.CRS("+proj=laea +lat_0=45 +lon_0=-100 +ellps=sphere +units=m +no_defs")
        assert not _is_wgs84_compatible_ellipsoid(crs)

    def test_accepts_wgs84_and_grs80(self):
        # EPSG:3857 is deliberately absent: Web Mercator is defined on a
        # sphere (a=b=6378137) and is dispatched by EPSG code without
        # consulting this helper, so the helper correctly reports it as
        # not WGS84-ellipsoid-compatible.
        from xrspatial.reproject._projections import _is_wgs84_compatible_ellipsoid

        for code in (3395, 32633, 5070, 6933, 3035, 3031):
            assert _is_wgs84_compatible_ellipsoid(pyproj.CRS(code)), code

    def test_accepts_explicit_wgs84_axes(self):
        from xrspatial.reproject._projections import _is_wgs84_compatible_ellipsoid

        crs = pyproj.CRS(
            "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
            "+a=6378137 +b=6356752.314245179 +units=m +no_defs"
        )
        assert _is_wgs84_compatible_ellipsoid(crs)

    def test_still_accepts_helmert_datums(self):
        # Datums with Helmert parameters stay "compatible": the dispatch
        # guards (try_numba_transform / transform_points) bail on them
        # separately via _get_datum_params.
        from xrspatial.reproject._projections import _is_wgs84_compatible_ellipsoid

        crs = pyproj.CRS(27700)  # OSGB36 / Airy
        assert _is_wgs84_compatible_ellipsoid(crs)


class TestSphereCrsesBailToPyproj:
    def test_modis_sinusoidal_no_fast_path(self):
        from xrspatial.reproject._projections import try_numba_transform

        geo = pyproj.CRS(4326)
        modis = pyproj.CRS(MODIS_SINU)
        cb = (0.0, 4.0e6, 1.0e5, 4.1e6)
        assert try_numba_transform(geo, modis, cb, (4, 4)) is None
        assert try_numba_transform(modis, geo, (0.0, 40.0, 1.0, 41.0), (4, 4)) is None

    def test_sphere_aea_no_fast_path(self):
        from xrspatial.reproject._projections import transform_points

        geo = pyproj.CRS(4326)
        crs = pyproj.CRS(SPHERE_AEA)
        lons = np.array([-100.0, -90.0])
        lats = np.array([30.0, 40.0])
        assert transform_points(geo, crs, lons, lats) is None
        # Sphere as the source goes through the other dispatch branch.
        assert transform_points(
            crs, geo, np.array([0.0]), np.array([0.0])
        ) is None

    def test_sphere_cea_no_fast_path(self):
        from xrspatial.reproject._projections import transform_points

        geo = pyproj.CRS(4326)
        crs = pyproj.CRS(SPHERE_CEA)
        lons = np.array([-100.0, 0.0])
        lats = np.array([30.0, 45.0])
        assert transform_points(geo, crs, lons, lats) is None

    def test_wgs84_pairs_keep_fast_path(self):
        from xrspatial.reproject._projections import transform_points

        geo = pyproj.CRS(4326)
        for code in (3857, 3395, 32633, 5070, 6933, 3035):
            res = transform_points(
                geo, pyproj.CRS(code), np.array([10.0]), np.array([45.0])
            )
            assert res is not None, code


class TestModisReprojectEndToEnd:
    def test_fast_path_matches_exact_pyproj_path(self):
        """4326 -> MODIS sinusoidal must agree with transform_precision=0.

        Before the guard, the default path ran the WGS84 sinusoidal
        kernel against the MODIS sphere and sampled source pixels up to
        ~19 km away from the exact pyproj path.
        """
        import xarray as xr

        from xrspatial.reproject import reproject

        ny, nx = 120, 160
        y = np.linspace(60, 30, ny)
        x = np.linspace(-20, 20, nx)
        xx, yy = np.meshgrid(x, y)
        data = np.sin(np.radians(xx * 4)) * np.cos(np.radians(yy * 4)) * 100
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exact = reproject(raster, MODIS_SINU, transform_precision=0)
            fast = reproject(raster, MODIS_SINU)
        mask = np.isfinite(exact.values) & np.isfinite(fast.values)
        assert mask.sum() > 0.5 * mask.size
        # Residual is the control-grid interpolation only (<0.1% of the
        # +/-100 data range). Pre-fix the max difference was ~150.
        diff = np.abs(exact.values[mask] - fast.values[mask])
        assert float(diff.max()) < 1.0


@pytest.mark.skipif(not HAS_CUPY, reason="CUDA/cupy not available")
class TestCudaSphereGuard:
    def test_cuda_modis_sinusoidal_no_fast_path(self):
        from xrspatial.reproject._projections_cuda import try_cuda_transform

        geo = pyproj.CRS(4326)
        modis = pyproj.CRS(MODIS_SINU)
        cb = (0.0, 4.0e6, 1.0e5, 4.1e6)
        assert try_cuda_transform(geo, modis, cb, (4, 4)) is None

    def test_cuda_wgs84_fast_path_still_active(self):
        from xrspatial.reproject._projections_cuda import try_cuda_transform

        geo = pyproj.CRS(4326)
        merc = pyproj.CRS(3857)
        cb = (0.0, 0.0, 1.0e5, 1.0e5)
        assert try_cuda_transform(geo, merc, cb, (4, 4)) is not None
