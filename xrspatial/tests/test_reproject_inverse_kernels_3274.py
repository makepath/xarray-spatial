"""Regression tests for GH #3274.

Two inverse-kernel bugs in ``xrspatial/reproject/_projections.py``:

1. ``_laea_inv_point`` (oblique/equatorial) normalized the input by
   ``x / (a * xmf)``, stripping the ``rq`` factor that the angular
   distance ``sce = 2 * asin(0.5 * rho / rq)`` divides out. The double
   division inflated distances by ~0.11% of the distance from the
   projection origin (up to 2.6 km for EPSG:3035 over Europe).
2. ``_authalic_apa`` carried inverse-series coefficients that did not
   invert ``_authalic_q`` (wrong at leading order in the second term:
   17/360 vs Snyder/PROJ's 23/360), putting every authalic-latitude
   inverse (AEA, CEA, LAEA) off by up to ~4.8 m.

The CUDA kernels in ``_projections_cuda.py`` shared both bugs.
"""
from __future__ import annotations

import math

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


def _inverse_parity_max_err_m(crs_code_or_obj, lons, lats):
    """Max error (metres) of the numba inverse fast path vs pyproj.

    Forward-projects (lons, lats) with pyproj, runs the projected points
    through ``transform_points`` back to EPSG:4326, and compares against
    pyproj's own inverse.
    """
    from xrspatial.reproject._projections import transform_points

    crs = pyproj.CRS(crs_code_or_obj)
    geo = pyproj.CRS(4326)
    fwd = pyproj.Transformer.from_crs(geo, crs, always_xy=True)
    px, py = fwd.transform(lons, lats)
    px = np.asarray(px)
    py = np.asarray(py)
    mask = np.isfinite(px) & np.isfinite(py)
    px, py = px[mask], py[mask]

    result = transform_points(crs, geo, px, py)
    assert result is not None, "expected a numba fast path for this pair"
    tx, ty = result

    inv = pyproj.Transformer.from_crs(crs, geo, always_xy=True)
    rx, ry = inv.transform(px, py)
    rx = np.asarray(rx)
    ry = np.asarray(ry)
    err_deg_x = (tx - rx) * np.cos(np.radians(ry))
    err_deg_y = ty - ry
    return float(np.hypot(err_deg_x, err_deg_y).max() * 111320.0)


class TestAuthalicInverseSeries:
    def test_round_trips_authalic_q(self):
        """phi -> q -> beta -> series must return phi to ~1e-9 rad.

        The old coefficients were off by up to 7.5e-7 rad (~4.8 m).
        """
        from xrspatial.reproject._projections import (_authalic_apa, _authalic_q, _WGS84_E)

        apa = _authalic_apa(_WGS84_E)
        qp = _authalic_q(1.0, _WGS84_E)
        worst = 0.0
        for phi_deg in np.linspace(-89.5, 89.5, 359):
            phi = math.radians(phi_deg)
            beta = math.asin(_authalic_q(math.sin(phi), _WGS84_E) / qp)
            t = 2.0 * beta
            phi_back = beta + sum(
                c * math.sin((k + 1) * t) for k, c in enumerate(apa[:5])
            )
            worst = max(worst, abs(phi_back - phi))
        # 5e-9 rad is ~3 cm on the ground; the correct series sits at
        # ~2.5e-10 rad, the broken one at 7.5e-7 rad.
        assert worst < 5e-9

    def test_matches_proj_pj_authset_coefficients(self):
        from xrspatial.reproject._projections import _authalic_apa, _WGS84_E2, _WGS84_E

        e2 = _WGS84_E2
        e4 = e2 * e2
        e6 = e4 * e2
        apa = _authalic_apa(_WGS84_E)
        assert apa[0] == pytest.approx(e2 / 3 + 31 * e4 / 180 + 517 * e6 / 5040, rel=1e-15)
        assert apa[1] == pytest.approx(23 * e4 / 360 + 251 * e6 / 3780, rel=1e-15)
        assert apa[2] == pytest.approx(761 * e6 / 45360, rel=1e-15)


class TestInverseKernelParityVsPyproj:
    """Inverse fast paths must match pyproj at the cm level.

    Tolerances are set an order of magnitude above the measured parity
    (~1.6 mm) and three orders below the pre-fix errors (4.8 m / 2.6 km).
    """

    def test_laea_3035_inverse(self):
        rng = np.random.default_rng(3274)
        lons = rng.uniform(-10, 30, 500)
        lats = rng.uniform(35, 70, 500)
        assert _inverse_parity_max_err_m(3035, lons, lats) < 0.05

    def test_cea_6933_inverse(self):
        rng = np.random.default_rng(3274)
        lons = rng.uniform(-179, 179, 500)
        lats = rng.uniform(-85, 85, 500)
        assert _inverse_parity_max_err_m(6933, lons, lats) < 0.05

    def test_aea_wgs84_inverse(self):
        # EPSG:5070 is NAD83; pyproj inserts a ~1-2 m WGS84->NAD83 step
        # that the fast path deliberately skips, so the regression test
        # uses an AEA definition on WGS84 directly.
        aea = pyproj.CRS(
            "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
            "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        )
        rng = np.random.default_rng(3274)
        lons = rng.uniform(-120, -70, 500)
        lats = rng.uniform(25, 49, 500)
        assert _inverse_parity_max_err_m(aea, lons, lats) < 0.05

    def test_laea_polar_inverse_unchanged(self):
        # The polar LAEA branch never had the rq bug; pin its parity so
        # the fix does not disturb it.
        laea_n = pyproj.CRS(
            "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs"
        )
        rng = np.random.default_rng(3274)
        lons = rng.uniform(-180, 180, 500)
        lats = rng.uniform(55, 89, 500)
        assert _inverse_parity_max_err_m(laea_n, lons, lats) < 0.05

    def test_laea_3035_forward_unchanged(self):
        from xrspatial.reproject._projections import transform_points

        rng = np.random.default_rng(3274)
        lons = rng.uniform(-10, 30, 500)
        lats = rng.uniform(35, 70, 500)
        geo = pyproj.CRS(4326)
        crs = pyproj.CRS(3035)
        tx, ty = transform_points(geo, crs, lons, lats)
        fwd = pyproj.Transformer.from_crs(geo, crs, always_xy=True)
        rx, ry = fwd.transform(lons, lats)
        err = np.hypot(tx - np.asarray(rx), ty - np.asarray(ry))
        assert float(err.max()) < 1e-3


class TestLaeaReprojectEndToEnd:
    def test_fast_path_matches_exact_pyproj_path(self):
        """reproject(4326 -> 3035) fast path vs transform_precision=0.

        transform_precision=0 is the documented exact-pyproj escape
        hatch; before the fix the fast path sampled source pixels up to
        2.6 km away from where the exact path sampled them.
        """
        import xarray as xr

        from xrspatial.reproject import reproject

        ny, nx = 200, 250
        y = np.linspace(70, 35, ny)
        x = np.linspace(-10, 30, nx)
        xx, yy = np.meshgrid(x, y)
        data = np.sin(np.radians(xx * 3)) * np.cos(np.radians(yy * 3)) * 1000
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )

        exact = reproject(raster, 'EPSG:3035', transform_precision=0)
        fast = reproject(raster, 'EPSG:3035')
        mask = np.isfinite(exact.values) & np.isfinite(fast.values)
        assert mask.sum() > 0.5 * mask.size
        # The remaining difference is the control-grid interpolation of
        # the approximate transform, well under a thousandth of the data
        # range. Pre-fix this was ~1.6 of a +/-1000 range at 7 km pixels.
        diff = np.abs(exact.values[mask] - fast.values[mask])
        assert float(diff.max()) < 0.05


@pytest.mark.skipif(not HAS_CUPY, reason="CUDA/cupy not available")
class TestCudaLaeaInverseParity:
    def test_cuda_laea_inverse_matches_pyproj(self):
        """try_cuda_transform for a 3035 chunk must match pyproj per pixel."""
        import cupy as cp

        from xrspatial.reproject._projections_cuda import try_cuda_transform

        geo = pyproj.CRS(4326)
        crs = pyproj.CRS(3035)
        # A chunk over central Europe, far enough from the projection
        # origin that the old bug showed km-scale errors.
        fwd = pyproj.Transformer.from_crs(geo, crs, always_xy=True)
        x0, y0 = fwd.transform(25.0, 40.0)
        h = w = 16
        cb = (x0 - 80000, y0 - 80000, x0 + 80000, y0 + 80000)
        result = try_cuda_transform(geo, crs, cb, (h, w))
        assert result is not None
        src_y, src_x = (cp.asnumpy(a) for a in result)

        res_x = (cb[2] - cb[0]) / w
        res_y = (cb[3] - cb[1]) / h
        xs = cb[0] + (np.arange(w) + 0.5) * res_x
        ys = cb[3] - (np.arange(h) + 0.5) * res_y
        xx, yy = np.meshgrid(xs, ys)
        inv = pyproj.Transformer.from_crs(crs, geo, always_xy=True)
        rlon, rlat = inv.transform(xx, yy)
        err_m = np.hypot(
            (src_x - rlon) * np.cos(np.radians(rlat)), src_y - rlat
        ) * 111320.0
        assert float(err_m.max()) < 0.05

    def test_cuda_matches_cpu_fast_path(self):
        import cupy as cp

        from xrspatial.reproject._projections import try_numba_transform
        from xrspatial.reproject._projections_cuda import try_cuda_transform

        geo = pyproj.CRS(4326)
        crs = pyproj.CRS(3035)
        cb = (4000000.0, 2000000.0, 4500000.0, 2500000.0)
        shape = (32, 32)
        cpu = try_numba_transform(geo, crs, cb, shape)
        gpu = try_cuda_transform(geo, crs, cb, shape)
        assert cpu is not None and gpu is not None
        np.testing.assert_allclose(cp.asnumpy(gpu[0]), cpu[0], atol=1e-9)
        np.testing.assert_allclose(cp.asnumpy(gpu[1]), cpu[1], atol=1e-9)


class TestLaeaEquatorialInverse:
    def test_laea_equit_inverse(self):
        """EQUIT-mode LAEA (lat_0=0) shares the fixed OBLIQ code path."""
        laea_eq = pyproj.CRS(
            "+proj=laea +lat_0=0 +lon_0=20 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs"
        )
        rng = np.random.default_rng(3274)
        lons = rng.uniform(-20, 60, 500)
        lats = rng.uniform(-40, 40, 500)
        assert _inverse_parity_max_err_m(laea_eq, lons, lats) < 0.05
