"""Regression tests for GH #3697.

The numba fast path in ``xrspatial/reproject/_projections.py`` used to claim
CRS definitions whose PROJ parameters it does not model, and then project
them with the wrong constants. Four separate defects:

1. ``_lcc_params`` / ``_aea_params`` defaulted ``lat_2`` to a ``lat_1`` that
   had already been converted to radians, so ``math.radians`` ran twice and
   a one-standard-parallel definition got a cone constant for a parallel
   pair that does not exist.
2. ``_aea_params``, ``_cea_params``, ``_laea_params``, ``_stere_params``,
   ``_sinu_params`` and ``_sterea_params`` ignored ``units``, reading a CRS
   in feet as metres.
3. Every extractor ignored ``+axis=``, so a west/south oriented grid came
   out negated and swapped.
4. ``_is_wgs84_compatible_ellipsoid`` read a missing ``+datum=`` key as
   "WGS84", letting datums that only share the GRS80/WGS84 ellipsoid skip
   their shift.
"""
import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.reproject import _projections as P
from xrspatial.reproject import reproject

pyproj = pytest.importorskip('pyproj')


WGS84 = 'EPSG:4326'


def _crs_or_skip(spec):
    """Build a CRS, skipping if this PROJ build's database lacks the code.

    Several CRSs used below (EPSG:10481, 9549, 9207) are recent additions
    to the EPSG registry, and PROJ ships a snapshot of that registry. An
    older PROJ should skip rather than error out of an unrelated
    assertion.
    """
    try:
        return pyproj.CRS.from_user_input(spec)
    except Exception:
        pytest.skip(f'{spec} is not in this PROJ database')


def _lon_field_error(crs_spec, lon_c, lat_c, span=0.5, n=401, out=32):
    """Max georeferencing error, in metres, of a reprojected raster.

    The source stores its own longitude as the pixel value and the output
    grid is pinned explicitly, so every output pixel should carry the
    longitude pyproj assigns to that pixel's target coordinates. The source
    is a smooth ramp sampled far finer than the output, so what is left is
    georeferencing error rather than resampling error.
    """
    tgt = _crs_or_skip(crs_spec)
    src_crs = _crs_or_skip(4326)
    fwd = pyproj.Transformer.from_crs(src_crs, tgt, always_xy=True)
    inv = pyproj.Transformer.from_crs(tgt, src_crs, always_xy=True)

    lons = np.linspace(lon_c - span, lon_c + span, n)
    lats = np.linspace(lat_c + span, lat_c - span, n)
    src = xr.DataArray(
        np.tile(lons, (n, 1)), dims=['y', 'x'],
        coords={'y': lats, 'x': lons}, attrs={'crs': WGS84},
    )

    cx, cy = fwd.transform(lon_c, lat_c)
    ex, ey = fwd.transform(
        np.array([lon_c - span / 2, lon_c + span / 2]),
        np.array([lat_c - span / 2, lat_c + span / 2]),
    )
    half = max(abs(ex[1] - ex[0]), abs(ey[1] - ey[0])) / 2
    result = reproject(
        src, tgt, bounds=(cx - half, cy - half, cx + half, cy + half),
        width=out, height=out, resampling='bilinear',
    )

    gx, gy = np.meshgrid(result.coords['x'].values, result.coords['y'].values)
    true_lon, _ = inv.transform(gx.ravel(), gy.ravel())
    got_lon = np.asarray(result.data).ravel()
    mask = np.isfinite(got_lon) & np.isfinite(true_lon)
    assert mask.any(), f"no valid output pixels for {crs_spec}"
    dlon = np.abs(got_lon[mask] - np.asarray(true_lon)[mask])
    return float(np.max(dlon) * 111320.0 * math.cos(math.radians(lat_c)))


# ---------------------------------------------------------------------------
# 1. lat_2 fallback
# ---------------------------------------------------------------------------

class TestLccSingleStandardParallel:
    """A 1SP Lambert Conformal Conic must use n = sin(lat_1)."""

    def test_cone_constant_is_sin_lat1(self):
        # +proj=lcc +lat_1=45 +lat_0=45 with no lat_2 at all.
        crs = _crs_or_skip(
            '+proj=lcc +lat_1=45 +lat_0=45 +lon_0=-100 +k_0=1 '
            '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        )
        assert 'lat_2' not in P._crs_to_dict(crs)
        params = P._lcc_params(crs)
        assert params is not None
        n = params[1]
        assert n == pytest.approx(math.sin(math.radians(45)), abs=1e-12)

    @pytest.mark.parametrize('epsg,lon,lat', [
        ('EPSG:8325', -120.25, 44.75),   # NAD83(2011) / Oregon Mitchell zone
        ('EPSG:9549', 6.82, 45.18),      # LTF2004(C)
    ])
    def test_reproject_matches_pyproj(self, epsg, lon, lat):
        assert _lon_field_error(epsg, lon, lat) < 1.0

    def test_two_parallel_lcc_still_uses_both(self):
        # EPSG:2154 carries lat_1 and lat_2, so the 2SP branch must run and
        # produce a cone constant strictly between the two parallels.
        crs = _crs_or_skip(2154)
        d = P._crs_to_dict(crs)
        assert d['lat_1'] != d['lat_2']
        n = P._lcc_params(crs)[1]
        assert math.sin(math.radians(d['lat_2'])) < n
        assert n < math.sin(math.radians(d['lat_1']))
        assert _lon_field_error('EPSG:2154', 3.0, 46.5) < 1.0

    def test_aea_single_standard_parallel(self):
        crs = _crs_or_skip(
            '+proj=aea +lat_1=45 +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 '
            '+datum=WGS84 +units=m +no_defs'
        )
        d = P._crs_to_dict(crs)
        if 'lat_2' in d:
            pytest.skip('this PROJ build emits an explicit lat_2')
        n = P._aea_params(crs)[1]
        assert n == pytest.approx(math.sin(math.radians(45)), abs=1e-12)


# ---------------------------------------------------------------------------
# 2. non-metre units
# ---------------------------------------------------------------------------

class TestNonMetricUnitsFallBack:
    """Extractors with no to_meter factor must decline a non-metre CRS."""

    @pytest.mark.parametrize('proj,extractor', [
        ('+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96',
         '_aea_params'),
        ('+proj=cea +lon_0=0 +lat_ts=30', '_cea_params'),
        ('+proj=laea +lat_0=45 +lon_0=-100', '_laea_params'),
        ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45', '_stere_params'),
        ('+proj=sinu +lon_0=0', '_sinu_params'),
        ('+proj=sterea +lat_0=52 +lon_0=5 +k=0.9999', '_sterea_params'),
    ])
    def test_feet_declined_metres_accepted(self, proj, extractor):
        fn = getattr(P, extractor)
        base = proj + ' +x_0=0 +y_0=0 +datum=WGS84 +no_defs'
        metric = pyproj.CRS.from_user_input(base + ' +units=m')
        feet = pyproj.CRS.from_user_input(base + ' +units=us-ft')
        assert fn(metric) is not None, 'metre CRS must keep the fast path'
        assert fn(feet) is None, 'foot CRS must fall back to pyproj'

    def test_reproject_us_survey_foot_albers(self):
        # EPSG:10481 is +proj=aea +units=us-ft; it used to land 6594 km out.
        assert _lon_field_error('EPSG:10481', -100.0, 31.25) < 1.0

    def test_lcc_and_tmerc_keep_their_foot_support(self):
        # These two do return a to_meter factor, so feet stay on the fast
        # path. EPSG:2260 is tmerc in US survey feet, EPSG:2261 is lcc.
        assert P._tmerc_params(pyproj.CRS.from_epsg(2260)) is not None
        lcc_ft = pyproj.CRS.from_user_input(
            '+proj=lcc +lat_1=41.03 +lat_2=40.66 +lat_0=40.16 +lon_0=-74 '
            '+x_0=300000 +y_0=0 +datum=WGS84 +units=us-ft +no_defs'
        )
        assert P._lcc_params(lcc_ft) is not None


# ---------------------------------------------------------------------------
# 3. axis order / direction
# ---------------------------------------------------------------------------

class TestAxisOrientation:
    """A grid that is not east/north must not take the fast path."""

    def test_wsu_declined(self):
        # EPSG:2048 is the South African Lo19 grid: +axis=wsu.
        crs = _crs_or_skip(2048)
        assert P._crs_to_dict(crs).get('axis') == 'wsu'
        assert P._tmerc_params(crs) is None
        assert not P._is_wgs84_compatible_ellipsoid(crs)

    def test_enu_still_accepted(self):
        crs = _crs_or_skip(32633)
        assert P._is_wgs84_compatible_ellipsoid(crs)

    def test_reproject_south_oriented_grid(self):
        assert _lon_field_error('EPSG:2048', 19.0, -29.0) < 1.0


# ---------------------------------------------------------------------------
# 4. datum sharing the GRS80 / WGS84 ellipsoid
# ---------------------------------------------------------------------------

class TestUnaliasedDatumGuard:
    """A datum PROJ has no +datum= alias for must still be measured."""

    # Lower bounds on the measured offset, well under the real values (the
    # offset varies across each area of use) but far above
    # _MAX_DATUM_OFFSET_M so the assertion pins the behaviour, not a
    # PROJ-version-specific number.
    @pytest.mark.parametrize('epsg,min_offset_m', [
        (2100, 200.0),    # GGRS87 / Greek Grid
        (2039, 50.0),     # Israel 1993 / Israeli TM Grid
        (9207, 100.0),    # VN-2000 / TM-3 104-30
    ])
    def test_shifted_datum_declined(self, epsg, min_offset_m):
        crs = _crs_or_skip(epsg)
        d = P._crs_to_dict(crs)
        # The dict carries no datum name, which is exactly what used to
        # make the old name test read these as WGS84.
        assert d.get('datum', '') == ''
        assert d.get('ellps') in ('GRS80', 'WGS84')
        offset = P._datum_offset_from_wgs84(crs)
        assert offset is not None
        assert offset > min_offset_m
        assert not P._is_wgs84_compatible_ellipsoid(crs)

    @pytest.mark.parametrize('epsg', [
        32633,   # WGS 84 / UTM zone 33N
        2154,    # RGF93 / Lambert-93   (ETRS89-realisation, ~0 m)
        3035,    # ETRS89-extended / LAEA Europe
        5070,    # NAD83 / Conus Albers (deliberately treated as WGS84)
        6350,    # NAD83(2011) / Conus Albers
    ])
    def test_aligned_datum_keeps_fast_path(self, epsg):
        crs = _crs_or_skip(epsg)
        assert P._is_wgs84_compatible_ellipsoid(crs)
        offset = P._datum_offset_from_wgs84(crs)
        if offset is not None:
            assert offset <= P._MAX_DATUM_OFFSET_M

    @pytest.mark.parametrize('epsg,lon,lat', [
        (2100, 24.0, 38.0),
        (2039, 35.2, 31.7),
    ])
    def test_reproject_matches_pyproj(self, epsg, lon, lat):
        assert _lon_field_error(f'EPSG:{epsg}', lon, lat) < 1.0

    def test_lite_crs_has_no_geodetic_base(self):
        # LiteCRS carries no geodetic_crs, so the measurement returns None
        # and the curated built-in table keeps its verdict.
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        assert P._datum_offset_from_wgs84(LiteCRS(3857)) is None

    def test_measurement_failure_fails_closed(self, monkeypatch):
        # A CRS we could have measured but could not must be rejected, not
        # waved through -- an unverifiable datum belongs on the pyproj path.
        crs = _crs_or_skip(32633)
        assert P._is_wgs84_compatible_ellipsoid(crs)

        def _boom(*args, **kwargs):
            raise RuntimeError('transform unavailable')

        P._datum_offset_from_wgs84.cache_clear()
        monkeypatch.setattr(pyproj.Transformer, 'from_crs', _boom)
        try:
            assert P._datum_offset_from_wgs84(crs) == math.inf
            assert not P._is_wgs84_compatible_ellipsoid(crs)
        finally:
            P._datum_offset_from_wgs84.cache_clear()

    def test_axis_order_only_crs_keeps_fast_path(self):
        # EPSG:2193 and EPSG:3346 are north/east in WKT but reproject
        # correctly because always_xy=True normalizes the ordering. The
        # axis guard must not reject them; see the note in
        # _is_wgs84_compatible_ellipsoid.
        for epsg in (2193, 3346):
            crs = _crs_or_skip(epsg)
            assert [a.direction for a in crs.axis_info] == ['north', 'east']
            assert P._crs_to_dict(crs).get('axis') is None
            assert P._is_wgs84_compatible_ellipsoid(crs)
            assert P._tmerc_params(crs) is not None


# ---------------------------------------------------------------------------
# Backend parity: the guard runs on the CPU and CUDA dispatchers alike
# ---------------------------------------------------------------------------

class TestBackendParity:
    """Both dispatchers share the extractors, so both must decline."""

    @pytest.mark.parametrize('epsg', [2100, 2048, 10481])
    def test_cpu_dispatcher_declines(self, epsg):
        tgt = _crs_or_skip(epsg)
        src = pyproj.CRS.from_epsg(4326)
        fwd = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
        aou = tgt.area_of_use
        cx, cy = fwd.transform((aou.west + aou.east) / 2,
                               (aou.south + aou.north) / 2)
        bounds = (cx - 1000, cy - 1000, cx + 1000, cy + 1000)
        assert P.try_numba_transform(src, tgt, bounds, (8, 8)) is None

    @pytest.mark.parametrize('epsg', [2100, 2048, 10481])
    def test_cuda_dispatcher_declines(self, epsg):
        cuda = pytest.importorskip('numba.cuda')
        if not cuda.is_available():
            pytest.skip('CUDA not available')
        pytest.importorskip('cupy')
        from xrspatial.reproject._projections_cuda import try_cuda_transform

        tgt = _crs_or_skip(epsg)
        src = pyproj.CRS.from_epsg(4326)
        fwd = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
        aou = tgt.area_of_use
        cx, cy = fwd.transform((aou.west + aou.east) / 2,
                               (aou.south + aou.north) / 2)
        bounds = (cx - 1000, cy - 1000, cx + 1000, cy + 1000)
        assert try_cuda_transform(src, tgt, bounds, (8, 8)) is None

    @pytest.mark.parametrize('epsg,lon,lat', [
        (8325, -120.25, 44.75),
        (2100, 24.0, 38.0),
    ])
    def test_dask_matches_numpy(self, epsg, lon, lat):
        da = pytest.importorskip('dask.array')
        tgt = _crs_or_skip(epsg)
        src_crs = _crs_or_skip(4326)
        fwd = pyproj.Transformer.from_crs(src_crs, tgt, always_xy=True)

        n, span = 201, 0.4
        lons = np.linspace(lon - span, lon + span, n)
        lats = np.linspace(lat + span, lat - span, n)
        values = np.tile(lons, (n, 1))
        coords = {'y': lats, 'x': lons}

        cx, cy = fwd.transform(lon, lat)
        bounds = (cx - 5000, cy - 5000, cx + 5000, cy + 5000)

        eager = reproject(
            xr.DataArray(values, dims=['y', 'x'], coords=coords,
                         attrs={'crs': WGS84}),
            tgt, bounds=bounds, width=24, height=24, resampling='bilinear',
        )
        lazy = reproject(
            xr.DataArray(da.from_array(values, chunks=(64, 64)),
                         dims=['y', 'x'], coords=coords,
                         attrs={'crs': WGS84}),
            tgt, bounds=bounds, width=24, height=24, resampling='bilinear',
        )
        np.testing.assert_allclose(
            np.asarray(eager.data), np.asarray(lazy.data.compute()),
            rtol=0, atol=1e-9,
        )
