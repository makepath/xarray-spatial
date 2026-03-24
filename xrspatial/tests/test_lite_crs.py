"""Tests for the lightweight CRS class (xrspatial.reproject._lite_crs)."""
from __future__ import annotations

import pytest

from xrspatial.reproject._lite_crs import CRS


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------
class TestCRSConstruction:
    def test_from_epsg_int(self):
        crs = CRS(4326)
        assert crs.to_epsg() == 4326

    def test_from_epsg_classmethod(self):
        crs = CRS.from_epsg(3857)
        assert crs.to_epsg() == 3857

    def test_from_authority_string(self):
        crs = CRS("EPSG:4326")
        assert crs.to_epsg() == 4326

    def test_unknown_epsg_raises(self):
        with pytest.raises(ValueError, match="not in the built-in table"):
            CRS(99999)

    def test_to_authority(self):
        crs = CRS(4326)
        assert crs.to_authority() == ("EPSG", "4326")

    def test_is_geographic_true(self):
        crs = CRS(4326)
        assert crs.is_geographic is True

    def test_is_geographic_false(self):
        crs = CRS(3857)
        assert crs.is_geographic is False


# -----------------------------------------------------------------------
# Equality & hashing
# -----------------------------------------------------------------------
class TestCRSEquality:
    def test_equal_same_epsg(self):
        assert CRS(4326) == CRS("EPSG:4326")

    def test_not_equal_different_epsg(self):
        assert CRS(4326) != CRS(3857)

    def test_hash_equal(self):
        assert hash(CRS(4326)) == hash(CRS.from_epsg(4326))

    def test_hash_in_set(self):
        s = {CRS(4326), CRS(3857), CRS("EPSG:4326")}
        assert len(s) == 2


# -----------------------------------------------------------------------
# to_dict for every named EPSG code
# -----------------------------------------------------------------------
class TestCRSToDict:
    def test_geographic_4326(self):
        d = CRS(4326).to_dict()
        assert d["proj"] == "longlat"
        assert d["datum"] == "WGS84"
        assert not any(k.startswith("_") for k in d)

    def test_utm_north_zone_32(self):
        d = CRS(32632).to_dict()
        assert d["proj"] == "utm"
        assert d["zone"] == 32
        assert d.get("south") is None or d.get("south") is False

    def test_utm_south_zone_55(self):
        d = CRS(32755).to_dict()
        assert d["proj"] == "utm"
        assert d["zone"] == 55
        assert d["south"] is True

    def test_utm_nad83_zone_10(self):
        d = CRS(26910).to_dict()
        assert d["proj"] == "utm"
        assert d["zone"] == 10
        assert d["datum"] == "NAD83"

    def test_lcc_2154(self):
        d = CRS(2154).to_dict()
        assert d["proj"] == "lcc"
        assert d["lon_0"] == 3
        assert d["lat_1"] == 49
        assert d["lat_2"] == 44

    def test_aea_5070(self):
        d = CRS(5070).to_dict()
        assert d["proj"] == "aea"
        assert d["lon_0"] == -96
        assert d["lat_1"] == 29.5
        assert d["lat_2"] == 45.5

    def test_web_mercator_3857(self):
        d = CRS(3857).to_dict()
        assert d["proj"] == "merc"
        assert d["datum"] == "WGS84"

    def test_laea_3035(self):
        d = CRS(3035).to_dict()
        assert d["proj"] == "laea"
        assert d["lon_0"] == 10
        assert d["lat_0"] == 52

    def test_stere_3031(self):
        d = CRS(3031).to_dict()
        assert d["proj"] == "stere"
        assert d["lat_0"] == -90
        assert d["lat_ts"] == -71

    def test_sterea_28992(self):
        d = CRS(28992).to_dict()
        assert d["proj"] == "sterea"
        assert d["k_0"] == 0.9999079

    def test_cea_6933(self):
        d = CRS(6933).to_dict()
        assert d["proj"] == "cea"
        assert d["lat_ts"] == 30


# -----------------------------------------------------------------------
# WKT round-trip
# -----------------------------------------------------------------------
class TestCRSWktRoundTrip:
    def test_roundtrip_geographic(self):
        crs = CRS(4326)
        wkt = crs.to_wkt()
        crs2 = CRS.from_wkt(wkt)
        assert crs2.to_epsg() == 4326

    def test_roundtrip_projected(self):
        crs = CRS(32632)
        wkt = crs.to_wkt()
        crs2 = CRS.from_wkt(wkt)
        assert crs2.to_epsg() == 32632

    def test_roundtrip_all_named(self):
        named_codes = [
            4326, 4269, 4267,
            3857, 3395,
            2154, 5070, 3035,
            3031, 3413, 3996,
            28992, 6933,
        ]
        for code in named_codes:
            crs = CRS(code)
            wkt = crs.to_wkt()
            crs2 = CRS.from_wkt(wkt)
            assert crs2.to_epsg() == code, f"round-trip failed for EPSG:{code}"

    def test_wkt_contains_authority(self):
        crs = CRS(4326)
        wkt = crs.to_wkt()
        assert 'AUTHORITY["EPSG","4326"]' in wkt


# -----------------------------------------------------------------------
# Two-tier CRS resolution (_crs_utils integration)
# -----------------------------------------------------------------------
try:
    import pyproj as _pyproj_mod
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

from xrspatial.reproject._crs_utils import _resolve_crs, _crs_from_wkt


class TestTwoTierResolution:
    def test_resolve_crs_int_uses_lite(self):
        result = _resolve_crs(4326)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_resolve_crs_string_uses_lite(self):
        result = _resolve_crs("EPSG:32632")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 32632

    @pytest.mark.skipif(not _HAS_PYPROJ, reason="pyproj not installed")
    def test_resolve_crs_unknown_falls_back(self):
        result = _resolve_crs(2193)
        assert not isinstance(result, CRS)
        assert hasattr(result, "to_epsg")
        assert result.to_epsg() == 2193

    def test_resolve_crs_none_returns_none(self):
        assert _resolve_crs(None) is None

    def test_resolve_crs_passes_through_lite_crs(self):
        lite = CRS(4326)
        result = _resolve_crs(lite)
        assert result is lite

    @pytest.mark.skipif(not _HAS_PYPROJ, reason="pyproj not installed")
    def test_resolve_crs_passes_through_pyproj(self):
        pp = _pyproj_mod.CRS.from_epsg(4326)
        result = _resolve_crs(pp)
        assert result is pp

    def test_crs_from_wkt_lite(self):
        wkt = CRS(4326).to_wkt()
        result = _crs_from_wkt(wkt)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326


# -----------------------------------------------------------------------
# Validate against pyproj (skipped when pyproj not installed)
# -----------------------------------------------------------------------
pyproj = pytest.importorskip("pyproj", reason="pyproj not installed")

# All named codes + a selection of UTM zones
_VALIDATE_CODES = [
    4326, 4269, 4267,
    3857, 3395,
    2154, 5070, 3035,
    3031, 3413, 3996,
    28992, 6933,
    32601, 32632, 32660,
    32701, 32755, 32760,
    26901, 26910, 26923,
]


# -----------------------------------------------------------------------
# transform_points
# -----------------------------------------------------------------------
import numpy as np
from xrspatial.reproject._projections import transform_points


class TestTransformPoints:
    def test_wgs84_to_web_mercator(self):
        xs = np.array([0.0, 10.0, -75.5])
        ys = np.array([0.0, 45.0, 40.0])
        src = pyproj.CRS.from_epsg(4326)
        tgt = pyproj.CRS.from_epsg(3857)
        result = transform_points(src, tgt, xs, ys)
        assert result is not None
        tx, ty = result
        # lon=0 lat=0 -> (0, 0) in Web Mercator
        assert abs(tx[0]) < 1.0
        assert abs(ty[0]) < 1.0
        # lon=10 -> positive easting
        assert tx[1] > 1e6

    def test_wgs84_to_utm_zone32(self):
        # Central meridian of UTM zone 32 is 9 deg E
        xs = np.array([9.0])
        ys = np.array([0.0])
        src = pyproj.CRS.from_epsg(4326)
        tgt = pyproj.CRS.from_epsg(32632)
        result = transform_points(src, tgt, xs, ys)
        assert result is not None
        tx, ty = result
        # On the central meridian, easting should be 500000
        assert abs(tx[0] - 500000.0) < 1.0

    def test_unsupported_pair_returns_none(self):
        # Two projected CRS -> no fast path
        src = pyproj.CRS.from_epsg(32632)
        tgt = pyproj.CRS.from_epsg(32633)
        result = transform_points(src, tgt, [500000], [0])
        assert result is None

    def test_matches_pyproj_transformer(self):
        # WGS84 -> Albers Equal Area (EPSG:5070), 20 random points
        rng = np.random.RandomState(42)
        xs = rng.uniform(-120, -70, 20)
        ys = rng.uniform(25, 50, 20)
        src = pyproj.CRS.from_epsg(4326)
        tgt = pyproj.CRS.from_epsg(5070)
        result = transform_points(src, tgt, xs, ys)
        assert result is not None
        tx, ty = result
        # Compare against pyproj
        transformer = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
        ref_x, ref_y = transformer.transform(xs, ys)
        # Tolerance is ~1 m because transform_points skips datum shifts
        # (metre-level error is sub-pixel for boundary estimation).
        np.testing.assert_allclose(tx, ref_x, atol=1.0)
        np.testing.assert_allclose(ty, ref_y, atol=1.0)


class TestCRSMatchesPyproj:
    @pytest.mark.parametrize("epsg", _VALIDATE_CODES)
    def test_to_dict_proj_key_matches(self, epsg):
        lite = CRS(epsg).to_dict()
        ref = pyproj.CRS.from_epsg(epsg).to_dict()
        assert lite["proj"] == ref["proj"], (
            f"EPSG:{epsg} proj mismatch: {lite['proj']} != {ref['proj']}"
        )

    @pytest.mark.parametrize("epsg", _VALIDATE_CODES)
    def test_is_geographic_matches(self, epsg):
        lite_geo = CRS(epsg).is_geographic
        ref_geo = pyproj.CRS.from_epsg(epsg).is_geographic
        assert lite_geo == ref_geo, (
            f"EPSG:{epsg} is_geographic mismatch: {lite_geo} != {ref_geo}"
        )

    @pytest.mark.parametrize("epsg", _VALIDATE_CODES)
    def test_equality_with_pyproj(self, epsg):
        a = CRS(epsg)
        b = CRS.from_epsg(epsg)
        assert a == b
