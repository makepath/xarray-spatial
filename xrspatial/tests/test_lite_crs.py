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
        with pytest.raises(ValueError, match="Unknown EPSG"):
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
