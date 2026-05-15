"""Regression tests for issue #1929.

The CRS resolution path used to swallow ``pyproj`` parse failures with
only a warning, then write the original unvalidatable string verbatim
into ``GTCitationGeoKey``. A caller who passed
``to_geotiff(crs="EPSG:4326")`` on a host without pyproj ended up with
the literal string ``"EPSG:4326"`` in the citation field; non-libgeotiff
readers drop the projection in that case. The same hole existed for
typo'd PROJ strings and free-form garbage even with pyproj installed.

The fix:

* Adds ``_validate_crs_fallback`` in ``_crs.py``. It refuses to land any
  non-WKT-shaped string in the citation when the caller has not opted
  in via ``allow_unparseable_crs=True``.
* Adds ``allow_unparseable_crs`` to ``to_geotiff``, ``write_geotiff_gpu``,
  and the dispatch through to ``_write_vrt_tiled``.

The validator is intentionally cheap (a ``str.startswith`` over the WKT
root keywords) so it stays in the hot write path.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._crs import (
    _looks_like_wkt,
    _validate_crs_fallback,
    _WKT_ROOT_KEYWORDS,
)


def _make_da() -> xr.DataArray:
    """Plain numpy-backed DataArray with no CRS attrs."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


class TestLooksLikeWkt:
    """Structural WKT recognition. Cheap and pyproj-free."""

    @pytest.mark.parametrize("root", _WKT_ROOT_KEYWORDS)
    def test_each_root_keyword_recognised(self, root):
        assert _looks_like_wkt(f'{root}["WGS 84", ...]')

    def test_leading_whitespace_tolerated(self):
        assert _looks_like_wkt('   PROJCS["UTM"]')

    def test_case_insensitive(self):
        assert _looks_like_wkt('projcs["UTM"]')
        assert _looks_like_wkt('GeOgCs["WGS 84"]')

    def test_epsg_token_rejected(self):
        assert not _looks_like_wkt("EPSG:4326")

    def test_proj_string_rejected(self):
        assert not _looks_like_wkt("+proj=utm +zone=10")

    def test_garbage_rejected(self):
        assert not _looks_like_wkt("not a CRS at all")

    def test_empty_string_rejected(self):
        assert not _looks_like_wkt("")

    def test_non_string_rejected(self):
        assert not _looks_like_wkt(None)
        assert not _looks_like_wkt(4326)


class TestValidateCrsFallback:
    """Direct unit tests on the validator helper."""

    def test_none_fallback_returns(self):
        _validate_crs_fallback(None, allow_unparseable_crs=False)

    def test_wkt_shaped_returns(self):
        _validate_crs_fallback(
            'PROJCS["test", ...]', allow_unparseable_crs=False
        )

    def test_non_wkt_raises(self):
        with pytest.raises(ValueError, match="GTCitationGeoKey"):
            _validate_crs_fallback("EPSG:4326", allow_unparseable_crs=False)

    def test_opt_in_allows_non_wkt(self):
        _validate_crs_fallback("EPSG:4326", allow_unparseable_crs=True)

    def test_message_names_the_offending_string(self):
        with pytest.raises(ValueError) as exc:
            _validate_crs_fallback("frobnicate", allow_unparseable_crs=False)
        assert "frobnicate" in str(exc.value)
        assert "allow_unparseable_crs" in str(exc.value)


class TestToGeotiffCrsFailClosed:
    """End-to-end through the public ``to_geotiff`` entry point.

    Default behaviour now raises on unvalidatable CRS strings, and the
    opt-in restores the pre-#1929 citation-only write.
    """

    def test_epsg_int_unchanged(self, tmp_path):
        """An int EPSG kwarg never lands in the fallback path."""
        out = str(tmp_path / "epsg_int_1929.tif")
        to_geotiff(_make_da(), out, crs=4326)
        assert os.path.exists(out)

    def test_valid_wkt_unchanged(self, tmp_path):
        """A WKT-shaped string is accepted by the structural check
        even without pyproj-driven validation."""
        out = str(tmp_path / "wkt_shaped_1929.tif")
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
            '6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]]'
        )
        to_geotiff(_make_da(), out, crs=wkt)
        assert os.path.exists(out)

    def test_epsg_token_string_via_kwarg(self, tmp_path):
        """``crs="EPSG:4326"`` resolves to int 4326 when pyproj is
        installed (so ``wkt_fallback`` stays None). Without pyproj the
        validator would refuse. We assert the success path here; the
        no-pyproj path is exercised by the validator unit tests."""
        out = str(tmp_path / "epsg_token_1929.tif")
        pytest.importorskip("pyproj")
        to_geotiff(_make_da(), out, crs="EPSG:4326")
        assert os.path.exists(out)

    def test_garbage_string_kwarg_raises(self, tmp_path):
        """A free-form non-WKT, non-PROJ string raises by default."""
        out = str(tmp_path / "garbage_kwarg_1929.tif")
        # Even with pyproj installed, ``"absolute-garbage"`` fails to
        # parse, ``_wkt_to_epsg`` returns None with a warning, the
        # writer lands it as ``wkt_fallback``, and the validator
        # refuses it. Filter the warning so it does not pollute the
        # test report.
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(_make_da(), out, crs="absolute-garbage")

    def test_opt_in_allows_garbage(self, tmp_path):
        """``allow_unparseable_crs=True`` restores the citation-only
        write. The file still emits a ``UserWarning`` from the geokey
        builder, mirroring the pre-#1929 behaviour."""
        out = str(tmp_path / "opt_in_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_da(), out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)

    def test_garbage_string_attr_raises(self, tmp_path):
        """The same guard fires when the bad string arrives via
        ``attrs['crs']`` instead of the ``crs=`` kwarg."""
        out = str(tmp_path / "garbage_attr_1929.tif")
        da = _make_da()
        da.attrs["crs"] = "still-garbage"
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(da, out)

    def test_no_crs_at_all_unchanged(self, tmp_path):
        """No CRS supplied means no GTCitationGeoKey is written; the
        validator is a no-op."""
        out = str(tmp_path / "no_crs_1929.tif")
        to_geotiff(_make_da(), out)
        assert os.path.exists(out)

    def test_message_recommends_alternatives(self, tmp_path):
        """The error message points users at the four options:
        EPSG int, real WKT, install pyproj, or opt in."""
        out = str(tmp_path / "msg_check_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError) as exc:
                to_geotiff(_make_da(), out, crs="bogus")
        msg = str(exc.value)
        assert "EPSG" in msg
        assert "WKT" in msg
        assert "allow_unparseable_crs" in msg
