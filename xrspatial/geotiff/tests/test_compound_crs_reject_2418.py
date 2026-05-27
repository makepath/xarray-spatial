"""Reject compound EPSG codes on the stable-EPSG writer path (#2418).

The writer's stable-path emits only ``GeographicTypeGeoKey`` (2048) and
``ProjectedCSTypeGeoKey`` (3072). Both are spec'd to hold a 2D
horizontal CRS code. When a compound CRS (horizontal + vertical) is
passed as an integer EPSG, pyproj reports it as ``is_geographic`` or
``is_projected`` based on its horizontal sub-CRS, and the writer stores
the compound code in the horizontal slot. The resulting GeoTIFF reads
back through rasterio / GDAL as the horizontal sub-CRS (vertical
component silently dropped) or as no CRS at all.

These tests pin three behaviours:

1. The stable-EPSG path raises ``NonRepresentableEPSGCRSError`` for
   compound codes (EPSG:6349, EPSG:5498, EPSG:7415, EPSG:8360), at the
   kwarg entry point and via ``DataArray.attrs['crs']``.
2. The WKT fallback path still accepts compound CRSs via WKT --
   ``_wkt_to_epsg`` now returns None for compound CRSs so the writer
   routes through the user-defined CRS / citation path instead of
   writing a compound EPSG into a horizontal GeoKey.
3. Plain horizontal CRSs (EPSG:4326, EPSG:32633) still round-trip
   through rasterio with the input code intact, so the fix is scoped
   to the broken case.

Tests use rasterio (not xrspatial's own reader) for the round-trip
assertions, since the xrspatial reader masks the corruption by
re-resolving the stored integer through pyproj.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import NonRepresentableEPSGCRSError, open_geotiff, to_geotiff
from xrspatial.geotiff._crs import _reject_non_representable_epsg, _validate_crs_arg

pyproj = pytest.importorskip("pyproj")
rasterio = pytest.importorskip("rasterio")


# Compound EPSG codes that previously slipped through the validator.
# 6349: NAD83(2011) + NAVD88 height (compound, horizontal=geographic)
# 5498: NAD83 + NAVD88 height (compound, horizontal=geographic)
# 7415: Amersfoort/RD New + NAP height (compound, horizontal=projected)
# 8360: ETRS89 + EVRF2000 Austria height (compound, horizontal=geographic)
COMPOUND_EPSG_CODES = [6349, 5498, 7415, 8360]


def _square(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_validate_crs_arg_rejects_compound_epsg(epsg):
    """``_validate_crs_arg`` raises for compound EPSG codes."""
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        _validate_crs_arg(epsg)


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_to_geotiff_rejects_compound_epsg_kwarg(epsg):
    """End-to-end ``to_geotiff(crs=<compound>)`` raises."""
    buf = io.BytesIO()
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        to_geotiff(_square(), buf, crs=epsg)


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_to_geotiff_rejects_compound_epsg_attrs(epsg):
    """End-to-end ``DataArray.attrs['crs'] = <compound>`` raises."""
    arr = _square()
    arr.attrs['crs'] = epsg
    buf = io.BytesIO()
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        to_geotiff(arr, buf)


def test_reject_non_representable_epsg_helper():
    """The helper raises directly on a compound pyproj CRS object."""
    from pyproj import CRS
    crs_obj = CRS.from_epsg(6349)
    assert crs_obj.is_compound is True
    with pytest.raises(NonRepresentableEPSGCRSError, match="6349"):
        _reject_non_representable_epsg(6349, crs_obj)


def test_reject_non_representable_epsg_passes_horizontal():
    """The helper does not raise for plain horizontal CRSs."""
    from pyproj import CRS
    _reject_non_representable_epsg(4326, CRS.from_epsg(4326))  # geographic 2D
    _reject_non_representable_epsg(32633, CRS.from_epsg(32633))  # projected
    _reject_non_representable_epsg(3857, CRS.from_epsg(3857))  # web mercator


@pytest.mark.parametrize("epsg", [4326, 32633, 3857, 4269])
def test_horizontal_crs_still_round_trips_through_rasterio(epsg, tmp_path):
    """Regression: scoped fix doesn't break the normal stable-EPSG path."""
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / f"tmp_2418_{epsg}.tif"
    to_geotiff(arr, str(path), crs=epsg)
    with rasterio.open(str(path)) as src:
        assert src.crs is not None
        assert src.crs.to_epsg() == epsg


def test_compound_wkt_takes_user_defined_path(tmp_path):
    """A compound CRS passed as WKT writes via the citation path (#1768).

    ``_wkt_to_epsg`` returns None for a compound CRS so the writer
    treats it as a user-defined CRS rather than emitting a compound
    EPSG into the horizontal-CRS GeoKey slot.
    """
    from pyproj import CRS
    wkt = CRS.from_epsg(6349).to_wkt()
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_compound_wkt.tif"
    # The WKT-only path emits a UserWarning per #1768; we don't care
    # about that here, just that the write succeeds and the CRS info
    # survives in xrspatial's own attrs.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_geotiff(arr, str(path), crs=wkt)
    da = open_geotiff(str(path))
    # ``crs_wkt`` carries the full compound CRS string back through the
    # xrspatial reader (the WKT-only path's contract).
    assert da.attrs.get('crs_wkt') is not None
    assert 'COMPOUND' in da.attrs['crs_wkt'].upper()


def test_compound_epsg_corruption_surfaces_when_validator_bypassed(
    tmp_path, monkeypatch,
):
    """If the validator is bypassed, the file IS corrupted (regression).

    Monkey-patches ``_validate_crs_arg`` and
    ``_model_type_from_epsg``'s compound check to no-ops, then writes
    EPSG:6349 through the integer path. Asserts the file rasterio
    reads back is not EPSG:6349 -- the exact corruption the fix
    prevents. If a future refactor drops either guard, this test
    starts failing instead of the writer silently corrupting files.
    """
    from xrspatial.geotiff import _crs as _crs_mod
    from xrspatial.geotiff import _geotags as _geotags_mod

    # Bypass the validator at both gates so we exercise the broken
    # write path the production code now refuses.
    monkeypatch.setattr(
        _crs_mod, "_reject_non_representable_epsg", lambda *a, **kw: None
    )

    def _model_type_without_compound_check(crs_epsg):
        # Reproduce the pre-fix behaviour: only branch on is_geographic.
        from pyproj import CRS
        crs = CRS.from_epsg(crs_epsg)
        if crs.is_geographic:
            return _geotags_mod.MODEL_TYPE_GEOGRAPHIC
        return _geotags_mod.MODEL_TYPE_PROJECTED

    monkeypatch.setattr(
        _geotags_mod, "_model_type_from_epsg", _model_type_without_compound_check
    )

    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_bypass_demo.tif"
    to_geotiff(arr, str(path), crs=6349)

    # Pin the corruption: rasterio either fails to recover the EPSG
    # entirely (``to_epsg() is None``) or recovers the wrong one
    # (the horizontal sub-CRS, with the vertical component dropped).
    # Both outcomes are caught by ``rasterio_epsg != 6349``.
    with rasterio.open(str(path)) as src:
        rasterio_epsg = src.crs.to_epsg() if src.crs is not None else None
    assert rasterio_epsg != 6349, (
        f"rasterio recovered EPSG:6349 from a write that bypassed the "
        f"validator: got {rasterio_epsg}. Either the writer is now "
        "correctly emitting compound GeoKeys (good, drop the test) or "
        "the bypass no longer reaches the broken path (unlikely)."
    )

    # Sanity: when validation is back in place, the same write raises.
    monkeypatch.undo()
    with pytest.raises(NonRepresentableEPSGCRSError):
        to_geotiff(arr, str(path), crs=6349)


def test_geographic_3d_crs_still_works(tmp_path):
    """EPSG:4979 (3D WGS84) is geographic, not compound: still allowed.

    The fix is narrow -- it only rejects compound CRSs, which are the
    demonstrably-broken case. 3D-geographic and other non-2D-horizontal
    EPSGs that round-trip cleanly through rasterio remain allowed.
    """
    from pyproj import CRS
    crs_obj = CRS.from_epsg(4979)
    assert crs_obj.is_compound is False
    # Should not raise from the validator.
    _validate_crs_arg(4979)
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_4979.tif"
    to_geotiff(arr, str(path), crs=4979)
    with rasterio.open(str(path)) as src:
        # rasterio resolves EPSG:4979 correctly.
        assert src.crs is not None
        assert src.crs.to_epsg() == 4979


def test_compound_epsg_error_message_points_to_workaround():
    """Error message names the compound CRS and offers a route forward."""
    with pytest.raises(NonRepresentableEPSGCRSError) as exc:
        _validate_crs_arg(6349)
    msg = str(exc.value)
    assert "6349" in msg
    assert "compound" in msg.lower()
    # The message should point at both the WKT fallback and the
    # horizontal-sub-CRS workaround so the caller can act on it.
    assert "WKT" in msg or "wkt" in msg
    assert "horizontal" in msg.lower()
