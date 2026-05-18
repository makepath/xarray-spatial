"""Canonical round-trip invariants for the geotiff module (issue #1986).

This module enumerates the supported round-trip cases and pins the
invariant per case. Two invariants are in scope:

* **byte-equivalent for pixels** -- ``np.array_equal`` between the
  source array and the array read back after one or more
  ``read -> write -> read`` cycles. Listed attrs are bit-equal.
* **semantic-equivalent** -- the array and attrs match up to documented
  normalisations: dtype promotion at the int-with-nodata boundary
  (sentinel pixels become NaN), CRS string -> int EPSG, transform tuple
  comparison up to float precision.

Every case also asserts **fixed-point convergence**: after the first
``read -> write -> read`` produces ``da1``, the second cycle producing
``da2`` must match ``da1`` exactly. One round is enough to detect drift
because the writer is deterministic given the same input attrs.

Cases NOT covered here (deferred to follow-up PRs):

* float with non-NaN declared nodata -- requires the masked / declared
  nodata split from issue #1988 to express the invariant cleanly.
* planar multiband -- the writer emits chunky only.
* overviews / COG -- separate complexity, deferred.
* VRT round-trip -- XML normalisation rules deferred.

The per-incident round-trip test files
(``test_metadata_round_trip_1484.py``,
``test_int_coords_round_trip_hotfix_1962.py``, etc.) stay as regression
markers for their bug numbers. This module is the canonical contract
the writer must satisfy going forward.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_gt() -> GeoTransform:
    return GeoTransform(
        origin_x=500000.0, origin_y=4000000.0,
        pixel_width=30.0, pixel_height=-30.0,
    )


def _read_write_read(da: xr.DataArray, tmp_path, tag: str) -> xr.DataArray:
    """Run one ``write -> read`` cycle on ``da`` and return the new DataArray."""
    path = str(tmp_path / f"rt_{tag}_1986.tif")
    to_geotiff(da, path, compression='none', tiled=False)
    return open_geotiff(path)


# Canonical attrs whose values must lock across a write -> read cycle
# whenever both reads have the key.
_LOCKED_ATTRS = ('crs', 'transform', 'nodata', 'raster_type')


def _assert_fixed_point(da1: xr.DataArray, da2: xr.DataArray) -> None:
    """``da1`` and ``da2`` come from two consecutive write -> read cycles.

    They must agree on pixels, dtype, dims, the attrs key set, and the
    values of the canonical attrs listed in ``_LOCKED_ATTRS``. Other
    attrs (best-effort pass-through) are only checked for presence.
    """
    assert da1.dtype == da2.dtype, (
        f"dtype drift: {da1.dtype} -> {da2.dtype}")
    assert da1.dims == da2.dims, (
        f"dims drift: {da1.dims} -> {da2.dims}")
    if np.issubdtype(da1.dtype, np.floating):
        np.testing.assert_array_equal(
            np.isnan(da1.values), np.isnan(da2.values))
        mask = ~np.isnan(da1.values)
        np.testing.assert_array_equal(
            da1.values[mask], da2.values[mask])
    else:
        np.testing.assert_array_equal(da1.values, da2.values)
    assert set(da1.attrs) == set(da2.attrs), (
        f"attrs key drift: {set(da1.attrs) ^ set(da2.attrs)}")
    for key in _LOCKED_ATTRS:
        if key in da1.attrs:
            v1, v2 = da1.attrs[key], da2.attrs[key]
            if key == 'transform':
                # Transform is a tuple of floats; compare up to float
                # precision rather than bit-equal.
                assert len(v1) == len(v2), (
                    f"transform length drift: {len(v1)} -> {len(v2)}")
                for a, b in zip(v1, v2):
                    assert a == pytest.approx(b, abs=1e-15, rel=1e-12), (
                        f"transform value drift: {v1} -> {v2}")
            else:
                assert v1 == v2, (
                    f"attrs[{key!r}] drift: {v1!r} -> {v2!r}")


# ---------------------------------------------------------------------------
# Case: int single-band, no nodata
# Invariant: byte-equivalent for pixels and dtype.
# ---------------------------------------------------------------------------

class TestIntSingleBandNoNodata:
    """``int32`` raster with no declared nodata. Canonical byte-equivalent
    case: the file should round-trip pixels and dtype with no drift."""

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint16])
    def test_byte_equivalent_pixels(self, tmp_path, dtype):
        arr = np.arange(20, dtype=dtype).reshape(4, 5)
        path = str(tmp_path / f"int_nond_{np.dtype(dtype).name}_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=32610,
              compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert da1.dtype == dtype
        np.testing.assert_array_equal(da1.values, arr)
        assert 'nodata' not in da1.attrs, (
            "int raster without declared nodata must not gain a sentinel")

        da2 = _read_write_read(da1, tmp_path, f"int_nond_{np.dtype(dtype).name}")
        np.testing.assert_array_equal(da2.values, arr)
        assert da2.dtype == dtype
        assert 'nodata' not in da2.attrs

        _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Case: float single-band, no nodata
# Invariant: byte-equivalent for pixels.
# ---------------------------------------------------------------------------

class TestFloatSingleBandNoNodata:
    """``float32`` / ``float64`` raster with no declared nodata. Byte-
    equivalent for pixels, including any NaN entries the source happens
    to carry (NaN-as-data, not NaN-as-sentinel)."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_byte_equivalent_pixels(self, tmp_path, dtype):
        arr = np.linspace(-1.0, 1.0, 20, dtype=dtype).reshape(4, 5)
        path = str(tmp_path / f"float_nond_{np.dtype(dtype).name}_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=32610,
              compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert da1.dtype == dtype
        np.testing.assert_array_equal(da1.values, arr)

        da2 = _read_write_read(da1, tmp_path, f"float_nond_{np.dtype(dtype).name}")
        np.testing.assert_array_equal(da2.values, arr)
        assert da2.dtype == dtype

        _assert_fixed_point(da1, da2)

    def test_nan_in_data_preserved(self, tmp_path):
        """Source pixels that are already NaN survive the round-trip when
        no nodata was declared. The writer does not invent a sentinel."""
        arr = np.array([[0.0, np.nan, 2.0],
                        [3.0, 4.0, np.nan]], dtype=np.float32)
        path = str(tmp_path / "float_nan_in_data_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=32610,
              compression='none', tiled=False)

        da1 = open_geotiff(path)
        np.testing.assert_array_equal(np.isnan(da1.values), np.isnan(arr))
        mask = ~np.isnan(arr)
        np.testing.assert_array_equal(da1.values[mask], arr[mask])
        assert 'nodata' not in da1.attrs, (
            "NaN-as-data must not get a synthetic GDAL_NODATA tag")

        da2 = _read_write_read(da1, tmp_path, "float_nan_in_data")
        _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Case: int single-band with declared nodata
# Invariant: documented drift -- dtype promotes to float64 with NaN at
# the sentinel positions. Valid pixels are byte-equal. The sentinel
# value survives in attrs['nodata'].
# ---------------------------------------------------------------------------

class TestIntWithDeclaredNodata:
    """Integer raster with a declared nodata sentinel.

    Current behaviour: the reader promotes integer-with-sentinel rasters
    to ``float64`` and masks sentinel pixels to NaN. This module
    documents that drift as the contract for now. A future write path
    informed by issue #1988's masked / declared nodata split may restore
    the original int dtype on the next read; that change will tighten
    the invariant to byte-equivalent.
    """

    def test_int32_sentinel_promotes_and_masks(self, tmp_path):
        arr = np.array([[1, 2, 3], [-9999, 5, 6]], dtype=np.int32)
        path = str(tmp_path / "int32_nd_1986.tif")
        write(arr, path, nodata=-9999, geo_transform=_default_gt(),
              crs_epsg=4326, compression='none', tiled=False)

        da1 = open_geotiff(path)
        # Dtype drift: int -> float64 with NaN at sentinel.
        assert da1.dtype == np.float64
        assert np.isnan(da1.values[1, 0])
        # Valid pixels are byte-equal as floats.
        valid = ~np.isnan(da1.values)
        np.testing.assert_array_equal(
            da1.values[valid].astype(np.int32),
            arr[arr != -9999],
        )
        # Sentinel value preserved in attrs.
        assert da1.attrs.get('nodata') == -9999

        # Fixed point: a second cycle from da1 reproduces da1 exactly.
        da2 = _read_write_read(da1, tmp_path, "int32_nd")
        _assert_fixed_point(da1, da2)
        assert da2.attrs.get('nodata') == da1.attrs.get('nodata')

    def test_uint16_sentinel_promotes_and_masks(self, tmp_path):
        arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
        path = str(tmp_path / "uint16_nd_1986.tif")
        write(arr, path, nodata=65535, geo_transform=_default_gt(),
              crs_epsg=4326, compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert da1.dtype == np.float64
        assert np.isnan(da1.values[1, 0])
        assert da1.attrs.get('nodata') == 65535

        da2 = _read_write_read(da1, tmp_path, "uint16_nd")
        _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Case: multiband chunky
# Invariant: byte-equivalent for pixels, dtype, dims, samples-per-pixel.
# ---------------------------------------------------------------------------

class TestMultibandChunky:
    """Multi-band chunky (interleaved) raster. The writer emits chunky
    by default; planar layout is not currently supported and is out of
    scope for this PR."""

    @pytest.mark.parametrize("nbands", [2, 3, 4])
    def test_byte_equivalent_pixels(self, tmp_path, nbands):
        arr = np.arange(4 * 5 * nbands, dtype=np.uint8).reshape(4, 5, nbands)
        path = str(tmp_path / f"mb_chunky_{nbands}_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=32610,
              compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert da1.shape == arr.shape
        assert da1.dtype == np.uint8
        assert da1.dims == ('y', 'x', 'band')
        np.testing.assert_array_equal(np.asarray(da1.values), arr)
        assert 'nodata' not in da1.attrs

        da2 = _read_write_read(da1, tmp_path, f"mb_chunky_{nbands}")
        np.testing.assert_array_equal(np.asarray(da2.values), arr)
        assert 'nodata' not in da2.attrs
        _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Case: no-georef (no ModelPixelScale, no ModelTiepoint, no GeoKeys)
# Invariant: semantic-equivalent. The transform attr is absent on read,
# stays absent through round-trip, and integer pixel coords survive.
# ---------------------------------------------------------------------------

class TestNoGeorefSemantic:
    """File with no GeoTIFF tags reads back with integer pixel coords
    and no ``transform`` attr. The writer must not synthesise a
    transform from those integer coords (issue #1949 fix).

    This case is intentionally narrower than
    ``test_no_georef_writer_round_trip_1949.py``: that file is the
    incident regression; here we assert the canonical contract.
    """

    def test_no_transform_survives_round_trip(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        src = str(tmp_path / "no_georef_src_1986.tif")
        arr = np.random.default_rng(seed=1986).random((4, 5)).astype(np.float32)
        tifffile.imwrite(src, arr, photometric='minisblack',
                         planarconfig='contig', metadata=None)

        da1 = open_geotiff(src)
        assert da1.dtype == np.float32
        assert 'transform' not in da1.attrs
        assert 'crs' not in da1.attrs
        np.testing.assert_array_equal(da1.values, arr)

        da2 = _read_write_read(da1, tmp_path, "no_georef")
        assert 'transform' not in da2.attrs
        assert 'crs' not in da2.attrs
        np.testing.assert_array_equal(da2.values, arr)
        _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Case: PixelIsPoint (GeoTIFF raster_type = 2)
# Invariant: byte-equivalent for raster_type attr and pixels.
# ---------------------------------------------------------------------------

class TestPixelIsPointRoundTrip:
    """``raster_type=2`` (PixelIsPoint) must survive read -> write -> read.

    The reader maps the GeoKey RasterTypeGeoKey (1025) value 2 to
    ``attrs['raster_type'] == 'point'``; the writer must re-emit
    RasterTypeGeoKey = 2 when that attr is present.
    """

    def test_raster_type_point_preserved(self, tmp_path):
        arr = np.arange(20, dtype=np.int32).reshape(4, 5)
        path = str(tmp_path / "pip_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=4326,
              raster_type=2, compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert da1.attrs.get('raster_type') == 'point'
        np.testing.assert_array_equal(da1.values, arr)

        da2 = _read_write_read(da1, tmp_path, "pip")
        assert da2.attrs.get('raster_type') == 'point'
        np.testing.assert_array_equal(da2.values, arr)
        _assert_fixed_point(da1, da2)

    def test_default_raster_type_area_absent_from_attrs(self, tmp_path):
        """``raster_type=1`` (PixelIsArea, the default) does not appear
        as an explicit attr; absence is the canonical signal."""
        arr = np.arange(20, dtype=np.int32).reshape(4, 5)
        path = str(tmp_path / "pia_1986.tif")
        write(arr, path, geo_transform=_default_gt(), crs_epsg=4326,
              raster_type=1, compression='none', tiled=False)

        da1 = open_geotiff(path)
        assert 'raster_type' not in da1.attrs, (
            "PixelIsArea default must not surface as an explicit "
            "'raster_type' attr; absence is the canonical signal")

        da2 = _read_write_read(da1, tmp_path, "pia")
        assert 'raster_type' not in da2.attrs
        _assert_fixed_point(da1, da2)
