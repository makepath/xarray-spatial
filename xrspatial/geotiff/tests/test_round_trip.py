"""Canonical round-trip invariants for the geotiff module.

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

Corpus-backed cases (using the golden corpus fixtures):

* planar multiband -- in: ``PLANARCONFIG=2`` (separate). Out: chunky.
  The writer emits chunky only, so the on-disk layout drifts but pixel
  bytes survive (verified against the planar-separate RGB fixture).
* overviews (internal IFD) -- base IFD pixels round-trip byte-equal
  and the overview factor list is preserved when the writer is asked
  to re-emit the same pyramid. Per-pixel overview parity is verified
  by the oracle suite, not here.
* COG layout -- base IFD pixels round-trip byte-equal and overview
  factors are preserved; the GDAL ``LAYOUT=COG`` ghost-IFD marker
  does NOT re-emit (xrspatial's writer does not synthesise the
  ghost-IFD layout block).
* sparse tiled -- elided zero tiles materialise as zeros on read; the
  rewrite is a normal tiled GeoTIFF whose pixels match those zeros.
* VRT mosaic -- read of a ``.vrt`` is semantically equivalent to a
  ``np.concatenate`` of the source pixels; the rewrite is a plain
  GeoTIFF whose pixels match.

Cases NOT covered here:

* float with non-NaN declared nodata -- requires the masked / declared
  nodata split to express the invariant cleanly.

The per-incident round-trip test coverage (e.g. ``unit/test_metadata.py``)
stays as a regression marker for the underlying bugs. This module is the
canonical contract the writer must satisfy going forward.
"""
from __future__ import annotations

import math
import os
import pathlib

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from xrspatial.geotiff import open_geotiff, to_geotiff, write_vrt
from xrspatial.geotiff._geotags import _NO_GEOREF_KEY, GeoTransform
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
    built on a masked / declared nodata split may restore the original
    int dtype on the next read; that change will tighten the invariant
    to byte-equivalent.
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
    by default; planar layout is not currently supported."""

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
    transform from those integer coords.

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


# ---------------------------------------------------------------------------
# Corpus-backed cases (golden corpus fixtures)
#
# The fixtures referenced below ship with the golden corpus. The tests
# below pull each fixture, run it through a ``read -> write -> read``
# cycle, and pin the canonical invariant. No new fixtures are added;
# coverage is extended by reusing the corpus.
#
# Several corpus fixtures still carry deprecated geographic attrs
# (``geog_citation``, ``angular_units``, ``semi_major_axis``,
# ``inv_flattening``). Reading them emits ``DeprecationWarning`` which
# is informative noise here, not a regression -- the warning module
# (``test_attrs_pr7_deprecate_geographic_1984.py``) already pins that
# contract. Filter them out for these cases so the round-trip
# assertions stay readable. Once the deprecated attrs are removed,
# this filter goes away.
# ---------------------------------------------------------------------------

# Class-level filter applied to every corpus-backed test class below.
# The fixtures emit deprecation warnings on read for the deprecated
# geographic-GeoKey attrs; that contract is locked elsewhere and is
# noise for the round-trip tests.
_CORPUS_DEPRECATION_FILTER = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:xrspatial.geotiff"
)

_CORPUS_FIXTURES_DIR = (
    pathlib.Path(__file__).resolve().parent / "golden_corpus" / "fixtures"
)


def _corpus_fixture(name: str) -> pathlib.Path:
    """Return a corpus fixture path, skipping if it has not been generated.

    The corpus is built by ``python -m
    xrspatial.geotiff.tests.golden_corpus.generate``; CI runs that step
    before the test suite. Locally, the fixtures may be absent on a
    fresh checkout, in which case the affected case skips with a
    pointer to the generator command.
    """
    p = _CORPUS_FIXTURES_DIR / name
    if not p.exists():
        pytest.skip(
            f"corpus fixture {name!r} not generated; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )
    return p


# ---------------------------------------------------------------------------
# Case: planar-separate multiband (corpus fixture ``planar_separate_uint8_rgb``)
# Invariant: byte-equivalent for pixels and dtype; PlanarConfiguration
# drifts because the writer emits chunky only. The reader normalises
# both layouts to ``(y, x, band)`` so the in-memory shape survives.
# ---------------------------------------------------------------------------

@_CORPUS_DEPRECATION_FILTER
class TestPlanarMultibandFromCorpus:
    """Planar-separate (``PLANARCONFIG=2``) RGB raster from the corpus.

    On read, the reader normalises both planar and chunky layouts to
    ``(y, x, band)``. On write, ``to_geotiff`` only emits chunky. The
    re-read therefore comes back chunky, but the in-memory pixels and
    dtype are byte-equal to the first read. This is the explicit
    contract for the planar case: layout drift on disk, byte-equal
    pixels in memory.
    """

    FIXTURE_NAME = "planar_separate_uint8_rgb.tif"

    def test_byte_equivalent_pixels_layout_drifts(self, tmp_path):
        src = _corpus_fixture(self.FIXTURE_NAME)
        da1 = open_geotiff(str(src))
        assert da1.dims == ('y', 'x', 'band')
        assert da1.dtype == np.uint8
        assert da1.shape[-1] == 3

        da2 = _read_write_read(da1, tmp_path, "planar_corpus")
        assert da2.dims == da1.dims
        assert da2.dtype == da1.dtype
        assert da2.shape == da1.shape
        np.testing.assert_array_equal(
            np.asarray(da2.values), np.asarray(da1.values))
        # Fixed point: corpus fixtures carry deprecated geographic attrs
        # (``geog_citation`` etc.) that the writer cannot reconstruct,
        # so the first cycle (``src -> da1 -> da2``) cannot satisfy
        # ``_assert_fixed_point``. Run a second cycle from ``da2`` and
        # assert convergence from there: once the deprecated attrs have
        # dropped off, the writer must hold the fixed point.
        da3 = _read_write_read(da2, tmp_path, "planar_corpus_2")
        _assert_fixed_point(da2, da3)


# ---------------------------------------------------------------------------
# Case: internal-IFD overviews (corpus fixture ``overview_internal_uint16``)
# Invariant: base IFD pixels are byte-equal; overview factors are
# preserved when the writer is asked to emit the same pyramid.
# ---------------------------------------------------------------------------

@_CORPUS_DEPRECATION_FILTER
class TestOverviewInternalFromCorpus:
    """Internal-IFD overview pyramid from the corpus.

    ``open_geotiff`` (without ``overview_level=``) returns the base IFD
    only. The round-trip asserts:

    * base pixels and dtype byte-equal after a write that re-emits the
      same overview factors;
    * the writer reports the same factors back on rasterio probe.

    Overview *pixel* equivalence is semantic, not byte-equal: the
    writer recomputes overviews from the base bytes through its own
    reducer rather than copying the original IFDs. This module pins
    factor preservation; pixel-level overview parity belongs to the
    oracle suite.
    """

    FIXTURE_NAME = "overview_internal_uint16.tif"

    def test_base_pixels_and_factors_preserved(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        src = _corpus_fixture(self.FIXTURE_NAME)
        da1 = open_geotiff(str(src))
        assert da1.dtype == np.uint16
        with rasterio.open(str(src)) as h:
            src_factors = h.overviews(1)
        assert src_factors, (
            f"corpus fixture {self.FIXTURE_NAME} must declare overview "
            f"factors; got {src_factors!r}"
        )

        out = tmp_path / "overview_internal_rt_1986.tif"
        to_geotiff(da1, str(out), compression='none',
                   cog=True, overview_levels=list(src_factors))
        with rasterio.open(str(out)) as h:
            out_factors = h.overviews(1)
            out_base = h.read(1)
        assert out_factors == src_factors, (
            f"overview factor drift: {src_factors!r} -> {out_factors!r}"
        )
        np.testing.assert_array_equal(out_base, np.asarray(da1.values))

        da2 = open_geotiff(str(out))
        # Fixed point from da2 onward (deprecated geographic attrs on
        # the corpus fixture mean the first cycle drops keys).
        out2 = tmp_path / "overview_internal_rt2_1986.tif"
        to_geotiff(da2, str(out2), compression='none',
                   cog=True, overview_levels=list(src_factors))
        da3 = open_geotiff(str(out2))
        _assert_fixed_point(da2, da3)


# ---------------------------------------------------------------------------
# Case: COG layout (corpus fixture ``cog_internal_overview_uint16``)
# Invariant: base IFD pixels byte-equal; overview factors preserved.
# The GDAL ``LAYOUT=COG`` ghost-IFD marker does NOT round-trip --
# xrspatial's writer does not emit it, even when ``cog=True``. That
# is documented drift in this case; downstream consumers that need the
# marker should write through GDAL.
# ---------------------------------------------------------------------------

@_CORPUS_DEPRECATION_FILTER
class TestCOGFromCorpus:
    """COG fixture from the corpus.

    Round-trip via ``to_geotiff(cog=True)`` preserves the base IFD
    bytes and the overview factor list. The ``LAYOUT=COG``
    ``IMAGE_STRUCTURE`` marker (a GDAL artefact) does not survive
    because the xrspatial writer does not emit a ghost-IFD layout
    block. That is the documented semantic drift for this case.
    """

    FIXTURE_NAME = "cog_internal_overview_uint16.tif"

    def test_cog_base_and_factors_preserved_marker_drifts(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        src = _corpus_fixture(self.FIXTURE_NAME)
        da1 = open_geotiff(str(src))
        # The corpus fixture's ``LAYOUT=COG`` marker is already pinned by
        # ``golden_corpus/test_overview_cog.test_cog_fixture_carries_cog_layout_marker``.
        # Here we only need the factor list to drive the rewrite.
        with rasterio.open(str(src)) as h:
            src_factors = h.overviews(1)

        out = tmp_path / "cog_rt_1986.tif"
        to_geotiff(da1, str(out), compression='none',
                   cog=True, overview_levels=list(src_factors))
        with rasterio.open(str(out)) as h:
            out_factors = h.overviews(1)
            out_base = h.read(1)
            out_layout = h.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        assert out_factors == src_factors
        np.testing.assert_array_equal(out_base, np.asarray(da1.values))
        assert out_layout != "COG", (
            "xrspatial writer must not silently mint a LAYOUT=COG ghost "
            "marker; that block is a GDAL artefact. If a future writer "
            "starts emitting it, tighten this invariant rather than "
            "letting the marker drift back in unannounced."
        )

        da2 = open_geotiff(str(out))
        # Fixed point from da2 onward (deprecated geographic attrs on
        # the corpus fixture mean the first cycle drops keys).
        out2 = tmp_path / "cog_rt2_1986.tif"
        to_geotiff(da2, str(out2), compression='none',
                   cog=True, overview_levels=list(src_factors))
        da3 = open_geotiff(str(out2))
        _assert_fixed_point(da2, da3)


# ---------------------------------------------------------------------------
# Case: sparse tiled (corpus fixture ``sparse_tiled_uint16``)
# Invariant: byte-equivalent for pixels (all zeros materialised); the
# rewrite is a normal tiled GeoTIFF whose pixels match. Sparseness on
# disk does NOT round-trip -- the rewrite writes every tile.
# ---------------------------------------------------------------------------

@_CORPUS_DEPRECATION_FILTER
class TestSparseTiledFromCorpus:
    """Sparse-tiled fixture from the corpus.

    The reader materialises every elided tile as zeros, so the
    in-memory array is a plain all-zero raster. The rewrite is a normal
    tiled GeoTIFF that carries non-zero TileByteCounts; sparseness on
    disk does not round-trip. Pixel bytes do.
    """

    FIXTURE_NAME = "sparse_tiled_uint16.tif"

    def test_zeros_materialise_and_round_trip(self, tmp_path):
        src = _corpus_fixture(self.FIXTURE_NAME)
        da1 = open_geotiff(str(src))
        assert da1.dtype == np.uint16
        assert (np.asarray(da1.values) == 0).all(), (
            "sparse fixture must materialise as an all-zero raster "
            "on read; non-zero pixels mean the elided-tile decoder "
            "regressed"
        )

        da2 = _read_write_read(da1, tmp_path, "sparse_corpus")
        np.testing.assert_array_equal(
            np.asarray(da2.values), np.asarray(da1.values))
        # Fixed point from da2 onward (deprecated geographic attrs on
        # the corpus fixture mean the first cycle drops keys).
        da3 = _read_write_read(da2, tmp_path, "sparse_corpus_2")
        _assert_fixed_point(da2, da3)


# ---------------------------------------------------------------------------
# Case: VRT mosaic (built from corpus fixtures ``dtype_uint8`` / ``dtype_uint16``)
# Invariant: ``open_geotiff(vrt)`` is semantically equivalent to a
# ``np.concatenate`` of the source pixels; the rewrite is a plain
# GeoTIFF whose pixels match the VRT read byte-for-byte.
# ---------------------------------------------------------------------------

@_CORPUS_DEPRECATION_FILTER
class TestVRTRoundTripFromCorpus:
    """VRT mosaic round-trip.

    The VRT writer takes two GeoTIFF sources from the corpus and wires
    up a horizontal mosaic; ``open_geotiff`` resolves the VRT and
    returns the concatenated pixels. The round-trip writes that
    in-memory array back as a plain GeoTIFF (no VRT) and asserts the
    re-read matches the original VRT read byte-for-byte. The VRT XML
    itself does not round-trip -- the writer emits a single TIFF, not
    a VRT pointing at sources. Use ``write_vrt`` explicitly when a VRT
    is the desired output.
    """

    SOURCE_FIXTURES = ("dtype_uint8.tif", "dtype_uint16.tif")

    @pytest.mark.parametrize("source_name", SOURCE_FIXTURES)
    def test_vrt_mosaic_round_trips_as_geotiff(self, tmp_path, source_name):
        rasterio = pytest.importorskip("rasterio")

        src_path = _corpus_fixture(source_name)
        with rasterio.open(str(src_path)) as h:
            data = h.read(1)
            profile = h.profile.copy()
            t = h.transform
        pw, ox, oy, ph = float(t.a), float(t.c), float(t.f), float(t.e)
        _, width = data.shape

        left = tmp_path / "vrt_left_1986.tif"
        right = tmp_path / "vrt_right_1986.tif"
        pl = profile.copy()
        pl["transform"] = rasterio.transform.Affine(pw, 0.0, ox, 0.0, ph, oy)
        pr = profile.copy()
        pr["transform"] = rasterio.transform.Affine(
            pw, 0.0, ox + pw * width, 0.0, ph, oy)
        with rasterio.open(str(left), "w", **pl) as dst:
            dst.write(data, 1)
        with rasterio.open(str(right), "w", **pr) as dst:
            dst.write(data, 1)

        vrt = tmp_path / "vrt_mosaic_1986.vrt"
        write_vrt(str(vrt), [str(left), str(right)])

        da1 = open_geotiff(str(vrt))
        expected = np.concatenate([data, data], axis=1)
        np.testing.assert_array_equal(np.asarray(da1.values), expected)

        out = tmp_path / "vrt_rt_1986.tif"
        to_geotiff(da1, str(out), compression='none')
        da2 = open_geotiff(str(out))
        np.testing.assert_array_equal(
            np.asarray(da2.values), np.asarray(da1.values))
        assert da2.dtype == da1.dtype
        # Unlike the other corpus-backed cases, ``da1`` here is the
        # VRT-resolved view of two rasterio-written GeoTIFFs (not the
        # corpus fixtures themselves), so it does not carry the
        # deprecated geographic attrs. The first cycle already holds the
        # fixed point.
        _assert_fixed_point(da1, da2)

# ===========================================================================
# Hypothesis property tests for write/read round trip (#2134)
# Source: test_roundtrip_properties.py
# ===========================================================================


hypothesis = pytest.importorskip("hypothesis")


# ---------------------------------------------------------------------------
# Profile registration
# ---------------------------------------------------------------------------

_COMMON_SUPPRESS = [
    HealthCheck.too_slow,
    HealthCheck.function_scoped_fixture,
    # The strategy uses ``assume`` to reject ill-typed combinations; on
    # narrow draws the filter rate can climb above the default threshold
    # without indicating a real strategy bug.
    HealthCheck.filter_too_much,
]

settings.register_profile(
    "reduced",
    max_examples=50,
    deadline=None,
    derandomize=True,
    suppress_health_check=_COMMON_SUPPRESS,
)
settings.register_profile(
    "local",
    max_examples=200,
    deadline=None,
    suppress_health_check=_COMMON_SUPPRESS,
)


# ---------------------------------------------------------------------------
# Strategy axes
# ---------------------------------------------------------------------------

COORD_DTYPES = ['int32', 'int64', 'float32', 'float64']
AXIS_DIRECTIONS = ['asc_asc', 'asc_desc', 'desc_asc', 'desc_desc']
SHAPES = [(1, 1), (1, 8), (8, 1), (4, 5), (16, 16)]
GEOREF_MODES = ['crs_only', 'transform_only', 'both', 'neither']
NODATA_MODES = ['in_range', 'out_of_range', 'fractional', 'nan', 'none']
BAND_LAYOUTS = ['band_first', 'band_last', 'no_band']
PIXEL_DTYPES = ['uint8', 'int16', 'int32', 'float32', 'float64']
CRS_CHOICES = [4326, 3857, 32633, 26910]


def _make_coord(direction: str, length: int, dtype_name: str) -> np.ndarray:
    """Build a 1D coord of ``length`` cells in the requested direction.

    Floats land on a regular grid so ``coords_to_transform`` accepts
    them; ints are an arange (the no-georef placeholder pattern the
    writer recognises via the #2120 marker).
    """
    dtype = np.dtype(dtype_name)
    if dtype.kind in ('i', 'u'):
        base = np.arange(length, dtype=dtype)
    else:
        # Pick a step that's exactly representable so the regularity
        # check passes; 1.0 / 0.5 are safe for both float32 and float64.
        base = np.arange(length, dtype=np.float64) * 1.0 + 100.0
        base = base.astype(dtype)
    if direction == 'desc':
        base = base[::-1].copy()
    return base


def _shape_for_layout(shape_2d: tuple[int, int], layout: str, n_bands: int):
    h, w = shape_2d
    if layout == 'no_band':
        return ('y', 'x'), (h, w)
    if layout == 'band_first':
        return ('band', 'y', 'x'), (n_bands, h, w)
    if layout == 'band_last':
        return ('y', 'x', 'band'), (h, w, n_bands)
    raise ValueError(f"bad layout {layout!r}")


def _build_pixels(shape: tuple[int, ...], dtype_name: str, seed: int) -> np.ndarray:
    """Build deterministic pixel data, avoiding the dtype extremes."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype_name)
    size = int(np.prod(shape))
    if dtype.kind == 'f':
        arr = rng.standard_normal(size).astype(dtype).reshape(shape)
    else:
        info = np.iinfo(dtype)
        # Stay clear of the extremes; the in_range nodata strategy
        # picks from outside the sampled span so the sentinel doesn't
        # collide with real data.
        lo = max(info.min, -100)
        hi = min(info.max, 100)
        arr = rng.integers(low=lo, high=hi, size=size, dtype=dtype).reshape(shape)
    return arr


def _pick_nodata(mode: str, dtype_name: str, rng: np.random.Generator):
    """Return a nodata value compatible with the dtype, or ``None``.

    ``in_range`` -- sentinel inside the dtype range but outside the
    pixel sample range (so no real pixel happens to equal it).
    ``out_of_range`` -- only valid for floats (would raise for ints).
    ``fractional`` -- only valid for floats.
    ``nan`` -- only valid for floats; returned as ``float('nan')``.
    ``none`` -- no sentinel.

    Returns ``None`` for the ``none`` case, ``float('nan')`` for the
    ``nan`` case, or a Python ``int`` / ``float`` scalar otherwise.
    """
    dtype = np.dtype(dtype_name)
    if mode == 'none':
        return None
    if mode == 'nan':
        return float('nan')
    if mode == 'fractional':
        return float(rng.uniform(0.1, 0.9))
    if mode == 'out_of_range':
        # Only meaningful for floats; the writer rejects int casts that
        # would lose information. Use a value the float dtype can hold
        # but no integer dtype can. Pair this mode with float dtypes only.
        return 1e30
    if mode == 'in_range':
        if dtype.kind == 'f':
            return -9999.0
        info = np.iinfo(dtype)
        # Pick a sentinel safely inside the dtype range, outside the
        # pixel sample range used by _build_pixels (which is |x| <= 100).
        candidate = max(info.min, -32768) if info.min < 0 else info.max
        # For unsigned dtypes ``info.min == 0`` so the candidate is
        # ``info.max``; for signed dtypes it's close to ``info.min``.
        return int(candidate)
    raise ValueError(f"bad nodata mode {mode!r}")


def _is_legal_combo(spec: dict) -> bool:
    """Filter combinations the writer is documented to reject.

    The strategy ``assume``s on these; rejected draws don't count
    against the example budget for invariant testing, but they do
    eat one slot of strategy generation time. Keep the filter set
    minimal so the example budget mostly hits the legal interior.
    """
    dtype = np.dtype(spec['pixel_dtype'])
    nodata_mode = spec['nodata_mode']
    # NaN, fractional, and out_of_range nodata require float dtype.
    if nodata_mode in ('nan', 'fractional', 'out_of_range') and dtype.kind != 'f':
        return False
    return True


# ---------------------------------------------------------------------------
# Composite strategy
# ---------------------------------------------------------------------------

@st.composite
def _round_trip_spec(draw):
    coord_dtype = draw(st.sampled_from(COORD_DTYPES))
    axis_dir = draw(st.sampled_from(AXIS_DIRECTIONS))
    shape = draw(st.sampled_from(SHAPES))
    georef = draw(st.sampled_from(GEOREF_MODES))
    nodata_mode = draw(st.sampled_from(NODATA_MODES))
    band_layout = draw(st.sampled_from(BAND_LAYOUTS))
    pixel_dtype = draw(st.sampled_from(PIXEL_DTYPES))
    # Only draw the dependent axes when they're actually consumed.
    # ``crs_epsg`` only matters when a CRS is going to be passed to
    # the writer; ``n_bands`` only matters when the layout has a band
    # axis. Conditional draws keep the strategy slot count tight.
    crs_epsg = (
        draw(st.sampled_from(CRS_CHOICES))
        if georef in ('crs_only', 'both')
        else None
    )
    n_bands = (
        draw(st.integers(min_value=2, max_value=3))
        if band_layout != 'no_band'
        else 1
    )
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    spec = dict(
        coord_dtype=coord_dtype,
        axis_dir=axis_dir,
        shape=shape,
        georef=georef,
        nodata_mode=nodata_mode,
        band_layout=band_layout,
        pixel_dtype=pixel_dtype,
        crs_epsg=crs_epsg,
        n_bands=n_bands,
        seed=seed,
    )
    assume(_is_legal_combo(spec))
    return spec


# ---------------------------------------------------------------------------
# Build / write / read helpers
# ---------------------------------------------------------------------------

def _build_dataarray(spec: dict) -> xr.DataArray:
    """Materialise a DataArray from a strategy draw.

    The georef mode controls whether the DataArray carries coords:

    * ``transform_only`` / ``both`` -- spatial coords with the chosen
      direction and dtype. Integer coords trigger the writer's
      no-georef path (#2120 marker on read), so this case effectively
      collapses to the same coverage as ``crs_only`` / ``neither``
      when coord dtype is int.
    * ``crs_only`` -- no spatial coords; the writer can't derive a
      transform, the on-disk file has CRS GeoKeys but no
      transform/scale/tiepoint tags.
    * ``neither`` -- no coords, no CRS kwarg. Round-trip should restore
      the same no-georef state.
    """
    h, w = spec['shape']
    dims, full_shape = _shape_for_layout(spec['shape'], spec['band_layout'],
                                         spec['n_bands'])
    pixels = _build_pixels(full_shape, spec['pixel_dtype'], spec['seed'])

    needs_coords = spec['georef'] in ('transform_only', 'both')
    coords = None
    if needs_coords:
        ax_x, ax_y = spec['axis_dir'].split('_')
        x = _make_coord(ax_x, w, spec['coord_dtype'])
        y = _make_coord(ax_y, h, spec['coord_dtype'])
        coords = {'y': y, 'x': x}
        if 'band' in dims:
            coords['band'] = np.arange(spec['n_bands'], dtype=np.int64)

    return xr.DataArray(pixels, dims=dims, coords=coords)


def _writer_kwargs(spec: dict, rng: np.random.Generator) -> dict:
    kwargs = dict(compression='none', tiled=False)
    if spec['georef'] in ('crs_only', 'both'):
        kwargs['crs'] = spec['crs_epsg']
    if spec['nodata_mode'] != 'none':
        nd = _pick_nodata(spec['nodata_mode'], spec['pixel_dtype'], rng)
        if nd is not None:
            kwargs['nodata'] = nd
    return kwargs


def _read_array(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values)


def _compare_pixels(a: np.ndarray, b: np.ndarray) -> None:
    """NaN-aware bit-equal pixel compare."""
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        a_f = a.astype(np.float64, copy=False)
        b_f = b.astype(np.float64, copy=False)
        nan_a = np.isnan(a_f)
        nan_b = np.isnan(b_f)
        if not np.array_equal(nan_a, nan_b):
            raise AssertionError("NaN mask drift between cycles")
        mask = ~nan_a
        np.testing.assert_array_equal(a_f[mask], b_f[mask])
    else:
        np.testing.assert_array_equal(a, b)


# Attrs whose values must match between two consecutive read results
# once the writer has canonicalised them. Other attrs (best-effort
# pass-through) are only checked for presence.
_PROPERTY_LOCKED_ATTRS = ('crs', 'transform', 'nodata', 'raster_type',
                          _NO_GEOREF_KEY)


def _assert_property_fixed_point(da1: xr.DataArray, da2: xr.DataArray) -> None:
    """Two consecutive write -> read results must agree on pixels,
    dtype, dims, and the canonical attrs.
    """
    assert da1.dtype == da2.dtype, f"dtype drift: {da1.dtype} -> {da2.dtype}"
    assert da1.dims == da2.dims, f"dims drift: {da1.dims} -> {da2.dims}"
    _compare_pixels(_read_array(da1), _read_array(da2))
    assert set(da1.attrs) == set(da2.attrs), (
        f"attrs key drift: {set(da1.attrs) ^ set(da2.attrs)}"
    )
    for key in _PROPERTY_LOCKED_ATTRS:
        if key in da1.attrs:
            v1 = da1.attrs[key]
            v2 = da2.attrs[key]
            if key == 'transform':
                assert len(v1) == len(v2)
                for a, b in zip(v1, v2):
                    assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9), (
                        f"transform drift: {v1} -> {v2}"
                    )
            elif key == 'nodata':
                if isinstance(v1, float) and math.isnan(v1):
                    assert isinstance(v2, float) and math.isnan(v2)
                else:
                    assert v1 == v2, f"nodata drift: {v1!r} -> {v2!r}"
            else:
                assert v1 == v2, f"attrs[{key!r}] drift: {v1!r} -> {v2!r}"


# ---------------------------------------------------------------------------
# Property: round-trip on the numpy backend
# ---------------------------------------------------------------------------

@settings(
    parent=settings.get_profile('local'),
)
@given(spec=_round_trip_spec())
def test_round_trip_fixed_point_numpy(tmp_path_factory, spec):
    """For every legal draw on the numpy backend, two consecutive
    write -> read cycles produce DataArrays that agree on the canonical
    attrs and pixel bytes.

    Skips the draw with ``assume`` when the writer is documented to
    refuse the combination (e.g. fractional / NaN nodata paired with an
    int pixel dtype). The intent is to lock the metadata round-trip
    contract, not to enumerate every documented refusal. Each skip is
    tagged with a Hypothesis ``event(...)`` so the stats output records
    which refusal class fired -- a regression that bumps the skip rate
    will surface in CI.
    """
    rng = np.random.default_rng(spec['seed'])
    da0 = _build_dataarray(spec)
    kwargs = _writer_kwargs(spec, rng)

    tmp = tmp_path_factory.mktemp("rt_2134")
    p1 = str(tmp / 'rt1.tif')
    p2 = str(tmp / 'rt2.tif')

    try:
        to_geotiff(da0, p1, **kwargs)
    except (ValueError, TypeError) as exc:
        # The writer rejects some specific combinations up front (e.g.
        # a 1x1 raster with no transform attr but with float coords
        # whose step is undefined). Those refusals are documented
        # behaviour, not round-trip failures. Tag the skip class so a
        # regression that pushes the rate up shows in Hypothesis stats.
        event(f"writer_rejected:{type(exc).__name__}")
        assume(False)
        return  # pragma: no cover

    try:
        da1 = open_geotiff(p1)
        # ``nodata=`` is not re-passed on the second cycle: the read
        # result carries the sentinel in attrs['nodata'] and the writer
        # picks it up there. Re-passing would double up the kwarg.
        to_geotiff(da1, p2, compression='none', tiled=False)
        da2 = open_geotiff(p2)
        _assert_property_fixed_point(da1, da2)
    finally:
        # Drop the tmp files eagerly so a 200-example session doesn't
        # leave 400 .tif files on disk until session teardown. The
        # mktemp directory itself is cleaned up by pytest.
        for p in (p1, p2):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Property: round-trip on the dask backend
# ---------------------------------------------------------------------------

@settings(
    parent=settings.get_profile('reduced'),
)
@given(spec=_round_trip_spec())
def test_round_trip_fixed_point_dask(tmp_path_factory, spec):
    """Same property as ``test_round_trip_fixed_point_numpy`` but the
    initial DataArray is wrapped in dask chunks so the streaming write
    path is exercised.

    Inherits the ``reduced`` profile (50 examples). The numpy property's
    200-example budget already covers the strategy interior; this pass
    exists to catch drift specific to the streaming writer.
    """
    pytest.importorskip('dask')

    rng = np.random.default_rng(spec['seed'])
    da0 = _build_dataarray(spec)

    # Pick chunks that actually split at least one axis when the shape
    # allows it; otherwise a single chunk reproduces the eager path
    # and the test would be redundant.
    h, w = spec['shape']
    if spec['band_layout'] == 'band_first':
        chunks = {'band': spec['n_bands'], 'y': max(h // 2, 1),
                  'x': max(w // 2, 1)}
    elif spec['band_layout'] == 'band_last':
        chunks = {'y': max(h // 2, 1), 'x': max(w // 2, 1),
                  'band': spec['n_bands']}
    else:
        chunks = {'y': max(h // 2, 1), 'x': max(w // 2, 1)}
    da0 = da0.chunk(chunks)

    kwargs = _writer_kwargs(spec, rng)

    tmp = tmp_path_factory.mktemp("rt_2134_dask")
    p1 = str(tmp / 'rt1.tif')
    p2 = str(tmp / 'rt2.tif')

    try:
        to_geotiff(da0, p1, **kwargs)
    except (ValueError, TypeError) as exc:
        event(f"writer_rejected:{type(exc).__name__}")
        assume(False)
        return  # pragma: no cover

    try:
        da1 = open_geotiff(p1)
        to_geotiff(da1, p2, compression='none', tiled=False)
        da2 = open_geotiff(p2)
        _assert_property_fixed_point(da1, da2)
    finally:
        for p in (p1, p2):
            try:
                os.unlink(p)
            except OSError:
                pass
