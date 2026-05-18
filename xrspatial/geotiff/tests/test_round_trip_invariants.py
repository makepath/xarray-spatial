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

Corpus-backed cases (using the #1930 golden corpus fixtures):

* planar multiband -- in: ``PLANARCONFIG=2`` (separate). Out: chunky.
  The writer emits chunky only, so the on-disk layout drifts but pixel
  bytes survive (verified against the planar-separate RGB fixture).
* overviews (internal IFD) -- base IFD pixels round-trip byte-equal;
  the overview pyramid is rewritten by the reducer rather than copied,
  so overview pixels are semantically equivalent only.
* COG layout -- base IFD pixels round-trip byte-equal; the
  ``LAYOUT=COG`` marker re-appears because the writer is in COG mode.
* sparse tiled -- elided zero tiles materialise as zeros on read; the
  rewrite is a normal tiled GeoTIFF whose pixels match those zeros.
* VRT mosaic -- read of a ``.vrt`` is semantically equivalent to a
  ``np.concatenate`` of the source pixels; the rewrite is a plain
  GeoTIFF whose pixels match.

Cases NOT covered here (deferred to follow-up PRs):

* float with non-NaN declared nodata -- requires the masked / declared
  nodata split from issue #1988 to express the invariant cleanly.

The per-incident round-trip test files
(``test_metadata_round_trip_1484.py``,
``test_int_coords_round_trip_hotfix_1962.py``, etc.) stay as regression
markers for their bug numbers. This module is the canonical contract
the writer must satisfy going forward.
"""
from __future__ import annotations

import pathlib

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


# ---------------------------------------------------------------------------
# Corpus-backed cases (#1930 golden corpus fixtures)
#
# The fixtures referenced below ship with the golden corpus from issue
# #1930. The tests below pull each fixture, run it through a
# ``read -> write -> read`` cycle, and pin the canonical invariant.
# Per the issue's constraint, no new fixtures are added; coverage is
# extended by reusing the corpus.
#
# Several corpus fixtures still carry the #1984 deprecated geographic
# attrs (``geog_citation``, ``angular_units``, ``semi_major_axis``,
# ``inv_flattening``). Reading them emits ``DeprecationWarning`` which
# is informative noise here, not a regression -- the warning module
# (``test_attrs_pr7_deprecate_geographic_1984.py``) already pins that
# contract. Filter them out for these cases so the round-trip
# assertions stay readable. Once the deprecated attrs are removed
# (next #1984 PR), this filter goes away.
# ---------------------------------------------------------------------------

# Class-level filter applied to every corpus-backed test class below.
# The fixtures emit deprecation warnings on read for the #1984
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
        # Fixed point: corpus fixtures carry deprecated #1984 attrs
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
    oracle (issue #1930).
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
        # Fixed point from da2 onward (deprecated #1984 attrs on the
        # corpus fixture mean the first cycle drops keys).
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
        with rasterio.open(str(src)) as h:
            src_factors = h.overviews(1)
            src_layout = h.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        assert src_layout == "COG", (
            f"corpus COG fixture must carry LAYOUT=COG marker, got "
            f"{src_layout!r}"
        )

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
        # Fixed point from da2 onward (deprecated #1984 attrs on the
        # corpus fixture mean the first cycle drops keys).
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
        # Fixed point from da2 onward (deprecated #1984 attrs on the
        # corpus fixture mean the first cycle drops keys).
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
        from xrspatial.geotiff import write_vrt

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
        _assert_fixed_point(da1, da2)
