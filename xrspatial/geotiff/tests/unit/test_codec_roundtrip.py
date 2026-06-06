"""Round-trip and parameter coverage for the codec public API.

The sibling ``test_compression.py``
in this directory is intentionally scoped to PackBits kernel unit tests
(its own docstring says so); write/read round-trips through the public
``to_geotiff`` / ``open_geotiff`` API land here.

Coverage is grouped by codec:

* JPEG: low-level codec, JPEGTables splice helper, writer round-trips
  (internal-only ``allow_internal_only_jpeg=True`` path), validation,
  rasterio-driven GDAL tiled JPEG read-back, ``to_geotiff`` rejection
  of the public ``compression='jpeg'`` path.
* JPEG 2000: codec round-trip via glymur, writer round-trip, dispatcher
  parity, availability flag.
* LERC: codec round-trip, writer round-trip, ``max_z_error`` budget
  semantics (lossless at 0, bounded error otherwise), dask streaming
  path, valid-mask handling, availability flag.
* LZ4: codec round-trip, writer round-trip, predictor interaction,
  ``compression_level`` boundaries (eager + dask), availability flag.
* Generic ``compression_level``: zstd / deflate round-trip and size
  monotonicity, LZW silent-ignore, out-of-range rejection.
* ``_write_geotiff_gpu`` docstring drift.

LERC and JPEG 2000 sections use ``pytest.importorskip`` (or skip
markers) because the optional codec libraries are not part of the base
install.
"""
from __future__ import annotations

import importlib.util
import io
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff._compression import (  # Compression tag constants; Availability flags; Codec functions (dispatcher + per-codec compress/decompress)  # noqa: E501
    COMPRESSION_JPEG, COMPRESSION_JPEG2000, COMPRESSION_LERC, COMPRESSION_LZ4, JPEG2000_AVAILABLE,
    LERC_AVAILABLE, LZ4_AVAILABLE, _splice_jpeg_tables, compress, decompress, jpeg2000_compress,
    jpeg2000_decompress, jpeg_compress, jpeg_decompress, lerc_compress, lerc_decompress,
    lerc_decompress_with_mask, lz4_compress, lz4_decompress)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import _compression_tag, write

_HAS_DASK = importlib.util.find_spec("dask") is not None
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_da(seed: int = 0, shape: tuple = (64, 64)) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float32)
    return xr.DataArray(arr, dims=["y", "x"])


def _make_compressible_smooth_small(shape: tuple = (128, 128)) -> xr.DataArray:
    """Smooth gradient + small noise on a 128x128 surface.

    Used by ``TestLz4CompressionLevel.test_lz4_higher_level_not_larger``
    where lz4's level effect shows up at modest size. The
    ``TestCompressionLevelEffect._make_compressible_large`` (512x512,
    no noise) helper sits on the test class because zstd / deflate
    need a larger smooth surface for the level gap to dominate codec
    heuristic noise.
    """
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0: shape[0], 0: shape[1]]
    arr = ((y + x).astype(np.float32)
           + rng.standard_normal(shape).astype(np.float32) * 0.01)
    return xr.DataArray(arr, dims=["y", "x"])


def _smooth_surface(h: int = 64, w: int = 64) -> np.ndarray:
    """Smooth float32 surface LERC can quantise efficiently."""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 4.0, h, dtype=np.float32),
        np.linspace(0.0, 4.0, w, dtype=np.float32),
        indexing="ij",
    )
    return (np.sin(yy) + np.cos(xx) * 0.5).astype(np.float32) * 10.0


def _make_geo_da(arr: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(arr.shape[0]),
                "x": np.arange(arr.shape[1])},
        attrs={"crs": 4326},
    )


# ===========================================================================
# JPEG
# ===========================================================================


class TestJpegCodec:
    """Low-level JPEG compress/decompress round trips."""

    def test_grayscale_round_trip(self):
        rng = np.random.RandomState(1050)
        arr = rng.randint(0, 256, (32, 32), dtype=np.uint8)
        compressed = jpeg_compress(arr.tobytes(), 32, 32, samples=1)
        decoded = np.frombuffer(
            jpeg_decompress(compressed, 32, 32, samples=1), dtype=np.uint8
        ).reshape(32, 32)
        assert decoded.shape == arr.shape
        assert np.abs(decoded.astype(int) - arr.astype(int)).mean() < 10

    def test_rgb_round_trip(self):
        # Smooth gradient -- random noise is the worst case for JPEG.
        y = np.linspace(50, 200, 32, dtype=np.uint8)
        x = np.linspace(50, 200, 32, dtype=np.uint8)
        r = np.outer(y, np.ones(32, dtype=np.uint8))
        g = np.outer(np.ones(32, dtype=np.uint8), x)
        b = np.full((32, 32), 128, dtype=np.uint8)
        arr = np.stack([r, g, b], axis=2)
        compressed = jpeg_compress(arr.tobytes(), 32, 32, samples=3)
        decoded = np.frombuffer(
            jpeg_decompress(compressed, 32, 32, samples=3), dtype=np.uint8
        ).reshape(32, 32, 3)
        assert decoded.shape == arr.shape
        assert np.abs(decoded.astype(int) - arr.astype(int)).mean() < 10

    def test_quality_affects_size(self):
        rng = np.random.RandomState(1050)
        arr = rng.randint(0, 256, (32, 32), dtype=np.uint8)
        data = arr.tobytes()
        low_q = jpeg_compress(data, 32, 32, samples=1, quality=10)
        high_q = jpeg_compress(data, 32, 32, samples=1, quality=95)
        assert len(low_q) < len(high_q)

    def test_invalid_samples_raises(self):
        with pytest.raises(ValueError, match="1 or 3 bands"):
            jpeg_compress(b"\x00" * 64, 4, 4, samples=2)


class TestJpegCompressionTag:
    """JPEG is wired into the writer's compression-tag map."""

    def test_jpeg_tag_value(self):
        assert _compression_tag("jpeg") == COMPRESSION_JPEG
        assert _compression_tag("JPEG") == COMPRESSION_JPEG

    def test_tag_value_is_7(self):
        assert COMPRESSION_JPEG == 7


class TestJpegWriteRoundTrip:
    """Write JPEG-compressed GeoTIFFs (internal-only path) and read back."""

    @pytest.mark.parametrize(
        "tiled, shape, samples",
        [
            pytest.param(True, (32, 32), 1, id="jpeg[uint8-gray-tiled]"),
            pytest.param(False, (32, 32), 1, id="jpeg[uint8-gray-stripped]"),
        ],
    )
    def test_jpeg_grayscale_roundtrip(self, tmp_path, tiled, shape, samples):
        rng = np.random.RandomState(1050)
        expected = rng.randint(50, 200, shape, dtype=np.uint8)
        path = str(tmp_path / f"jpeg_gray_tiled-{tiled}.tif")
        kwargs = {"compression": "jpeg", "tiled": tiled,
                  "allow_internal_only_jpeg": True}
        if tiled:
            kwargs["tile_size"] = 16
        write(expected, path, **kwargs)
        arr, _ = read_to_array(path, allow_internal_only_jpeg=True)
        assert arr.shape == expected.shape
        assert arr.dtype == np.uint8
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10

    def test_jpeg_rgb_tiled(self, tmp_path):
        y = np.linspace(50, 200, 32, dtype=np.uint8)
        x = np.linspace(50, 200, 32, dtype=np.uint8)
        r = np.outer(y, np.ones(32, dtype=np.uint8))
        g = np.outer(np.ones(32, dtype=np.uint8), x)
        b = np.full((32, 32), 128, dtype=np.uint8)
        expected = np.stack([r, g, b], axis=2)
        path = str(tmp_path / "jpeg_rgb_tiled.tif")
        write(expected, path, compression="jpeg", tiled=True, tile_size=16,
              allow_internal_only_jpeg=True)
        arr, _ = read_to_array(path, allow_internal_only_jpeg=True)
        assert arr.shape == expected.shape
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10


class TestJpegValidation:
    """JPEG rejects invalid input dtypes / band counts."""

    @pytest.mark.parametrize(
        "arr_factory, match",
        [
            pytest.param(lambda: np.zeros((8, 8), dtype=np.float32),
                         "uint8", id="jpeg[reject-float32]"),
            pytest.param(lambda: np.zeros((8, 8), dtype=np.uint16),
                         "uint8", id="jpeg[reject-uint16]"),
            pytest.param(lambda: np.zeros((8, 8, 4), dtype=np.uint8),
                         "1 or 3 bands", id="jpeg[reject-4-band]"),
        ],
    )
    def test_jpeg_rejects_invalid_input(self, tmp_path, arr_factory, match):
        arr = arr_factory()
        path = str(tmp_path / "bad.tif")
        with pytest.raises(ValueError, match=match):
            write(arr, path, compression="jpeg",
                  allow_internal_only_jpeg=True)


class TestPublicToGeotiffJpegRejected:
    """``to_geotiff`` refuses ``compression='jpeg'`` (lacks JPEGTables).

    The CPU encoder writes self-contained JFIF streams without the TIFF
    JPEGTables tag (347), so libtiff / GDAL / rasterio cannot decode
    the file. Round-tripping internally via Pillow would mask the
    interop break, so the public writer rejects the codec until a
    JPEGTables-aware encoder is in place.
    """

    def test_to_geotiff_jpeg_rejected(self, tmp_path):
        from xrspatial.geotiff import to_geotiff

        rng = np.random.RandomState(1050)
        data = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        da = xr.DataArray(
            data, dims=["y", "x"],
            coords={"y": np.arange(32, dtype=float),
                    "x": np.arange(32, dtype=float)},
        )
        path = str(tmp_path / "api.tif")
        with pytest.raises(ValueError, match="JPEGTables"):
            to_geotiff(da, path, compression="jpeg", tile_size=16)


class TestJpegTablesSplice:
    """JPEGTables splice helper used for tiled JPEG TIFFs."""

    def _build_pil_jpeg_split(self, size=16):
        from PIL import Image

        rng = np.random.RandomState(1502)
        arr = rng.randint(50, 200, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        full = buf.getvalue()
        sos = full.index(b"\xff\xda")
        tables = b"\xff\xd8" + full[2:sos] + b"\xff\xd9"
        fragment = b"\xff\xd8" + full[sos:]
        return tables, fragment, size

    def test_splice_reconstructs_complete_jpeg(self):
        from PIL import Image

        tables, fragment, size = self._build_pil_jpeg_split()
        spliced = _splice_jpeg_tables(fragment, tables)
        decoded = Image.open(io.BytesIO(spliced))
        decoded.load()
        assert decoded.size == (size, size)

    def test_splice_passthrough_on_empty_tables(self):
        payload = b"\xff\xd8\xff\xd9"
        assert _splice_jpeg_tables(payload, b"") == payload
        assert _splice_jpeg_tables(payload, None) == payload

    def test_splice_passthrough_on_invalid_input(self):
        # No SOI -> return unchanged so libjpeg's own error surfaces.
        assert _splice_jpeg_tables(b"no soi", b"\xff\xd8\xff\xd9") == b"no soi"

    def test_jpeg_decompress_accepts_jpeg_tables_kwarg(self):
        tables, fragment, _ = self._build_pil_jpeg_split()
        out = jpeg_decompress(fragment, 16, 16, samples=3, jpeg_tables=tables)
        assert len(out) == 16 * 16 * 3


@pytest.mark.skipif(
    not _HAS_RASTERIO,
    reason="rasterio required to write GDAL-style tiled JPEG TIFFs",
)
class TestGdalTiledJpegRead:
    """Read GDAL-style tiled JPEG TIFFs that use the JPEGTables tag."""

    def _gradient_rgb(self, size=128):
        y = np.linspace(20, 240, size, dtype=np.uint8)
        x = np.linspace(20, 240, size, dtype=np.uint8)
        r = np.broadcast_to(y[:, None], (size, size)).astype(np.uint8)
        g = np.broadcast_to(x[None, :], (size, size)).astype(np.uint8)
        b = np.full((size, size), 128, dtype=np.uint8)
        return np.stack([r, g, b], axis=0)  # rasterio wants (bands, H, W)

    def test_tiled_ycbcr_jpeg(self, tmp_path):
        import rasterio as rio

        from xrspatial.geotiff._header import TAG_JPEG_TABLES, parse_all_ifds, parse_header

        size = 128
        data = self._gradient_rgb(size)
        path = str(tmp_path / "tiled_jpeg_ycbcr.tif")
        with rio.open(
            path, "w", driver="GTiff", height=size, width=size, count=3,
            dtype="uint8", tiled=True, blockxsize=64, blockysize=64,
            compress="JPEG", photometric="YCBCR",
        ) as dst:
            dst.write(data)

        with open(path, "rb") as f:
            blob = f.read()
        hdr = parse_header(blob)
        ifds = parse_all_ifds(blob, hdr)
        assert TAG_JPEG_TABLES in ifds[0].entries
        assert ifds[0].jpeg_tables is not None
        assert ifds[0].jpeg_tables[:2] == b"\xff\xd8"

        arr, _ = read_to_array(path, allow_internal_only_jpeg=True)
        assert arr.shape == (size, size, 3)
        assert arr.dtype == np.uint8

        with rio.open(path) as src:
            ref = src.read()  # (bands, H, W)
        ref = np.transpose(ref, (1, 2, 0))
        # JPEG q=75 + 4:2:0 chroma subsampling shows ~1-3 mean error on
        # smooth gradients; allow a generous 5.
        assert np.abs(arr.astype(int) - ref.astype(int)).mean() < 5

    def test_tiled_grayscale_jpeg(self, tmp_path):
        import rasterio as rio

        size = 96
        y = np.linspace(20, 240, size, dtype=np.uint8)
        gray = np.broadcast_to(y[:, None], (size, size)).astype(np.uint8)

        path = str(tmp_path / "tiled_jpeg_gray.tif")
        with rio.open(
            path, "w", driver="GTiff", height=size, width=size, count=1,
            dtype="uint8", tiled=True, blockxsize=32, blockysize=32,
            compress="JPEG",
        ) as dst:
            dst.write(gray, 1)

        arr, _ = read_to_array(path, allow_internal_only_jpeg=True)
        assert arr.shape == (size, size)

        with rio.open(path) as src:
            ref = src.read(1)
        assert np.abs(arr.astype(int) - ref.astype(int)).mean() < 5


# ===========================================================================
# JPEG 2000
# ===========================================================================


_jpeg2000_only = pytest.mark.skipif(
    not JPEG2000_AVAILABLE, reason="glymur not installed",
)


@_jpeg2000_only
class TestJpeg2000Codec:
    """CPU JPEG 2000 codec round-trip via glymur."""

    @pytest.mark.parametrize(
        "dtype, shape, samples",
        [
            pytest.param("uint8", (8, 8), 1, id="jpeg2000[uint8-1band]"),
            pytest.param("uint16", (8, 8), 1, id="jpeg2000[uint16-1band]"),
            pytest.param("uint8", (8, 8), 3, id="jpeg2000[uint8-3band]"),
        ],
    )
    def test_jpeg2000_codec_roundtrip(self, dtype, shape, samples):
        h, w = shape
        if samples == 1:
            arr = np.arange(h * w, dtype=dtype).reshape(h, w)
        else:
            arr = np.arange(h * w * samples, dtype=dtype).reshape(h, w, samples)
        compressed = jpeg2000_compress(
            arr.tobytes(), w, h, samples=samples,
            dtype=np.dtype(dtype), lossless=True,
        )
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        decoded = jpeg2000_decompress(compressed, w, h, samples)
        if samples == 1:
            result = np.frombuffer(decoded, dtype=dtype).reshape(h, w)
        else:
            result = np.frombuffer(decoded, dtype=dtype).reshape(h, w, samples)
        np.testing.assert_array_equal(result, arr)

    def test_jpeg2000_single_pixel(self):
        arr = np.array([[42]], dtype=np.uint8)
        compressed = jpeg2000_compress(
            arr.tobytes(), 1, 1, samples=1, dtype=np.dtype("uint8"),
            lossless=True,
        )
        decompressed = jpeg2000_decompress(compressed, 1, 1, 1)
        assert np.frombuffer(decompressed, dtype=np.uint8)[0] == 42

    def test_jpeg2000_lossy_at_most_as_large_as_lossless(self):
        rng = np.random.RandomState(1048)
        arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
        lossless = jpeg2000_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype("uint8"),
            lossless=True,
        )
        lossy = jpeg2000_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype("uint8"),
            lossless=False,
        )
        assert len(lossy) <= len(lossless)

    def test_jpeg2000_dispatcher_decompress(self):
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        compressed = jpeg2000_compress(
            arr.tobytes(), 4, 4, samples=1, dtype=np.dtype("uint8"),
            lossless=True,
        )
        result = decompress(compressed, COMPRESSION_JPEG2000,
                            width=4, height=4, samples=1)
        np.testing.assert_array_equal(result.reshape(4, 4), arr)


@_jpeg2000_only
class TestJpeg2000WriteRoundTrip:
    """Write-read round-trip using the TIFF writer with JPEG 2000."""

    @pytest.mark.parametrize(
        "dtype, tiled",
        [
            pytest.param("uint8", True, id="jpeg2000[uint8-tiled]"),
            pytest.param("uint16", True, id="jpeg2000[uint16-tiled]"),
            pytest.param("uint8", False, id="jpeg2000[uint8-stripped]"),
        ],
    )
    def test_jpeg2000_writer_roundtrip(self, tmp_path, dtype, tiled):
        expected = np.arange(64, dtype=dtype).reshape(8, 8)
        path = str(tmp_path / f"j2k_{dtype}_tiled-{tiled}.tif")
        kwargs = {"compression": "jpeg2000", "tiled": tiled}
        if tiled:
            kwargs["tile_size"] = 8
        write(expected, path, **kwargs)
        arr, _ = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_jpeg2000_with_geo_info(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform

        expected = np.ones((8, 8), dtype=np.uint8) * 100
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / "j2k_geo.tif")
        write(expected, path, compression="jpeg2000", tiled=True, tile_size=8,
              geo_transform=gt, crs_epsg=4326, nodata=0)
        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)
        assert geo.crs_epsg == 4326

    def test_jpeg2000_public_api_roundtrip(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = _make_geo_da(data)
        path = str(tmp_path / "j2k_api.tif")
        # Experimental codec; pass the opt-in so the round-trip
        # exercises the encode path rather than the rejection gate.
        to_geotiff(da, path, compression="jpeg2000",
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, data)


class TestJpeg2000Availability:
    """Availability flag and error handling (always runs)."""

    def test_compression_constant(self):
        assert COMPRESSION_JPEG2000 == 34712

    def test_compression_tag_mapping(self):
        assert _compression_tag("jpeg2000") == 34712
        assert _compression_tag("j2k") == 34712

    def test_unavailable_raises_import_error(self):
        import xrspatial.geotiff._compression as comp_mod

        orig = comp_mod.JPEG2000_AVAILABLE
        comp_mod.JPEG2000_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="glymur"):
                comp_mod.jpeg2000_decompress(b"\x00", 1, 1, 1)
            with pytest.raises(ImportError, match="glymur"):
                comp_mod.jpeg2000_compress(
                    b"\x00", 1, 1, dtype=np.dtype("uint8"),
                )
        finally:
            comp_mod.JPEG2000_AVAILABLE = orig


# ===========================================================================
# LERC
# ===========================================================================


_lerc_only = pytest.mark.skipif(
    not LERC_AVAILABLE, reason="lerc not installed",
)


@_lerc_only
class TestLercCodec:
    """CPU LERC codec round-trip."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param("uint8", id="lerc[uint8-lossless]"),
            pytest.param("uint16", id="lerc[uint16-lossless]"),
            pytest.param("float32", id="lerc[float32-lossless]"),
        ],
    )
    def test_lerc_codec_roundtrip_lossless(self, dtype):
        arr = np.arange(64, dtype=dtype).reshape(8, 8)
        compressed = lerc_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype(dtype),
            max_z_error=0.0,
        )
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        decompressed = lerc_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=dtype).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_lerc_lossy_within_tolerance(self):
        rng = np.random.RandomState(1052)
        arr = rng.rand(32, 32).astype(np.float32) * 100
        max_err = 0.5
        compressed = lerc_compress(
            arr.tobytes(), 32, 32, samples=1, dtype=np.dtype("float32"),
            max_z_error=max_err,
        )
        decompressed = lerc_decompress(compressed, 32, 32, 1)
        result = np.frombuffer(decompressed, dtype=np.float32).reshape(32, 32)
        np.testing.assert_array_less(np.abs(result - arr), max_err + 1e-6)

    def test_lerc_lossy_smaller_than_lossless(self):
        rng = np.random.RandomState(1052)
        arr = rng.rand(64, 64).astype(np.float32) * 1000
        lossless = lerc_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype("float32"),
            max_z_error=0.0,
        )
        lossy = lerc_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype("float32"),
            max_z_error=1.0,
        )
        assert len(lossy) <= len(lossless)

    def test_lerc_dispatcher_decompress(self):
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        compressed = lerc_compress(
            arr.tobytes(), 4, 4, samples=1, dtype=np.dtype("float32"),
            max_z_error=0.0,
        )
        result = decompress(compressed, COMPRESSION_LERC,
                            width=4, height=4, samples=1)
        # ``decompress`` returns uint8; reinterpret to float32.
        result_f = np.frombuffer(result.tobytes(), dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(result_f, arr)


@_lerc_only
class TestLercWriteRoundTrip:
    """Write-read round-trip using the TIFF writer with LERC."""

    @pytest.mark.parametrize(
        "dtype, tiled",
        [
            pytest.param("uint8", True, id="lerc[uint8-tiled]"),
            pytest.param("float32", True, id="lerc[float32-tiled]"),
            pytest.param("float32", False, id="lerc[float32-stripped]"),
        ],
    )
    def test_lerc_writer_roundtrip(self, tmp_path, dtype, tiled):
        expected = np.arange(64, dtype=dtype).reshape(8, 8)
        path = str(tmp_path / f"lerc_{dtype}_tiled-{tiled}.tif")
        kwargs = {"compression": "lerc", "tiled": tiled}
        if tiled:
            kwargs["tile_size"] = 8
        write(expected, path, **kwargs)
        arr, _ = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_lerc_public_api_roundtrip(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = _make_geo_da(data)
        path = str(tmp_path / "lerc_api.tif")
        to_geotiff(da, path, compression="lerc",
                   allow_experimental_codecs=True)
        # The read side is also gated on the opt-in.
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, data)


@_lerc_only
class TestLercMaxZError:
    """``max_z_error`` budget semantics on LERC writes.

    Numerical tolerance assertions are pinned verbatim from the source
    file (``test_lerc_max_z_error.py``): every per-pixel error claim
    matches ``budget + 1e-7`` slack and the bit-exact lossless guarantee
    is kept as ``assert_array_equal``.
    """

    def test_lerc_max_z_error_lossless_bit_exact(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        arr = _smooth_surface()
        da = _make_geo_da(arr)
        path = str(tmp_path / "lerc_lossless.tif")
        to_geotiff(da, path, compression="lerc", max_z_error=0.0,
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, arr)

    def test_lerc_max_z_error_lossy_smaller_and_bounded(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        arr = _smooth_surface(128, 128)
        da = _make_geo_da(arr)
        lossless_path = str(tmp_path / "lerc_lossless.tif")
        lossy_path = str(tmp_path / "lerc_lossy.tif")
        to_geotiff(da, lossless_path, compression="lerc", max_z_error=0.0,
                   allow_experimental_codecs=True)
        to_geotiff(da, lossy_path, compression="lerc", max_z_error=0.05,
                   allow_experimental_codecs=True)
        lossless_size = os.path.getsize(lossless_path)
        lossy_size = os.path.getsize(lossy_path)
        assert lossy_size < lossless_size, (
            f"expected lossy file smaller, "
            f"got lossless={lossless_size} lossy={lossy_size}"
        )
        result = open_geotiff(lossy_path,
                              allow_experimental_codecs=True).values
        max_err = float(np.max(np.abs(result - arr)))
        assert max_err <= 0.05 + 1e-7, (
            f"per-pixel error {max_err} exceeds budget"
        )

    def test_lerc_max_z_error_dask_streaming(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        dask = pytest.importorskip("dask.array")
        arr = _smooth_surface(64, 64)
        darr = dask.from_array(arr, chunks=(32, 32))
        da = xr.DataArray(
            darr, dims=["y", "x"],
            coords={"y": np.arange(64), "x": np.arange(64)},
            attrs={"crs": 4326},
        )
        path = str(tmp_path / "lerc_dask.tif")
        to_geotiff(da, path, compression="lerc", max_z_error=0.05,
                   tile_size=32, allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True).values
        max_err = float(np.max(np.abs(result - arr)))
        assert max_err <= 0.05 + 1e-7

    def test_lerc_max_z_error_with_non_lerc_codec_raises(self, tmp_path):
        from xrspatial.geotiff import to_geotiff

        arr = _smooth_surface(16, 16)
        da = _make_geo_da(arr)
        path = str(tmp_path / "should_not_exist.tif")
        with pytest.raises(ValueError, match="max_z_error"):
            to_geotiff(da, path, compression="zstd", max_z_error=0.1)

    def test_lerc_negative_max_z_error_raises(self, tmp_path):
        from xrspatial.geotiff import to_geotiff

        arr = _smooth_surface(16, 16)
        da = _make_geo_da(arr)
        path = str(tmp_path / "should_not_exist.tif")
        with pytest.raises(ValueError, match="max_z_error"):
            to_geotiff(da, path, compression="lerc", max_z_error=-0.01,
                       allow_experimental_codecs=True)

    def test_lerc_max_z_error_zero_allowed_with_other_codec(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        # Default 0.0 must not error out for any other codec.
        arr = _smooth_surface(16, 16)
        da = _make_geo_da(arr)
        path = str(tmp_path / "zstd_default.tif")
        to_geotiff(da, path, compression="zstd", max_z_error=0.0)
        result = open_geotiff(path, allow_experimental_codecs=True).values
        np.testing.assert_array_equal(result, arr)


@_lerc_only
class TestLercDecompressWithMask:
    """Direct tests of the ``lerc_decompress_with_mask`` wrapper."""

    def _encode_with_mask(self, arr: np.ndarray,
                          mask: np.ndarray | None,
                          max_z_error: float = 0.0) -> bytes:
        import lerc

        has_mask = mask is not None
        result = lerc.encode(arr, 1, has_mask, mask, max_z_error, 1)
        assert result[0] == 0, f"lerc.encode returned error {result[0]}"
        return bytes(result[2])

    def test_lerc_no_mask_returns_none(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        blob = self._encode_with_mask(arr, None)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(out, arr)
        assert mask is None

    def test_lerc_all_valid_mask_collapses_to_none(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        valid = np.ones((3, 4), dtype=np.uint8)
        blob = self._encode_with_mask(arr, valid)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(out, arr)
        # All-True mask carries no information; treated as no mask.
        assert mask is None

    def test_lerc_partial_mask_returns_array(self):
        arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        valid = np.array([[1, 0, 1]], dtype=np.uint8)
        blob = self._encode_with_mask(arr, valid)
        decoded_bytes, mask = lerc_decompress_with_mask(blob)
        out = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(1, 3)
        # LERC zero-fills masked positions in the data array.
        np.testing.assert_array_equal(
            out, np.array([[1.0, 0.0, 3.0]], dtype=np.float32),
        )
        assert mask is not None
        np.testing.assert_array_equal(
            np.asarray(mask).reshape(1, 3),
            np.array([[1, 0, 1]], dtype=np.uint8),
        )

    def test_lerc_legacy_decompress_drops_mask(self):
        arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        valid = np.array([[1, 0, 1]], dtype=np.uint8)
        blob = self._encode_with_mask(arr, valid)
        out_bytes = lerc_decompress(blob)
        assert isinstance(out_bytes, bytes)
        out = np.frombuffer(out_bytes, dtype=np.float32).reshape(1, 3)
        np.testing.assert_array_equal(
            out, np.array([[1.0, 0.0, 3.0]], dtype=np.float32),
        )


@pytest.fixture
def lerc_writer_with_mask(monkeypatch):
    """Patch ``lerc_compress`` so the writer can embed a per-pixel mask.

    The production writer hard-codes ``hasMask=False`` because there is
    no plumbing for per-pixel mask data. The fixture swaps in an encoder
    that derives the mask from a holder ``invalid`` predicate: given the
    decoded pixel array it returns a uint8 valid mask (1=valid, 0=invalid)
    of the same shape. Tests specify masking by predicate so the mask
    matches whatever tile padding the writer emits.
    """
    import lerc

    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype("float32"), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = lerc.encode(arr, samples, has_mask, mask, max_z_error, 1)
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}"
            )
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


@_lerc_only
class TestLercTiffRoundTripWithMask:
    """End-to-end TIFF round-trips that exercise the reader's mask merge."""

    @pytest.mark.parametrize(
        "dtype, nodata, invalid_positions",
        [
            pytest.param(
                "float32", float("nan"), {(0, 1), (5, 4)},
                id="lerc-mask[float32-nan]",
            ),
            pytest.param(
                "float32", -9999.0, {(0, 1), (3, 3), (7, 7)},
                id="lerc-mask[float32-sentinel]",
            ),
            pytest.param(
                "uint16", 65535, {(0, 1), (4, 4)},
                id="lerc-mask[uint16-sentinel]",
            ),
        ],
    )
    def test_lerc_valid_mask_roundtrip(self, tmp_path, lerc_writer_with_mask,
                                       dtype, nodata, invalid_positions):
        if dtype == "uint16":
            arr = (np.arange(1, 65, dtype=np.uint16) * 100).reshape(8, 8)
        else:
            arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)

        # ``positions=invalid_positions`` binds the parametrise value at
        # function-definition time. Without the default-arg bind, the
        # closure would resolve ``invalid_positions`` lazily from the
        # surrounding scope, which is a classic late-binding bug if the
        # test is ever refactored to share the predicate between cases.
        def invalid_pred(a, positions=invalid_positions):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / f"lerc_mask_{dtype}.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=nodata)
        out, _ = read_to_array(path, allow_experimental_codecs=True)

        for r in range(8):
            for c in range(8):
                if (r, c) in invalid_positions:
                    if isinstance(nodata, float) and np.isnan(nodata):
                        assert np.isnan(out[r, c]), \
                            f"expected NaN at ({r},{c}), got {out[r, c]}"
                    else:
                        expected_sentinel = np.array(
                            nodata, dtype=arr.dtype,
                        )
                        assert out[r, c] == expected_sentinel, (
                            f"expected sentinel {nodata!r} at "
                            f"({r},{c}), got {out[r, c]}"
                        )
                else:
                    assert out[r, c] == arr[r, c], \
                        f"value mismatch at ({r},{c})"

    def test_lerc_no_mask_roundtrip_bit_exact(self, tmp_path):
        """All-valid LERC file (no mask) still round-trips bit-exactly."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / "lerc_no_mask.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8)
        out, _ = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(out, arr)


class TestLercAvailability:
    """Availability flag and error handling (always runs)."""

    def test_compression_constant(self):
        assert COMPRESSION_LERC == 34887

    def test_compression_tag_mapping(self):
        assert _compression_tag("lerc") == 34887

    def test_unavailable_raises_import_error(self):
        import xrspatial.geotiff._compression as comp_mod

        orig = comp_mod.LERC_AVAILABLE
        comp_mod.LERC_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="lerc"):
                comp_mod.lerc_decompress(b"\x00")
            with pytest.raises(ImportError, match="lerc"):
                comp_mod.lerc_compress(b"\x00", 1, 1)
        finally:
            comp_mod.LERC_AVAILABLE = orig


# ===========================================================================
# LZ4
# ===========================================================================


_lz4_only = pytest.mark.skipif(
    not LZ4_AVAILABLE, reason="lz4 not installed",
)


@_lz4_only
class TestLz4Codec:
    """CPU LZ4 codec round-trip via python-lz4."""

    @pytest.mark.parametrize(
        "payload_factory",
        [
            pytest.param(lambda: b"hello world! " * 100,
                         id="lz4[repetitive-text]"),
            pytest.param(lambda: bytes(range(256)) * 10,
                         id="lz4[binary-256]"),
            pytest.param(lambda: b"",
                         id="lz4[empty]"),
            pytest.param(
                lambda: bytes(
                    np.random.RandomState(1051).randint(
                        0, 256, size=50000, dtype=np.uint8,
                    )
                ),
                id="lz4[random-50k]",
            ),
        ],
    )
    def test_lz4_codec_roundtrip(self, payload_factory):
        data = payload_factory()
        compressed = lz4_compress(data)
        assert lz4_decompress(compressed) == data

    def test_lz4_dispatcher_roundtrip(self):
        data = b"test data " * 50
        compressed = compress(data, COMPRESSION_LZ4)
        result = decompress(compressed, COMPRESSION_LZ4)
        assert result.tobytes() == data


@_lz4_only
class TestLz4WriteRoundTrip:
    """Write-read round-trip using the TIFF writer with LZ4."""

    @pytest.mark.parametrize(
        "dtype, tiled, predictor",
        [
            pytest.param("uint8", True, False, id="lz4[uint8-tiled]"),
            pytest.param("float32", True, False, id="lz4[float32-tiled]"),
            pytest.param("uint8", False, False, id="lz4[uint8-stripped]"),
            pytest.param("float32", True, True,
                         id="lz4[float32-tiled-predictor]"),
        ],
    )
    def test_lz4_writer_roundtrip(self, tmp_path, dtype, tiled, predictor):
        if dtype == "uint8":
            expected = np.arange(64, dtype=dtype).reshape(8, 8)
            shape = (8, 8)
        else:
            expected = (np.random.RandomState(1051)
                        .rand(16, 16).astype(np.float32))
            shape = (16, 16)
        path = str(tmp_path / f"lz4_{dtype}.tif")
        kwargs = {"compression": "lz4", "tiled": tiled}
        if tiled:
            kwargs["tile_size"] = 8
        if predictor:
            kwargs["predictor"] = True
        write(expected, path, **kwargs)
        arr, _ = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)
        assert arr.shape == shape

    def test_lz4_public_api_roundtrip(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = _make_geo_da(data)
        path = str(tmp_path / "lz4_api.tif")
        to_geotiff(da, path, compression="lz4",
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, data)


class TestLz4Availability:
    """Availability flag and error handling (always runs)."""

    def test_compression_constant(self):
        assert COMPRESSION_LZ4 == 50004

    def test_compression_tag_mapping(self):
        assert _compression_tag("lz4") == 50004

    def test_unavailable_raises_import_error(self):
        import xrspatial.geotiff._compression as comp_mod

        orig = comp_mod.LZ4_AVAILABLE
        comp_mod.LZ4_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="lz4"):
                comp_mod.lz4_decompress(b"\x00")
            with pytest.raises(ImportError, match="lz4"):
                comp_mod.lz4_compress(b"\x00")
        finally:
            comp_mod.LZ4_AVAILABLE = orig


@_lz4_only
class TestLz4CompressionLevel:
    """Parameter coverage for ``compression_level`` with ``compression='lz4'``.

    The level-validation map in ``xrspatial.geotiff.__init__`` advertises
    a ``(0, 16)`` valid range for ``lz4``. Only ``deflate`` ``(1, 9)``
    and ``zstd`` ``(1, 22)`` had direct round-trip + boundary-error
    tests, so this section pins lz4 coverage at the eager numpy path and
    the dask streaming path. lz4 is lossless across its level range.
    """

    @pytest.mark.parametrize(
        "level",
        [
            pytest.param(0, id="lz4-level[0]"),
            pytest.param(1, id="lz4-level[1]"),
            pytest.param(9, id="lz4-level[9]"),
            pytest.param(16, id="lz4-level[16]"),
        ],
    )
    def test_lz4_level_round_trip(self, level, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=level)
        path = str(tmp_path / f"lz4_level_{level}.tif")
        to_geotiff(da, path, compression="lz4",
                   compression_level=level,
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, da.values)

    def test_lz4_default_level_round_trip(self, tmp_path):
        """``compression_level=None`` falls through to ``lz4_compress``'s
        default (``level=0``). Pin the no-arg path so a future signature
        change is caught."""
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=99)
        path = str(tmp_path / "lz4_default.tif")
        to_geotiff(da, path, compression="lz4",
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, da.values)

    def test_lz4_higher_level_not_larger(self, tmp_path):
        from xrspatial.geotiff import to_geotiff

        da = _make_compressible_smooth_small()
        path_lo = str(tmp_path / "lz4_lo.tif")
        path_hi = str(tmp_path / "lz4_hi.tif")
        to_geotiff(da, path_lo, compression="lz4", compression_level=0,
                   allow_experimental_codecs=True)
        to_geotiff(da, path_hi, compression="lz4", compression_level=16,
                   allow_experimental_codecs=True)
        size_lo = os.path.getsize(path_lo)
        size_hi = os.path.getsize(path_hi)
        # Allow equality: small or already-compressed payloads can land
        # at the same byte count. The contract is "no worse".
        assert size_hi <= size_lo, (
            f"Expected level-16 file ({size_hi}) <= level-0 file ({size_lo})"
        )

    @pytest.mark.parametrize(
        "level",
        [
            pytest.param(-1, id="lz4-level[-1]"),
            pytest.param(-10, id="lz4-level[-10]"),
            pytest.param(17, id="lz4-level[17]"),
            pytest.param(100, id="lz4-level[100]"),
        ],
    )
    def test_lz4_out_of_range_level_raises_eager(self, level, tmp_path):
        from xrspatial.geotiff import to_geotiff

        da = _make_da()
        path = str(tmp_path / f"lz4_bad_{level}.tif")
        with pytest.raises(ValueError, match="compression_level"):
            to_geotiff(da, path, compression="lz4",
                       compression_level=level,
                       allow_experimental_codecs=True)

    def test_lz4_out_of_range_message_includes_range(self, tmp_path):
        """Error message advertises the valid (0, 16) range so callers
        know the bound."""
        from xrspatial.geotiff import to_geotiff

        da = _make_da()
        path = str(tmp_path / "lz4_bad_range_msg.tif")
        with pytest.raises(ValueError, match=r"lz4.*\(valid:\s*0-16\)"):
            to_geotiff(da, path, compression="lz4",
                       compression_level=999,
                       allow_experimental_codecs=True)


@_lz4_only
@pytest.mark.skipif(not _HAS_DASK, reason="dask required")
class TestLz4CompressionLevelDask:
    """Dask streaming branch has its own ``_LEVEL_RANGES`` check; cover both
    accept and reject branches there."""

    def _make_dask_da(self, shape=(64, 64), chunks=(16, 16)):
        import dask.array as da_mod

        rng = np.random.default_rng(7)
        arr = rng.standard_normal(shape).astype(np.float32)
        return xr.DataArray(
            da_mod.from_array(arr, chunks=chunks),
            dims=["y", "x"],
        ), arr

    @pytest.mark.parametrize(
        "level",
        [
            pytest.param(0, id="lz4-dask[0]"),
            pytest.param(1, id="lz4-dask[1]"),
            pytest.param(8, id="lz4-dask[8]"),
            pytest.param(16, id="lz4-dask[16]"),
        ],
    )
    def test_lz4_dask_streaming_level_round_trip(self, level, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        dask_da, np_arr = self._make_dask_da()
        path = str(tmp_path / f"lz4_dask_level_{level}.tif")
        to_geotiff(dask_da, path, compression="lz4",
                   compression_level=level, tile_size=16,
                   allow_experimental_codecs=True)
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, np_arr)

    @pytest.mark.parametrize(
        "level",
        [
            pytest.param(-1, id="lz4-dask[-1]"),
            pytest.param(17, id="lz4-dask[17]"),
            pytest.param(50, id="lz4-dask[50]"),
        ],
    )
    def test_lz4_dask_streaming_out_of_range_raises(self, level, tmp_path):
        from xrspatial.geotiff import to_geotiff

        dask_da, _ = self._make_dask_da()
        path = str(tmp_path / f"lz4_dask_bad_{level}.tif")
        with pytest.raises(ValueError, match="compression_level"):
            to_geotiff(dask_da, path, compression="lz4",
                       compression_level=level, tile_size=16,
                       allow_experimental_codecs=True)


# ===========================================================================
# Generic compression_level coverage (zstd / deflate / lzw)
# ===========================================================================


class TestCompressionLevelRoundTrip:
    """zstd and deflate survive write/read across their advertised ranges."""

    @pytest.mark.parametrize(
        "codec, level",
        [
            pytest.param("zstd", 1, id="zstd-level[1]"),
            pytest.param("zstd", 22, id="zstd-level[22]"),
            pytest.param("deflate", 1, id="deflate-level[1]"),
            pytest.param("deflate", 9, id="deflate-level[9]"),
        ],
    )
    def test_level_round_trip(self, tmp_path, codec, level):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=level)
        path = str(tmp_path / f"{codec}{level}.tif")
        to_geotiff(da, path, compression=codec, compression_level=level)
        result = open_geotiff(path)
        np.testing.assert_allclose(result.values, da.values)


class TestCompressionLevelEffect:
    """Higher compression level produces a smaller-or-equal file."""

    def _make_compressible_large(self, shape=(512, 512)):
        # Large + smooth so the level gap is dominated by real
        # compression work, not codec heuristic noise on tiny inputs.
        y, x = np.mgrid[0: shape[0], 0: shape[1]]
        arr = (y + x).astype(np.float32)
        return xr.DataArray(arr, dims=["y", "x"])

    @pytest.mark.parametrize(
        "codec, low, high",
        [
            pytest.param("zstd", 1, 22, id="zstd-level[1-vs-22]"),
            pytest.param("deflate", 1, 9, id="deflate-level[1-vs-9]"),
        ],
    )
    def test_higher_level_not_larger(self, tmp_path, codec, low, high):
        from xrspatial.geotiff import to_geotiff

        da = self._make_compressible_large()
        path_lo = str(tmp_path / f"{codec}_lo.tif")
        path_hi = str(tmp_path / f"{codec}_hi.tif")
        to_geotiff(da, path_lo, compression=codec, compression_level=low)
        to_geotiff(da, path_hi, compression=codec, compression_level=high)
        size_lo = os.path.getsize(path_lo)
        size_hi = os.path.getsize(path_hi)
        assert size_hi <= size_lo, (
            f"Expected level-{high} file ({size_hi}) "
            f"<= level-{low} file ({size_lo})"
        )


class TestCompressionLevelDefault:
    """``compression_level=None`` uses the codec default."""

    def test_zstd_none_uses_default(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=5)
        path = str(tmp_path / "zstd_default.tif")
        to_geotiff(da, path, compression="zstd", compression_level=None)
        assert os.path.getsize(path) > 0
        result = open_geotiff(path)
        np.testing.assert_allclose(result.values, da.values)

    def test_deflate_omitted_uses_default(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=6)
        path = str(tmp_path / "deflate_default.tif")
        to_geotiff(da, path, compression="deflate")
        assert os.path.getsize(path) > 0
        result = open_geotiff(path)
        np.testing.assert_allclose(result.values, da.values)


class TestLzwIgnoresLevel:
    """LZW has no level concept; passing one is a silent no-op."""

    def test_lzw_with_level_does_not_raise(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, to_geotiff

        da = _make_da(seed=7)
        path = str(tmp_path / "lzw_level.tif")
        to_geotiff(da, path, compression="lzw", compression_level=5)
        assert os.path.getsize(path) > 0
        result = open_geotiff(path)
        np.testing.assert_allclose(result.values, da.values)


class TestCompressionLevelOutOfRange:
    @pytest.mark.parametrize(
        "codec, level",
        [
            pytest.param("zstd", 0, id="zstd-level[0-reject]"),
            pytest.param("zstd", 23, id="zstd-level[23-reject]"),
            pytest.param("zstd", -1, id="zstd-level[-1-reject]"),
            pytest.param("deflate", 0, id="deflate-level[0-reject]"),
            pytest.param("deflate", 10, id="deflate-level[10-reject]"),
        ],
    )
    def test_out_of_range_raises(self, tmp_path, codec, level):
        from xrspatial.geotiff import to_geotiff

        da = _make_da()
        path = str(tmp_path / f"{codec}_bad_{level}.tif")
        with pytest.raises(ValueError, match="compression_level"):
            to_geotiff(da, path, compression=codec, compression_level=level)


# ===========================================================================
# _write_geotiff_gpu compression docstring drift
# ===========================================================================


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)

# Codecs to exercise end-to-end through the GPU writer. ``jpeg`` is
# excluded here because (a) ``to_geotiff`` rejects it at runtime and
# (b) the JPEG round-trip is covered by the GPU codec tests with
# appropriate uint8 RGB data; keeping it out of this parametrize
# avoids exercising the JPEG path on dtype/shape combinations that
# aren't representative.
_GPU_FALLBACK_CODECS = (
    "lzw", "packbits", "lz4", "lerc", "jpeg2000", "j2k",
)


def test_write_geotiff_gpu_docstring_lists_full_codec_set():
    """The ``compression`` docstring block lists every codec
    ``to_geotiff`` accepts."""
    from xrspatial.geotiff import _write_geotiff_gpu

    doc = _write_geotiff_gpu.__doc__
    assert doc is not None, "_write_geotiff_gpu lost its docstring"
    block_start = doc.index("compression : str")
    block_end = doc.index("compression_level", block_start)
    block = doc[block_start:block_end]
    for codec in (
        "'none'", "'deflate'", "'lzw'", "'jpeg'", "'packbits'",
        "'zstd'", "'lz4'", "'jpeg2000'", "'j2k'", "'lerc'",
    ):
        assert codec in block, (
            f"compression docstring missing {codec}; current block:\n{block}"
        )


@_gpu_only
@pytest.mark.parametrize("codec", _GPU_FALLBACK_CODECS)
def test_write_geotiff_gpu_accepts_cpu_fallback_codecs(tmp_path, codec):
    """Codecs without a GPU encoder still write successfully via CPU."""
    import cupy

    from xrspatial.geotiff import _write_geotiff_gpu

    if codec in ("jpeg2000", "j2k"):
        arr_cpu = np.random.RandomState(0).randint(
            0, 65535, size=(64, 64), dtype=np.uint16,
        )
    else:
        arr_cpu = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    da = xr.DataArray(
        cupy.asarray(arr_cpu), dims=["y", "x"],
        coords={"y": np.arange(64.0, 0, -1), "x": np.arange(64.0)},
        attrs={"crs": 4326,
               "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 64.0)},
    )
    path = str(tmp_path / f"out_{codec}.tif")
    _write_geotiff_gpu(da, path, compression=codec,
                       allow_experimental_codecs=True)
    assert os.path.exists(path), (
        f"_write_geotiff_gpu(compression={codec!r}) failed to write a file"
    )
    assert os.path.getsize(path) > 0


class TestDispatcherErrorBranches:
    """Error branches of the ``compress``/``decompress`` dispatchers (#2995).

    The byte-stream ``compress()`` entry point cannot encode the block
    codecs (JPEG / JPEG2000 / LERC) because they need width / height /
    samples, so it rejects them with a redirect message. Both dispatchers
    reject unknown tags via their ``else`` branch. These guards had no
    test before #2995.
    """

    # An arbitrary tag value that is not any defined COMPRESSION_* constant.
    UNSUPPORTED_TAG = 99999

    def test_compress_rejects_jpeg(self):
        with pytest.raises(ValueError, match="Use jpeg_compress"):
            compress(b"\x00" * 16, COMPRESSION_JPEG)

    def test_compress_rejects_jpeg2000(self):
        with pytest.raises(ValueError, match="Use jpeg2000_compress"):
            compress(b"\x00" * 16, COMPRESSION_JPEG2000)

    def test_compress_rejects_lerc(self):
        with pytest.raises(ValueError, match="Use lerc_compress"):
            compress(b"\x00" * 16, COMPRESSION_LERC)

    def test_compress_rejects_unknown_tag(self):
        with pytest.raises(ValueError, match="Unsupported compression type"):
            compress(b"\x00" * 16, self.UNSUPPORTED_TAG)

    def test_decompress_rejects_unknown_tag(self):
        with pytest.raises(ValueError, match="Unsupported compression type"):
            decompress(b"\x00" * 16, self.UNSUPPORTED_TAG)
