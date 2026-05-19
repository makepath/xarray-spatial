"""Issue #2138: push-down byte-affecting validation for the
array-level write entry points (``_write`` / ``_write_streaming``)
and byte-parity with ``to_geotiff``.

The public ``to_geotiff`` wrapper runs several checks before its
array-level callees:

* compression-name validation against ``_VALID_COMPRESSIONS``
* JPEG-in-TIFF opt-in gate (issue #1845)
* ``max_z_error`` sign + LERC-only pairing
* ``crs_epsg`` bool rejection
* unparseable-CRS fail-closed (issue #1929)
* NaN-to-sentinel rewrite with a defensive copy
* ``float16`` / ``bool_`` auto-promotion

This file covers the push-down: each of those gaps must now fire
inside ``_write`` / ``_write_streaming`` so a direct caller cannot
bypass them. It also covers byte parity between ``_write`` and the
matching ``to_geotiff(xr.DataArray(...))`` call for every entry in
``_VALID_COMPRESSIONS`` -- if the wrapper and the lower-level
function disagree on a single byte, a direct caller silently
produces a different file.
"""
from __future__ import annotations

import os

import dask.array as dsk
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._writer import _write, _write_streaming
from xrspatial.geotiff._reader import _read_to_array


def _make_uint8_band(seed: int = 2138, shape=(32, 32)) -> np.ndarray:
    """Deterministic 2D uint8 array used by the byte-parity tests."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, shape, dtype=np.uint8)


def _make_float32_band(seed: int = 2138, shape=(32, 32)) -> np.ndarray:
    """Deterministic 2D float32 array for codecs that require floats (LERC)."""
    rng = np.random.RandomState(seed)
    return rng.rand(*shape).astype(np.float32)


def _bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Gap #1: compression name validation
# ---------------------------------------------------------------------------


class TestCompressionNamePushdown:
    """``_write`` must reject unknown compression names with the canonical
    list, the same way ``to_geotiff`` does. Before #2138 the array-level
    entry point relied on ``_compression_tag`` which raised but without
    the canonical list."""

    def test_write_rejects_unknown_compression(self, tmp_path):
        arr = _make_uint8_band()
        out = str(tmp_path / "tmp_2138_unknown_comp.tif")
        with pytest.raises(ValueError) as excinfo:
            _write(arr, out, compression="zstandard")
        msg = str(excinfo.value)
        assert "zstandard" in msg
        # Canonical list is part of the new wording.
        assert "zstd" in msg

    def test_write_streaming_rejects_unknown_compression(self, tmp_path):
        arr = dsk.from_array(_make_uint8_band(), chunks=(16, 16))
        out = str(tmp_path / "tmp_2138_unknown_comp_streaming.tif")
        with pytest.raises(ValueError, match="zstandard"):
            _write_streaming(arr, out, compression="zstandard")


# ---------------------------------------------------------------------------
# Gap #2: JPEG opt-in gate
# ---------------------------------------------------------------------------


class TestJpegOptInPushdown:
    """``_write`` must refuse ``compression='jpeg'`` unless the caller
    opts in, mirroring ``to_geotiff``'s gate. Before #2138 direct
    callers could silently produce a JFIF-tile file that other readers
    reject."""

    def test_write_rejects_jpeg_without_opt_in(self, tmp_path):
        arr = _make_uint8_band()
        out = str(tmp_path / "tmp_2138_jpeg_no_optin.tif")
        with pytest.raises(ValueError, match="allow_internal_only_jpeg"):
            _write(arr, out, compression="jpeg")

    def test_write_accepts_jpeg_with_opt_in(self, tmp_path):
        arr = _make_uint8_band()
        out = str(tmp_path / "tmp_2138_jpeg_optin.tif")
        _write(arr, out, compression="jpeg",
               allow_internal_only_jpeg=True)
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_write_streaming_rejects_jpeg_without_opt_in(self, tmp_path):
        arr = dsk.from_array(_make_uint8_band(), chunks=(16, 16))
        out = str(tmp_path / "tmp_2138_jpeg_streaming.tif")
        with pytest.raises(ValueError, match="allow_internal_only_jpeg"):
            _write_streaming(arr, out, compression="jpeg")


# ---------------------------------------------------------------------------
# Gap #3: max_z_error sign + codec pairing
# ---------------------------------------------------------------------------


class TestMaxZErrorPushdown:
    def test_write_rejects_negative_max_z_error(self, tmp_path):
        arr = _make_float32_band()
        out = str(tmp_path / "tmp_2138_negative_mze.tif")
        with pytest.raises(ValueError, match="max_z_error"):
            _write(arr, out, compression="lerc", max_z_error=-0.01)

    def test_write_rejects_max_z_error_on_non_lerc(self, tmp_path):
        arr = _make_float32_band()
        out = str(tmp_path / "tmp_2138_mze_zstd.tif")
        with pytest.raises(ValueError, match="max_z_error"):
            _write(arr, out, compression="zstd", max_z_error=0.05)

    def test_write_streaming_rejects_negative_max_z_error(self, tmp_path):
        arr = dsk.from_array(_make_float32_band(), chunks=(16, 16))
        out = str(tmp_path / "tmp_2138_streaming_neg_mze.tif")
        with pytest.raises(ValueError, match="max_z_error"):
            _write_streaming(arr, out, compression="lerc",
                             max_z_error=-0.01)


# ---------------------------------------------------------------------------
# Gap #4: crs_epsg bool rejection
# ---------------------------------------------------------------------------


class TestCrsEpsgBoolPushdown:
    """``crs_epsg=True`` would otherwise be written as ``EPSG=1`` because
    ``bool`` is an ``int`` subclass in Python. Both the public wrapper
    and the array-level entry points must reject it."""

    def test_write_rejects_bool_crs_epsg(self, tmp_path):
        arr = _make_uint8_band()
        out = str(tmp_path / "tmp_2138_bool_crs.tif")
        with pytest.raises(ValueError, match="bool"):
            _write(arr, out, crs_epsg=True)

    def test_write_rejects_false_crs_epsg(self, tmp_path):
        arr = _make_uint8_band()
        out = str(tmp_path / "tmp_2138_false_crs.tif")
        with pytest.raises(ValueError, match="bool"):
            _write(arr, out, crs_epsg=False)

    def test_write_streaming_rejects_bool_crs_epsg(self, tmp_path):
        arr = dsk.from_array(_make_uint8_band(), chunks=(16, 16))
        out = str(tmp_path / "tmp_2138_streaming_bool_crs.tif")
        with pytest.raises(ValueError, match="bool"):
            _write_streaming(arr, out, crs_epsg=True)


# ---------------------------------------------------------------------------
# Gap #5: defensive copy on NaN-to-sentinel rewrite
# ---------------------------------------------------------------------------


class TestNanToSentinelDefensiveCopy:
    """``to_geotiff`` rewrites NaN pixels to the nodata sentinel via
    ``arr.copy()`` so the caller's buffer is never mutated. Direct
    callers of ``_write`` used to skip this and write NaN bytes to
    disk. Push the rewrite (and the defensive copy) down so the
    invariant holds at every entry point."""

    def test_write_does_not_mutate_caller_buffer(self, tmp_path):
        # Float32 array with a real NaN and a non-NaN nodata sentinel.
        arr = np.full((8, 8), 1.5, dtype=np.float32)
        arr[2, 3] = np.nan
        original = arr.copy()
        out = str(tmp_path / "tmp_2138_no_mutate.tif")
        _write(arr, out, nodata=-9999.0, compression="zstd")
        # Caller's buffer must still carry the NaN it started with.
        np.testing.assert_array_equal(np.isnan(arr), np.isnan(original))
        # And the non-NaN positions must be untouched.
        finite = ~np.isnan(original)
        np.testing.assert_array_equal(arr[finite], original[finite])

    def test_write_writes_sentinel_in_file(self, tmp_path):
        arr = np.full((8, 8), 1.5, dtype=np.float32)
        arr[2, 3] = np.nan
        out = str(tmp_path / "tmp_2138_sentinel.tif")
        _write(arr, out, nodata=-9999.0, compression="zstd")
        # ``mask_nodata`` defaults to True on ``open_geotiff`` so the
        # sentinel comes back as NaN. Use ``_read_to_array`` (the raw
        # buffer) to confirm the sentinel actually hit disk.
        decoded, _ = _read_to_array(out)
        assert decoded[2, 3] == np.float32(-9999.0)


# ---------------------------------------------------------------------------
# Gap #7: float16 / bool_ auto-promotion
# ---------------------------------------------------------------------------


class TestDtypePromotionPushdown:
    def test_write_promotes_float16(self, tmp_path):
        # Float16 is not a TIFF SampleFormat; the wrapper promotes to
        # float32 before encode, and the push-down means a direct
        # caller gets the same behaviour rather than a dtype-mapper
        # ``ValueError``.
        arr = (np.linspace(0, 1, 64, dtype=np.float16).reshape(8, 8))
        out = str(tmp_path / "tmp_2138_float16.tif")
        _write(arr, out, compression="zstd")
        decoded, _ = _read_to_array(out)
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, arr.astype(np.float32))

    def test_write_promotes_bool(self, tmp_path):
        arr = np.array([[True, False], [False, True]], dtype=np.bool_)
        out = str(tmp_path / "tmp_2138_bool.tif")
        _write(arr, out, compression="zstd")
        decoded, _ = _read_to_array(out)
        assert decoded.dtype == np.uint8
        np.testing.assert_array_equal(decoded, arr.astype(np.uint8))


# ---------------------------------------------------------------------------
# Byte-parity: _write vs to_geotiff
# ---------------------------------------------------------------------------


# JPEG omitted from the byte-parity sweep on purpose: it requires the
# opt-in, which the wrapper emits a runtime warning for, and JPEG is
# lossy so trivial seed changes can shift bytes. ``_write`` is exercised
# elsewhere; the parity sweep covers the lossless codec set that direct
# callers reach for first.
_PARITY_CODECS = (
    "none",
    "deflate",
    "lzw",
    "packbits",
    "zstd",
    "lz4",
)


@pytest.mark.parametrize("compression", _PARITY_CODECS)
def test_write_vs_to_geotiff_byte_parity_uint8(compression, tmp_path):
    """``_write(arr, ...)`` and ``to_geotiff(xr.DataArray(arr), ...)``
    must produce byte-identical files for every entry in
    ``_VALID_COMPRESSIONS`` that round-trips losslessly. A divergence
    here is exactly the silent-different-file footgun #2138 names.
    """
    arr = _make_uint8_band(seed=2138 + hash(compression) % 1000)
    out_direct = str(tmp_path / f"tmp_2138_direct_{compression}.tif")
    out_wrapper = str(tmp_path / f"tmp_2138_wrapper_{compression}.tif")
    _write(arr, out_direct, compression=compression, tiled=True,
           tile_size=16)
    to_geotiff(xr.DataArray(arr, dims=("y", "x")), out_wrapper,
               compression=compression, tiled=True, tile_size=16)
    assert _bytes(out_direct) == _bytes(out_wrapper), (
        f"byte-parity violated for compression={compression!r}: "
        f"_write and to_geotiff produced different output files."
    )


@pytest.mark.parametrize("compression", ("zstd", "deflate", "lzw"))
def test_write_streaming_vs_to_geotiff_byte_parity_uint8(
        compression, tmp_path):
    """Same idea for the dask streaming path. ``to_geotiff`` on a
    dask-backed DataArray dispatches into ``_write_streaming``; feed
    ``_write_streaming`` and the wrapper the same dask source and a
    matching tile geometry and they must agree byte-for-byte."""
    raw = _make_uint8_band(seed=4276 + hash(compression) % 1000,
                           shape=(48, 48))
    chunks = (16, 16)
    dask_arr = dsk.from_array(raw, chunks=chunks)

    out_direct = str(
        tmp_path / f"tmp_2138_direct_streaming_{compression}.tif"
    )
    out_wrapper = str(
        tmp_path / f"tmp_2138_wrapper_streaming_{compression}.tif"
    )

    _write_streaming(dask_arr, out_direct, compression=compression,
                     tiled=True, tile_size=16)
    to_geotiff(xr.DataArray(dask_arr, dims=("y", "x")), out_wrapper,
               compression=compression, tiled=True, tile_size=16)
    assert _bytes(out_direct) == _bytes(out_wrapper), (
        f"byte-parity violated for streaming compression={compression!r}"
    )


def test_write_lerc_lossless_round_trip(tmp_path):
    """LERC with ``max_z_error=0`` is lossless. Confirm the codec
    survives the push-down and still round-trips bit-exactly when the
    pairing check passes."""
    arr = _make_float32_band()
    out = str(tmp_path / "tmp_2138_lerc_lossless.tif")
    _write(arr, out, compression="lerc", max_z_error=0.0)
    decoded, _ = _read_to_array(out)
    np.testing.assert_array_equal(decoded, arr)


def test_aliases_match_underscore_names():
    """``write`` / ``write_streaming`` / ``read_to_array`` must be the
    exact same objects as their underscore-prefixed canonical names so
    backward-compatible internal callers do not silently dispatch
    into stale copies."""
    from xrspatial.geotiff import _reader, _writer
    assert _writer.write is _writer._write
    assert _writer.write_streaming is _writer._write_streaming
    assert _reader.read_to_array is _reader._read_to_array


def test_write_not_leaked_into_public_namespace():
    """The array-level write entry points are module-private. They
    must not appear as attributes of ``xrspatial.geotiff`` (the
    documented public surface is ``to_geotiff``). Mirrors the #1708
    contract for ``read_to_array``."""
    import xrspatial.geotiff as g

    for name in ('write', 'write_streaming', '_write', '_write_streaming'):
        assert not hasattr(g, name), (
            f"{name!r} leaked into xrspatial.geotiff's public namespace. "
            "The supported public eager-write entry point is to_geotiff. "
            "Internal callers should import the array-level function "
            "from xrspatial.geotiff._writer directly. See issue #2138."
        )
