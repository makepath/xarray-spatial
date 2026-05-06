"""Regression tests for multi-sample predictor decode.

Issue #1220: the GPU predictor=2 decode path passed
`width=tile_width * samples` and `bytes_per_sample=itemsize * samples`
to `_predictor_decode_kernel`, causing `row_bytes` to be
`tile_width * samples**2 * itemsize` instead of
`tile_width * samples * itemsize`. That walks the cumulative-sum loop
past the end of each tile row and, on the last tile, past the end of
the `d_decomp` buffer (silent OOB GPU write).

Issue #1247: the CPU predictor=3 (floating-point) decode path called
`fp_predictor_decode(chunk, width, height, bytes_per_sample * samples)`,
which de-interleaved the row into `bytes_per_sample * samples` byte
lanes of length `width` instead of `bytes_per_sample` lanes of length
`width * samples` per TIFF Technical Note 3. Reading a GDAL-written
multi-band float TIFF with predictor=3 returned garbage pixel values.

These tests cover both bugs: GPU vs CPU parity for predictor=2, and
CPU correctness against a hand-built TN3-compliant predictor=3 buffer.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zlib

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


try:
    import cupy  # noqa: F401
    from numba import cuda
    _HAS_GPU = cuda.is_available()
except Exception:
    _HAS_GPU = False


gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="Requires CuPy + CUDA for GPU decode path")


def _gpu_to_numpy(da: xr.DataArray) -> np.ndarray:
    """Pull a CuPy-backed DataArray to host memory."""
    arr = da.data
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


@gpu_only
@pytest.mark.parametrize("samples,dtype_str", [
    (3, "uint8"),   # RGB uint8
    (4, "uint8"),   # RGBA uint8
    (3, "uint16"),  # RGB uint16
])
def test_gpu_predictor2_multisample_matches_cpu(tmp_path, samples, dtype_str):
    """GPU decode of a tiled multi-sample TIFF with predictor=2 matches CPU.

    Before the fix, the GPU predictor=2 kernel over-indexed the
    decompressed buffer by a factor of `samples` on the inner loop,
    corrupting the decoded pixels and writing past the end of
    ``d_decomp`` on the last tile.
    """
    dtype = np.dtype(dtype_str)
    h, w = 16, 16
    rng = np.random.RandomState(42)
    if dtype.kind == 'u' and dtype.itemsize == 1:
        data = rng.randint(0, 256, size=(h, w, samples), dtype=dtype)
    else:
        high = np.iinfo(dtype).max if dtype.kind in ('u', 'i') else 1000
        data = rng.randint(0, high, size=(h, w, samples), dtype=dtype)

    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / f"rgb_pred_{samples}_{dtype_str}.tif")
    # tile_size smaller than image so we exercise more than one tile.
    to_geotiff(da, path, compression='deflate', tile_size=8, predictor=True)

    cpu_arr = open_geotiff(path).values
    assert cpu_arr.shape == (h, w, samples)
    assert cpu_arr.dtype == dtype
    # Sanity: CPU round-trip reproduces the source exactly.
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_da = open_geotiff(path, gpu=True)
    gpu_arr = _gpu_to_numpy(gpu_da)

    assert gpu_arr.shape == cpu_arr.shape
    assert gpu_arr.dtype == cpu_arr.dtype
    # Byte-for-byte equality is the whole point: predictor=2 is exact.
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@gpu_only
def test_gpu_predictor2_multisample_uneven_tiles(tmp_path):
    """Image size not a multiple of tile size, multi-sample, predictor=2.

    Exercises partial edge tiles which share the same predictor decode
    path and are most likely to trip any lingering stride mismatch.
    """
    h, w, samples = 20, 20, 3  # 20 is not a multiple of 8
    rng = np.random.RandomState(7)
    data = rng.randint(0, 256, size=(h, w, samples), dtype=np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / "rgb_pred_uneven.tif")
    to_geotiff(da, path, compression='deflate', tile_size=8, predictor=True)

    cpu_arr = open_geotiff(path).values
    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))

    np.testing.assert_array_equal(gpu_arr, cpu_arr)
    np.testing.assert_array_equal(gpu_arr, data)


# ---------------------------------------------------------------------------
# Issue #1247: CPU fp_predictor_decode wrong for multi-band predictor=3
# ---------------------------------------------------------------------------

def _tn3_encode_row(row_bytes: np.ndarray, floats_per_row: int,
                    bytes_per_sample: int) -> np.ndarray:
    """Apply TIFF TN3 predictor=3 encoding to one row of raw pixel bytes.

    Transposes the row into ``bytes_per_sample`` byte lanes of length
    ``floats_per_row`` (MSB-first lane), then takes the byte-wise
    horizontal difference across the whole transposed row.

    This mirrors what libtiff / GDAL write to disk for a predictor=3
    chunky-multi-band float raster.
    """
    n = floats_per_row * bytes_per_sample
    tmp = np.empty(n, dtype=np.uint8)
    for f_idx in range(floats_per_row):
        for b in range(bytes_per_sample):
            lane = bytes_per_sample - 1 - b
            tmp[lane * floats_per_row + f_idx] = row_bytes[f_idx * bytes_per_sample + b]
    out = tmp.copy()
    for i in range(n - 1, 0, -1):
        out[i] = np.uint8((int(tmp[i]) - int(tmp[i - 1])) & 0xFF)
    return out


def _build_predictor3_stripped_tiff(arr: np.ndarray) -> bytes:
    """Build a minimal stripped deflate+predictor=3 TIFF for ``arr``.

    ``arr`` must be (height, width, samples) float32 or float64.  Each
    row is TN3-encoded then deflated into a single strip-per-row.  The
    resulting TIFF mimics what GDAL produces for multi-band float
    rasters with ``COMPRESS=DEFLATE`` and ``PREDICTOR=3``.
    """
    assert arr.ndim == 3
    assert arr.dtype.kind == 'f'
    height, width, samples = arr.shape
    bps = arr.dtype.itemsize
    bits_per_sample = bps * 8
    floats_per_row = width * samples

    # Raw bytes (little-endian float, native layout) per row, TN3-encoded,
    # then deflated.  One strip per row keeps the test simple.
    strip_blobs = []
    raw_bytes = np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()
    row_raw_bytes = floats_per_row * bps
    for r in range(height):
        row = raw_bytes[r * row_raw_bytes:(r + 1) * row_raw_bytes]
        encoded = _tn3_encode_row(row, floats_per_row, bps)
        strip_blobs.append(zlib.compress(encoded.tobytes(), 6))

    # --- Tag list (sorted by tag id) ---
    bo = '<'
    tags: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tags.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tags.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tags.append((tag, 3, len(vals),
                     struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_longs(tag, vals):
        tags.append((tag, 4, len(vals),
                     struct.pack(f'{bo}{len(vals)}I', *vals)))

    add_short(256, width)                                 # ImageWidth
    add_short(257, height)                                # ImageLength
    add_shorts(258, [bits_per_sample] * samples)          # BitsPerSample
    add_short(259, 8)                                     # Compression = deflate
    add_short(262, 2 if samples >= 3 else 1)              # PhotometricInterpretation
    add_long(273, 0)                                      # StripOffsets (patched)
    add_short(277, samples)                               # SamplesPerPixel
    add_short(278, 1)                                     # RowsPerStrip (one row)
    add_longs(279, [len(b) for b in strip_blobs])         # StripByteCounts
    add_short(284, 1)                                     # PlanarConfiguration = chunky
    add_short(317, 3)                                     # Predictor = 3 (float)
    add_shorts(339, [3] * samples)                        # SampleFormat = float

    tags.sort(key=lambda t: t[0])

    # --- Compute layout ---
    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4  # count + entries + next IFD
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_overflow_offsets = {}
    for tag, typ, count, raw in tags:
        if len(raw) > 4:
            tag_overflow_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_overflow_offsets[tag] = None

    # StripOffsets: one per row, each pointing at the deflated strip.
    pixel_start = overflow_start + len(overflow_buf)
    strip_offsets = []
    pos = 0
    for blob in strip_blobs:
        strip_offsets.append(pixel_start + pos)
        pos += len(blob)
    # Replace the StripOffsets entry with the real offsets
    patched = []
    for tag, typ, count, raw in tags:
        if tag == 273:
            if len(strip_offsets) == 1:
                new_raw = struct.pack(f'{bo}I', strip_offsets[0])
                patched.append((tag, 4, 1, new_raw))
            else:
                new_raw = struct.pack(f'{bo}{len(strip_offsets)}I',
                                       *strip_offsets)
                patched.append((tag, 4, len(strip_offsets), new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tags = patched

    # Rebuild overflow with patched StripOffsets
    overflow_buf = bytearray()
    tag_overflow_offsets = {}
    for tag, typ, count, raw in tags:
        if len(raw) > 4:
            tag_overflow_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_overflow_offsets[tag] = None

    # If the overflow size changed when patching StripOffsets, shift strip
    # offsets, rebuild tags + overflow, and repeat until stable.
    for _ in range(3):
        new_pixel_start = overflow_start + len(overflow_buf)
        if new_pixel_start == pixel_start:
            break
        shift = new_pixel_start - pixel_start
        strip_offsets = [off + shift for off in strip_offsets]
        patched2 = []
        for tag, typ, count, raw in tags:
            if tag == 273:
                if len(strip_offsets) == 1:
                    new_raw = struct.pack(f'{bo}I', strip_offsets[0])
                    patched2.append((tag, 4, 1, new_raw))
                else:
                    new_raw = struct.pack(f'{bo}{len(strip_offsets)}I',
                                           *strip_offsets)
                    patched2.append((tag, 4, len(strip_offsets), new_raw))
            else:
                patched2.append((tag, typ, count, raw))
        tags = patched2
        pixel_start = new_pixel_start

        overflow_buf = bytearray()
        tag_overflow_offsets = {}
        for tag, typ, count, raw in tags:
            if len(raw) > 4:
                tag_overflow_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_overflow_offsets[tag] = None

    # --- Serialize ---
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tags:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_overflow_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))  # no next IFD
    out.extend(overflow_buf)
    for blob in strip_blobs:
        out.extend(blob)

    return bytes(out)


@pytest.mark.parametrize("samples,dtype_str", [
    (3, "float32"),   # RGB float32 -- the common GDAL multi-band case
    (4, "float32"),   # RGBA float32
    (3, "float64"),   # RGB float64
    (2, "float32"),   # 2-band float (common for complex reals)
])
def test_cpu_predictor3_multisample_reads_correctly_1247(
        tmp_path, samples, dtype_str):
    """CPU decode of a TN3-compliant multi-band predictor=3 TIFF.

    Regression test for issue #1247.  Before the fix,
    ``fp_predictor_decode`` was called with
    ``bytes_per_sample * samples`` as the lane count, which swizzles
    the byte lanes the wrong way for chunky multi-band data.  The
    decoded pixels differed from the original by roughly half the
    bytes in a tile.
    """
    dtype = np.dtype(dtype_str)
    h, w = 5, 4  # small so the brute-force encoder stays fast
    rng = np.random.RandomState(1247)
    data = rng.uniform(-1000.0, 1000.0, size=(h, w, samples)).astype(dtype)

    path = str(tmp_path / f"tn3_pred3_{samples}_{dtype_str}_1247.tif")
    with open(path, 'wb') as f:
        f.write(_build_predictor3_stripped_tiff(data))

    decoded = open_geotiff(path).values
    assert decoded.shape == (h, w, samples)
    assert decoded.dtype == dtype
    np.testing.assert_array_equal(decoded, data)


def test_cpu_predictor3_single_sample_still_works_1247(tmp_path):
    """Single-band predictor=3 should keep working after the fix.

    The pre-fix and post-fix call signatures are numerically equivalent
    when ``samples == 1`` (``width*1 == width`` and
    ``bytes_per_sample*1 == bytes_per_sample``).  This test guards
    against regressions in the single-band path while the dispatch is
    being refactored.
    """
    h, w = 4, 4
    rng = np.random.RandomState(11247)
    data = rng.uniform(-10.0, 10.0, size=(h, w, 1)).astype(np.float32)

    path = str(tmp_path / "tn3_pred3_single_1247.tif")
    with open(path, 'wb') as f:
        f.write(_build_predictor3_stripped_tiff(data))

    decoded = open_geotiff(path).values
    # Single-band files round-trip through open_geotiff as a 2D array.
    assert decoded.shape in ((h, w), (h, w, 1))
    np.testing.assert_array_equal(decoded.reshape(h, w), data[..., 0])


def test_apply_predictor3_matches_tn3_reference_1247():
    """``_apply_predictor`` with pred=3 inverts TN3 encoding exactly.

    Unit-test of the dispatch fix at the ``_apply_predictor`` level.
    Builds a TN3-encoded buffer for multi-band float32 and checks that
    ``_apply_predictor(chunk, 3, width, height, bytes_per_sample,
    samples=samples)`` returns the original bytes.  This is the
    narrowest possible regression test for #1247 and does not depend
    on the TIFF header / reader plumbing.
    """
    from xrspatial.geotiff._reader import _apply_predictor

    h, w, samples = 3, 5, 3
    bps = 4  # float32
    rng = np.random.RandomState(31247)
    data = rng.uniform(-100.0, 100.0,
                       size=(h, w, samples)).astype(np.float32)
    raw = np.frombuffer(data.tobytes(), dtype=np.uint8).copy()

    floats_per_row = w * samples
    encoded_rows = [
        _tn3_encode_row(raw[r * floats_per_row * bps:
                            (r + 1) * floats_per_row * bps],
                        floats_per_row, bps)
        for r in range(h)
    ]
    encoded = np.concatenate(encoded_rows)
    decoded = _apply_predictor(encoded.copy(), 3, w, h, bps, samples=samples)

    np.testing.assert_array_equal(decoded, raw)


# ---------------------------------------------------------------------------
# Issue #1479: GPU predictor=3 multi-sample coverage gap
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sample-level predictor=2 (libtiff/GDAL convention) for multi-byte dtypes
# ---------------------------------------------------------------------------
#
# Per TIFF Technical Note: predictor=2 differences are taken between
# adjacent same-component samples in the sample's natural bit width
# (uint16 wraps at 65536, uint32 at 2^32, ...).  A byte-wise
# implementation drops the inter-byte carry for any sample wider than
# one byte, so xrspatial must read uint16/int16/uint32/int32 TIFFs
# written with predictor=2 by libtiff-compatible tools (rasterio, GDAL,
# tifffile) without corruption, and write TIFFs that those tools can
# read back without corruption.

try:
    import tifffile as _tifffile  # noqa: F401
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False


tifffile_required = pytest.mark.skipif(
    not _HAS_TIFFFILE, reason="Requires the tifffile package")


@tifffile_required
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32", "int32"])
def test_predictor2_reads_libtiff_multibyte_correctly(tmp_path, dtype_str):
    """xrspatial reads predictor=2 TIFFs with multi-byte samples correctly.

    The bug: the byte-wise predictor decode dropped inter-byte carry,
    so libtiff-style sample-wise differences came back as garbage.  We
    write a known array via tifffile (which uses libtiff conventions)
    and verify xrspatial reads it back exactly.
    """
    import tifffile

    dtype = np.dtype(dtype_str)
    arr = np.array([[1000, 2000, 3000, 4000],
                    [5000, 6000, 7000, 8000],
                    [9000, 10000, 11000, 12000],
                    [13000, 14000, 15000, 16000]], dtype=dtype)

    path = str(tmp_path / f"libtiff_pred2_{dtype_str}.tif")
    tifffile.imwrite(path, arr, compression='deflate', predictor=2)

    out = open_geotiff(path).values
    np.testing.assert_array_equal(out, arr)


@tifffile_required
def test_predictor2_reads_libtiff_multiband_uint16(tmp_path):
    """Multi-band chunky uint16 with predictor=2 round-trips through tifffile."""
    import tifffile

    arr = (np.arange(48).reshape(4, 4, 3) * 100).astype(np.uint16)
    path = str(tmp_path / "libtiff_pred2_rgb_uint16.tif")
    tifffile.imwrite(path, arr, compression='deflate',
                     predictor=2, photometric='rgb')

    out = open_geotiff(path).values
    np.testing.assert_array_equal(out, arr)


@tifffile_required
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32", "int32"])
def test_predictor2_writer_interops_with_libtiff(tmp_path, dtype_str):
    """xrspatial-written predictor=2 TIFFs decode correctly under tifffile.

    The encoder must produce sample-level differences so that
    libtiff/GDAL/rasterio can decode the file.  Round-trip through
    xrspatial alone is not enough -- a byte-wise encoder paired with a
    byte-wise decoder agrees with itself but corrupts the file for
    everyone else.
    """
    import tifffile

    dtype = np.dtype(dtype_str)
    arr = (np.arange(16, dtype=dtype) * 250).reshape(4, 4)
    da = xr.DataArray(arr, dims=('y', 'x'))

    path = str(tmp_path / f"xrs_pred2_{dtype_str}.tif")
    to_geotiff(da, path, compression='deflate', predictor=2)

    out_xrs = open_geotiff(path).values
    np.testing.assert_array_equal(out_xrs, arr)

    out_tiff = tifffile.imread(path)
    np.testing.assert_array_equal(out_tiff, arr)


@gpu_only
@tifffile_required
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"])
def test_gpu_predictor2_multibyte_matches_cpu(tmp_path, dtype_str):
    """GPU decode of predictor=2 with multi-byte samples matches CPU.

    Regression for the same sample-level differencing fix on the GPU
    kernels.  The image is written via tifffile (libtiff convention) so
    both backends must walk the predictor inverse at sample resolution.
    """
    import tifffile

    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(42)
    high = np.iinfo(dtype).max // 4
    low = np.iinfo(dtype).min // 4 if dtype.kind == 'i' else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)

    path = str(tmp_path / f"gpu_pred2_{dtype_str}.tif")
    tifffile.imwrite(path, data, compression='deflate', predictor=2,
                     tile=(16, 16))

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@gpu_only
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"])
def test_gpu_predictor2_multibyte_writer_round_trip(tmp_path, dtype_str):
    """xrspatial writer + GPU reader round-trip for multi-byte predictor=2."""
    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(7)
    high = np.iinfo(dtype).max // 4
    low = np.iinfo(dtype).min // 4 if dtype.kind == 'i' else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)
    da = xr.DataArray(data, dims=['y', 'x'])

    path = str(tmp_path / f"gpu_pred2_writer_{dtype_str}.tif")
    to_geotiff(da, path, compression='deflate', tile_size=16, predictor=2)

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@gpu_only
@tifffile_required
def test_gpu_predictor2_multiband_uint16_matches_cpu(tmp_path):
    """GPU decode of multi-band uint16 predictor=2 matches CPU and source."""
    import tifffile

    arr = (np.arange(32 * 32 * 3).reshape(32, 32, 3) % 50000).astype(np.uint16)
    path = str(tmp_path / "gpu_pred2_rgb_uint16.tif")
    tifffile.imwrite(path, arr, compression='deflate', predictor=2,
                     photometric='rgb', tile=(16, 16))

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, arr)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@gpu_only
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"])
def test_gpu_predictor2_writer_round_trip(tmp_path, dtype_str):
    """xrspatial writer + GPU encode path round-trip for multi-byte predictor=2.

    With ``gpu=True`` the writer takes the CUDA encode path; the file
    that lands on disk must still decode correctly on both CPU and GPU
    readers.
    """
    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(1234)
    high = np.iinfo(dtype).max // 4
    low = np.iinfo(dtype).min // 4 if dtype.kind == 'i' else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)
    da = xr.DataArray(data, dims=['y', 'x'])

    path = str(tmp_path / f"gpu_pred2_enc_{dtype_str}.tif")
    to_geotiff(da, path, compression='deflate', tile_size=16,
               predictor=2, gpu=True)

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@gpu_only
@pytest.mark.parametrize("samples,dtype_str", [
    (3, "float32"),
    (4, "float32"),
    (3, "float64"),
    (4, "float64"),
])
def test_gpu_predictor3_multisample_matches_cpu_1479(
        tmp_path, samples, dtype_str):
    """GPU decode of a tiled multi-sample float TIFF with predictor=3.

    Regression coverage for issue #1479.  The GPU
    ``_fp_predictor_decode_kernel`` correctly handles multi-sample float
    rasters, but predictor=3 was not exercised by CI (only predictor=2
    multi-sample was).  This test round-trips random float data through
    ``to_geotiff`` with ``predictor=3`` and asserts the GPU decode is
    bit-exactly equal to the CPU decode and to the source array.
    """
    dtype = np.dtype(dtype_str)
    h, w = 64, 64
    rng = np.random.RandomState(1479)
    data = rng.uniform(-1000.0, 1000.0,
                       size=(h, w, samples)).astype(dtype)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / f"fp_pred3_{samples}_{dtype_str}_1479.tif")
    to_geotiff(da, path, compression='deflate', predictor=3,
               tiled=True, tile_size=32)

    cpu_arr = open_geotiff(path).values
    assert cpu_arr.shape == (h, w, samples)
    assert cpu_arr.dtype == dtype
    # CPU round-trip must reproduce the source exactly (predictor=3 is
    # lossless).
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_da = open_geotiff(path, gpu=True)
    gpu_arr = _gpu_to_numpy(gpu_da)

    assert gpu_arr.shape == cpu_arr.shape
    assert gpu_arr.dtype == cpu_arr.dtype
    # Bit-for-bit GPU vs CPU parity is the regression we're guarding.
    np.testing.assert_array_equal(gpu_arr, cpu_arr)
    np.testing.assert_array_equal(gpu_arr, data)
