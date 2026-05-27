"""TIFF Predictor (2 / 3) read and write coverage.

One parametrised home for predictor=2
(horizontal differencing) and predictor=3 (floating-point, TIFF Technical
Note 3) on the CPU read and write paths.

Coverage is grouped by behaviour:

* Predictor=2 round-trips: big- and little-endian decode across uint8 /
  int8 / int16 / uint16 / uint32 / int32; stripped and tiled layouts;
  libtiff/GDAL/tifffile interop for multi-byte samples (sample-level
  differences, not byte-wise), including multi-band chunky data.
* Predictor=3 round-trips: big- and little-endian decode for float32 /
  float64; stripped and tiled layouts; writer end-to-end (with
  ``deflate`` / ``zstd``, tiled / stripped, dask streaming, multi-band);
  bit-level value fidelity at 1024x1024; predictor / compression
  interaction (``compression='none'`` suppresses the tag).
* Predictor=3 multi-sample: hand-built TN3-compliant
  multi-band stripped TIFFs decode correctly; single-band path stays
  consistent; the ``_apply_predictor`` dispatch helper inverts TN3
  encoding exactly.
* Validation: predictor=3 paired with an integer ``SampleFormat`` is
  rejected at every read site (eager numpy, dask) and at the writer
  (``normalize_predictor``). Legitimate combinations (predictor=1 /
  predictor=2 / predictor=3 + float) remain no-ops.
* Encoder API: ``normalize_predictor`` bool|int -> int mapping and
  rejection of out-of-range values; legacy ``predictor=True``/`False`
  semantics; the perf gate that predictor=3 stays within 2x of
  predictor=2 on smooth float data (opt-in via env var).

GPU predictor variants are intentionally out of scope here -- the
dedicated GPU predictor coverage lives in
``xrspatial/geotiff/tests/gpu/test_codec.py``. GPU regressions
(predictor=2 int8 tiled/stripped, predictor=3 BE GPU, predictor=2/3
multi-sample GPU parity) live there so the CPU and GPU coverage stay
co-located by behaviour rather than hardware.
"""
from __future__ import annotations

import importlib.util
import os
import struct
import zlib

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
                                       TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_PREDICTOR,
                                       TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
                                       TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._validation import _validate_predictor_sample_format
from xrspatial.geotiff._writer import (_assemble_standard_layout, _write_stripped,
                                       normalize_predictor)

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# GPU gate (co-located so GPU regressions stay next to the CPU baselines).
# These tests skip when no CUDA device is available.
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:  # pragma: no cover - import-time errors only
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")

# tifffile needs imagecodecs for the predictor=3 decode path; the
# predictor=3 read section skips cleanly when the optional dep is
# missing.
imagecodecs_required = pytest.mark.skipif(
    importlib.util.find_spec("imagecodecs") is None,
    reason="imagecodecs is required for tifffile predictor=3 round-trips",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gpu_to_numpy(da: xr.DataArray) -> np.ndarray:
    """Pull a CuPy-backed DataArray to host memory."""
    arr = da.data
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _smooth_float(shape, dtype):
    """Smooth surface where FP predictor is expected to help compression."""
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(dtype)
    return (np.sin(x / 40) * np.cos(y / 40) * 100 + (x + y) / 4).astype(dtype)


def _da_xy(arr: np.ndarray) -> xr.DataArray:
    """Wrap a 2D / 3D ndarray as a ``to_geotiff``-compatible DataArray.

    Note: all callers in this module pass 2D arrays. The 3D branch is
    here for safety and intentionally indexes the trailing two axes
    (``arr.shape[-2:]``) so the (band, y, x) layout maps the right
    sizes to the y/x coords. The pre-consolidation helper used
    ``arr.shape[:2]`` which would have given (band, y) on a 3D input;
    that path was never exercised, but leaving the latent bug in
    place felt worse than quietly fixing it.
    """
    h, w = arr.shape[:2] if arr.ndim == 2 else arr.shape[-2:]
    coords = {
        "x": np.arange(w, dtype=np.float64) * 10.0,
        "y": np.arange(h, dtype=np.float64) * 10.0,
    }
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=("y", "x"), coords=coords)
    return xr.DataArray(arr, dims=("band", "y", "x"), coords=coords)


def _signed_int8_grid(h: int = 16, w: int = 16) -> np.ndarray:
    """Deterministic int8 raster that walks the signed wraparound.

    A bug that decodes the byte stream as unsigned would produce a
    different cumulative sum than the signed reference.
    """
    info = np.iinfo(np.int8)
    rng = np.random.RandomState(0x1781)
    return rng.randint(info.min, info.max + 1, size=(h, w)).astype(np.int8)


def _read_predictor_tag(path):
    """Read the TIFF Predictor tag (id=317) directly from a file's IFD."""
    with open(path, "rb") as f:
        header = f.read(8)
    assert header[:2] == b"II", "test fixture writes little-endian"
    magic = struct.unpack("<H", header[2:4])[0]
    assert magic == 42, "classic TIFF expected"
    ifd_offset = struct.unpack("<I", header[4:8])[0]

    with open(path, "rb") as f:
        f.seek(ifd_offset)
        n_entries = struct.unpack("<H", f.read(2))[0]
        for _ in range(n_entries):
            entry = f.read(12)
            tag, type_id, count = struct.unpack("<HHI", entry[:8])
            if tag == 317:
                return struct.unpack("<H", entry[8:10])[0]
    return None  # tag absent => predictor 1 (none)


# ---------------------------------------------------------------------------
# Builders for malformed / hand-built TIFFs.
# ---------------------------------------------------------------------------


def _build_predictor3_uint32_tiff(arr: np.ndarray) -> bytes:
    """Build a malformed TIFF: predictor=3 + uint32 (integer) sample format.

    Uses the in-repo ``_assemble_standard_layout`` so we can write tags
    the public writer would reject. Compression is COMPRESSION_NONE so
    the strip bytes are exactly the raw integer values; the bug is then
    visible by comparing the round-tripped values against the originals.
    """
    rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
    bits_per_sample, _ = numpy_to_tiff_dtype(arr.dtype)
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, arr.shape[1]),
        (TAG_IMAGE_LENGTH, LONG, 1, arr.shape[0]),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
        (TAG_COMPRESSION, SHORT, 1, COMPRESSION_NONE),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, 1),  # UINT
        (TAG_PREDICTOR, SHORT, 1, 3),  # floating-point predictor
        (TAG_ROWS_PER_STRIP, SHORT, 1, arr.shape[0]),
        (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    parts = [(arr, arr.shape[1], arr.shape[0], rel_off, bc, chunks)]
    return _assemble_standard_layout(8, [tags], parts, bigtiff=False)


def _tn3_encode_row(row_bytes: np.ndarray, floats_per_row: int,
                    bytes_per_sample: int) -> np.ndarray:
    """Apply TIFF TN3 predictor=3 encoding to one row of raw pixel bytes.

    Transposes the row into ``bytes_per_sample`` byte lanes of length
    ``floats_per_row`` (MSB-first lane), then takes the byte-wise
    horizontal difference across the whole transposed row.  Mirrors what
    libtiff / GDAL write to disk for a predictor=3 chunky-multi-band
    float raster.
    """
    n = floats_per_row * bytes_per_sample
    tmp = np.empty(n, dtype=np.uint8)
    for f_idx in range(floats_per_row):
        for b in range(bytes_per_sample):
            lane = bytes_per_sample - 1 - b
            tmp[lane * floats_per_row + f_idx] = (
                row_bytes[f_idx * bytes_per_sample + b]
            )
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
    assert arr.dtype.kind == "f"
    height, width, samples = arr.shape
    bps = arr.dtype.itemsize
    bits_per_sample = bps * 8
    floats_per_row = width * samples

    strip_blobs = []
    raw_bytes = np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()
    row_raw_bytes = floats_per_row * bps
    for r in range(height):
        row = raw_bytes[r * row_raw_bytes:(r + 1) * row_raw_bytes]
        encoded = _tn3_encode_row(row, floats_per_row, bps)
        strip_blobs.append(zlib.compress(encoded.tobytes(), 6))

    bo = "<"
    tags: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tags.append((tag, 3, 1, struct.pack(f"{bo}H", val)))

    def add_long(tag, val):
        tags.append((tag, 4, 1, struct.pack(f"{bo}I", val)))

    def add_shorts(tag, vals):
        tags.append((tag, 3, len(vals),
                     struct.pack(f"{bo}{len(vals)}H", *vals)))

    def add_longs(tag, vals):
        tags.append((tag, 4, len(vals),
                     struct.pack(f"{bo}{len(vals)}I", *vals)))

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bits_per_sample] * samples)
    add_short(259, 8)
    add_short(262, 2 if samples >= 3 else 1)
    add_long(273, 0)  # StripOffsets (patched after layout)
    add_short(277, samples)
    add_short(278, 1)
    add_longs(279, [len(b) for b in strip_blobs])
    add_short(284, 1)
    add_short(317, 3)
    add_shorts(339, [3] * samples)

    tags.sort(key=lambda t: t[0])

    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_overflow_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tags:
        if len(raw) > 4:
            tag_overflow_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_overflow_offsets[tag] = None

    pixel_start = overflow_start + len(overflow_buf)
    strip_offsets = []
    pos = 0
    for blob in strip_blobs:
        strip_offsets.append(pixel_start + pos)
        pos += len(blob)

    def _patch_strip_offsets(in_tags, offs):
        patched = []
        for tag, typ, count, raw in in_tags:
            if tag == 273:
                if len(offs) == 1:
                    new_raw = struct.pack(f"{bo}I", offs[0])
                    patched.append((tag, 4, 1, new_raw))
                else:
                    new_raw = struct.pack(f"{bo}{len(offs)}I", *offs)
                    patched.append((tag, 4, len(offs), new_raw))
            else:
                patched.append((tag, typ, count, raw))
        return patched

    tags = _patch_strip_offsets(tags, strip_offsets)

    overflow_buf = bytearray()
    tag_overflow_offsets = {}
    for tag, _typ, _count, raw in tags:
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
        tags = _patch_strip_offsets(tags, strip_offsets)
        pixel_start = new_pixel_start

        overflow_buf = bytearray()
        tag_overflow_offsets = {}
        for tag, _typ, _count, raw in tags:
            if len(raw) > 4:
                tag_overflow_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_overflow_offsets[tag] = None

    out = bytearray()
    out.extend(b"II")
    out.extend(struct.pack(f"{bo}H", 42))
    out.extend(struct.pack(f"{bo}I", ifd_start))
    out.extend(struct.pack(f"{bo}H", num_entries))

    for tag, typ, count, raw in tags:
        out.extend(struct.pack(f"{bo}HHI", tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b"\x00"))
        else:
            ptr = overflow_start + tag_overflow_offsets[tag]
            out.extend(struct.pack(f"{bo}I", ptr))

    out.extend(struct.pack(f"{bo}I", 0))  # no next IFD
    out.extend(overflow_buf)
    for blob in strip_blobs:
        out.extend(blob)

    return bytes(out)


# ===========================================================================
# Section 1: Predictor=2 read round-trips (endianness x dtype x layout)
# ===========================================================================
#
# The predictor=2 decode runs sample-wise via a numpy
# view at the file's byte order. Numba's nopython mode rejects arrays
# with a non-native byte order, so the multi-byte big-endian path needs
# a byteswap around the kernel call. The uint8 byte-wise kernel never
# needed swapping. The int8 sample-format=2 branch is a separate code
# path that historically slipped through both buckets.


@pytest.mark.parametrize(
    "dtype,byteorder",
    [
        (np.uint8, "<"),
        (np.uint8, ">"),
        (np.int8, "<"),
        (np.int8, ">"),
        (np.uint16, "<"),
        (np.uint16, ">"),
        (np.int16, "<"),
        (np.int16, ">"),
        (np.uint32, "<"),
        (np.uint32, ">"),
        (np.int32, "<"),
        (np.int32, ">"),
    ],
    ids=[
        "uint8-le", "uint8-be",
        "int8-le", "int8-be",
        "uint16-le", "uint16-be",
        "int16-le", "int16-be",
        "uint32-le", "uint32-be",
        "int32-le", "int32-be",
    ],
)
def test_predictor2_round_trip_stripped(tmp_path, dtype, byteorder):
    """predictor=2 stripped layout decodes back to the original array.

    Covers every CPU dtype the project supports for predictor=2:

    * uint8 / int8: byte-wise kernel path.
    * uint16/int16/uint32/int32: multi-byte sample-wise kernel path,
      which requires the BE byteswap around the kernel call.
    """
    dt = np.dtype(dtype)
    if dt.kind == "u":
        info = np.iinfo(dt)
        arr = np.linspace(0, info.max // 2, 64,
                          dtype=np.int64).astype(dt).reshape(8, 8)
    elif dt == np.int8:
        arr = _signed_int8_grid(8, 8)
    else:
        info = np.iinfo(dt)
        arr = np.linspace(info.min // 2, info.max // 2, 64,
                          dtype=np.int64).astype(dt).reshape(8, 8)

    label = "be" if byteorder == ">" else "le"
    path = tmp_path / f"pred2_{label}_{dt.name}.tif"
    tifffile.imwrite(str(path), arr, byteorder=byteorder, predictor=2,
                     compression="deflate")

    out, _ = read_to_array(str(path))
    assert out.dtype == dt
    np.testing.assert_array_equal(out, arr)


def test_predictor2_round_trip_tiled_int8(tmp_path):
    """Tiled layout (separate ``_decode_tile`` path) handles int8 + pred=2."""
    arr = _signed_int8_grid(32, 48)
    path = tmp_path / "pred2_int8_tiled.tif"
    tifffile.imwrite(str(path), arr, predictor=2, compression="deflate",
                     tile=(16, 16))

    out, _ = read_to_array(str(path))
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, arr)


@_gpu_only
@pytest.mark.parametrize("tiled", [True, False],
                         ids=["tiled", "stripped"])
def test_gpu_predictor2_int8_matches_cpu(tmp_path, tiled):
    """GPU decode of int8 + predictor=2 matches CPU baseline.

    Stripped layout falls back to CPU; tiled exercises the GPU decoder.
    Either way, the decoded result must match the source.
    """
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    arr = _signed_int8_grid(32, 48) if tiled else _signed_int8_grid()
    path = tmp_path / f"pred2_int8_{'tiled' if tiled else 'stripped'}_gpu.tif"
    kwargs = {"predictor": 2, "compression": "deflate"}
    if tiled:
        kwargs["tile"] = (16, 16)
    tifffile.imwrite(str(path), arr, **kwargs)

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.int8
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


# ===========================================================================
# Section 2: Predictor=2 libtiff / GDAL interop (sample-level differencing)
# ===========================================================================
#
# Per TIFF Technical Note: predictor=2 differences are taken between
# adjacent same-component samples in the sample's natural bit width
# (uint16 wraps at 65536, uint32 at 2^32, ...).  A byte-wise
# implementation drops the inter-byte carry for any sample wider than
# one byte, so xrspatial must read TIFFs written with predictor=2 by
# libtiff-compatible tools without corruption, and write TIFFs that
# those tools can read back without corruption.


@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32", "int32"],
                         ids=lambda v: f"pred2-libtiff-{v}")
def test_predictor2_reads_libtiff_multibyte_correctly(tmp_path, dtype_str):
    """xrspatial reads predictor=2 TIFFs with multi-byte samples correctly."""
    dtype = np.dtype(dtype_str)
    arr = np.array([[1000, 2000, 3000, 4000],
                    [5000, 6000, 7000, 8000],
                    [9000, 10000, 11000, 12000],
                    [13000, 14000, 15000, 16000]], dtype=dtype)

    path = str(tmp_path / f"libtiff_pred2_{dtype_str}.tif")
    tifffile.imwrite(path, arr, compression="deflate", predictor=2)

    out = open_geotiff(path).values
    np.testing.assert_array_equal(out, arr)


def test_predictor2_reads_libtiff_multiband_uint16(tmp_path):
    """Multi-band chunky uint16 with predictor=2 round-trips through tifffile."""
    arr = (np.arange(48).reshape(4, 4, 3) * 100).astype(np.uint16)
    path = str(tmp_path / "libtiff_pred2_rgb_uint16.tif")
    tifffile.imwrite(path, arr, compression="deflate",
                     predictor=2, photometric="rgb")

    out = open_geotiff(path).values
    np.testing.assert_array_equal(out, arr)


@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32", "int32"],
                         ids=lambda v: f"pred2-writer-{v}")
def test_predictor2_writer_interops_with_libtiff(tmp_path, dtype_str):
    """xrspatial-written predictor=2 TIFFs decode correctly under tifffile.

    The encoder must produce sample-level differences so that
    libtiff/GDAL/rasterio can decode the file.  Round-trip through
    xrspatial alone is not enough -- a byte-wise encoder paired with a
    byte-wise decoder agrees with itself but corrupts the file for
    everyone else.
    """
    dtype = np.dtype(dtype_str)
    arr = (np.arange(16, dtype=dtype) * 250).reshape(4, 4)
    da = xr.DataArray(arr, dims=("y", "x"))

    path = str(tmp_path / f"xrs_pred2_{dtype_str}.tif")
    to_geotiff(da, path, compression="deflate", predictor=2)

    out_xrs = open_geotiff(path).values
    np.testing.assert_array_equal(out_xrs, arr)

    out_tiff = tifffile.imread(path)
    np.testing.assert_array_equal(out_tiff, arr)


# ===========================================================================
# Section 3: Predictor=2 multi-sample (GPU multi-sample bug)
# ===========================================================================
#
# The GPU predictor=2 decode path previously passed ``width=tile_width
# * samples`` and ``bytes_per_sample=itemsize * samples`` to the kernel,
# making ``row_bytes`` ``tile_width * samples**2 * itemsize`` instead of
# ``tile_width * samples * itemsize``. That walks past the end of each
# tile row and, on the last tile, past the end of the buffer.


@_gpu_only
@pytest.mark.parametrize(
    "samples,dtype_str",
    [
        (3, "uint8"),
        (4, "uint8"),
        (3, "uint16"),
    ],
    ids=lambda v: v if isinstance(v, str) else f"s{v}",
)
def test_gpu_predictor2_multisample_matches_cpu(tmp_path, samples, dtype_str):
    """GPU decode of a tiled multi-sample TIFF with predictor=2 matches CPU."""
    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(42)
    if dtype.kind == "u" and dtype.itemsize == 1:
        data = rng.randint(0, 256, size=(h, w, samples), dtype=dtype)
    else:
        high = np.iinfo(dtype).max if dtype.kind in ("u", "i") else 1000
        data = rng.randint(0, high, size=(h, w, samples), dtype=dtype)

    da = xr.DataArray(data, dims=["y", "x", "band"])

    path = str(tmp_path / f"rgb_pred_{samples}_{dtype_str}.tif")
    to_geotiff(da, path, compression="deflate", tile_size=16, predictor=True)

    cpu_arr = open_geotiff(path).values
    assert cpu_arr.shape == (h, w, samples)
    assert cpu_arr.dtype == dtype
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_da = open_geotiff(path, gpu=True)
    gpu_arr = _gpu_to_numpy(gpu_da)

    assert gpu_arr.shape == cpu_arr.shape
    assert gpu_arr.dtype == cpu_arr.dtype
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@_gpu_only
def test_gpu_predictor2_multisample_uneven_tiles(tmp_path):
    """Image size not a multiple of tile size, multi-sample, predictor=2."""
    h, w, samples = 40, 40, 3
    rng = np.random.RandomState(7)
    data = rng.randint(0, 256, size=(h, w, samples), dtype=np.uint8)
    da = xr.DataArray(data, dims=["y", "x", "band"])

    path = str(tmp_path / "rgb_pred_uneven.tif")
    to_geotiff(da, path, compression="deflate", tile_size=16, predictor=True)

    cpu_arr = open_geotiff(path).values
    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))

    np.testing.assert_array_equal(gpu_arr, cpu_arr)
    np.testing.assert_array_equal(gpu_arr, data)


@_gpu_only
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"],
                         ids=lambda v: f"gpu-pred2-{v}")
def test_gpu_predictor2_multibyte_matches_cpu(tmp_path, dtype_str):
    """GPU decode of predictor=2 with multi-byte samples matches CPU."""
    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(42)
    high = np.iinfo(dtype).max // 4
    low = np.iinfo(dtype).min // 4 if dtype.kind == "i" else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)

    path = str(tmp_path / f"gpu_pred2_{dtype_str}.tif")
    tifffile.imwrite(path, data, compression="deflate", predictor=2,
                     tile=(16, 16))

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@_gpu_only
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"],
                         ids=lambda v: f"gpu-pred2-writer-{v}")
def test_gpu_predictor2_multibyte_writer_round_trip(tmp_path, dtype_str):
    """xrspatial writer + GPU reader round-trip for multi-byte predictor=2."""
    dtype = np.dtype(dtype_str)
    h, w = 32, 32
    rng = np.random.RandomState(7)
    high = np.iinfo(dtype).max // 4
    low = np.iinfo(dtype).min // 4 if dtype.kind == "i" else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)
    da = xr.DataArray(data, dims=["y", "x"])

    path = str(tmp_path / f"gpu_pred2_writer_{dtype_str}.tif")
    to_geotiff(da, path, compression="deflate", tile_size=16, predictor=2)

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@_gpu_only
def test_gpu_predictor2_multiband_uint16_matches_cpu(tmp_path):
    """GPU decode of multi-band uint16 predictor=2 matches CPU and source."""
    arr = (np.arange(32 * 32 * 3).reshape(32, 32, 3) % 50000).astype(np.uint16)
    path = str(tmp_path / "gpu_pred2_rgb_uint16.tif")
    tifffile.imwrite(path, arr, compression="deflate", predictor=2,
                     photometric="rgb", tile=(16, 16))

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, arr)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


@_gpu_only
@pytest.mark.parametrize("dtype_str", ["uint16", "int16", "uint32"],
                         ids=lambda v: f"gpu-pred2-encoder-{v}")
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
    low = np.iinfo(dtype).min // 4 if dtype.kind == "i" else 0
    data = rng.randint(low, high, size=(h, w), dtype=dtype)
    da = xr.DataArray(data, dims=["y", "x"])

    path = str(tmp_path / f"gpu_pred2_enc_{dtype_str}.tif")
    to_geotiff(da, path, compression="deflate", tile_size=16,
               predictor=2, gpu=True)

    cpu_arr = open_geotiff(path).values
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


# ===========================================================================
# Section 4: Predictor=3 (floating-point) read round-trips
# ===========================================================================
#
# The floating-point predictor (TIFF Tech Note 3) byte-swizzles each row
# into ``bytes_per_sample`` lanes, MSB-first.  The decoder un-transposes
# those lanes back to the file's native byte order; the byte position of
# the MSB differs between BE (index 0) and LE (index ``bps-1``).
#
# Before the fix the un-transpose always wrote MSB at index ``bps-1``,
# so big-endian predictor=3 files decoded to garbage values even though
# they came back as a clean float array (no error, just wrong numbers).


@imagecodecs_required
@pytest.mark.parametrize(
    "dtype,byteorder",
    [
        (np.float32, "<"),
        (np.float32, ">"),
        (np.float64, "<"),
        (np.float64, ">"),
    ],
    ids=["float32-le", "float32-be", "float64-le", "float64-be"],
)
def test_predictor3_round_trip_stripped(tmp_path, dtype, byteorder):
    """predictor=3 stripped layout decodes back to the original array."""
    dt = np.dtype(dtype)
    if byteorder == ">":
        # Use the structured BE fixture from the original regression to
        # walk the MSB-first transpose deterministically.
        arr = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [1.5, 2.5, 3.5, 4.5],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=dt,
        )
    else:
        arr = np.linspace(-100.0, 100.0, 64, dtype=dt).reshape(8, 8)

    label = "be" if byteorder == ">" else "le"
    path = tmp_path / f"pred3_{label}_{dt.name}.tif"
    tifffile.imwrite(str(path), arr, byteorder=byteorder, predictor=3,
                     compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


@imagecodecs_required
def test_predictor3_round_trip_tiled_big_endian(tmp_path):
    """BE predictor=3 with tiled layout, multiple tiles per row."""
    rng = np.random.RandomState(20260506)
    arr = rng.standard_normal((32, 48)).astype(np.float32)
    path = tmp_path / "be_pred3_tiled.tif"
    tifffile.imwrite(str(path), arr, byteorder=">", predictor=3,
                     compression="deflate", tile=(16, 16))

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


@imagecodecs_required
@_gpu_only
def test_gpu_predictor3_big_endian_matches_cpu(tmp_path):
    """The GPU decode path also handles big-endian predictor=3."""
    rng = np.random.RandomState(20260506)
    arr = rng.standard_normal((32, 48)).astype(np.float32)
    path = tmp_path / "be_pred3_gpu.tif"
    tifffile.imwrite(str(path), arr, byteorder=">", predictor=3,
                     compression="deflate", tile=(16, 16))

    cpu = open_geotiff(str(path)).values
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = open_geotiff(str(path), gpu=True)
    gpu_arr = gpu_da.data
    if hasattr(gpu_arr, "get"):
        gpu_arr = gpu_arr.get()
    else:
        gpu_arr = np.asarray(gpu_arr)
    np.testing.assert_array_equal(gpu_arr, arr)


# ===========================================================================
# Section 5: Predictor=3 writer end-to-end
# ===========================================================================
#
# ``to_geotiff`` previously accepted ``predictor: bool`` and
# emitted only TIFF predictor 2.  Predictor 3 (byte-swizzled
# differencing per TN3) gives noticeably better deflate/zstd ratios on
# float data and is what most GDAL/rasterio workflows use for elevation
# rasters.


@pytest.mark.parametrize("dtype", [np.float32, np.float64],
                         ids=lambda v: np.dtype(v).name)
@pytest.mark.parametrize("compression", ["deflate", "zstd"])
@pytest.mark.parametrize("tiled", [True, False],
                         ids=["tiled", "stripped"])
def test_predictor3_writer_round_trip(tmp_path, dtype, compression, tiled):
    arr = _smooth_float((96, 128), dtype)
    da = _da_xy(arr)
    path = tmp_path / f"fp_pred_{np.dtype(dtype).name}_{compression}.tif"
    to_geotiff(da, str(path), compression=compression, predictor=3,
               tiled=tiled)

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)
    assert _read_predictor_tag(str(path)) == 3


def test_predictor3_better_than_predictor2_on_smooth_floats(tmp_path):
    """FP predictor exists precisely because it compresses smooth floats better."""
    arr = _smooth_float((512, 512), np.float32)
    da = _da_xy(arr)
    p2 = tmp_path / "pred2_smooth.tif"
    p3 = tmp_path / "pred3_smooth.tif"
    to_geotiff(da, str(p2), compression="deflate", predictor=2)
    to_geotiff(da, str(p3), compression="deflate", predictor=3)

    assert p3.stat().st_size < p2.stat().st_size


def test_predictor_legacy_bool_unchanged(tmp_path):
    """``predictor=True`` keeps emitting TIFF predictor 2."""
    arr = _smooth_float((32, 32), np.float32)
    da = _da_xy(arr)
    path = tmp_path / "legacy_true.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=True)
    assert _read_predictor_tag(str(path)) == 2

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_predictor_false_emits_no_tag(tmp_path):
    arr = _smooth_float((32, 32), np.float32)
    da = _da_xy(arr)
    path = tmp_path / "legacy_false.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=False)
    tag = _read_predictor_tag(str(path))
    assert tag in (None, 1)


def test_predictor3_with_compression_none_is_silent(tmp_path):
    """compression='none' suppresses any predictor (matches predictor=2 behavior)."""
    arr = _smooth_float((16, 16), np.float32)
    da = _da_xy(arr)
    path = tmp_path / "pred3_nocomp.tif"
    to_geotiff(da, str(path), compression="none", predictor=3)

    tag = _read_predictor_tag(str(path))
    assert tag in (None, 1), \
        "predictor must be suppressed when compression=none"

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_predictor3_streaming_dask(tmp_path):
    """Dask-backed input takes the streaming path; predictor=3 must work."""
    da_module = pytest.importorskip("dask.array")
    arr = _smooth_float((128, 192), np.float32)
    dask_arr = da_module.from_array(arr, chunks=(64, 96))
    da = xr.DataArray(
        dask_arr, dims=("y", "x"),
        coords={"x": np.arange(192, dtype=np.float64) * 10.0,
                "y": np.arange(128, dtype=np.float64) * 10.0},
    )
    path = tmp_path / "pred3_streaming.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=3,
               tile_size=64)

    assert _read_predictor_tag(str(path)) == 3
    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_predictor3_multiband_round_trip(tmp_path):
    """Multi-band float predictor=3 round-trip.

    This checks the write side round-trips correctly for the
    multi-band case where the row swizzle
    has to use ``width * samples`` lanes, not ``width``.
    """
    h, w = 48, 64
    arr = np.stack([
        _smooth_float((h, w), np.float32),
        _smooth_float((h, w), np.float32) * 1.5,
        _smooth_float((h, w), np.float32) - 10.0,
    ], axis=0)  # (3, h, w)
    da = xr.DataArray(
        arr, dims=("band", "y", "x"),
        coords={"x": np.arange(w, dtype=np.float64) * 10.0,
                "y": np.arange(h, dtype=np.float64) * 10.0},
    )
    path = tmp_path / "pred3_multiband.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=3)

    assert _read_predictor_tag(str(path)) == 3
    out = open_geotiff(str(path))
    if out.ndim == 3 and out.shape[-1] == 3:
        out_arr = np.moveaxis(out.values, -1, 0)
    else:
        out_arr = out.values
    np.testing.assert_array_equal(out_arr, arr)


def test_predictor3_large_round_trip_value_exact(tmp_path):
    """1024x1024 float32 deflate+predictor=3 round-trips with no value drift.

    The encode path was refactored to dispatch the per-row kernel from
    inside an ``@ngjit`` wrapper instead of from a Python ``for`` loop.
    Guards against any silent corruption by asserting the output array
    is byte-for-byte identical: dtype must match, and a uint8 view of
    the bytes must compare equal so the check catches signed-zero
    drift, NaN payload changes, and any other bit-level divergence that
    ``assert_array_equal`` would mask.
    """
    h, w = 1024, 1024
    arr = _smooth_float((h, w), np.float32)
    da = _da_xy(arr)
    path = tmp_path / "pred3_large_round_trip.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=3)

    assert _read_predictor_tag(str(path)) == 3
    out = open_geotiff(str(path))
    out_arr = np.ascontiguousarray(out.values)
    assert out_arr.dtype == arr.dtype, (
        f"dtype drift: in={arr.dtype}, out={out_arr.dtype}"
    )
    assert out_arr.shape == arr.shape
    assert out_arr.tobytes() == arr.tobytes(), (
        "predictor=3 round-trip diverged at the bit level "
        "(signed zero, NaN payload, or actual corruption)"
    )


def test_predictor3_encode_within_2x_of_predictor2(tmp_path):
    """Loose regression check: predictor=3 encode is within 2x of predictor=2.

    Before the ngjit row-loop refactor, predictor=3 was ~2.5x slower than
    predictor=2 because the row loop was in Python. Opt-in via
    ``XRSPATIAL_RUN_PERF_TESTS=1`` -- shared CI runners, CPU throttling,
    debug builds, and noisy filesystems all make absolute wall-clock
    timings flaky, so the test stays off by default.
    """
    if os.environ.get("XRSPATIAL_RUN_PERF_TESTS") != "1":
        pytest.skip(
            "set XRSPATIAL_RUN_PERF_TESTS=1 to run wall-clock perf tests")

    import time

    arr = _smooth_float((1024, 1024), np.float32)
    da = _da_xy(arr)
    p2 = tmp_path / "pred2_timing.tif"
    p3 = tmp_path / "pred3_timing.tif"

    # Warm up numba
    to_geotiff(da, str(p2), compression="deflate", predictor=2)
    to_geotiff(da, str(p3), compression="deflate", predictor=3)

    t0 = time.perf_counter()
    to_geotiff(da, str(p2), compression="deflate", predictor=2)
    t_p2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    to_geotiff(da, str(p3), compression="deflate", predictor=3)
    t_p3 = time.perf_counter() - t0

    assert t_p3 < 2.0 * t_p2, (
        f"predictor=3 ({t_p3*1000:.1f} ms) is more than 2x slower than "
        f"predictor=2 ({t_p2*1000:.1f} ms); ngjit row loop may have regressed"
    )


# ===========================================================================
# Section 6: Predictor=3 multi-sample
# ===========================================================================
#
# The CPU predictor=3 decode path used to call
# ``fp_predictor_decode(chunk, width, height, bytes_per_sample * samples)``,
# which de-interleaved the row into ``bytes_per_sample * samples`` byte
# lanes of length ``width`` instead of ``bytes_per_sample`` lanes of
# length ``width * samples`` per TN3. Reading a GDAL-written multi-band
# float TIFF with predictor=3 returned garbage values.


@pytest.mark.parametrize(
    "samples,dtype_str",
    [
        (3, "float32"),
        (4, "float32"),
        (3, "float64"),
        (2, "float32"),
    ],
    ids=lambda v: v if isinstance(v, str) else f"s{v}",
)
def test_cpu_predictor3_multisample_reads_correctly(
        tmp_path, samples, dtype_str):
    """CPU decode of a TN3-compliant multi-band predictor=3 TIFF.

    Before the fix,
    ``fp_predictor_decode`` was called with
    ``bytes_per_sample * samples`` as the lane count, which swizzles the
    byte lanes the wrong way for chunky multi-band data.  The decoded
    pixels differed from the original by roughly half the bytes in a
    tile.
    """
    dtype = np.dtype(dtype_str)
    h, w = 5, 4  # small so the brute-force encoder stays fast
    rng = np.random.RandomState(1247)
    data = rng.uniform(-1000.0, 1000.0,
                       size=(h, w, samples)).astype(dtype)

    path = str(tmp_path / f"tn3_pred3_{samples}_{dtype_str}.tif")
    with open(path, "wb") as f:
        f.write(_build_predictor3_stripped_tiff(data))

    decoded = open_geotiff(path).values
    assert decoded.shape == (h, w, samples)
    assert decoded.dtype == dtype
    np.testing.assert_array_equal(decoded, data)


def test_cpu_predictor3_single_sample_still_works(tmp_path):
    """Single-band predictor=3 should keep working after the fix.

    The pre-fix and post-fix call signatures are numerically equivalent
    when ``samples == 1``.  This test guards against regressions in the
    single-band path while the dispatch is being refactored.
    """
    h, w = 4, 4
    rng = np.random.RandomState(11247)
    data = rng.uniform(-10.0, 10.0, size=(h, w, 1)).astype(np.float32)

    path = str(tmp_path / "tn3_pred3_single.tif")
    with open(path, "wb") as f:
        f.write(_build_predictor3_stripped_tiff(data))

    decoded = open_geotiff(path).values
    assert decoded.shape in ((h, w), (h, w, 1))
    np.testing.assert_array_equal(decoded.reshape(h, w), data[..., 0])


def test_apply_predictor3_matches_tn3_reference():
    """``_apply_predictor`` with pred=3 inverts TN3 encoding exactly.

    Unit-test of the dispatch fix at the ``_apply_predictor`` level.
    Builds a TN3-encoded buffer for multi-band float32 and checks that
    ``_apply_predictor(chunk, 3, width, height, bytes_per_sample,
    samples=samples)`` returns the original bytes.  This is the
    narrowest possible regression test for the dispatch and does not
    depend on the TIFF header / reader plumbing.
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


@_gpu_only
@pytest.mark.parametrize(
    "samples,dtype_str",
    [
        (3, "float32"),
        (4, "float32"),
        (3, "float64"),
        (4, "float64"),
    ],
    ids=lambda v: v if isinstance(v, str) else f"s{v}",
)
def test_gpu_predictor3_multisample_matches_cpu(
        tmp_path, samples, dtype_str):
    """GPU decode of a tiled multi-sample float TIFF with predictor=3.

    The GPU ``_fp_predictor_decode_kernel`` correctly handles
    multi-sample float rasters; this exercises the predictor=3 path
    (predictor=2 multi-sample is covered separately).
    """
    dtype = np.dtype(dtype_str)
    h, w = 64, 64
    rng = np.random.RandomState(1479)
    data = rng.uniform(-1000.0, 1000.0,
                       size=(h, w, samples)).astype(dtype)
    da = xr.DataArray(data, dims=["y", "x", "band"])

    path = str(tmp_path / f"fp_pred3_{samples}_{dtype_str}.tif")
    to_geotiff(da, path, compression="deflate", predictor=3,
               tiled=True, tile_size=32)

    cpu_arr = open_geotiff(path).values
    assert cpu_arr.shape == (h, w, samples)
    assert cpu_arr.dtype == dtype
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_da = open_geotiff(path, gpu=True)
    gpu_arr = _gpu_to_numpy(gpu_da)

    assert gpu_arr.shape == cpu_arr.shape
    assert gpu_arr.dtype == cpu_arr.dtype
    np.testing.assert_array_equal(gpu_arr, cpu_arr)
    np.testing.assert_array_equal(gpu_arr, data)


# ===========================================================================
# Section 7: Predictor=3 + integer SampleFormat validator
# ===========================================================================
#
# A malformed TIFF that claims Predictor=3 paired with an integer
# SampleFormat used to decode silently to garbage. The byte-swizzle
# unshuffle ran on integer bytes and produced values that look like
# valid integers, with no warning. The writer side already rejects the
# combination in ``_writer._resolve_predictor``; these tests assert
# reader symmetry via the validator in ``_validation``.


class TestPredictor3IntegerSampleFormatRejected:
    """The validator rejects predictor=3 with non-float sample formats."""

    def test_helper_rejects_pred3_sf1_uint(self):
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 1)

    def test_helper_rejects_pred3_sf2_int(self):
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 2)

    def test_helper_rejects_pred3_sf4_undefined(self):
        # SampleFormat=4 (UNDEFINED) is treated as uint by the dtype map;
        # routing it through predictor=3 also produces garbage.
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 4)

    def test_helper_accepts_pred3_sf3_float(self):
        # The legitimate combination remains a no-op.
        _validate_predictor_sample_format(3, 3)

    def test_helper_accepts_pred1_with_any_sf(self):
        # Predictor=1 (none) is sample-format-agnostic.
        _validate_predictor_sample_format(1, 1)
        _validate_predictor_sample_format(1, 2)
        _validate_predictor_sample_format(1, 3)

    def test_helper_accepts_pred2_with_any_sf(self):
        # Predictor=2 (horizontal) is sample-format-agnostic.
        _validate_predictor_sample_format(2, 1)
        _validate_predictor_sample_format(2, 2)
        _validate_predictor_sample_format(2, 3)

    def test_helper_normalizes_tuple_predictor(self):
        # IFD.predictor delegates to get_value, which returns a tuple for
        # a malformed Predictor tag with count > 1. The validator must
        # still fire on the (3,) + non-float pair.
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format((3,), 1)
        # Empty tuple falls back to predictor=1 (none) -> no-op.
        _validate_predictor_sample_format((), 1)
        # Non-3 tuple predictor + non-float sample_format -> no-op.
        _validate_predictor_sample_format((2,), 1)


class TestEagerReadRejectsMalformedFile:
    """``open_geotiff`` raises on the malformed predictor=3 + uint32 file."""

    def test_open_geotiff_eager_raises(self, tmp_path):
        arr = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32.tif"
        path.write_bytes(_build_predictor3_uint32_tiff(arr))

        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path))

    def test_open_geotiff_dask_raises(self, tmp_path):
        arr = np.array(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32_dask.tif"
        path.write_bytes(_build_predictor3_uint32_tiff(arr))

        from xrspatial.geotiff import read_geotiff_dask
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_dask(str(path), chunks=64)


# ===========================================================================
# Section 8: Writer-side predictor validation and normalisation
# ===========================================================================


def test_predictor3_rejects_integer_dtype(tmp_path):
    arr = np.zeros((8, 8), dtype=np.int32)
    da = _da_xy(arr)
    path = tmp_path / "bad_int_pred3.tif"
    with pytest.raises(ValueError, match="predictor=3"):
        to_geotiff(da, str(path), compression="deflate", predictor=3)


def test_normalize_predictor_table():
    """The bool|int -> int normalization mapping."""
    f32 = np.dtype("float32")
    i32 = np.dtype("int32")
    deflate = 8  # COMPRESSION_DEFLATE

    assert normalize_predictor(False, f32, deflate) == 1
    assert normalize_predictor(0, f32, deflate) == 1
    assert normalize_predictor(1, f32, deflate) == 1
    assert normalize_predictor(True, f32, deflate) == 2
    assert normalize_predictor(2, f32, deflate) == 2
    assert normalize_predictor(3, f32, deflate) == 3

    with pytest.raises(ValueError, match="predictor=3"):
        normalize_predictor(3, i32, deflate)

    with pytest.raises(ValueError, match="predictor must be"):
        normalize_predictor(99, f32, deflate)
