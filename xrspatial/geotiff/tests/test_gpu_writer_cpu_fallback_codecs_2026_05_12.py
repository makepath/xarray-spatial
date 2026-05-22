"""Coverage for ``write_geotiff_gpu`` codecs that route to a CPU fallback.

Test coverage gap sweep 2026-05-12 (pass 11). Closes a Cat 4 HIGH
parameter-coverage gap.

``write_geotiff_gpu`` documents nine ``compression=`` modes. The
existing ``test_gpu_writer_compression_modes_2026_05_11.py`` pins
round-trip behaviour for four (``none``, ``deflate``, ``zstd``,
``jpeg``).  The remaining five route through their own branches in
``gpu_compress_tiles`` (``xrspatial/geotiff/_gpu_decode.py:2974-3019``)
and fall through to the CPU compressor when no GPU accelerator is
available:

* ``compression='lzw'`` -> trailing ``cpu_compress`` fallback branch.
* ``compression='packbits'`` -> trailing ``cpu_compress`` fallback.
* ``compression='lz4'`` -> trailing ``cpu_compress`` fallback.
* ``compression='lerc'`` -> dedicated ``lerc_compress`` branch (CPU
  only -- nvCOMP has no LERC backend).
* ``compression='jpeg2000'``/``'j2k'`` -> ``_nvjpeg2k_batch_encode``
  with ``jpeg2000_compress`` fallback.

These code paths shipped without targeted tests; a regression
dropping any codec from the dispatch, swapping the routed
compression tag, or returning the wrong tile-buffer slice would not
surface against the existing suite. See issue #1706.

The tests are *additive* -- no source changes are made. They:

* Round-trip pixel equality on integer / float input through the
  GPU writer.
* Pin the TIFF Compression tag emitted in the IFD (catches the
  wrong-tag class of regression that the internal reader would miss
  because it consults the same lookup).
* Assert pixel equality of GPU-written and CPU-written outputs for
  the lossless codecs so a divergence between the GPU's CPU-fallback
  branch and the CPU writer's matching codec call site surfaces.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff, write_geotiff_gpu
from xrspatial.geotiff._compression import JPEG2000_AVAILABLE, LERC_AVAILABLE, LZ4_AVAILABLE
from xrspatial.geotiff._header import parse_header, parse_ifd

# --------------------------------------------------------------------------
# GPU gating
# --------------------------------------------------------------------------


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
_lerc_only = pytest.mark.skipif(
    not LERC_AVAILABLE, reason="lerc not installed",
)
_lz4_only = pytest.mark.skipif(
    not LZ4_AVAILABLE, reason="lz4 not installed",
)
_j2k_only = pytest.mark.skipif(
    not JPEG2000_AVAILABLE, reason="glymur not installed",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


_TIFF_COMPRESSION_TAG = 259
# Pinned per the TIFF specification + ``_writer._compression_tag`` table.
# Changes here would mask a wrong-codec-tag regression rather than fail it.
_COMPRESSION_TAGS = {
    'lzw': 5,
    'packbits': 32773,
    'lerc': 34887,
    'lz4': 50004,
    'jpeg2000': 34712,
    'j2k': 34712,
}


def _read_compression_tag(path: str) -> int:
    """Return the TIFF Compression (tag 259) value from *path*."""
    with open(path, 'rb') as f:
        data = f.read()
    hdr = parse_header(data)
    ifd = parse_ifd(data, hdr.first_ifd_offset, hdr)
    entry = ifd.entries[_TIFF_COMPRESSION_TAG]
    val = entry.value
    if isinstance(val, (tuple, list)):
        return int(val[0])
    return int(val)


def _make_int_da(h=32, w=32, dtype=np.uint16):
    """Build a deterministic CuPy-backed integer DataArray.

    Integer dtypes exercise the lossless path for every codec under
    test. uint16 covers the common case for satellite imagery; the
    explicit dtype keeps the round-trip assertion straightforward.
    """
    import cupy
    arr = (np.arange(h * w, dtype=np.int64) % 1000).astype(dtype).reshape(h, w)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x'),
        coords={'y': np.arange(h), 'x': np.arange(w)},
    ), arr


def _make_float_da(h=32, w=32, dtype=np.float32):
    """Build a deterministic CuPy-backed float DataArray.

    LERC needs a float input for the lossless-with-max-z-error=0 path
    to be a meaningful test (it accepts integers too but the LERC
    float-quant logic is what historically had bugs).
    """
    import cupy
    arr = (np.arange(h * w, dtype=np.float32) * 0.5).reshape(h, w).astype(dtype)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x'),
        coords={'y': np.arange(h), 'x': np.arange(w)},
    ), arr


def _make_uint8_rgb_da(h=32, w=32):
    """Build a CuPy-backed uint8 3-band RGB DataArray."""
    import cupy
    y = np.linspace(0, 240, h, dtype=np.uint8)
    x = np.linspace(0, 240, w, dtype=np.uint8)
    r = np.broadcast_to(y[:, None], (h, w)).astype(np.uint8)
    g = np.broadcast_to(x[None, :], (h, w)).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x', 'band'),
        coords={'y': np.arange(h), 'x': np.arange(w), 'band': [1, 2, 3]},
    ), arr


# --------------------------------------------------------------------------
# Cat 4 HIGH: lzw round-trip + tag pin
# --------------------------------------------------------------------------


@_gpu_only
def test_write_geotiff_gpu_lzw_roundtrip(tmp_path):
    """``compression='lzw'`` round-trips pixel-exact via the CPU fallback."""
    da, arr = _make_int_da()
    path = str(tmp_path / "lzw_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='lzw')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_lzw_compression_tag(tmp_path):
    """``compression='lzw'`` emits TIFF Compression tag 5."""
    da, _ = _make_int_da()
    path = str(tmp_path / "lzw_tag.tif")

    write_geotiff_gpu(da, path, compression='lzw')

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lzw']


# --------------------------------------------------------------------------
# Cat 4 HIGH: packbits round-trip + tag pin
# --------------------------------------------------------------------------


@_gpu_only
def test_write_geotiff_gpu_packbits_roundtrip(tmp_path):
    """``compression='packbits'`` round-trips pixel-exact."""
    da, arr = _make_int_da()
    path = str(tmp_path / "packbits_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='packbits')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_packbits_compression_tag(tmp_path):
    """``compression='packbits'`` emits TIFF Compression tag 32773."""
    da, _ = _make_int_da()
    path = str(tmp_path / "packbits_tag.tif")

    write_geotiff_gpu(da, path, compression='packbits')

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['packbits']


# --------------------------------------------------------------------------
# Cat 4 HIGH: lz4 round-trip + tag pin
# --------------------------------------------------------------------------


@_gpu_only
@_lz4_only
def test_write_geotiff_gpu_lz4_roundtrip(tmp_path):
    """``compression='lz4'`` round-trips pixel-exact via the nvCOMP-or-CPU
    fallback path.

    LZ4 has an nvCOMP backend (``_nvcomp_batch_compress`` tries it
    first); if it returns ``None`` the trailing CPU ``compress``
    fallback fires. Either branch must produce a TIFF whose pixels
    decode losslessly to the input.
    """
    da, arr = _make_int_da()
    path = str(tmp_path / "lz4_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='lz4', allow_experimental_codecs=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_lz4_only
def test_write_geotiff_gpu_lz4_compression_tag(tmp_path):
    """``compression='lz4'`` emits TIFF Compression tag 50004."""
    da, _ = _make_int_da()
    path = str(tmp_path / "lz4_tag.tif")

    write_geotiff_gpu(da, path, compression='lz4', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lz4']


# --------------------------------------------------------------------------
# Cat 4 HIGH: lerc round-trip + tag pin
# --------------------------------------------------------------------------


@_gpu_only
@_lerc_only
def test_write_geotiff_gpu_lerc_float_lossless_roundtrip(tmp_path):
    """``compression='lerc'`` with default ``max_z_error=0.0`` is lossless
    on a float input and round-trips pixel-exact.

    LERC is the documented float-friendly codec on the GPU writer
    (with ``max_z_error=0`` -- non-zero is explicitly rejected up the
    call chain). The GPU writer routes lerc through ``lerc_compress``
    on CPU; this test pins that the CPU fallback wiring is correct
    and that the file round-trips.
    """
    da, arr = _make_float_da(dtype=np.float32)
    path = str(tmp_path / "lerc_float.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_lerc_only
def test_write_geotiff_gpu_lerc_int_roundtrip(tmp_path):
    """``compression='lerc'`` on integer input round-trips pixel-exact.

    Integer LERC exercises a different quantisation branch inside the
    CPU lerc_compress helper than the float path; both feed through
    the same GPU writer dispatch branch but historic LERC bugs have
    landed in only one dtype kind at a time.
    """
    da, arr = _make_int_da(dtype=np.uint16)
    path = str(tmp_path / "lerc_int.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_lerc_only
def test_write_geotiff_gpu_lerc_compression_tag(tmp_path):
    """``compression='lerc'`` emits TIFF Compression tag 34887."""
    da, _ = _make_float_da()
    path = str(tmp_path / "lerc_tag.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lerc']


# --------------------------------------------------------------------------
# Cat 4 HIGH: jpeg2000 / j2k round-trip + tag pin
# --------------------------------------------------------------------------


@_gpu_only
@_j2k_only
def test_write_geotiff_gpu_jpeg2000_uint8_lossless_roundtrip(tmp_path):
    """``compression='jpeg2000'`` on uint8 single-band round-trips pixel-exact.

    JPEG 2000 routes through ``_nvjpeg2k_batch_encode`` first and
    falls back to glymur's ``jpeg2000_compress`` when nvJPEG2K is not
    loadable. The CPU ``test_jpeg2000`` suite covers the codec
    directly; this test pins the GPU writer's dispatch to it.

    The compression-only ``lossless=True`` default in
    ``jpeg2000_compress`` keeps the round-trip pixel-exact, so a
    regression in the wiring that fell through to a different codec
    branch (or rendered the input via lossy mode) would surface here.
    """
    da, arr = _make_int_da(dtype=np.uint8)
    path = str(tmp_path / "j2k_uint8.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_j2k_only
def test_write_geotiff_gpu_jpeg2000_rgb_roundtrip(tmp_path):
    """``compression='jpeg2000'`` on a 3-band uint8 raster round-trips
    pixel-exact.

    Multi-band JPEG 2000 exercises a different glymur call signature
    (samples=3 vs. samples=1). A regression that ignored the
    ``samples_per_pixel`` argument would silently produce a 1-band
    file or a corrupted RGB file.
    """
    da, arr = _make_uint8_rgb_da()
    path = str(tmp_path / "j2k_rgb.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.shape == arr.shape


@_gpu_only
@_j2k_only
def test_write_geotiff_gpu_j2k_alias_matches_jpeg2000(tmp_path):
    """The ``'j2k'`` alias produces a file equivalent to ``'jpeg2000'``.

    ``_compression_tag`` maps both names to ``COMPRESSION_JPEG2000``
    (34712). A regression that diverged the two aliases on the GPU
    writer would silently drop one or the other depending on which
    name the user passed.
    """
    da, arr = _make_int_da(dtype=np.uint8)
    j2k_path = str(tmp_path / "alias_j2k.tif")
    jpeg2k_path = str(tmp_path / "alias_jpeg2000.tif")

    write_geotiff_gpu(da, j2k_path, compression='j2k', allow_experimental_codecs=True)
    write_geotiff_gpu(da, jpeg2k_path, compression='jpeg2000', allow_experimental_codecs=True)

    assert _read_compression_tag(j2k_path) == _COMPRESSION_TAGS['j2k']
    assert _read_compression_tag(jpeg2k_path) == _COMPRESSION_TAGS['jpeg2000']
    np.testing.assert_array_equal(open_geotiff(j2k_path).values, arr)
    np.testing.assert_array_equal(open_geotiff(jpeg2k_path).values, arr)


@_gpu_only
@_j2k_only
def test_write_geotiff_gpu_jpeg2000_compression_tag(tmp_path):
    """``compression='jpeg2000'`` emits TIFF Compression tag 34712."""
    da, _ = _make_int_da(dtype=np.uint8)
    path = str(tmp_path / "j2k_tag.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['jpeg2000']


# --------------------------------------------------------------------------
# Cross-backend parity (CPU vs GPU lossless codecs)
# --------------------------------------------------------------------------


@_gpu_only
@pytest.mark.parametrize("compression", ['lzw', 'packbits'])
def test_write_geotiff_gpu_cpu_parity_lossless(tmp_path, compression):
    """The GPU writer's CPU fallback produces a file that decodes to the
    same array as ``to_geotiff(gpu=False)`` for the same input.

    The GPU writer's lossless fallback codecs land in the same
    ``cpu_compress`` helper that ``to_geotiff`` invokes, so the decoded
    arrays must match pixel-exact. A wiring regression that fed a
    different tile slice or mis-routed the predictor would surface here
    as a pixel-level disagreement that the GPU-only round-trip does
    not catch.

    LZ4 is omitted from the parametrise list because its routing
    through ``_nvcomp_batch_compress`` makes the byte-stream codec
    choice depend on the host's nvCOMP install: the test would assert
    nvCOMP-vs-CPU LZ4 parity, not GPU-writer vs CPU-writer parity.
    LERC and JPEG 2000 are also omitted because their compression
    branches go through dedicated CPU helpers (``lerc_compress``,
    ``jpeg2000_compress``) and either backend can take the codec call
    site independently -- pixel parity is still asserted by the
    per-codec round-trip tests above.
    """
    da, arr = _make_int_da()
    gpu_path = str(tmp_path / f"gpu_{compression}.tif")
    cpu_path = str(tmp_path / f"cpu_{compression}.tif")

    write_geotiff_gpu(da, gpu_path, compression=compression)

    # Build a CPU DataArray with the same input. to_geotiff(gpu=False)
    # writes through the matching CPU codec branch.
    cpu_da = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(arr.shape[0]), 'x': np.arange(arr.shape[1])},
    )
    to_geotiff(cpu_da, cpu_path, compression=compression, gpu=False)

    gpu_out = open_geotiff(gpu_path).values
    cpu_out = open_geotiff(cpu_path).values

    np.testing.assert_array_equal(gpu_out, arr)
    np.testing.assert_array_equal(cpu_out, arr)
    np.testing.assert_array_equal(gpu_out, cpu_out)


# --------------------------------------------------------------------------
# Dispatch through to_geotiff(gpu=True, compression=...) routes here
# --------------------------------------------------------------------------


@_gpu_only
@pytest.mark.parametrize("compression", ['lzw', 'packbits'])
def test_to_geotiff_gpu_true_dispatches_through_fallback_codec(
    tmp_path, compression,
):
    """``to_geotiff(gpu=True, compression='lzw'/'packbits')`` should still
    succeed (rather than rejecting the codec at the dispatch layer).

    ``to_geotiff`` rejects ``compression='jpeg'`` outright and forwards
    everything else to ``write_geotiff_gpu`` when ``gpu=True``. The
    GPU writer documents the CPU-fallback codecs as accepted, so the
    auto-dispatch path must round-trip them too. A regression that
    added a fallback-codec rejection in the dispatch (a la jpeg)
    would silently break callers who pass the same kwargs to both
    backends.
    """
    da, arr = _make_int_da()
    path = str(tmp_path / f"dispatch_{compression}.tif")

    to_geotiff(da, path, compression=compression, gpu=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert _read_compression_tag(path) == _COMPRESSION_TAGS[compression]
