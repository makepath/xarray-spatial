"""Coverage for ``write_geotiff_gpu`` compression modes.

The GPU writer documents four ``compression=`` modes: ``'zstd'``
(default, "fastest on GPU"), ``'deflate'``, ``'jpeg'`` (nvJPEG with
Pillow fallback), and ``'none'``. The existing test suite exercises
only ``'none'`` and ``'deflate'`` with direct round-trip assertions.

* ``'zstd'`` is the default and is hit implicitly by tests that omit
  ``compression=``, but no test asserts pixel fidelity for the zstd
  path. A regression in the nvCOMP zstd encoder (or in the writer's
  zstd codec-tag wiring) would not surface against the implicit
  callers because they only assert metadata-level properties.

* ``'jpeg'`` routes to ``_nvjpeg_batch_encode`` with a CPU Pillow
  fallback. Neither code path is exercised through ``write_geotiff_gpu``
  anywhere else in the suite. ``to_geotiff(compression='jpeg')``
  rejects the CPU path with the JPEGTables interop error, so the only
  way to reach the GPU JPEG encoder via the public API is through
  ``write_geotiff_gpu``.

This module closes the Cat 4 HIGH parameter-coverage gap by pinning a
round-trip test for each documented mode (zstd, deflate, jpeg, none)
plus a parametrised TIFF compression-tag check that the file header
advertises the right codec.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    write_geotiff_gpu,
)
from xrspatial.geotiff import _gpu_decode
from xrspatial.geotiff._header import parse_header, parse_ifd


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


def _nvjpeg_available() -> bool:
    """True when libnvjpeg can be loaded; ``_nvjpeg_batch_encode`` will
    actually fire instead of silently falling back to Pillow."""
    if not _HAS_GPU:
        return False
    try:
        return _gpu_decode._get_nvjpeg() is not None
    except Exception:
        return False


_HAS_NVJPEG = _nvjpeg_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
_nvjpeg_only = pytest.mark.skipif(
    not _HAS_NVJPEG, reason="libnvjpeg required for nvJPEG encode path",
)


class _CallSpy:
    """Counts forwarded calls to a wrapped callable."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._fn(*args, **kwargs)


# Compression-tag IDs from the TIFF specification, mirroring the table
# in ``_writer._compression_tag``. Pinned here so an accidental change
# to the codec-tag wiring is caught.
_TIFF_COMPRESSION_TAG = 259
_COMPRESSION_TAGS = {
    'none': 1,
    'deflate': 8,
    'jpeg': 7,
    'zstd': 50000,
}


def _read_compression_tag(path: str) -> int:
    """Return the TIFF Compression (tag 259) value from *path*."""
    with open(path, 'rb') as f:
        data = f.read()
    hdr = parse_header(data)
    ifd = parse_ifd(data, hdr.first_ifd_offset, hdr)
    entry = ifd.entries[_TIFF_COMPRESSION_TAG]
    val = entry.value
    # value is either an int scalar or a 1-tuple depending on count;
    # the TIFF spec allows count=1 to be inlined.
    if isinstance(val, (tuple, list)):
        return int(val[0])
    return int(val)


def _make_int_da(h=64, w=64, dtype=np.int32):
    """Build a deterministic CuPy-backed DataArray for lossless codecs."""
    import cupy
    arr = (np.arange(h * w, dtype=np.int64) % 1000).astype(dtype).reshape(h, w)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x'),
        coords={'y': np.arange(h), 'x': np.arange(w)},
    ), arr


def _make_rgb_uint8_da(h=64, w=64):
    """Build a CuPy-backed uint8 3-band DataArray with a smooth gradient.

    JPEG is lossy; random noise is the worst case and makes round-trip
    tests platform/library-sensitive. A deterministic smooth gradient
    (mirroring ``test_jpeg.py``'s ``_gradient_rgb``) keeps the
    quantisation error well below 10 absolute units per channel even at
    default quality, so a tight tolerance is achievable.
    """
    import cupy
    y = np.linspace(20, 240, h, dtype=np.uint8)
    x = np.linspace(20, 240, w, dtype=np.uint8)
    r = np.broadcast_to(y[:, None], (h, w)).astype(np.uint8)
    g = np.broadcast_to(x[None, :], (h, w)).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x', 'band'),
        coords={'y': np.arange(h), 'x': np.arange(w), 'band': [1, 2, 3]},
    ), arr


def _make_mono_uint8_da(h=64, w=64):
    """Single-band uint8 smooth gradient."""
    import cupy
    y = np.linspace(20, 240, h, dtype=np.uint8)
    x = np.linspace(20, 240, w, dtype=np.uint8)
    arr = ((y[:, None].astype(np.int32) + x[None, :].astype(np.int32)) // 2
           ).astype(np.uint8)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x'),
        coords={'y': np.arange(h), 'x': np.arange(w)},
    ), arr


# ---------------------------------------------------------------------------
# Cat 4 HIGH: zstd is the documented default, never round-tripped explicitly
# ---------------------------------------------------------------------------

@_gpu_only
def test_write_geotiff_gpu_zstd_roundtrip(tmp_path):
    """Default ``compression='zstd'`` round-trips pixel-exact.

    The GPU writer advertises zstd as the fastest GPU codec and uses it
    as the default. nvCOMP zstd is lossless, so the read-back must
    equal the input bit-for-bit.
    """
    da, arr = _make_int_da()
    path = str(tmp_path / "zstd_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='zstd')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_zstd_default_matches_explicit(tmp_path):
    """Omitting ``compression=`` selects the zstd codec.

    Pins the default so a silent change to the default codec (eg. to
    'deflate') would fail this test. We assert that

    (a) both files advertise the zstd compression tag in their IFD, and
    (b) the decoded pixel arrays are identical.

    We deliberately do not require byte-for-byte identity of the on-disk
    files: the writer is free to vary tile ordering or padding between
    runs, and the test would become brittle. The compression-tag pin
    plus the decoded-array equality is enough to catch a default-codec
    swap.
    """
    da, arr = _make_int_da()
    default_path = str(tmp_path / "default.tif")
    explicit_path = str(tmp_path / "explicit_zstd.tif")

    write_geotiff_gpu(da, default_path)
    write_geotiff_gpu(da, explicit_path, compression='zstd')

    assert _read_compression_tag(default_path) == _COMPRESSION_TAGS['zstd']
    assert _read_compression_tag(explicit_path) == _COMPRESSION_TAGS['zstd']

    default_out = open_geotiff(default_path).values
    explicit_out = open_geotiff(explicit_path).values
    np.testing.assert_array_equal(default_out, arr)
    np.testing.assert_array_equal(default_out, explicit_out)


# ---------------------------------------------------------------------------
# Cat 4 HIGH: jpeg is documented but never round-tripped
# ---------------------------------------------------------------------------

@_gpu_only
def test_write_geotiff_gpu_jpeg_rgb_roundtrip(tmp_path):
    """``compression='jpeg'`` round-trips a 3-band uint8 RGB raster.

    Uses a deterministic smooth gradient (the worst-case-for-JPEG random
    input was replaced per Copilot review on #1647). At default quality
    plus 4:2:0 chroma subsampling a smooth RGB gradient round-trips with
    mean-abs error well under 5 absolute units per channel; we allow 8
    as a small platform-variance buffer.
    """
    da, arr = _make_rgb_uint8_da()
    path = str(tmp_path / "jpeg_rgb.tif")

    # Issue #1845: the JPEG encode path is opt-in. The writer also
    # emits a GeoTIFFFallbackWarning, which is the documented contract.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    out = open_geotiff(path)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    diff = np.abs(out.values.astype(np.int32) - arr.astype(np.int32))
    assert diff.mean() < 8, (
        f"JPEG round-trip mean diff {diff.mean()} suggests encoder/decoder break"
    )


@_gpu_only
def test_write_geotiff_gpu_jpeg_uint8_single_band_roundtrip(tmp_path):
    """``compression='jpeg'`` round-trips a 1-band uint8 (greyscale)
    raster.

    Single-band JPEG exercises a different nvJPEG path (luminance-only
    vs. RGB) and the Pillow fallback's monochrome branch. Smooth
    gradient keeps the round-trip error tight.
    """
    da, arr = _make_mono_uint8_da()
    path = str(tmp_path / "jpeg_mono.tif")

    # Issue #1845: opt-in flag required; warning fires.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    out = open_geotiff(path)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    diff = np.abs(out.values.astype(np.int32) - arr.astype(np.int32))
    assert diff.mean() < 5


@_nvjpeg_only
def test_write_geotiff_gpu_jpeg_uses_nvjpeg_when_available(tmp_path,
                                                          monkeypatch):
    """When libnvjpeg is present the writer must hit ``_nvjpeg_batch_encode``,
    not silently fall back to Pillow.

    The encode path inside ``gpu_compress_tiles`` tries nvJPEG first and
    only falls back when it returns ``None``. A silent regression that
    breaks nvJPEG would still produce a valid file via Pillow, so the
    round-trip tests above can't catch it. Here we spy on the encoder
    and assert the GPU path actually fired.
    """
    spy = _CallSpy(_gpu_decode._nvjpeg_batch_encode)
    monkeypatch.setattr(_gpu_decode, "_nvjpeg_batch_encode", spy)

    da, _ = _make_rgb_uint8_da()
    path = str(tmp_path / "jpeg_nvjpeg_spy.tif")

    # Issue #1845: opt-in flag required; warning fires.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    assert spy.calls >= 1, (
        "libnvjpeg is loadable but _nvjpeg_batch_encode was never called; "
        "the JPEG path silently fell through to the Pillow fallback"
    )
    # Sanity: a file still got written.
    assert _read_compression_tag(path) == _COMPRESSION_TAGS['jpeg']


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: compression-tag header check across all documented modes
# ---------------------------------------------------------------------------

@_gpu_only
@pytest.mark.parametrize("compression", ['none', 'deflate', 'zstd'])
def test_write_geotiff_gpu_compression_tag(tmp_path, compression):
    """The TIFF Compression tag in the output matches the requested
    codec.

    A regression that wired the writer to a different codec tag would
    produce files that decode correctly through the internal reader
    (it inspects the same wired tag) but break interop with GDAL /
    rasterio / libtiff.
    """
    da, _ = _make_int_da()
    path = str(tmp_path / f"compression_tag_{compression}.tif")

    write_geotiff_gpu(da, path, compression=compression)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS[compression]


@_gpu_only
def test_write_geotiff_gpu_jpeg_compression_tag(tmp_path):
    """The JPEG compression tag (7) is written for uint8 RGB input."""
    da, _ = _make_rgb_uint8_da()
    path = str(tmp_path / "jpeg_tag.tif")

    # Issue #1845: opt-in flag required; warning fires.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['jpeg']


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: explicit deflate round-trip (already covered indirectly
# but no test in the suite asserts pixel equality on the GPU writer
# deflate path with a non-COG/non-overview layout).
# ---------------------------------------------------------------------------

@_gpu_only
def test_write_geotiff_gpu_deflate_roundtrip(tmp_path):
    """``compression='deflate'`` round-trips pixel-exact for the plain
    (non-COG) GPU writer path.

    The existing deflate coverage on the GPU writer runs through the
    COG path or through NaN-sentinel scenarios. This test pins the
    plain tiled-deflate layout against a deterministic integer raster.
    """
    da, arr = _make_int_da()
    path = str(tmp_path / "deflate_plain.tif")

    write_geotiff_gpu(da, path, compression='deflate')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert _read_compression_tag(path) == _COMPRESSION_TAGS['deflate']


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM: none / uncompressed round-trip
# ---------------------------------------------------------------------------

@_gpu_only
def test_write_geotiff_gpu_none_roundtrip(tmp_path):
    """``compression='none'`` round-trips pixel-exact.

    The GPU writer still chunks the image into tile buffers even when
    no codec is applied; this test pins that the no-codec assembly
    path emits a valid, readable file.
    """
    da, arr = _make_int_da()
    path = str(tmp_path / "none_plain.tif")

    write_geotiff_gpu(da, path, compression='none')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert _read_compression_tag(path) == _COMPRESSION_TAGS['none']


# ---------------------------------------------------------------------------
# Cross-codec parity: pixel-exact for lossless codecs
# ---------------------------------------------------------------------------

@_gpu_only
def test_write_geotiff_gpu_lossless_codecs_agree(tmp_path):
    """zstd / deflate / none must produce pixel-identical read-backs.

    The codecs are lossless, so for the same input the decoded
    pixel arrays must match exactly. Catches regressions where a codec
    path silently corrupts data (eg. wrong predictor wiring).
    """
    da, arr = _make_int_da()
    paths = {
        codec: str(tmp_path / f"parity_{codec}.tif")
        for codec in ('none', 'deflate', 'zstd')
    }
    for codec, path in paths.items():
        write_geotiff_gpu(da, path, compression=codec)

    reads = {codec: open_geotiff(path).values for codec, path in paths.items()}

    np.testing.assert_array_equal(reads['none'], arr)
    np.testing.assert_array_equal(reads['deflate'], reads['none'])
    np.testing.assert_array_equal(reads['zstd'], reads['none'])
