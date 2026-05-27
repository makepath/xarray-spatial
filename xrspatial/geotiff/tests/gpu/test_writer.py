"""GPU writer tests.

Folds the GPU-only writer test files into one parametrised home under
``xrspatial/geotiff/tests/gpu/``. Sections in source-order below:

* ``test_gpu_writer_attrs_1563.py`` -- write-path attrs parity for the
  GPU writer (crs_wkt, image_description, extra_samples, colormap,
  extra_tags, resolution, gdal_metadata, transform, nodata).
* ``test_gpu_writer_band_first_1580.py`` -- ``(band, y, x)`` layout
  remap on the GPU writer + ``to_geotiff`` auto-dispatch parity.
* ``test_gpu_writer_compression_modes_2026_05_11.py`` -- round-trip and
  compression-tag pins for the four documented GPU codecs (zstd,
  deflate, jpeg, none) plus the nvJPEG spy.
* ``test_gpu_writer_cpu_fallback_codecs_2026_05_12.py`` -- CPU-fallback
  codecs on the GPU writer (lzw, packbits, lz4, lerc, jpeg2000 / j2k).
* ``test_gpu_writer_nan_sentinel_1599.py`` -- NaN-to-sentinel rewrite on
  the GPU write path so external readers see the same valid-pixel set.
* ``test_gpu_writer_overview_inplace_1948.py`` -- in-place
  ``cupy.putmask`` rewrite for the COG overview loop + buffer-aliasing
  guard.
* ``test_gpu_writer_overview_mode_and_compression_level_1740.py`` --
  ``overview_resampling='mode'`` dispatch and ``compression_level=``
  accepted-but-ignored contract on the GPU writer.
* ``test_to_geotiff_gpu_fallback_1674.py`` -- ``to_geotiff(gpu=True)``
  exception-classification contract (ImportError + GPU-signal
  RuntimeError fall back with warning; everything else propagates).

The per-file ``_gpu_only`` / ``_gpu_available`` helpers are replaced
with the shared ``requires_gpu`` marker from ``_helpers/markers.py``
(aliased ``_gpu_only`` for brevity in this module). The CuPy-aware
``test_to_geotiff_gpu_fallback_1674.py`` tests never required a real
GPU (they monkeypatch ``write_geotiff_gpu``); they keep running on
non-GPU hosts as before.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import tempfile
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (GeoTIFFFallbackWarning, _gpu_decode, open_geotiff, to_geotiff,
                               write_geotiff_gpu)
from xrspatial.geotiff._compression import JPEG2000_AVAILABLE, LERC_AVAILABLE, LZ4_AVAILABLE
from xrspatial.geotiff._geotags import GeoTransform, _epsg_to_wkt
from xrspatial.geotiff._header import parse_header, parse_ifd
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import _block_reduce_2d, write
from xrspatial.geotiff.tests.conftest import requires_gpu as _gpu_only

# ---------------------------------------------------------------------------
# nvJPEG capability gate (only used by the compression-modes section).
# ---------------------------------------------------------------------------


def _nvjpeg_available() -> bool:
    """True when libnvjpeg can be loaded; ``_nvjpeg_batch_encode`` will
    actually fire instead of silently falling back to Pillow."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy  # noqa: F401
        if not cupy.cuda.is_available():
            return False
        return _gpu_decode._get_nvjpeg() is not None
    except Exception:
        return False


_nvjpeg_only = pytest.mark.skipif(
    not _nvjpeg_available(), reason="libnvjpeg required for nvJPEG encode path",
)
_lerc_only = pytest.mark.skipif(
    not LERC_AVAILABLE, reason="lerc not installed",
)
_lz4_only = pytest.mark.skipif(
    not LZ4_AVAILABLE, reason="lz4 not installed",
)
_j2k_only = pytest.mark.skipif(
    not JPEG2000_AVAILABLE, reason="glymur not installed",
)


# ===========================================================================
# Section 1: write-path attrs parity (was test_gpu_writer_attrs_1563.py)
# ===========================================================================


@_gpu_only
def test_crs_wkt_only_attr_round_trips(tmp_path):
    """When the source CRS only resolves to WKT (no EPSG attr), the GPU
    writer must still emit GeoKeys so the file has a CRS at all."""
    import cupy
    wkt = _epsg_to_wkt(4326)
    if wkt is None:
        pytest.skip("pyproj not available")
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs_wkt': wkt},
    )
    out = str(tmp_path / 'crs_wkt_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    # The WKT should resolve back to EPSG 4326 on read.
    assert rd.attrs.get('crs') == 4326, (
        f"crs_wkt was dropped on the GPU write path; "
        f"got attrs={sorted(rd.attrs.keys())}"
    )


@_gpu_only
def test_image_description_round_trips_via_gpu_writer(tmp_path):
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'image_description': 'gpu-attr-test-1563'},
    )
    out = str(tmp_path / 'desc_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('image_description') == 'gpu-attr-test-1563'


@_gpu_only
def test_extra_samples_single_band_writer_compat(tmp_path):
    """Single-band write with ``extra_samples`` set must not crash, even
    though TIFF 6.0 only surfaces ExtraSamples on multi-sample images
    (the reader drops it for 1-sample files). The multi-band case is
    covered by ``test_extra_samples_round_trips_multiband_via_gpu_writer``
    below.
    """
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'extra_samples': [1]},
    )
    out = str(tmp_path / 'es_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('crs') == 4326


@_gpu_only
def test_extra_samples_round_trips_multiband_via_gpu_writer(tmp_path):
    """Multi-band write: ExtraSamples surfaces on the reader because
    SamplesPerPixel > 1. The assembler auto-synthesizes the tag for
    multi-band minisblack (same behaviour as the CPU writer), so we
    just assert the attr appears -- pinning that the GPU path reaches
    the multi-band writer code without dropping ExtraSamples entirely.
    """
    import cupy
    arr = np.zeros((4, 5, 2), dtype=np.uint8)
    arr[..., 1] = 255
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x', 'band'],
        coords={'y': np.arange(4.0), 'x': np.arange(5.0)},
        attrs={'crs': 4326},
    )
    out = str(tmp_path / 'es_multi_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    es = rd.attrs.get('extra_samples')
    assert es is not None, (
        f"extra_samples dropped on multi-band GPU write; "
        f"attrs={sorted(rd.attrs.keys())}"
    )


@_gpu_only
def test_colormap_round_trips_via_gpu_writer(tmp_path):
    """A uint8 raster with a 768-entry colormap (3*256 uint16 values for
    R/G/B) must surface ``attrs['colormap']`` after the GPU round-trip --
    the synthesized tag 320 entry rides in ``extra_tags`` and the read
    path projects it back onto the friendly attr."""
    import cupy
    palette = []
    for ch_offset in (0, 1, 2):
        for i in range(256):
            palette.append((i * 257 + ch_offset) & 0xFFFF)
    assert len(palette) == 768
    pixels = np.array([[0, 1, 2, 254, 255],
                       [10, 20, 30, 40, 50]], dtype=np.uint8)
    da_gpu = xr.DataArray(
        cupy.asarray(pixels),
        dims=['y', 'x'],
        coords={'y': np.arange(2.0), 'x': np.arange(5.0)},
        attrs={'crs': 4326, 'colormap': palette},
    )
    out = str(tmp_path / 'cmap_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    rd_cmap = rd.attrs.get('colormap')
    assert rd_cmap is not None, (
        f"colormap dropped on GPU write; attrs={sorted(rd.attrs.keys())}"
    )
    assert tuple(rd_cmap) == tuple(palette)


@_gpu_only
def test_extra_tags_custom_tag_round_trips_via_gpu_writer(tmp_path):
    """A user-supplied ``extra_tags`` entry (here: Software, tag 305,
    ASCII) must be forwarded to ``_assemble_tiff`` and reappear in
    ``attrs['extra_tags']`` on read."""
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    software = "xrspatial-1563-test"
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'extra_tags': [(305, 2, len(software) + 1, software)],
        },
    )
    out = str(tmp_path / 'extra_tags_1563.tif')
    # Rich-tag extra_tags is an experimental write surface. Opt in on
    # both write and read sides for the round-trip.
    write_geotiff_gpu(da_gpu, out, compression='none',
                      allow_experimental_codecs=True)

    rd = open_geotiff(out)
    et = rd.attrs.get('extra_tags') or []
    by_id = {t[0]: t for t in et}
    assert 305 in by_id, (
        f"extra_tags Software (305) dropped on GPU write; got ids "
        f"{sorted(by_id.keys())}"
    )
    value = by_id[305][3]
    # ASCII values may be returned as bytes or str; strip NUL terminator.
    if isinstance(value, bytes):
        value = value.decode('ascii')
    assert value.rstrip('\x00') == software


@_gpu_only
def test_resolution_tags_round_trip_via_gpu_writer(tmp_path):
    """``x_resolution`` / ``y_resolution`` / ``resolution_unit`` must
    survive a GPU write -> CPU read cycle. The writer maps the unit
    string back to its TIFF id (1=none, 2=inch, 3=centimeter)."""
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'x_resolution': 300.0,
            'y_resolution': 300.0,
            'resolution_unit': 'inch',
        },
    )
    out = str(tmp_path / 'res_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('x_resolution') == pytest.approx(300.0, rel=0.01), (
        f"x_resolution drift: got {rd.attrs.get('x_resolution')}"
    )
    assert rd.attrs.get('y_resolution') == pytest.approx(300.0, rel=0.01), (
        f"y_resolution drift: got {rd.attrs.get('y_resolution')}"
    )
    assert rd.attrs.get('resolution_unit') == 'inch', (
        f"resolution_unit drift: got {rd.attrs.get('resolution_unit')}"
    )


@_gpu_only
def test_gdal_metadata_round_trips_via_gpu_writer(tmp_path):
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326,
               'gdal_metadata': {'AREA_OR_POINT': 'Area',
                                 'CUSTOM_KEY': 'val_1563'}},
    )
    out = str(tmp_path / 'gdal_meta_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    meta = rd.attrs.get('gdal_metadata') or {}
    assert meta.get('CUSTOM_KEY') == 'val_1563', (
        f"gdal_metadata was dropped on the GPU write path; "
        f"got {meta}"
    )


@_gpu_only
def test_transform_attr_round_trip_bit_stable(tmp_path):
    """Reading a file with a fractional pixel size, then writing it back
    through ``write_geotiff_gpu`` must preserve ``attrs['transform']``
    bit-for-bit (the CPU writer guarantees this; the GPU writer used to
    drop the attr and recompute from coords, which drifts).
    """
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    gt = GeoTransform(
        origin_x=-122.123456789,
        origin_y=37.987654321,
        pixel_width=1.0 / 3600.0 + 1e-12,
        pixel_height=-(1.0 / 3600.0 + 1e-12),
    )
    src = str(tmp_path / 'frac_in_1563.tif')
    write(arr, src, geo_transform=gt, crs_epsg=4326,
          compression='none', tiled=False)
    eager_in = open_geotiff(src)

    da_gpu = open_geotiff(src, gpu=True)
    out = str(tmp_path / 'frac_out_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    after = open_geotiff(out)
    assert after.attrs['transform'] == eager_in.attrs['transform'], (
        f"transform drifted on GPU round-trip:\n"
        f"  in : {eager_in.attrs['transform']}\n"
        f"  out: {after.attrs['transform']}"
    )


@_gpu_only
def test_no_data_attr_still_round_trips_after_fix(tmp_path):
    """Regression guard: the existing nodata + raster_type pass-through
    still works after wiring in the new attrs."""
    import cupy
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(2.0), 'x': np.arange(2.0)},
        attrs={'crs': 4326, 'nodata': -9999.0, 'raster_type': 'point'},
    )
    out = str(tmp_path / 'nodata_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('nodata') == -9999.0
    assert rd.attrs.get('raster_type') == 'point'


# ===========================================================================
# Section 2: (band, y, x) layout remap (was test_gpu_writer_band_first_1580.py)
# ===========================================================================


@_gpu_only
@pytest.mark.parametrize("band_dim_name", ["band", "bands", "channel"])
def test_band_first_layout_written_correctly_via_write_geotiff_gpu(
        tmp_path, band_dim_name):
    """Direct call to write_geotiff_gpu with a (band, y, x) CuPy
    DataArray must produce the same file dimensions as the CPU writer
    would (height=y, width=x, samples=band).
    """
    import cupy
    rng = np.random.default_rng(seed=1580)
    arr_bhw = rng.integers(0, 255, (3, 16, 32), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=[band_dim_name, "y", "x"],
        coords={
            band_dim_name: np.arange(3),
            "y": np.arange(16, dtype=np.float64),
            "x": np.arange(32, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / f"band_first_1580_{band_dim_name}.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes["y"] == 16, (
        f"expected height=16, got {rd.sizes}; the writer is treating the "
        f"band axis as height"
    )
    assert rd.sizes["x"] == 32, f"expected width=32, got {rd.sizes}"
    assert rd.sizes["band"] == 3, (
        f"expected 3 bands, got {rd.sizes}; the writer is treating the "
        f"width as samples-per-pixel"
    )

    # Pixel values should match the source after the (band, y, x) ->
    # (y, x, band) remap.
    np.testing.assert_array_equal(rd.values, np.moveaxis(arr_bhw, 0, -1))


@_gpu_only
def test_band_first_layout_via_to_geotiff_auto_dispatch(tmp_path):
    """The user-facing path: pass a CuPy (band, y, x) DataArray to
    ``to_geotiff`` and let auto-detection pick the GPU writer.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 1)
    arr_bhw = rng.integers(0, 255, (2, 8, 12), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=["band", "y", "x"],
        coords={
            "band": np.arange(2),
            "y": np.arange(8, dtype=np.float64),
            "x": np.arange(12, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / "band_first_1580_autodispatch.tif")
    to_geotiff(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes == {"y": 8, "x": 12, "band": 2}, (
        f"auto-dispatched GPU write produced wrong dims: {rd.sizes}"
    )
    np.testing.assert_array_equal(rd.values, np.moveaxis(arr_bhw, 0, -1))


@_gpu_only
def test_yxbands_layout_unchanged(tmp_path):
    """Regression guard: the original (y, x, band) layout must still
    write correctly after the band-first remap was added.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 2)
    arr_yxb = rng.integers(0, 255, (8, 12, 2), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_yxb),
        dims=["y", "x", "band"],
        coords={
            "y": np.arange(8, dtype=np.float64),
            "x": np.arange(12, dtype=np.float64),
            "band": np.arange(2),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / "yxb_1580.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes == {"y": 8, "x": 12, "band": 2}
    np.testing.assert_array_equal(rd.values, arr_yxb)


@_gpu_only
def test_gpu_band_first_matches_cpu_byte_for_byte_on_pixel_values(tmp_path):
    """Cross-backend parity: GPU and CPU writers must emit the same
    pixel values for the same (band, y, x) input.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 3)
    arr_bhw = rng.integers(0, 255, (3, 24, 40), dtype=np.uint8)
    da_cpu = xr.DataArray(
        arr_bhw,
        dims=["band", "y", "x"],
        coords={"band": np.arange(3),
                "y": np.arange(24, dtype=np.float64),
                "x": np.arange(40, dtype=np.float64)},
        attrs={"crs": 4326},
    )
    da_gpu = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=["band", "y", "x"],
        coords={"band": np.arange(3),
                "y": np.arange(24, dtype=np.float64),
                "x": np.arange(40, dtype=np.float64)},
        attrs={"crs": 4326},
    )

    cpu_path = str(tmp_path / "band_first_1580_cpu.tif")
    gpu_path = str(tmp_path / "band_first_1580_gpu.tif")
    to_geotiff(da_cpu, cpu_path, compression="none", gpu=False)
    write_geotiff_gpu(da_gpu, gpu_path, compression="none")

    cpu_rd = open_geotiff(cpu_path).values
    gpu_rd = open_geotiff(gpu_path).values
    np.testing.assert_array_equal(cpu_rd, gpu_rd)


# ===========================================================================
# Section 3: compression modes (was test_gpu_writer_compression_modes_2026_05_11.py)
# ===========================================================================


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


@_gpu_only
def test_write_geotiff_gpu_jpeg_rgb_roundtrip(tmp_path):
    """``compression='jpeg'`` round-trips a 3-band uint8 RGB raster.

    Uses a deterministic smooth gradient rather than the
    worst-case-for-JPEG random input. At default quality
    plus 4:2:0 chroma subsampling a smooth RGB gradient round-trips with
    mean-abs error well under 5 absolute units per channel; we allow 8
    as a small platform-variance buffer.
    """
    da, arr = _make_rgb_uint8_da()
    path = str(tmp_path / "jpeg_rgb.tif")

    # The JPEG encode path is opt-in. The writer also emits a
    # GeoTIFFFallbackWarning, which is the documented contract.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    out = open_geotiff(path, allow_internal_only_jpeg=True)
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

    # Opt-in flag required; warning fires.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    out = open_geotiff(path, allow_internal_only_jpeg=True)
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

    # Opt-in flag required; warning fires.
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

    # Opt-in flag required; warning fires.
    with pytest.warns(Warning):
        write_geotiff_gpu(
            da, path, compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['jpeg']


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


# ===========================================================================
# Section 4: CPU-fallback codecs
# (was test_gpu_writer_cpu_fallback_codecs_2026_05_12.py)
# ===========================================================================


def _make_int_da_small(h=32, w=32, dtype=np.uint16):
    """Build a deterministic CuPy-backed integer DataArray for the
    CPU-fallback codec tests. Smaller than ``_make_int_da`` to keep the
    CPU LERC / glymur paths fast."""
    import cupy
    arr = (np.arange(h * w, dtype=np.int64) % 1000).astype(dtype).reshape(h, w)
    return xr.DataArray(
        cupy.asarray(arr),
        dims=('y', 'x'),
        coords={'y': np.arange(h), 'x': np.arange(w)},
    ), arr


def _make_float_da_small(h=32, w=32, dtype=np.float32):
    """Build a deterministic CuPy-backed float DataArray for LERC.

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


def _make_uint8_rgb_da_small(h=32, w=32):
    """Build a smaller CuPy-backed uint8 3-band RGB DataArray for j2k."""
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


@_gpu_only
def test_write_geotiff_gpu_lzw_roundtrip(tmp_path):
    """``compression='lzw'`` round-trips pixel-exact via the CPU fallback."""
    da, arr = _make_int_da_small()
    path = str(tmp_path / "lzw_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='lzw')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_lzw_compression_tag(tmp_path):
    """``compression='lzw'`` emits TIFF Compression tag 5."""
    da, _ = _make_int_da_small()
    path = str(tmp_path / "lzw_tag.tif")

    write_geotiff_gpu(da, path, compression='lzw')

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lzw']


@_gpu_only
def test_write_geotiff_gpu_packbits_roundtrip(tmp_path):
    """``compression='packbits'`` round-trips pixel-exact."""
    da, arr = _make_int_da_small()
    path = str(tmp_path / "packbits_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='packbits')

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_packbits_compression_tag(tmp_path):
    """``compression='packbits'`` emits TIFF Compression tag 32773."""
    da, _ = _make_int_da_small()
    path = str(tmp_path / "packbits_tag.tif")

    write_geotiff_gpu(da, path, compression='packbits')

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['packbits']


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
    da, arr = _make_int_da_small()
    path = str(tmp_path / "lz4_roundtrip.tif")

    write_geotiff_gpu(da, path, compression='lz4', allow_experimental_codecs=True)

    out = open_geotiff(path, allow_experimental_codecs=True)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_lz4_only
def test_write_geotiff_gpu_lz4_compression_tag(tmp_path):
    """``compression='lz4'`` emits TIFF Compression tag 50004."""
    da, _ = _make_int_da_small()
    path = str(tmp_path / "lz4_tag.tif")

    write_geotiff_gpu(da, path, compression='lz4', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lz4']


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
    da, arr = _make_float_da_small(dtype=np.float32)
    path = str(tmp_path / "lerc_float.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    out = open_geotiff(path, allow_experimental_codecs=True)
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
    da, arr = _make_int_da_small(dtype=np.uint16)
    path = str(tmp_path / "lerc_int.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    out = open_geotiff(path, allow_experimental_codecs=True)
    np.testing.assert_array_equal(out.values, arr)
    assert out.dtype == arr.dtype


@_gpu_only
@_lerc_only
def test_write_geotiff_gpu_lerc_compression_tag(tmp_path):
    """``compression='lerc'`` emits TIFF Compression tag 34887."""
    da, _ = _make_float_da_small()
    path = str(tmp_path / "lerc_tag.tif")

    write_geotiff_gpu(da, path, compression='lerc', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['lerc']


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
    da, arr = _make_int_da_small(dtype=np.uint8)
    path = str(tmp_path / "j2k_uint8.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    out = open_geotiff(path, allow_experimental_codecs=True)
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
    da, arr = _make_uint8_rgb_da_small()
    path = str(tmp_path / "j2k_rgb.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    out = open_geotiff(path, allow_experimental_codecs=True)
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
    da, arr = _make_int_da_small(dtype=np.uint8)
    j2k_path = str(tmp_path / "alias_j2k.tif")
    jpeg2k_path = str(tmp_path / "alias_jpeg2000.tif")

    write_geotiff_gpu(da, j2k_path, compression='j2k', allow_experimental_codecs=True)
    write_geotiff_gpu(da, jpeg2k_path, compression='jpeg2000', allow_experimental_codecs=True)

    assert _read_compression_tag(j2k_path) == _COMPRESSION_TAGS['j2k']
    assert _read_compression_tag(jpeg2k_path) == _COMPRESSION_TAGS['jpeg2000']
    np.testing.assert_array_equal(
        open_geotiff(j2k_path, allow_experimental_codecs=True).values, arr,
    )
    np.testing.assert_array_equal(
        open_geotiff(jpeg2k_path, allow_experimental_codecs=True).values, arr,
    )


@_gpu_only
@_j2k_only
def test_write_geotiff_gpu_jpeg2000_compression_tag(tmp_path):
    """``compression='jpeg2000'`` emits TIFF Compression tag 34712."""
    da, _ = _make_int_da_small(dtype=np.uint8)
    path = str(tmp_path / "j2k_tag.tif")

    write_geotiff_gpu(da, path, compression='jpeg2000', allow_experimental_codecs=True)

    assert _read_compression_tag(path) == _COMPRESSION_TAGS['jpeg2000']


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
    da, arr = _make_int_da_small()
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
    da, arr = _make_int_da_small()
    path = str(tmp_path / f"dispatch_{compression}.tif")

    to_geotiff(da, path, compression=compression, gpu=True)

    out = open_geotiff(path)
    np.testing.assert_array_equal(out.values, arr)
    assert _read_compression_tag(path) == _COMPRESSION_TAGS[compression]


# ===========================================================================
# Section 5: NaN-to-sentinel rewrite (was test_gpu_writer_nan_sentinel_1599.py)
# ===========================================================================


def _float_raster_with_nans(dtype=np.float32):
    h, w = 50, 60
    arr = np.arange(h * w, dtype=dtype).reshape(h, w)
    # Two disconnected NaN patches so single-pixel guards don't accidentally
    # mask the bug.
    arr[10:15, 20:25] = np.nan
    arr[40, 50] = np.nan
    y = np.arange(h, dtype=np.float64) * -0.1 + 50.0
    x = np.arange(w, dtype=np.float64) * 0.1 - 100.0
    return arr, y, x


@_gpu_only
def test_gpu_writer_substitutes_nan_with_sentinel(tmp_path):
    """GPU-written file stores the sentinel where the input held NaN."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_nan_sentinel.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    # The raw decoded bytes should contain the sentinel, not NaN.
    assert not np.isnan(raw).any(), \
        "GPU writer left NaN in file bytes despite nodata=-9999"
    nan_locations = np.isnan(arr)
    assert (raw[nan_locations] == -9999.0).all()
    # And the valid pixels round-trip untouched.
    assert np.array_equal(raw[~nan_locations], arr[~nan_locations])


@_gpu_only
def test_gpu_and_cpu_writers_byte_equivalent_on_nan_input(tmp_path):
    """For a float NaN input + finite sentinel, GPU and CPU writers
    must produce identical pixel data."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da_cpu = xr.DataArray(arr.copy(), dims=['y', 'x'],
                          coords={'y': y, 'x': x})
    da_gpu = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                          coords={'y': y, 'x': x})

    p_cpu = str(tmp_path / "cpu.tif")
    p_gpu = str(tmp_path / "gpu.tif")
    to_geotiff(da_cpu, p_cpu, crs=4326, nodata=-9999)
    write_geotiff_gpu(da_gpu, p_gpu, crs=4326, nodata=-9999)

    raw_cpu, _ = read_to_array(p_cpu)
    raw_gpu, _ = read_to_array(p_gpu)
    assert np.array_equal(raw_cpu, raw_gpu)


@_gpu_only
def test_gpu_writer_preserves_caller_cupy_buffer(tmp_path):
    """The NaN-to-sentinel rewrite must NOT mutate the caller's CuPy
    array. The CPU writer takes a defensive copy at the matching step;
    the GPU writer mirrors that behaviour."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    cp_arr = cp.asarray(arr.copy())
    da = xr.DataArray(cp_arr, dims=['y', 'x'], coords={'y': y, 'x': x})

    p = str(tmp_path / "gpu_preserve.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    after = cp.asnumpy(cp_arr)
    # NaNs must still be present at the original locations.
    assert np.isnan(after[10:15, 20:25]).all()
    assert np.isnan(after[40, 50])
    # And non-NaN cells must be unchanged.
    np.testing.assert_array_equal(
        after[~np.isnan(arr)], arr[~np.isnan(arr)])


@_gpu_only
def test_gpu_writer_no_rewrite_when_no_nans(tmp_path):
    """A NaN-free input must round-trip bit-exact with the GPU writer
    regardless of the nodata kwarg."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    # Replace NaNs with finite values so the array has no NaN at all.
    arr = np.where(np.isnan(arr), 1.5, arr).astype(np.float32)
    assert not np.isnan(arr).any()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_no_nans.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    assert np.array_equal(raw, arr)


@_gpu_only
def test_gpu_writer_nan_nodata_skips_substitution(tmp_path):
    """If the requested sentinel is NaN itself, no rewrite is needed.
    The file keeps the NaN bytes verbatim (CPU and GPU agree)."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_nan_sentinel_nan.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=float('nan'))

    raw, _ = read_to_array(p)
    # NaN pixels remain NaN; finite pixels remain finite.
    nan_locations = np.isnan(arr)
    assert np.isnan(raw[nan_locations]).all()
    np.testing.assert_array_equal(raw[~nan_locations], arr[~nan_locations])


@_gpu_only
def test_gpu_writer_external_reader_sees_correct_nodata_mask(tmp_path):
    """rasterio (and any other GDAL_NODATA-strict reader) must see the
    same valid-pixel set on CPU and GPU outputs. The GPU file used to
    report 100% valid pixels because the sentinel was never written
    into NaN positions."""
    rasterio = pytest.importorskip("rasterio")
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    n_nans = int(np.isnan(arr).sum())
    da_cpu = xr.DataArray(arr.copy(), dims=['y', 'x'],
                          coords={'y': y, 'x': x})
    da_gpu = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                          coords={'y': y, 'x': x})

    p_cpu = str(tmp_path / "cpu.tif")
    p_gpu = str(tmp_path / "gpu.tif")
    # Use deflate rather than the default zstd: some rasterio/GDAL builds
    # ship without ZSTD codec support and would fail this round-trip
    # for environment reasons unrelated to the nodata mask under test.
    to_geotiff(da_cpu, p_cpu, crs=4326, nodata=-9999, compression='deflate')
    write_geotiff_gpu(
        da_gpu, p_gpu, crs=4326, nodata=-9999, compression='deflate')

    with rasterio.open(p_cpu) as src:
        cpu_masked = src.read(1, masked=True)
    with rasterio.open(p_gpu) as src:
        gpu_masked = src.read(1, masked=True)

    expected_invalid = n_nans
    assert cpu_masked.size - cpu_masked.count() == expected_invalid
    assert gpu_masked.size - gpu_masked.count() == expected_invalid


@_gpu_only
def test_gpu_writer_multiband_nan_substitution(tmp_path):
    """The substitution must work for 3D (y, x, band) inputs as well."""
    import cupy as cp

    h, w, b = 30, 40, 3
    arr = np.arange(h * w * b, dtype=np.float32).reshape(h, w, b)
    arr[5:10, 8:12, 1] = np.nan
    arr[20, 30, 0] = np.nan
    y = np.arange(h, dtype=np.float64) * -0.1 + 50.0
    x = np.arange(w, dtype=np.float64) * 0.1 - 100.0
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x', 'band'],
                      coords={'y': y, 'x': x, 'band': np.arange(b)})

    p = str(tmp_path / "gpu_mb.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    nan_locations = np.isnan(arr)
    assert not np.isnan(raw).any()
    assert (raw[nan_locations] == -9999.0).all()
    np.testing.assert_array_equal(
        raw[~nan_locations], arr[~nan_locations])


# ===========================================================================
# Section 6: COG overview in-place putmask
# (was test_gpu_writer_overview_inplace_1948.py)
# ===========================================================================


def _make_float_raster_with_nodata_1948(height: int, width: int, sentinel: float):
    """Build a float32 raster whose sentinel pixels would poison overviews.

    Reserve a contiguous all-sentinel sub-rectangle so that every 2x
    decimation step still contains at least one fully-sentinel 2x2
    block. ``_block_reduce_2d_gpu`` masks the sentinel back to NaN
    before reducing, so a partially-sentinel block returns a valid mean
    and the overview pixel never becomes NaN; only a fully-sentinel
    block triggers the NaN->sentinel rewrite branch the fix touches.
    """
    import cupy

    arr = np.random.RandomState(1948).rand(height, width).astype(np.float32)
    # Bottom-right quadrant becomes all-sentinel so every 2x reduction
    # of that region stays all-sentinel and the NaN rewrite branch
    # fires at every overview level.
    arr[height // 2:, width // 2:] = sentinel
    return cupy.asarray(arr)


def test_gpu_writer_overview_loop_uses_putmask_1948():
    """Source-level guard against the redundant ``current.copy()``.

    The fix replaces the ``current = current.copy(); current[nan_mask] = ...``
    pair with ``cupy.putmask(current, nan_mask, ...)`` so the buffer
    ``make_overview_gpu`` returned is mutated in place. This guard
    fires if anyone reintroduces the redundant copy.

    Path note: this file lives under ``tests/gpu/`` (one level deeper
    than the original ``tests/`` home), so the walk back to
    ``_writers/gpu.py`` is ``parent.parent.parent``, not ``parent.parent``.
    """
    src_path = pathlib.Path(__file__).parent.parent.parent / "_writers" / "gpu.py"
    src = src_path.read_text()
    # The overview loop should call cupy.putmask, not current.copy() + indexed write.
    assert "cupy.putmask(" in src, (
        "write_geotiff_gpu overview loop should use cupy.putmask "
        "for the in-place NaN->sentinel rewrite (issue #1948)."
    )
    # Confirm the in-place pattern is the one inside the overview branch,
    # not somewhere unrelated. The pattern's location follows the
    # ``make_overview_gpu`` call and the ``nan_mask = cupy.isnan(current)``
    # gate; assert the order is preserved.
    idx_overview = src.find("make_overview_gpu(current")
    idx_putmask = src.find("cupy.putmask(", idx_overview)
    assert idx_overview != -1 and idx_putmask != -1, (
        "could not locate the overview-loop in-place rewrite"
    )
    # The legacy two-line pattern would have ``current = current.copy()``
    # right before the indexed write. Ensure the overview branch no
    # longer contains that exact line. Anchor the slice on the next
    # statement after the inner ``while`` (``parts.append(...)``) so
    # the window tracks the real loop body instead of a fixed
    # character count that drifts as surrounding code changes.
    idx_parts_append = src.find("parts.append(", idx_overview)
    assert idx_parts_append != -1, (
        "could not locate the ``parts.append(`` sentinel that closes "
        "the overview-loop body"
    )
    overview_branch = src[idx_overview:idx_parts_append]
    assert "current = current.copy()" not in overview_branch, (
        "overview loop should no longer copy the cupy buffer before "
        "the in-place sentinel rewrite (issue #1948)."
    )


@_gpu_only
def test_gpu_writer_cog_overview_sentinel_roundtrip_1948():
    """COG write -> CPU read preserves the sentinel through every overview.

    Regression check on the correctness of the in-place rewrite. If
    ``cupy.putmask`` ever drifts away from the ``current.copy()`` +
    indexed-write semantics (e.g. broadcasts the sentinel as the wrong
    dtype) the round-trip would surface NaN where the sentinel should
    be, breaking external-reader masking. The test is gated on a GPU
    because the overview loop only runs on the GPU writer code path.
    """
    sentinel = np.float32(-9999.0)
    height, width = 1024, 1024
    arr_gpu = _make_float_raster_with_nodata_1948(height, width, float(sentinel))

    da = xr.DataArray(
        arr_gpu, dims=["y", "x"],
        coords={"y": np.arange(height), "x": np.arange(width)},
        attrs={"crs": 4326, "nodata": float(sentinel)},
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_1948_cog.tif")
        write_geotiff_gpu(
            da, path, compression="deflate", tile_size=256,
            cog=True, overview_levels=[2, 4],
        )

        # Read full-resolution and the two overview levels.
        full = open_geotiff(path)
        ov1 = open_geotiff(path, overview_level=1)
        ov2 = open_geotiff(path, overview_level=2)

    # Full-resolution: sentinel pixels survive as NaN (the read path
    # masks the sentinel back to NaN since attrs['nodata'] is set).
    assert np.isnan(full.values).any(), (
        "full-resolution overview lost its sentinel pixels"
    )
    assert ov1.shape == (height // 2, width // 2), (
        f"overview level 1 shape {ov1.shape}, expected "
        f"({height // 2}, {width // 2})"
    )
    assert ov2.shape == (height // 4, width // 4), (
        f"overview level 2 shape {ov2.shape}, expected "
        f"({height // 4}, {width // 4})"
    )
    # Each overview must still mask the sentinel. With the prior
    # ``current.copy()`` pattern the test passed because the buffer was
    # copied; with ``cupy.putmask`` the test still passes because the
    # buffer is mutated in place. This guard fires only if the rewrite
    # is dropped entirely or sentinel dtype/promotion semantics change.
    assert np.isnan(ov1.values).any(), (
        "overview level 1 lost its sentinel pixels after the in-place rewrite"
    )
    assert np.isnan(ov2.values).any(), (
        "overview level 2 lost its sentinel pixels after the in-place rewrite"
    )


@_gpu_only
def test_gpu_writer_overview_uses_make_overview_gpu_fresh_buffer_1948():
    """``make_overview_gpu`` returns a freshly allocated cupy buffer.

    The in-place rewrite contract relies on ``make_overview_gpu``
    handing back a buffer that no caller-visible state aliases. This
    guard exercises every overview-method path inside
    ``_block_reduce_2d_gpu`` and confirms the returned buffer is a
    different cupy.ndarray than the input.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import make_overview_gpu

    rng = np.random.RandomState(1948)
    src_np = rng.rand(64, 64).astype(np.float32)
    src_gpu = cupy.asarray(src_np)

    for method in ("mean", "nearest", "min", "max", "median"):
        out = make_overview_gpu(src_gpu, method=method, nodata=None)
        assert int(out.data.ptr) != int(src_gpu.data.ptr), (
            f"make_overview_gpu(method={method!r}) returned an aliased buffer"
        )
        assert out.shape == (32, 32), (
            f"make_overview_gpu(method={method!r}) returned shape {out.shape}"
        )


# ===========================================================================
# Section 7: overview_resampling='mode' + compression_level kwarg
# (was test_gpu_writer_overview_mode_and_compression_level_1740.py)
# ===========================================================================


def _mode_4x4_uint8() -> np.ndarray:
    """4x4 uint8 raster with a deterministic mode per 2x2 block.

    Block layout:

        [ 1  1 | 5  5 ]
        [ 1  9 | 5  5 ]
        --------------
        [ 2  2 | 3  3 ]
        [ 2  7 | 3  8 ]

    Mode of each 2x2 block:
        [[ 1, 5 ],
         [ 2, 3 ]]
    """
    return np.array(
        [
            [1, 1, 5, 5],
            [1, 9, 5, 5],
            [2, 2, 3, 3],
            [2, 7, 3, 8],
        ],
        dtype=np.uint8,
    )


_MODE_4x4_EXPECTED = np.array([[1, 5], [2, 3]], dtype=np.uint8)


def _mode_8x8_uint8() -> np.ndarray:
    """8x8 uint8 raster with deterministic mode per 2x2 level-1 block."""
    rng = np.random.default_rng(seed=1740)
    # Use a small categorical range so ties are common; the GPU mode
    # branch falls back to the CPU implementation, so the result must
    # match _block_reduce_2d bit-for-bit.
    return rng.integers(0, 4, size=(8, 8), dtype=np.uint8)


@_gpu_only
def test_block_reduce_2d_gpu_mode_matches_cpu_4x4():
    """``_block_reduce_2d_gpu(method='mode')`` falls back to the CPU
    helper and must produce the same output as the CPU reducer on the
    same input."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _mode_4x4_uint8()
    arr_gpu = cupy.asarray(arr)
    out_gpu = _block_reduce_2d_gpu(arr_gpu, 'mode')
    np.testing.assert_array_equal(cupy.asnumpy(out_gpu), _MODE_4x4_EXPECTED)


@_gpu_only
def test_block_reduce_2d_gpu_mode_matches_cpu_random_8x8():
    """Bit-for-bit parity with the CPU helper across a random 8x8 input
    with many ties (small categorical range)."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _mode_8x8_uint8()
    cpu_out = _block_reduce_2d(arr, 'mode')
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr), 'mode')
    np.testing.assert_array_equal(cupy.asnumpy(gpu_out), cpu_out)
    # Output dtype is preserved.
    assert gpu_out.dtype == cpu_out.dtype == arr.dtype


@_gpu_only
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
def test_block_reduce_2d_gpu_mode_dtype_preserved(dtype):
    """Mode is dtype-preserving; the CPU fallback must not promote to
    float on the GPU branch."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    info = np.iinfo(dtype)
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(max(info.min, 0), min(info.max, 7) + 1,
                       size=(8, 8), dtype=dtype)
    out = _block_reduce_2d_gpu(cupy.asarray(arr), 'mode')
    assert out.dtype == arr.dtype


@_gpu_only
def test_write_geotiff_gpu_cog_overview_resampling_mode(tmp_path):
    """``write_geotiff_gpu(cog=True, overview_resampling='mode')`` writes
    a COG whose level-1 overview matches the closed-form 2x2 mode
    reduction.

    Exercises the full GPU make-overview path including the dispatch on
    ``method == 'mode'`` in ``_block_reduce_2d_gpu``.
    """
    import cupy

    arr = _mode_4x4_uint8()
    arr_gpu = cupy.asarray(arr)
    da = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(4.0, 0, -1), 'x': np.arange(4.0)},
    )
    p = str(tmp_path / 'cog_mode_gpu_1740.tif')
    write_geotiff_gpu(
        da, p, cog=True, compression='deflate', tiled=True,
        tile_size=16, overview_levels=[2],
        overview_resampling='mode',
    )

    # Full-res round-trip remains exact.
    full = open_geotiff(p)
    np.testing.assert_array_equal(np.asarray(full.data), arr)

    # Level-1 overview matches the closed-form 2x2 mode reduction.
    ov = open_geotiff(p, overview_level=1)
    assert ov.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(ov.data), _MODE_4x4_EXPECTED)


@_gpu_only
def test_to_geotiff_gpu_cog_overview_resampling_mode(tmp_path):
    """``to_geotiff(gpu=True, cog=True, overview_resampling='mode')``
    threads through to the GPU writer and produces the same overview as
    the explicit ``write_geotiff_gpu`` call."""
    import cupy

    arr = _mode_4x4_uint8()
    da = xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={'y': np.arange(4.0, 0, -1), 'x': np.arange(4.0)},
    )
    p = str(tmp_path / 'cog_mode_to_geotiff_gpu_1740.tif')
    to_geotiff(
        da, p, gpu=True, cog=True, compression='deflate', tiled=True,
        tile_size=16, overview_levels=[2],
        overview_resampling='mode',
    )

    ov = open_geotiff(p, overview_level=1)
    assert ov.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(ov.data), _MODE_4x4_EXPECTED)


@_gpu_only
def test_gpu_vs_cpu_mode_overview_pixel_parity(tmp_path):
    """The GPU writer's ``mode`` overview bytes match the CPU writer's
    output pixel-for-pixel.

    The GPU branch routes through the CPU ``_block_reduce_2d`` helper,
    so any drift between the two would either mean the GPU dispatch is
    swapping methods or the routing has changed. Both are regressions
    the read-back tests above might miss on small inputs.
    """
    import cupy

    arr = _mode_8x8_uint8()

    da_cpu = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
    )
    p_cpu = str(tmp_path / 'cog_mode_cpu_1740.tif')
    to_geotiff(
        da_cpu, p_cpu, cog=True, compression='deflate', tiled=True,
        tile_size=16, overview_levels=[2],
        overview_resampling='mode',
    )

    da_gpu = xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
    )
    p_gpu = str(tmp_path / 'cog_mode_gpu_via_to_geotiff_1740.tif')
    to_geotiff(
        da_gpu, p_gpu, gpu=True, cog=True, compression='deflate', tiled=True,
        tile_size=16, overview_levels=[2],
        overview_resampling='mode',
    )

    ov_cpu = np.asarray(open_geotiff(p_cpu, overview_level=1).data)
    ov_gpu = np.asarray(open_geotiff(p_gpu, overview_level=1).data)
    np.testing.assert_array_equal(ov_gpu, ov_cpu)


def _gpu_compression_level_da(arr: np.ndarray):
    """Wrap *arr* in a CuPy-backed DataArray with monotonic y/x coords."""
    import cupy
    return xr.DataArray(
        cupy.asarray(arr), dims=['y', 'x'],
        coords={
            'y': np.arange(arr.shape[0], dtype=np.float64)[::-1],
            'x': np.arange(arr.shape[1], dtype=np.float64),
        },
    )


@_gpu_only
@pytest.mark.parametrize("compression, in_range_level", [
    ('zstd', 1),
    ('zstd', 22),
    ('deflate', 1),
    ('deflate', 9),
])
def test_write_geotiff_gpu_compression_level_in_range_accepted(
        tmp_path, compression, in_range_level):
    """In-range ``compression_level`` values are accepted and the file
    still round-trips.

    The kwarg is documented as ignored, but ``to_geotiff(gpu=True, ...,
    compression_level=N)`` callers expect the level to flow through
    without error when N is a value the CPU writer would also accept.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / f'level_in_{compression}_{in_range_level}_1740.tif')

    # Must not raise.
    write_geotiff_gpu(
        da, p, compression=compression,
        compression_level=in_range_level,
        tile_size=32,
    )

    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
@pytest.mark.parametrize("compression, oor_level", [
    # CPU writer would reject these; the GPU writer must not.
    ('zstd', 999),
    ('zstd', -5),
    ('deflate', 50),
    ('deflate', 0),
])
def test_write_geotiff_gpu_compression_level_out_of_range_accepted(
        tmp_path, compression, oor_level):
    """Out-of-range ``compression_level`` values do not raise on the GPU
    writer (contrast with the CPU writer, which raises ``ValueError``).

    Pins the docstring contract: "Accepted for API compatibility but
    currently ignored -- nvCOMP does not expose level control". Without
    this test, a regression that wired the GPU writer up to the CPU
    range validator would change behaviour against the doc and break
    every ``to_geotiff(gpu=True, compression_level=X)`` caller for the
    in-range case and noisily for the out-of-range one.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / f'level_oor_{compression}_{oor_level}_1740.tif')

    # Must not raise -- the GPU writer ignores the level.
    write_geotiff_gpu(
        da, p, compression=compression,
        compression_level=oor_level,
        tile_size=32,
    )

    # And the file must still round-trip; the ignored level cannot
    # corrupt the output.
    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
def test_to_geotiff_gpu_compression_level_out_of_range_accepted(tmp_path):
    """``to_geotiff(gpu=True, compression_level=X)`` threads the kwarg
    through to ``write_geotiff_gpu`` and must not raise on an out-of-range
    value (the GPU writer ignores it).

    Without this test the dispatcher could be rewritten to validate
    levels before forwarding and silently break the documented contract.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = _gpu_compression_level_da(arr)
    p = str(tmp_path / 'level_oor_to_geotiff_gpu_1740.tif')

    # Must not raise even though 999 is out of zstd's 1-22 range.
    to_geotiff(
        da, p, gpu=True, compression='zstd',
        compression_level=999, tiled=True, tile_size=32,
    )

    out = open_geotiff(p)
    np.testing.assert_allclose(np.asarray(out.data), arr)


@_gpu_only
def test_to_geotiff_cpu_compression_level_out_of_range_raises(tmp_path):
    """Companion check: the CPU writer rejects out-of-range
    ``compression_level``. Pins the asymmetry the GPU tests above rely
    on -- if the CPU writer ever stopped raising, the GPU "ignored"
    contract becomes uninteresting and the GPU tests above would silently
    pass for the wrong reason.
    """
    arr = np.random.RandomState(1740).rand(32, 32).astype(np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={
            'y': np.arange(32.0, 0, -1),
            'x': np.arange(32.0),
        },
    )
    p = str(tmp_path / 'level_oor_to_geotiff_cpu_1740.tif')
    with pytest.raises(ValueError, match=r"compression_level=999"):
        to_geotiff(da, p, compression='zstd', compression_level=999,
                   tiled=True, tile_size=32)


# ===========================================================================
# Section 8: to_geotiff(gpu=True) fallback contract
# (was test_to_geotiff_gpu_fallback_1674.py)
#
# These tests monkeypatch ``write_geotiff_gpu`` so they do not need a
# real GPU; they exercise the CPU dispatcher's exception classification.
# ===========================================================================


@pytest.fixture
def clear_strict_env(monkeypatch):
    """Default mode: ``XRSPATIAL_GEOTIFF_STRICT`` is unset."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """Strict mode: ``XRSPATIAL_GEOTIFF_STRICT=1`` is set."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


@pytest.fixture
def cpu_data():
    """A small 2D numpy-backed DataArray suitable for the CPU writer."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(8, dtype=np.float64),
            'x': np.arange(8, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def _patch_gpu_writer_to_raise(monkeypatch, exc):
    """Replace ``write_geotiff_gpu`` (as resolved by ``to_geotiff``) with a
    stub that raises ``exc``.

    ``to_geotiff`` calls ``write_geotiff_gpu`` directly inside its own
    defining module (``_writers.eager``), so the patch targets the
    module-level name there. Patching ``xrspatial.geotiff``
    would only update the package re-export and would not intercept the
    actual call site.
    """
    from xrspatial.geotiff._writers import eager as g

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(g, 'write_geotiff_gpu', _boom, raising=True)


def test_runtime_error_without_gpu_signal_propagates(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """A bare ``RuntimeError`` from the GPU writer must NOT be swallowed.

    This is the regression target. Before the fix, the bare except caught
    every exception type and silently dropped the user onto the CPU
    pipeline, producing a different file from the one they asked for.
    """
    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("synthetic non-GPU error"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(RuntimeError, match="synthetic non-GPU error"):
        to_geotiff(cpu_data, str(path), gpu=True)


def test_value_error_propagates(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """A ``ValueError`` from inside the GPU writer must not be swallowed."""
    _patch_gpu_writer_to_raise(
        monkeypatch, ValueError("synthetic value error"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(ValueError, match="synthetic value error"):
        to_geotiff(cpu_data, str(path), gpu=True)


def test_import_error_falls_back_with_warning(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """``ImportError`` from the GPU writer triggers a CPU fallback.

    The user asked for ``gpu=True`` on a system without cupy. A
    ``GeoTIFFFallbackWarning`` makes the substitution visible and the
    text names the explicit request so users know which knob to tune.
    """
    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    path = tmp_path / "fallback.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    msg = str(fallback_warnings[0].message)
    # Explicit gpu=True wording: blame the request, not the data.
    assert 'to_geotiff(gpu=True)' in msg
    assert 'Data is on the GPU' not in msg
    assert 'ImportError' in msg
    assert 'no cupy' in msg


def test_import_error_strict_mode_reraises(
        tmp_path, cpu_data, set_strict_env, monkeypatch):
    """``XRSPATIAL_GEOTIFF_STRICT=1`` promotes the ``ImportError`` fallback
    to a re-raise so CI catches the case where the GPU path silently
    degrades to CPU compression."""
    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no nvCOMP"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(ImportError, match="no nvCOMP"):
        to_geotiff(cpu_data, str(path), gpu=True)


@pytest.mark.parametrize('msg', [
    "CUDA not available",
    "no device found",
    "nvCOMP library not loadable",
    "cuInit failed: no driver",
    "no GPU on this host",
])
def test_runtime_error_with_gpu_signal_falls_back(
        tmp_path, cpu_data, clear_strict_env, monkeypatch, msg):
    """A ``RuntimeError`` whose text names a GPU-availability problem is
    treated like ``ImportError``: fall back to CPU with a warning.

    Pattern matches keep the catch narrow without requiring a custom
    ``nvCompUnavailableError`` class. Anything that does not name CUDA /
    nvCOMP / no device / no GPU / cuInit is treated as a real bug and
    propagated by ``test_runtime_error_without_gpu_signal_propagates``.
    """
    _patch_gpu_writer_to_raise(monkeypatch, RuntimeError(msg))

    path = tmp_path / "fallback.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'RuntimeError' in text
    # Explicit gpu=True branch shares the same template as ImportError;
    # the auto-detected wording must never appear here.
    assert 'to_geotiff(gpu=True)' in text
    assert 'Data is on the GPU' not in text


def test_runtime_error_with_gpu_signal_strict_reraises(
        tmp_path, cpu_data, set_strict_env, monkeypatch):
    """Strict mode re-raises GPU-availability ``RuntimeError`` too."""
    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("CUDA not available"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(RuntimeError, match="CUDA not available"):
        to_geotiff(cpu_data, str(path), gpu=True)


def _make_synthetic_gpu_data():
    """Return a numpy-backed DataArray that ``_is_gpu_data`` will be
    patched to treat as GPU-resident."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(8, dtype=np.float64),
            'x': np.arange(8, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def test_auto_detected_gpu_fallback_warns(
        tmp_path, clear_strict_env, monkeypatch):
    """When ``gpu`` is auto-detected from CuPy-backed data, the same
    fallback rules apply: ``ImportError`` triggers the warning.

    Users whose data was CuPy-backed deserve a warning every time the
    GPU writer failed so they know their array was copied to host
    before the CPU writer wrote it. The warning text must blame the
    auto-detect path, not an ``gpu=True`` argument the caller never
    passed.
    """
    # Synthesise a "CuPy-looking" DataArray via _is_gpu_data's hook.
    # Easiest: patch _is_gpu_data to True in the writer module that
    # actually calls it (the to_geotiff body lives in _writers.eager).
    # The CPU fallback then operates on the numpy buffer underneath.
    from xrspatial.geotiff._writers import eager as g
    monkeypatch.setattr(g, '_is_gpu_data', lambda data: True, raising=True)

    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    da = _make_synthetic_gpu_data()

    path = tmp_path / "auto.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        # gpu defaults to None -> auto-detect path
        to_geotiff(da, str(path))

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    # Auto-detected branch wording: blame the data, not gpu=True.
    assert 'Data is on the GPU' in text
    assert 'to_geotiff(gpu=True)' not in text
    assert 'ImportError' in text
    assert 'no cupy' in text


def test_auto_detected_gpu_runtime_error_falls_back_with_warning(
        tmp_path, clear_strict_env, monkeypatch):
    """Same shape for the ``RuntimeError`` branch under auto-detect.

    Both fallback branches (ImportError, RuntimeError-with-GPU-signal)
    must use the same template so call sites do not diverge over time.
    """
    from xrspatial.geotiff._writers import eager as g

    monkeypatch.setattr(g, '_is_gpu_data', lambda data: True, raising=True)
    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("CUDA not available"))

    da = _make_synthetic_gpu_data()

    path = tmp_path / "auto_rt.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(da, str(path))

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'Data is on the GPU' in text
    assert 'to_geotiff(gpu=True)' not in text
    assert 'RuntimeError' in text
    assert 'CUDA not available' in text


def test_explicit_gpu_false_then_true_uses_explicit_template(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """``gpu=True`` plus non-CuPy data must use the explicit template
    even when ``_is_gpu_data`` would return False on its own.

    This pins down that the template is selected from ``gpu is None``,
    not from the resolved ``use_gpu`` value -- so passing ``gpu=True``
    on numpy data still attributes the fallback to the explicit flag.
    """
    from xrspatial import geotiff as g

    # Even if auto-detect would say "not GPU", the explicit request
    # should drive the wording.
    monkeypatch.setattr(g, '_is_gpu_data', lambda data: False, raising=True)
    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    path = tmp_path / "explicit.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'to_geotiff(gpu=True)' in text
    assert 'Data is on the GPU' not in text
