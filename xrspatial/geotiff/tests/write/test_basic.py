"""Generic writer paths.

Covers the eager ``to_geotiff`` / ``write_geotiff_gpu`` / ``write_vrt``
surface: round-trip basics, dtype x compression matrix, kwarg order
and return-path contracts, the uncompressed-tiled no-dead-alloc gate,
the writer layout monkeypatch contract, and the VRT writer surface
(path kwarg, CRS, bool / int nodata, int64, photometric, source
compatibility, tiled output).

Section banners below mark the topical sub-areas. Tests-only restructure
for epic #2390.
"""

from __future__ import annotations

import numpy as np
import pytest
import os
import platform
import xarray as xr
import inspect
import importlib.util
import io
import tracemalloc
import uuid
import warnings
import typing
import re
import glob

from xrspatial.geotiff import (
    _vrt as _vrt_module,
    _writer as writer_mod,
    open_geotiff,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)
from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._header import TAG_PHOTOMETRIC, parse_header, parse_ifd
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import _make_overview, _write_tiled, write
# ``write_vrt`` here is the private internal binding, aliased so it does
# not shadow the public re-export above. The only section that needs
# the private form is the writer-source-compat fold (see PR
# description for the why).
from xrspatial.geotiff._vrt import write_vrt as _priv_write_vrt
from xrspatial.geotiff.tests.conftest import requires_gpu


# -------------------------------------------------------------------------
# Section: writer round-trip basics
# -------------------------------------------------------------------------

class TestMakeOverview:
    def test_2x_decimation(self):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        ov = _make_overview(arr)
        assert ov.shape == (4, 4)
        # Check first value: mean of top-left 2x2 block
        expected = np.mean([0, 1, 8, 9])
        assert ov[0, 0] == pytest.approx(expected)

    def test_integer_rounding(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8]], dtype=np.uint8)
        ov = _make_overview(arr)
        assert ov.shape == (1, 2)
        assert ov.dtype == np.uint8


class TestWriteRoundTrip:
    def test_uncompressed_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'uncompressed.tif')
        write(expected, path, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_deflate_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'deflate.tif')
        write(expected, path, compression='deflate', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_uncompressed_tiled(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'tiled.tif')
        write(expected, path, compression='none', tiled=True, tile_size=4)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_deflate_tiled(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'deflate_tiled.tif')
        write(expected, path, compression='deflate', tiled=True, tile_size=4)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_lzw_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'lzw.tif')
        write(expected, path, compression='lzw', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_uint16(self, tmp_path):
        expected = np.arange(100, dtype=np.uint16).reshape(10, 10)
        path = str(tmp_path / 'uint16.tif')
        write(expected, path, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_with_geo_info(self, tmp_path):
        expected = np.ones((4, 4), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'geo.tif')
        write(expected, path, geo_transform=gt, crs_epsg=4326,
              nodata=-9999.0, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)
        assert geo.crs_epsg == 4326
        assert geo.transform.origin_x == pytest.approx(-120.0)
        assert geo.transform.pixel_width == pytest.approx(0.001)

    def test_predictor_deflate(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'predictor.tif')
        write(expected, path, compression='deflate', tiled=False, predictor=True)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)


class TestWriteInvalidInput:
    def test_unsupported_compression(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        # Issue #2138 pushed the canonical compression-list check from
        # ``to_geotiff`` down into ``_write`` so direct callers get the
        # same actionable error as the public wrapper. The wording
        # shifted from ``_compression_tag``'s "Unsupported compression"
        # to the wrapper's "Unknown compression" + canonical list.
        with pytest.raises(ValueError, match="(Unsupported|Unknown) compression"):
            write(arr, str(tmp_path / 'bad.tif'), compression='bzip2')


# -------------------------------------------------------------------------
# Section: writer dtype x compression matrix
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# T-5: dtype x compression matrix
# ---------------------------------------------------------------------------

DTYPES_T5 = [
    np.uint8, np.uint16, np.uint32,
    np.int16, np.int32, np.int64,
    np.float32, np.float64,
]
CODECS_T5 = ['none', 'deflate', 'lzw', 'zstd', 'lz4']


def _make_dtype_arr(dtype, h=32, w=32):
    """Make a small array with values that fit the dtype's positive range."""
    n = h * w
    dt = np.dtype(dtype)
    if dt.kind == 'f':
        # Non-trivial floats; include a few extreme-ish values.
        arr = np.linspace(-1e3, 1e3, n).astype(dt).reshape(h, w)
    elif dt.kind == 'u':
        # Stay below uint16 max so it fits any unsigned dtype here.
        arr = (np.arange(n) % 1000).astype(dt).reshape(h, w)
    else:  # signed int
        arr = ((np.arange(n) % 2000) - 1000).astype(dt).reshape(h, w)
    return arr


def _codec_supports(codec, dtype):
    """Return False for combos the writer rejects, True otherwise."""
    # JPEG is not in the parametrized codec list (only uint8/3-band).
    # All listed codecs accept any of the listed dtypes.
    return True


@pytest.mark.parametrize('codec', CODECS_T5)
@pytest.mark.parametrize('dtype', DTYPES_T5)
def test_dtype_codec_roundtrip_stripped(tmp_path, dtype, codec):
    """Round-trip every dtype x codec in stripped layout."""
    if not _codec_supports(codec, dtype):
        pytest.skip(f"{codec} does not support {np.dtype(dtype).name}")

    expected = _make_dtype_arr(dtype)
    path = str(tmp_path / f'1483_t5_strip_{np.dtype(dtype).name}_{codec}.tif')

    try:
        write(expected, path, compression=codec, tiled=False)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"codec {codec} not available: {e}")

    # Codecs in the experimental tier (LERC / J2K / LZ4) need the
    # read-side opt-in too (PR 4 of epic #2340). Tier 1 codecs ignore
    # the kwarg, so passing it unconditionally keeps the loop simple.
    arr, _geo = read_to_array(path, allow_experimental_codecs=True)
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == expected.dtype


@pytest.mark.parametrize('codec', CODECS_T5)
@pytest.mark.parametrize('dtype', DTYPES_T5)
def test_dtype_codec_roundtrip_tiled(tmp_path, dtype, codec):
    """Round-trip every dtype x codec in tiled layout."""
    if not _codec_supports(codec, dtype):
        pytest.skip(f"{codec} does not support {np.dtype(dtype).name}")

    expected = _make_dtype_arr(dtype)
    path = str(tmp_path / f'1483_t5_tile_{np.dtype(dtype).name}_{codec}.tif')

    try:
        write(expected, path, compression=codec, tiled=True, tile_size=16)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"codec {codec} not available: {e}")

    arr, _geo = read_to_array(path, allow_experimental_codecs=True)
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == expected.dtype


# ---------------------------------------------------------------------------
# T-6: NaN vs sentinel nodata
# ---------------------------------------------------------------------------

def _float_with_nan(h=8, w=8, dtype=np.float32):
    arr = np.linspace(0.0, 100.0, h * w, dtype=dtype).reshape(h, w)
    arr[0, 0] = np.nan
    arr[3, 5] = np.nan
    arr[-1, -1] = np.nan
    return arr


def test_nodata_nan_float_roundtrip(tmp_path):
    """nodata=NaN: NaN positions in the input round-trip as NaN."""
    expected = _float_with_nan(dtype=np.float32)
    path = str(tmp_path / '1483_t6_nodata_nan.tif')

    da = xr.DataArray(expected, dims=('y', 'x'))
    to_geotiff(da, path, nodata=float('nan'), compression='deflate')

    out = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(out.data), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(out.data[finite], expected[finite])


def test_nodata_sentinel_float_disk_vs_read(tmp_path):
    """nodata=-9999: NaN positions become sentinel on disk, NaN on read-back."""
    expected = _float_with_nan(dtype=np.float32)
    path = str(tmp_path / '1483_t6_nodata_sentinel.tif')

    da = xr.DataArray(expected, dims=('y', 'x'))
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    # On-disk values: NaN positions hold the sentinel.
    raw, _geo = read_to_array(path)
    nan_mask = np.isnan(expected)
    assert np.all(raw[nan_mask] == np.float32(-9999.0))
    # Non-NaN positions match.
    np.testing.assert_array_equal(raw[~nan_mask], expected[~nan_mask])

    # Read back through open_geotiff: sentinel becomes NaN again.
    out = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(out.data), nan_mask)
    np.testing.assert_array_equal(out.data[~nan_mask], expected[~nan_mask])
    assert out.attrs.get('nodata') == -9999.0


def test_nodata_uint8_sentinel(tmp_path):
    """nodata=255 for uint8: sentinel on disk, NaN on read (array promoted to float)."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8).copy()
    arr[0, 0] = 255
    arr[4, 4] = 255
    path = str(tmp_path / '1483_t6_nodata_uint8.tif')

    da = xr.DataArray(arr, dims=('y', 'x'))
    to_geotiff(da, path, nodata=255, compression='deflate')

    # On-disk: still uint8 with 255 in those slots.
    raw, _geo = read_to_array(path)
    assert raw.dtype == np.uint8
    assert raw[0, 0] == 255 and raw[4, 4] == 255
    np.testing.assert_array_equal(raw, arr)

    # Read-back: open_geotiff promotes integer with nodata to float + NaN.
    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    assert np.isnan(out.data[0, 0])
    assert np.isnan(out.data[4, 4])
    finite = ~np.isnan(out.data)
    np.testing.assert_array_equal(out.data[finite].astype(np.uint8),
                                  arr[finite])


# ---------------------------------------------------------------------------
# T-7: COG validity (rasterio-dependent)
# ---------------------------------------------------------------------------


def test_cog_layout_and_overviews(tmp_path):
    """A cog=True file is tiled, carries overviews, and (when rio-cogeo is
    installed) passes the COG validator.

    Note: xrspatial does not currently emit GDAL's IMAGE_STRUCTURE.LAYOUT=COG
    tag, so we don't assert that. Structural COG properties (tiled, overviews
    present, GDAL-readable) are what the writer actually guarantees.
    """
    rasterio = pytest.importorskip(
        'rasterio',
        reason='rasterio is optional; COG validity test skipped when missing',
    )
    h = w = 1024
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w) % 1000.0
    path = str(tmp_path / '1483_t7_cog.tif')

    da = xr.DataArray(arr, dims=('y', 'x'))
    to_geotiff(
        da, path, crs=4326, cog=True, compression='deflate', tile_size=256,
    )

    with rasterio.open(path) as src:
        assert src.is_tiled, "COG output must be tiled"
        # 1024x1024 with 256 tiles produces at least one halving.
        ovs = src.overviews(1)
        assert len(ovs) >= 1, f"expected at least one overview, got {ovs}"
        assert ovs[0] in (2, 4, 8, 16), f"unexpected first overview: {ovs}"
        # Each overview should be strictly larger than the previous (decimation
        # factors are monotonically increasing).
        assert all(b > a for a, b in zip(ovs, ovs[1:])), \
            f"overview decimations not monotonically increasing: {ovs}"
        # Sanity: full-resolution band should round-trip values.
        sample = src.read(1, window=((0, 4), (0, 4)))
        np.testing.assert_array_equal(sample, arr[:4, :4])

    # If rio-cogeo is installed, run its validator for the gold-standard check.
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        return
    valid, errors, _warnings = cog_validate(path, strict=False)
    assert valid, f"rio-cogeo cog_validate failed: errors={errors}"


# ---------------------------------------------------------------------------
# T-9: write-to-readonly directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='POSIX chmod semantics required',
)
@pytest.mark.skipif(
    hasattr(os, 'geteuid') and os.geteuid() == 0,
    reason='root bypasses directory permissions',
)
def test_write_to_readonly_dir_raises_oserror(tmp_path):
    """Writing into a chmod 0o555 directory must raise OSError/PermissionError."""
    ro_dir = tmp_path / '1483_t9_readonly'
    ro_dir.mkdir()
    target = str(ro_dir / 'out.tif')

    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(arr, dims=('y', 'x'))

    original_mode = ro_dir.stat().st_mode
    try:
        os.chmod(ro_dir, 0o555)
        with pytest.raises((OSError, PermissionError)):
            to_geotiff(da, target, compression='deflate')
    finally:
        os.chmod(ro_dir, original_mode)


# -------------------------------------------------------------------------
# Section: kwarg order / signature parity
# -------------------------------------------------------------------------

def test_writer_kwarg_order_matches_to_geotiff():
    """``write_geotiff_gpu`` lists its kwargs in the same order as
    ``to_geotiff``, modulo the ``gpu`` kwarg the GPU writer omits.

    Both signatures use keyword-only kwargs so positional callers are
    unaffected. The order still matters for IDE autocomplete, generated
    docs, and any caller that inspects ``inspect.signature``.
    """
    eager_params = list(inspect.signature(to_geotiff).parameters)
    gpu_params = list(inspect.signature(write_geotiff_gpu).parameters)

    # to_geotiff has ``gpu`` (auto-dispatch flag); write_geotiff_gpu does
    # not. Drop it from the comparison instead of asserting on the
    # missing kwarg directly, so unrelated future additions to either
    # signature still surface here.
    assert 'gpu' in eager_params
    assert 'gpu' not in gpu_params
    eager_params_no_gpu = [p for p in eager_params if p != 'gpu']

    assert gpu_params == eager_params_no_gpu, (
        "write_geotiff_gpu and to_geotiff kwarg order diverged.\n"
        f"  to_geotiff (with 'gpu' removed): {eager_params_no_gpu}\n"
        f"  write_geotiff_gpu:               {gpu_params}\n"
        "Reorder write_geotiff_gpu to match to_geotiff (see #1922)."
    )


def test_writer_kwarg_defaults_match_to_geotiff():
    """The kwargs both writers share also have identical defaults.

    A surprise-free dispatch ``to_geotiff(..., gpu=True)`` requires
    ``write_geotiff_gpu`` to default the same way for every kwarg the
    auto-dispatch entry point forwards (issue #1916 added
    ``allow_internal_only_jpeg`` to satisfy that contract; this test
    pins the broader parity).
    """
    eager_sig = inspect.signature(to_geotiff)
    gpu_sig = inspect.signature(write_geotiff_gpu)

    shared = set(eager_sig.parameters) & set(gpu_sig.parameters)
    # ``data`` and ``path`` are required positionals with no default;
    # comparing inspect.Parameter.empty against itself is fine.
    mismatches = []
    for name in sorted(shared):
        ed = eager_sig.parameters[name].default
        gd = gpu_sig.parameters[name].default
        if ed != gd:
            mismatches.append((name, ed, gd))
    assert not mismatches, (
        "write_geotiff_gpu and to_geotiff disagree on defaults: "
        f"{mismatches}"
    )


# -------------------------------------------------------------------------
# Section: return-path contract
# -------------------------------------------------------------------------

from .._helpers.markers import gpu_available as _gpu_available  # noqa: E402

_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def _small_da() -> xr.DataArray:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4)[::-1].astype(np.float64),
                "x": np.arange(4).astype(np.float64)},
        attrs={"crs": 4326},
    )


def test_to_geotiff_returns_string_path(tmp_path):
    """``to_geotiff`` returns the str path passed in."""
    da = _small_da()
    out = tmp_path / "test_1938_str.tif"
    rv = to_geotiff(da, str(out))
    assert isinstance(rv, str), (
        f"to_geotiff(str) must return a str, got {type(rv).__name__}"
    )
    assert rv == str(out)
    assert os.path.exists(rv)


def test_to_geotiff_returns_file_like(tmp_path):
    """``to_geotiff`` returns the file-like object passed in."""
    da = _small_da()
    buf = io.BytesIO()
    rv = to_geotiff(da, buf)
    assert rv is buf, (
        f"to_geotiff(BytesIO) must return the same file-like, "
        f"got {type(rv).__name__}"
    )
    # The buffer was actually written to.
    assert buf.tell() > 0 or len(buf.getvalue()) > 0


def test_to_geotiff_cog_returns_path(tmp_path):
    """COG path also returns the str path."""
    da = _small_da()
    out = tmp_path / "test_1938_cog.tif"
    rv = to_geotiff(da, str(out), cog=True, tile_size=16)
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_to_geotiff_dask_streaming_returns_path(tmp_path):
    """Dask-streaming write path also returns the str path."""
    import dask.array as da_arr

    arr = da_arr.arange(256, dtype=np.float32, chunks=64).reshape(16, 16)
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(16)[::-1].astype(np.float64),
                "x": np.arange(16).astype(np.float64)},
        attrs={"crs": 4326},
    )
    out = tmp_path / "test_1938_dask.tif"
    rv = to_geotiff(da, str(out))
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_write_vrt_returns_string_path(tmp_path):
    """``write_vrt`` (already conformant) keeps returning the str path."""
    # Create a source tif first.
    src = tmp_path / "src.tif"
    to_geotiff(_small_da(), str(src))
    vrt_path = tmp_path / "out.vrt"
    rv = write_vrt(str(vrt_path), [str(src)])
    assert isinstance(rv, str)
    assert rv == str(vrt_path)
    assert os.path.exists(rv)


@_gpu_only
def test_write_geotiff_gpu_returns_string_path(tmp_path):
    """GPU writer returns the str path (only runs with cupy + CUDA)."""
    import cupy

    arr_cpu = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr_gpu = cupy.asarray(arr_cpu)
    da = xr.DataArray(
        arr_gpu,
        dims=("y", "x"),
        coords={"y": np.arange(4)[::-1].astype(np.float64),
                "x": np.arange(4).astype(np.float64)},
        attrs={"crs": 4326},
    )
    out = tmp_path / "test_1938_gpu.tif"
    rv = write_geotiff_gpu(da, str(out))
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_writer_signatures_declare_path_return():
    """All three writers annotate the same return type.

    The annotation is a string under ``from __future__ import annotations``;
    pin the literal so the three writers cannot drift apart silently.
    """
    expected = {
        to_geotiff: "str | BinaryIO",
        write_geotiff_gpu: "str | BinaryIO",
        write_vrt: "str",
    }
    for fn, expected_ann in expected.items():
        sig = inspect.signature(fn)
        assert sig.return_annotation == expected_ann, (
            f"{fn.__name__} return annotation drifted: expected "
            f"{expected_ann!r}, got {sig.return_annotation!r}"
        )


def test_writer_returns_are_not_none(tmp_path):
    """None of the public writers may go back to returning ``None``."""
    # Use the ``tmp_path`` fixture (not ``tempfile.TemporaryDirectory``)
    # because ``write_vrt`` reads each source through the module-level
    # ``_MmapCache`` in ``_reader.py``, which keeps the file handle and
    # mmap of ``src.tif`` open after ``_FileSource.close()`` so repeated
    # reads of the same file stay cheap. On Windows that cached handle
    # blocks ``os.unlink`` (WinError 32), so a synchronous
    # ``TemporaryDirectory`` teardown raises before the test returns.
    # ``tmp_path`` defers cleanup to pytest's session-end sweep, which
    # tolerates the still-open handle the same way the other tests in
    # this file already do.
    da = _small_da()
    out = str(tmp_path / "out.tif")
    rv = to_geotiff(da, out)
    assert rv is not None
    src = str(tmp_path / "src.tif")
    to_geotiff(da, src)
    vrt_rv = write_vrt(str(tmp_path / "m.vrt"), [src])
    assert vrt_rv is not None


# -------------------------------------------------------------------------
# Section: uncompressed tiled: no dead allocation
# -------------------------------------------------------------------------

# Peak ``tracemalloc`` size, in multiples of the input raster size, that
# the uncompressed branch of ``_write_tiled`` must stay under. The dead
# bytearray drove peak to ~2.07x; the current implementation sits at
# ~1.06-1.12x across the cases below. 1.5x leaves room for unrelated
# refactors while still firmly catching the regression.
_PEAK_RATIO_LIMIT = 1.5


def test_uncompressed_tiled_round_trip_exact(tmp_path):
    rng = np.random.RandomState(20260512)
    h, w = 96, 144
    data = rng.randint(0, 200, size=(h, w)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    p = str(tmp_path / f"tmp_1736_uncomp_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=32, compression='none')
    assert os.path.exists(p)

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)
    assert out.shape == (h, w)


def test_uncompressed_tiled_round_trip_partial_edge_tiles(tmp_path):
    """Tile size that does not divide width/height exercises the
    zero-padded edge-tile branch inside the loop."""
    rng = np.random.RandomState(20260513)
    h, w = 50, 70  # 32 does not divide either; edges pad
    data = rng.randint(0, 60000, size=(h, w)).astype(np.uint16)
    da = xr.DataArray(data, dims=['y', 'x'])

    p = str(tmp_path / f"tmp_1736_edge_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=32, compression='none')

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)


def test_uncompressed_tiled_round_trip_multiband(tmp_path):
    rng = np.random.RandomState(20260514)
    h, w, b = 48, 80, 3
    data = rng.randint(0, 200, size=(h, w, b)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    p = str(tmp_path / f"tmp_1736_multi_{uuid.uuid4().hex[:8]}.tif")
    to_geotiff(da, p, tiled=True, tile_size=16, compression='none')

    out = open_geotiff(p)
    np.testing.assert_array_equal(out.data, data)


def _peak_ratio_for_write_tiled(data: np.ndarray, tile_size: int) -> float:
    """Return ``tracemalloc`` peak / ``data.nbytes`` for one
    ``_write_tiled`` call against the uncompressed branch.

    Allocations made before this call are excluded from peak by the
    ``reset_peak`` step, so the ratio reflects what ``_write_tiled``
    itself adds.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _write_tiled(data, COMPRESSION_NONE, 1, tile_size=tile_size)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / data.nbytes


def test_uncompressed_tiled_peak_memory_single_band():
    """Peak memory for the uncompressed branch should stay below
    ``_PEAK_RATIO_LIMIT * raster_bytes``.  Reintroducing the dead
    ``bytearray(n_tiles * tile_bytes)`` would push the ratio to ~2x
    and fail this check."""
    h, w = 1024, 1024  # 1 MB raw, exact tile divisor -> no edge padding
    data = np.random.RandomState(20260512).randint(
        0, 255, size=(h, w), dtype=np.uint8,
    )
    ratio = _peak_ratio_for_write_tiled(data, tile_size=256)
    assert ratio < _PEAK_RATIO_LIMIT, (
        f"_write_tiled peak memory {ratio:.2f}x raster exceeds the "
        f"{_PEAK_RATIO_LIMIT}x cap; the dead bytearray from #1736 may "
        f"have been reintroduced."
    )


def test_uncompressed_tiled_peak_memory_multiband():
    """3-band variant of the peak-memory check. ``samples == 3``
    triples the would-be dead buffer, so this case is even more
    sensitive to a regression."""
    h, w = 1024, 1024
    data = np.random.RandomState(20260513).randint(
        0, 255, size=(h, w, 3), dtype=np.uint8,
    )
    ratio = _peak_ratio_for_write_tiled(data, tile_size=256)
    assert ratio < _PEAK_RATIO_LIMIT, (
        f"_write_tiled peak memory {ratio:.2f}x raster exceeds the "
        f"{_PEAK_RATIO_LIMIT}x cap; the dead bytearray from #1736 may "
        f"have been reintroduced."
    )


# -------------------------------------------------------------------------
# Section: writer layout monkeypatch contract
# -------------------------------------------------------------------------

def _make_float32(h: int = 8, w: int = 8) -> xr.DataArray:
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "x": np.arange(w, dtype=np.float64),
            "y": np.arange(h, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )


@pytest.mark.parametrize(
    "helper_name",
    [
        "_promote_offsets_to_long8",
        "_assemble_standard_layout",
        "_assemble_cog_layout",
        "_resolve_photometric",
    ],
)
def test_assemble_tiff_resolves_helper_through_writer_module(
    monkeypatch, tmp_path, helper_name,
):
    """``_assemble_tiff`` must look up ``helper_name`` via ``_writer``.

    Replace the helper on the ``_writer`` module with a sentinel that
    records the call and delegates to the real implementation. If
    ``_assemble_tiff`` were to bind the helper at import time (rather
    than resolving it through ``_writer`` on each call), the sentinel
    would never fire and the assertion would fail.
    """
    real = getattr(writer_mod, helper_name)
    calls: list[tuple] = []

    def _wrapped(*args, **kwargs):
        calls.append((args, tuple(sorted(kwargs.items()))))
        return real(*args, **kwargs)

    monkeypatch.setattr(writer_mod, helper_name, _wrapped)

    da = _make_float32(8, 8)
    path = str(tmp_path / f"monkeypatch_{helper_name}_2248.tif")

    # ``_assemble_cog_layout`` only fires when at least one overview
    # is written; ``_promote_offsets_to_long8`` only fires when the
    # writer chooses BigTIFF. Pass the right kwargs per helper so each
    # one is exercised by ``_assemble_tiff`` on this call.
    if helper_name == "_assemble_cog_layout":
        to_geotiff(da, path, cog=True, overview_levels=[2])
    elif helper_name == "_promote_offsets_to_long8":
        to_geotiff(da, path, bigtiff=True)
    else:
        to_geotiff(da, path)

    assert calls, (
        f"_assemble_tiff did not call _writer.{helper_name}; the "
        f"monkeypatch on the _writer namespace was bypassed."
    )


# -------------------------------------------------------------------------
# Section: write_vrt path kwarg contract
# -------------------------------------------------------------------------

def _build_source_tif(tmp_path, name='src.tif'):
    """Create a small GeoTIFF used as the VRT's source file."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    p = str(tmp_path / name)
    to_geotiff(da, p)
    return p


def test_write_vrt_signature_first_arg_is_path():
    """Signature parity with to_geotiff / write_geotiff_gpu.

    The api-consistency sweep cares specifically about
    ``inspect.signature``: IDE autocomplete, mypy, and Sphinx-rendered
    docs all read the same source. Pinning the first param name here
    catches any future re-rename that re-introduces the drift.
    """
    sig = inspect.signature(write_vrt)
    params = list(sig.parameters)
    # ``path`` is the new canonical name, ``source_files`` follows.
    # ``vrt_path`` is kept as a keyword-only deprecated alias.
    assert params[0] == 'path'
    assert params[1] == 'source_files'
    assert 'vrt_path' in params
    # ``vrt_path`` is keyword-only (the alias should never be used
    # positionally going forward).
    assert sig.parameters['vrt_path'].kind == inspect.Parameter.KEYWORD_ONLY


def test_write_vrt_positional_path_works(tmp_path):
    """Positional ``write_vrt(path, sources)`` is unchanged.

    Existing callers ``write_vrt(some_path, sources)`` keep working
    after the rename because the new ``path`` parameter sits where
    ``vrt_path`` used to be. No deprecation warning should fire.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = write_vrt(out, [src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_path_kwarg_works(tmp_path):
    """Keyword ``write_vrt(path=..., source_files=...)`` works.

    A caller who passes everything by keyword (no positional args)
    cannot reach the function before #1946 because ``path`` did not
    exist; this is the path-symmetric counterpart to the existing
    ``write_vrt(vrt_path=...)`` test below.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = write_vrt(path=out, source_files=[src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_vrt_path_kwarg_emits_deprecation_warning(tmp_path):
    """``vrt_path=...`` works but emits ``DeprecationWarning``.

    Mirrors the existing ``crs_wkt`` deprecation in the same writer
    (#1715): old name still works, but caller sees a clear migration
    hint via the warning.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with pytest.warns(DeprecationWarning, match='vrt_path'):
        result = write_vrt(vrt_path=out, source_files=[src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_path_and_vrt_path_together_raises(tmp_path):
    """Both names supplied is ambiguous; refuse to pick one.

    Mirrors the ``crs`` / ``crs_wkt`` rule documented in the existing
    write_vrt source: passing both is rejected with TypeError
    regardless of whether the two values happen to match.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with pytest.raises(TypeError, match="path.*vrt_path"):
        write_vrt(path=out, vrt_path=out, source_files=[src])


def test_write_vrt_no_path_raises(tmp_path):
    """Neither ``path`` nor ``vrt_path`` -> TypeError.

    Before the shim, omitting the first positional argument raised
    ``TypeError: missing 1 required positional argument`` from CPython.
    The shim adds a sentinel default so the kwarg-only positional no
    longer triggers that automatic check; the explicit raise inside
    the shim preserves the pre-#1946 error semantics.
    """
    src = _build_source_tif(tmp_path)
    with pytest.raises(TypeError, match='path'):
        write_vrt(source_files=[src])


def test_write_vrt_explicit_path_none_raises(tmp_path):
    """``write_vrt(path=None, ...)`` is rejected with TypeError.

    The sentinel-default pattern (#1962 review) distinguishes "caller
    passed nothing" (sentinel) from "caller passed None explicitly".
    Explicit ``None`` is a bug in the caller's code, not a request to
    fall through to the deprecated ``vrt_path`` alias, so the shim
    raises with a clear message that names the offending kwarg.
    """
    src = _build_source_tif(tmp_path)
    with pytest.raises(TypeError, match="'path'.*None"):
        write_vrt(path=None, source_files=[src])


def test_write_vrt_positional_none_raises(tmp_path):
    """Positional ``write_vrt(None, sources)`` is rejected with TypeError.

    Same rationale as the keyword case: an explicit positional ``None``
    is rejected up front instead of crashing deep in
    ``os.path.dirname(os.path.abspath(None))``. Pinned because the
    pre-#1962 code accepted positional ``None`` and raised the wrong
    "missing required argument" error.
    """
    src = _build_source_tif(tmp_path)
    with pytest.raises(TypeError, match="'path'.*None"):
        write_vrt(None, [src])


def test_write_vrt_first_arg_name_matches_writer_trio():
    """Cross-sibling consistency: all three writers use the same
    destination kwarg name.

    The deep-sweep-api-consistency sweep keeps adding to the writer
    trio's parity contract. Pin the rule here so future re-renames
    that split the trio again will trip a test.
    """
    eager_first = list(
        inspect.signature(to_geotiff).parameters
    )[1]  # data, path -> index 1
    gpu_first = list(
        inspect.signature(write_geotiff_gpu).parameters
    )[1]
    vrt_first = list(
        inspect.signature(write_vrt).parameters
    )[0]  # path, source_files -> index 0
    assert eager_first == 'path'
    assert gpu_first == 'path'
    assert vrt_first == 'path'


def test_write_vrt_path_round_trip_matches_old(tmp_path):
    """The written VRT decodes the same regardless of which kwarg name
    the caller used.

    Smoke test that the shim does not silently drop or re-route any of
    the other kwargs while resolving ``path`` vs ``vrt_path``.
    """
    src = _build_source_tif(tmp_path)
    out_new = str(tmp_path / 'out_new.vrt')
    out_old = str(tmp_path / 'out_old.vrt')

    write_vrt(out_new, [src])
    with warnings.catch_warnings():
        # ignore the deprecation; we still need the legacy path to
        # produce a byte-identical mosaic.
        warnings.simplefilter('ignore', DeprecationWarning)
        write_vrt(vrt_path=out_old, source_files=[src])

    a_new = read_vrt(out_new)
    a_old = read_vrt(out_old)
    np.testing.assert_array_equal(np.asarray(a_new), np.asarray(a_old))


# -------------------------------------------------------------------------
# Section: write_vrt CRS propagation
# -------------------------------------------------------------------------


# --- Signature pins ---


def test_write_vrt_accepts_crs_kwarg():
    """``crs`` is in the signature and defaults to ``None``."""
    import inspect

    sig = inspect.signature(write_vrt)
    assert 'crs' in sig.parameters
    assert sig.parameters['crs'].default is None


def test_write_vrt_crs_annotation_matches_writer_trio():
    """``crs`` is annotated ``int | str | None``, identical to
    ``to_geotiff(..., crs=...)`` and ``write_geotiff_gpu(..., crs=...)``.
    """
    import inspect

    sig = inspect.signature(write_vrt)
    ann = str(sig.parameters['crs'].annotation)
    assert ann == 'int | str | None'


# --- Runtime: ``crs=<EPSG int>`` writes an EPSG-resolved WKT ---


def test_write_vrt_crs_epsg_int_writes_wkt_to_xml(tmp_path):
    """``crs=4326`` resolves to a WKT string in the VRT's <SRS> element.

    The current implementation forwards the WKT to ``_vrt.write_vrt``,
    which interpolates it into the <SRS> XML node. Reading the file
    back with ``read_vrt`` must therefore produce
    ``attrs['crs'] == 4326`` (because ``_wkt_to_epsg`` round-trips
    EPSG:4326's WKT cleanly).
    """
    src = _build_source_tif(tmp_path, 'epsg_int.tif')
    vrt_path = str(tmp_path / 'epsg_int.vrt')

    out = write_vrt(vrt_path, [src], crs=4326)
    assert out == vrt_path
    assert os.path.exists(vrt_path)

    da = read_vrt(vrt_path)
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_wkt_string(tmp_path):
    """``crs=<WKT string>`` passes the WKT through verbatim."""
    src = _build_source_tif(tmp_path, 'wkt.tif')
    vrt_path = str(tmp_path / 'wkt.vrt')

    # Build a WKT for EPSG:4326 directly via pyproj
    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    out = write_vrt(vrt_path, [src], crs=wkt)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    # WKT round-trips back to EPSG:4326 via _wkt_to_epsg
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_none_falls_through(tmp_path):
    """``crs=None`` (the default) picks the CRS from the first source."""
    src = _build_source_tif(tmp_path, 'none.tif')
    vrt_path = str(tmp_path / 'none.vrt')

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        out = write_vrt(vrt_path, [src], crs=None)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    # The source TIFF was written with EPSG:4326; VRT inherits it.
    assert da.attrs.get('crs') == 4326


def test_write_vrt_no_crs_kwarg_no_warning(tmp_path):
    """Omitting ``crs`` entirely (the most common call shape) does not
    emit any warning. The deprecation shim only fires when ``crs_wkt``
    is supplied explicitly."""
    src = _build_source_tif(tmp_path, 'no_kwarg.tif')
    vrt_path = str(tmp_path / 'no_kwarg.vrt')

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        write_vrt(vrt_path, [src])  # neither kwarg supplied
    assert os.path.exists(vrt_path)


# --- Deprecation shim: ``crs_wkt=`` still works but warns ---


def test_write_vrt_crs_wkt_deprecated_warns(tmp_path):
    """Passing ``crs_wkt=<wkt>`` emits ``DeprecationWarning`` but still
    produces a working VRT."""
    src = _build_source_tif(tmp_path, 'depr.tif')
    vrt_path = str(tmp_path / 'depr.vrt')

    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    with pytest.warns(DeprecationWarning, match='crs_wkt'):
        out = write_vrt(vrt_path, [src], crs_wkt=wkt)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_wkt_none_still_warns(tmp_path):
    """``crs_wkt=None`` (explicit) was a documented shape in the old
    signature -- it now warns because the caller is using the
    deprecated kwarg name, even if the value is None."""
    src = _build_source_tif(tmp_path, 'depr_none.tif')
    vrt_path = str(tmp_path / 'depr_none.vrt')

    with pytest.warns(DeprecationWarning, match='crs_wkt'):
        write_vrt(vrt_path, [src], crs_wkt=None)
    assert os.path.exists(vrt_path)


def test_write_vrt_both_crs_and_crs_wkt_rejected(tmp_path):
    """Passing both raises ``TypeError`` rather than silently picking
    one. The error message names both kwargs so the caller can fix
    their call quickly."""
    src = _build_source_tif(tmp_path, 'both.tif')
    vrt_path = str(tmp_path / 'both.vrt')

    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    with pytest.raises(TypeError, match='crs.*crs_wkt'):
        write_vrt(vrt_path, [src], crs=4326, crs_wkt=wkt)


# --- Cross-writer parity: same kwarg name on all three writers ---


def test_writer_trio_all_accept_crs_kwarg():
    """``crs`` is the canonical kwarg on every public writer in the trio.
    A caller forwarding ``crs=<value>`` to whichever writer matches the
    output extension never has to special-case the kwarg name (issue
    #1715)."""
    import inspect

    from xrspatial.geotiff import to_geotiff, write_geotiff_gpu, write_vrt

    for fn in (to_geotiff, write_geotiff_gpu, write_vrt):
        sig = inspect.signature(fn)
        assert 'crs' in sig.parameters, f"{fn.__name__} missing crs kwarg"
        assert (
            str(sig.parameters['crs'].annotation) == 'int | str | None'
        ), f"{fn.__name__}.crs annotation drift"


# --- Negative tests: bad input shapes ---


def test_write_vrt_crs_invalid_type_rejected(tmp_path):
    """``crs=<list>`` (or any non-int/str/None) raises ``TypeError`` from
    the public wrapper rather than from deep inside the writer."""
    src = _build_source_tif(tmp_path, 'bad_type.tif')
    vrt_path = str(tmp_path / 'bad_type.vrt')

    with pytest.raises(TypeError, match='crs must be'):
        write_vrt(vrt_path, [src], crs=[4326])


def test_write_vrt_crs_unparseable_string_rejected(tmp_path):
    """``crs='not a CRS'`` raises ``ValueError`` from the public
    wrapper (the WKT keyword heuristic recognises PROJCS/GEOGCS only;
    everything else is sent through pyproj which will reject it)."""
    src = _build_source_tif(tmp_path, 'bad_str.tif')
    vrt_path = str(tmp_path / 'bad_str.vrt')

    with pytest.raises(ValueError, match='Could not parse crs'):
        write_vrt(vrt_path, [src], crs='not-a-real-crs-string')


# -------------------------------------------------------------------------
# Section: write_vrt bool nodata
# -------------------------------------------------------------------------

@pytest.fixture
def uint8_da():
    """Small uint8 DataArray for nodata round-trip tests."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.fixture
def src_geotiff(uint8_da, tmp_path):
    """A real on-disk source GeoTIFF that write_vrt can point at."""
    path = str(tmp_path / "src_1921.tif")
    to_geotiff(uint8_da, path)
    return path


# ---------------------------------------------------------------------------
# write_vrt: the bug from issue #1921
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_write_vrt_rejects_bool_nodata(src_geotiff, tmp_path, bad):
    """``write_vrt`` raises ``TypeError`` for any bool nodata.

    Fixed in issue #1921 by routing the public ``write_vrt`` wrapper
    through ``_validate_nodata_arg`` and adding a defense-in-depth check
    inside the internal ``_vrt.write_vrt``.
    """
    vrt_path = str(tmp_path / "out_1921_bad.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_vrt(vrt_path, [src_geotiff], nodata=bad)


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_write_vrt_internal_rejects_bool_nodata(src_geotiff, tmp_path, bad):
    """Direct call to the internal ``_vrt.write_vrt`` also rejects bool.

    Defense-in-depth: the public wrapper's ``_validate_nodata_arg`` is
    skipped when callers reach the internal symbol directly (e.g. the
    multi-tile dask write path in ``_writers/eager.py`` that calls
    ``_vrt.write_vrt`` after writing per-tile GeoTIFFs, or a future
    split of the wrapper). Parametrize over both ``bool`` and
    ``np.bool_`` polarities so a refactor that narrows the internal
    guard to just ``bool`` surfaces here, not in user code. See #1921.
    """
    from xrspatial.geotiff._vrt import write_vrt as _internal_write_vrt
    vrt_path = str(tmp_path / "out_1921_internal.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _internal_write_vrt(vrt_path, [src_geotiff], nodata=bad)


@pytest.mark.parametrize(
    "good",
    [0, 0.0, -9999, 255, np.int16(-1), np.float32(0.5)],
)
def test_write_vrt_accepts_numeric_nodata(src_geotiff, tmp_path, good):
    """Numeric sentinels go through unchanged: the fix must not over-reject."""
    vrt_path = str(tmp_path / f"out_1921_numeric_{good!r}.vrt")
    write_vrt(vrt_path, [src_geotiff], nodata=good)
    with open(vrt_path) as f:
        content = f.read()
    # The exact format of the emitted nodata string is implementation
    # detail; we only assert no "True"/"False" leaked through.
    assert "<NoDataValue>True</NoDataValue>" not in content
    assert "<NoDataValue>False</NoDataValue>" not in content


def test_write_vrt_accepts_none_nodata(src_geotiff, tmp_path):
    """``nodata=None`` is the documented default and must keep working."""
    vrt_path = str(tmp_path / "out_1921_none.vrt")
    write_vrt(vrt_path, [src_geotiff], nodata=None)
    assert os.path.exists(vrt_path)


# ---------------------------------------------------------------------------
# write_geotiff_gpu: defense-in-depth parity
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_write_geotiff_gpu_rejects_bool_nodata(uint8_da, tmp_path, bad):
    """Direct ``write_geotiff_gpu`` call rejects bool nodata.

    The top-of-function ``_validate_nodata_arg`` call (added by #1973)
    fires first; the deeper ``build_geo_tags`` guard is a second line
    of defense. Pinning the behaviour so a refactor that drops the
    top-of-function call surfaces here, not deep inside the geotag
    builder.
    """
    from xrspatial.geotiff import write_geotiff_gpu
    path = str(tmp_path / "gpu_1921_bad.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_geotiff_gpu(uint8_da, path, nodata=bad)


@requires_gpu
def test_to_geotiff_gpu_dispatch_rejects_bool_nodata(uint8_da, tmp_path):
    """Auto-dispatch path: ``to_geotiff(gpu=True, nodata=True)``.

    The eager-side guard fires before dispatch, so the GPU writer never
    runs. Pin that ordering so a future refactor cannot accidentally
    skip the eager check on the GPU dispatch path.
    """
    path = str(tmp_path / "to_geotiff_gpu_1921.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(uint8_da, path, gpu=True, nodata=True)


# -------------------------------------------------------------------------
# Section: write_vrt int nodata
# -------------------------------------------------------------------------

def _nodata_annotation(fn):
    sig = inspect.signature(fn)
    return sig.parameters["nodata"].annotation


def test_write_vrt_public_nodata_accepts_int_annotation():
    """The public wrapper widens the annotation to include int."""
    ann = _nodata_annotation(write_vrt)
    # Allow either typing.Union[float, int, None] or float | int | None.
    if isinstance(ann, str):
        # Forward-referenced string annotation (rare here; defensive).
        assert "int" in ann, ann
        return
    if hasattr(typing, "get_args"):
        args = set(typing.get_args(ann))
        if args:
            assert int in args, args
            return
    # Fallback: stringify the annotation.
    assert "int" in str(ann), str(ann)


def test_write_vrt_internal_nodata_accepts_int_annotation():
    """The internal helper in `_vrt.py` mirrors the public surface."""
    ann = _nodata_annotation(_vrt_module.write_vrt)
    if isinstance(ann, str):
        assert "int" in ann, ann
        return
    if hasattr(typing, "get_args"):
        args = set(typing.get_args(ann))
        if args:
            assert int in args, args
            return
    assert "int" in str(ann), str(ann)


def test_write_vrt_int_nodata_round_trips(tmp_path):
    """An int nodata renders to ``<NoDataValue>`` and parses back the same."""
    # Build a tiny uint16 tile so the sentinel makes sense.
    arr = np.array([[100, 200, 65535],
                    [300, 400, 500]], dtype=np.uint16)
    da = xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5]),
            "x": np.array([0.5, 1.5, 2.5]),
        },
        attrs={"crs": 4326},
    )
    tif_path = tmp_path / "source.tif"
    to_geotiff(da, str(tif_path))

    vrt_path = tmp_path / "mosaic.vrt"
    # Passing an int sentinel must not raise; the surface should match
    # to_geotiff's "float, int, or None" contract.
    write_vrt(str(vrt_path), [str(tif_path)], nodata=65535)

    # Confirm the int round-trips through the parser back into a VRT band.
    parsed = _vrt_module.parse_vrt(
        vrt_path.read_text(), vrt_dir=str(tmp_path))
    band_nodata = parsed.bands[0].nodata
    assert band_nodata == 65535, band_nodata


# -------------------------------------------------------------------------
# Section: VRT writer: int64 source
# -------------------------------------------------------------------------

def _da(arr: np.ndarray) -> xr.DataArray:
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )


def _read_vrt_dtype_attr(vrt_path: str) -> str:
    """Extract the ``dataType`` attribute from the emitted VRT XML."""
    with open(vrt_path) as f:
        xml = f.read()
    m = re.search(r'dataType="([^"]+)"', xml)
    assert m is not None, f"no dataType attribute in VRT:\n{xml}"
    return m.group(1)


def test_uint64_vrt_writer_declares_uint64(tmp_path):
    big = np.iinfo(np.uint64).max
    arr = np.array([[1, 2], [big - 7, big]], dtype=np.uint64)
    vrt = tmp_path / 'u64_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    assert _read_vrt_dtype_attr(str(vrt)) == 'UInt64'


def test_int64_vrt_writer_declares_int64(tmp_path):
    info = np.iinfo(np.int64)
    arr = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    vrt = tmp_path / 'i64_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    assert _read_vrt_dtype_attr(str(vrt)) == 'Int64'


def test_uint64_vrt_round_trip(tmp_path):
    big = np.iinfo(np.uint64).max
    arr = np.array([[1, 2], [big - 7, big]], dtype=np.uint64)
    vrt = tmp_path / 'u64_rt_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    r = open_geotiff(str(vrt))
    assert r.dtype == np.uint64
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_int64_vrt_round_trip(tmp_path):
    info = np.iinfo(np.int64)
    arr = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    vrt = tmp_path / 'i64_rt_1833.vrt'
    to_geotiff(_da(arr), str(vrt))
    r = open_geotiff(str(vrt))
    assert r.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(r.values), arr)


# -------------------------------------------------------------------------
# Section: VRT writer: photometric tag
# -------------------------------------------------------------------------

def _read_primary_ifd(path: str):
    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    return parse_ifd(raw, hdr.first_ifd_offset, hdr)


def _tile_paths(vrt_path: str):
    stem = os.path.splitext(os.path.basename(vrt_path))[0]
    tiles_dir = os.path.join(
        os.path.dirname(os.path.abspath(vrt_path)),
        stem + '_tiles',
    )
    return sorted(glob.glob(os.path.join(tiles_dir, 'tile_*.tif')))


def test_vrt_writer_forwards_photometric_miniswhite_1861(tmp_path):
    """photometric='miniswhite' must tag every per-tile TIFF with
    PhotometricInterpretation = 0 (MinIsWhite)."""
    arr = np.zeros((48, 48), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    vrt_path = str(tmp_path / 'miniswhite_1861.vrt')

    to_geotiff(da, vrt_path, photometric='miniswhite', tile_size=16)

    tiles = _tile_paths(vrt_path)
    assert tiles, 'expected at least one per-tile TIFF under _tiles/'
    for tile in tiles:
        ifd = _read_primary_ifd(tile)
        assert ifd.get_value(TAG_PHOTOMETRIC) == 0, (
            f'tile {tile} has Photometric '
            f'{ifd.get_value(TAG_PHOTOMETRIC)}, expected 0 (MinIsWhite)'
        )


def test_vrt_writer_default_photometric_minisblack_1861(tmp_path):
    """Control: default photometric='auto' keeps per-tile TIFFs at
    PhotometricInterpretation = 1 (MinIsBlack)."""
    arr = np.zeros((48, 48), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    vrt_path = str(tmp_path / 'default_auto_1861.vrt')

    to_geotiff(da, vrt_path, tile_size=16)

    tiles = _tile_paths(vrt_path)
    assert tiles, 'expected at least one per-tile TIFF under _tiles/'
    for tile in tiles:
        ifd = _read_primary_ifd(tile)
        assert ifd.get_value(TAG_PHOTOMETRIC) == 1, (
            f'tile {tile} has Photometric '
            f'{ifd.get_value(TAG_PHOTOMETRIC)}, expected 1 (MinIsBlack)'
        )


# -------------------------------------------------------------------------
# Section: VRT writer: source compatibility
# -------------------------------------------------------------------------

def _unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f"vrt_1733_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_tif(path: str, *, h: int, w: int, dtype, bands: int = 1,
               px: float = 1.0, py: float = -1.0,
               origin_x: float = 0.0, origin_y: float = 100.0,
               crs: int | None = 4326) -> None:
    if bands == 1:
        arr = np.arange(h * w, dtype=dtype).reshape(h, w)
        dims = ['y', 'x']
    else:
        arr = np.arange(h * w * bands, dtype=dtype).reshape(h, w, bands)
        dims = ['y', 'x', 'band']
    y = origin_y + (np.arange(h) + 0.5) * py
    x = origin_x + (np.arange(w) + 0.5) * px
    coords = {'y': y, 'x': x}
    attrs = {}
    if crs is not None:
        attrs['crs'] = crs
    da = xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)
    to_geotiff(da, path, compression='none')


def test_mismatched_pixel_size_raises(tmp_path):
    d = _unique_dir(tmp_path, "px")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, px=1.0, py=-1.0)
    # Place b adjacent so the geometry would otherwise work, but the
    # pixel size disagrees.
    _write_tif(b, h=4, w=4, dtype=np.float32, px=2.0, py=-2.0,
               origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="pixel size"):
        _priv_write_vrt(vrt, [a, b])


def test_mismatched_dtype_raises(tmp_path):
    d = _unique_dir(tmp_path, "dtype")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    _write_tif(b, h=4, w=4, dtype=np.int16, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="dtype|sample_format|bps"):
        _priv_write_vrt(vrt, [a, b])


def test_mismatched_band_count_raises(tmp_path):
    d = _unique_dir(tmp_path, "bands")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, bands=1)
    _write_tif(b, h=4, w=4, dtype=np.float32, bands=3, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="band count"):
        _priv_write_vrt(vrt, [a, b])


def test_compatible_sources_succeed(tmp_path):
    d = _unique_dir(tmp_path, "ok")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    _priv_write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_pixel_size_within_tolerance_accepted(tmp_path):
    d = _unique_dir(tmp_path, "tol")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, px=1.0, py=-1.0)
    # Drift well below the 1e-6 relative tolerance.
    _write_tif(b, h=4, w=4, dtype=np.float32,
               px=1.0 + 1e-10, py=-1.0, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    _priv_write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_single_source_still_works(tmp_path):
    d = _unique_dir(tmp_path, "one")
    a = os.path.join(d, "a.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    vrt = os.path.join(d, "out.vrt")
    _priv_write_vrt(vrt, [a])
    assert os.path.exists(vrt)


def test_mismatched_crs_raises(tmp_path):
    # Two sources with different non-empty CRS values must be rejected,
    # otherwise the VRT would inherit the first source's CRS and silently
    # misproject the second.
    d = _unique_dir(tmp_path, "crs_diff")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=3857)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        _priv_write_vrt(vrt, [a, b])


def test_asymmetric_crs_raises_first_set_second_missing(tmp_path):
    # First source has a CRS, second is written without one. The VRT
    # would otherwise be tagged with the first source's CRS, which can
    # misplace data when the second source actually came from a
    # different (or unknown) projection.
    d = _unique_dir(tmp_path, "crs_first")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=None)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        _priv_write_vrt(vrt, [a, b])


def test_asymmetric_crs_raises_first_missing_second_set(tmp_path):
    # Symmetric case: first source missing a CRS, second has one. The
    # earlier guard only triggered when both sides were set, so this
    # would have silently produced an untagged VRT despite one source
    # carrying a known projection.
    d = _unique_dir(tmp_path, "crs_second")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=None)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=4326)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        _priv_write_vrt(vrt, [a, b])


def test_matching_crs_succeeds(tmp_path):
    # Sanity check: two sources with the same CRS should still be
    # accepted (defends against an overly aggressive equality check).
    d = _unique_dir(tmp_path, "crs_match")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=4326)
    vrt = os.path.join(d, "out.vrt")
    _priv_write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_both_missing_crs_succeeds(tmp_path):
    # If neither source has a CRS, the VRT just won't be tagged with one
    # and there's nothing to mis-tag. This must not raise.
    d = _unique_dir(tmp_path, "crs_both_missing")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=None)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=None)
    vrt = os.path.join(d, "out.vrt")
    _priv_write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


# -------------------------------------------------------------------------
# Section: VRT writer: tiled output
# -------------------------------------------------------------------------

@pytest.fixture
def sample_raster():
    """200x200 float32 raster with coords and CRS."""
    arr = np.random.default_rng(55).random((200, 200), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 200)  # north-to-south
    x = np.linspace(-106.0, -105.0, 200)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326, 'nodata': -9999.0})
    return da


class TestVrtOutputNumpy:
    def test_creates_vrt_and_tiles_dir(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'out_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
        tiles_dir = str(tmp_path / 'out_1083_tiles')
        assert os.path.isdir(tiles_dir)
        tile_files = os.listdir(tiles_dir)
        assert len(tile_files) > 0
        assert all(f.endswith('.tif') for f in tile_files)

    def test_round_trip_numpy(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rt_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tile_naming_convention(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'named_1083.vrt')
        to_geotiff(sample_raster, vrt_path, tile_size=128)
        tiles_dir = str(tmp_path / 'named_1083_tiles')
        files = sorted(os.listdir(tiles_dir))
        # 200x200 with tile_size=128 -> 2x2 grid (TIFF 6 spec requires
        # tile_size be a multiple of 16; 100 was rejected post-#1767).
        assert files == [
            'tile_00_00.tif', 'tile_00_01.tif',
            'tile_01_00.tif', 'tile_01_01.tif',
        ]

    def test_relative_paths_in_vrt(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rel_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        with open(vrt_path) as f:
            content = f.read()
        # Paths should be relative (no leading /)
        assert 'rel_1083_tiles/' in content
        assert str(tmp_path) not in content

    def test_compression_level_passed_to_tiles(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cl_1083.vrt')
        to_geotiff(sample_raster, vrt_path, compression='zstd',
                   compression_level=1)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)


class TestVrtOutputDask:
    def test_dask_round_trip(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'dask_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_dask_one_tile_per_chunk(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'chunks_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        tiles_dir = str(tmp_path / 'chunks_1083_tiles')
        # 200x200 chunked 100x100 -> 2x2 = 4 tiles
        assert len(os.listdir(tiles_dir)) == 4


class TestVrtEdgeCases:
    def test_cog_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cog_1083.vrt')
        with pytest.raises(ValueError, match='cog.*vrt|vrt.*cog|COG.*VRT|VRT.*COG|cog.*VRT|vrt.*COG'):  # noqa: E501
            to_geotiff(sample_raster, vrt_path, cog=True)

    def test_overview_levels_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'ovr_1083.vrt')
        with pytest.raises(ValueError, match='overview.*vrt|vrt.*overview|overview.*VRT|VRT.*overview'):  # noqa: E501
            to_geotiff(sample_raster, vrt_path, overview_levels=[2, 4])

    def test_nonempty_tiles_dir_raises(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'exist_1083_tiles'
        tiles_dir.mkdir()
        (tiles_dir / 'dummy.tif').write_text('x')
        vrt_path = str(tmp_path / 'exist_1083.vrt')
        with pytest.raises(FileExistsError):
            to_geotiff(sample_raster, vrt_path)

    def test_empty_tiles_dir_ok(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'empty_1083_tiles'
        tiles_dir.mkdir()
        vrt_path = str(tmp_path / 'empty_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
