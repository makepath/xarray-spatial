"""Tests for floating-point predictor (predictor=3) write path.

Issue #1313: ``to_geotiff`` previously accepted ``predictor: bool`` and
emitted only TIFF predictor 2 (horizontal differencing).  Predictor 3
(byte-swizzled differencing per TIFF Technical Note 3) gives noticeably
better deflate/zstd ratios on float data and is what most GDAL/rasterio
workflows use for elevation rasters.  These tests cover the new write
path end-to-end.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import normalize_predictor


def _smooth_float(shape, dtype):
    """Smooth surface where FP predictor is expected to help compression."""
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(dtype)
    return (np.sin(x / 40) * np.cos(y / 40) * 100 + (x + y) / 4).astype(dtype)


def _da(arr):
    h, w = arr.shape[:2]
    coords = {
        'x': np.arange(w, dtype=np.float64) * 10.0,
        'y': np.arange(h, dtype=np.float64) * 10.0,
    }
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=('y', 'x'), coords=coords)
    # 3D: (band, y, x)
    return xr.DataArray(arr, dims=('band', 'y', 'x'), coords=coords)


def _read_predictor_tag(path):
    """Read the TIFF Predictor tag (id=317) directly from a file's IFD."""
    with open(path, 'rb') as f:
        header = f.read(8)
    assert header[:2] == b'II', "test fixture writes little-endian"
    magic = struct.unpack('<H', header[2:4])[0]
    assert magic == 42, "classic TIFF expected"
    ifd_offset = struct.unpack('<I', header[4:8])[0]

    with open(path, 'rb') as f:
        f.seek(ifd_offset)
        n_entries = struct.unpack('<H', f.read(2))[0]
        for _ in range(n_entries):
            entry = f.read(12)
            tag, type_id, count = struct.unpack('<HHI', entry[:8])
            if tag == 317:
                return struct.unpack('<H', entry[8:10])[0]
    return None  # tag absent => predictor 1 (none)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('compression', ['deflate', 'zstd'])
@pytest.mark.parametrize('tiled', [True, False])
def test_predictor3_round_trip(tmp_path, dtype, compression, tiled):
    arr = _smooth_float((96, 128), dtype)
    da = _da(arr)
    path = tmp_path / f'fp_pred_1313_{np.dtype(dtype).name}_{compression}.tif'
    to_geotiff(da, str(path), compression=compression, predictor=3,
               tiled=tiled)

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)

    # File should advertise predictor=3 in the TIFF tag
    assert _read_predictor_tag(str(path)) == 3


def test_predictor3_better_than_predictor2_on_smooth_floats(tmp_path):
    arr = _smooth_float((512, 512), np.float32)
    da = _da(arr)
    p2 = tmp_path / 'pred2_1313.tif'
    p3 = tmp_path / 'pred3_1313.tif'
    to_geotiff(da, str(p2), compression='deflate', predictor=2)
    to_geotiff(da, str(p3), compression='deflate', predictor=3)

    # FP predictor exists precisely because it compresses smooth floats
    # better than horizontal differencing.  If this regresses, something
    # is wrong with the encoder wiring.
    assert p3.stat().st_size < p2.stat().st_size


def test_predictor3_rejects_integer_dtype(tmp_path):
    arr = np.zeros((8, 8), dtype=np.int32)
    da = _da(arr)
    path = tmp_path / 'bad_int_1313.tif'
    with pytest.raises(ValueError, match='predictor=3'):
        to_geotiff(da, str(path), compression='deflate', predictor=3)


def test_predictor_legacy_bool_unchanged(tmp_path):
    """``predictor=True`` keeps emitting TIFF predictor 2."""
    arr = _smooth_float((32, 32), np.float32)
    da = _da(arr)
    path = tmp_path / 'legacy_true_1313.tif'
    to_geotiff(da, str(path), compression='deflate', predictor=True)
    assert _read_predictor_tag(str(path)) == 2

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_predictor_false_emits_no_tag(tmp_path):
    arr = _smooth_float((32, 32), np.float32)
    da = _da(arr)
    path = tmp_path / 'legacy_false_1313.tif'
    to_geotiff(da, str(path), compression='deflate', predictor=False)
    # Tag absent OR set to 1 -> both mean "no predictor"
    tag = _read_predictor_tag(str(path))
    assert tag in (None, 1)


def test_predictor3_with_compression_none_is_silent(tmp_path):
    """compression='none' suppresses any predictor (matches predictor=2 behavior)."""
    arr = _smooth_float((16, 16), np.float32)
    da = _da(arr)
    path = tmp_path / 'pred3_nocomp_1313.tif'
    to_geotiff(da, str(path), compression='none', predictor=3)

    tag = _read_predictor_tag(str(path))
    assert tag in (None, 1), \
        "predictor must be suppressed when compression=none"

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_normalize_predictor_table():
    """The bool|int -> int normalization mapping."""
    f32 = np.dtype('float32')
    i32 = np.dtype('int32')
    deflate = 8  # COMPRESSION_DEFLATE

    assert normalize_predictor(False, f32, deflate) == 1
    assert normalize_predictor(0, f32, deflate) == 1
    assert normalize_predictor(1, f32, deflate) == 1
    assert normalize_predictor(True, f32, deflate) == 2
    assert normalize_predictor(2, f32, deflate) == 2
    assert normalize_predictor(3, f32, deflate) == 3

    with pytest.raises(ValueError, match='predictor=3'):
        normalize_predictor(3, i32, deflate)

    with pytest.raises(ValueError, match='predictor must be'):
        normalize_predictor(99, f32, deflate)


def test_predictor3_streaming_dask(tmp_path):
    """Dask-backed input takes the streaming path; predictor=3 must work."""
    da_module = pytest.importorskip('dask.array')
    arr = _smooth_float((128, 192), np.float32)
    dask_arr = da_module.from_array(arr, chunks=(64, 96))
    da = xr.DataArray(
        dask_arr, dims=('y', 'x'),
        coords={'x': np.arange(192, dtype=np.float64) * 10.0,
                'y': np.arange(128, dtype=np.float64) * 10.0},
    )
    path = tmp_path / 'pred3_streaming_1313.tif'
    to_geotiff(da, str(path), compression='deflate', predictor=3,
               tile_size=64)

    assert _read_predictor_tag(str(path)) == 3
    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_predictor3_multiband_round_trip(tmp_path):
    """Multi-band float predictor=3 round-trip.

    Issue #1247 fixed the read side; this checks the write side now
    round-trips correctly for the multi-band case where the row swizzle
    has to use ``width * samples`` lanes, not ``width``.
    """
    h, w = 48, 64
    arr = np.stack([
        _smooth_float((h, w), np.float32),
        _smooth_float((h, w), np.float32) * 1.5,
        _smooth_float((h, w), np.float32) - 10.0,
    ], axis=0)  # (3, h, w)
    da = xr.DataArray(
        arr, dims=('band', 'y', 'x'),
        coords={'x': np.arange(w, dtype=np.float64) * 10.0,
                'y': np.arange(h, dtype=np.float64) * 10.0},
    )
    path = tmp_path / 'pred3_multiband_1313.tif'
    to_geotiff(da, str(path), compression='deflate', predictor=3)

    assert _read_predictor_tag(str(path)) == 3
    out = open_geotiff(str(path))
    # open_geotiff returns (y, x, band) for multi-band; reorder for compare
    if out.ndim == 3 and out.shape[-1] == 3:
        out_arr = np.moveaxis(out.values, -1, 0)
    else:
        out_arr = out.values
    np.testing.assert_array_equal(out_arr, arr)


def test_predictor3_large_round_trip_value_exact(tmp_path):
    """1024x1024 float32 deflate+predictor=3 round-trips with no value drift.

    The encode path was refactored to dispatch the per-row kernel from
    inside an ``@ngjit`` wrapper instead of from a Python ``for`` loop.
    Guards against any silent corruption from the refactor by asserting
    the output array is byte-for-byte identical to the input: dtype must
    match, and a ``uint8`` view of the bytes must compare equal so the
    check catches signed-zero drift, NaN payload changes, and any other
    bit-level divergence that ``assert_array_equal`` would mask.
    """
    h, w = 1024, 1024
    arr = _smooth_float((h, w), np.float32)
    da = _da(arr)
    path = tmp_path / 'pred3_large_round_trip_1313.tif'
    to_geotiff(da, str(path), compression='deflate', predictor=3)

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
    predictor=2 because the row loop was in Python.  Opt-in via
    ``XRSPATIAL_RUN_PERF_TESTS=1`` -- shared CI runners, CPU throttling,
    debug builds, and noisy filesystems all make absolute wall-clock
    timings flaky, so the test stays off by default. Matches the
    convention from ``test_streaming_write_parallel.py``.
    """
    if os.environ.get('XRSPATIAL_RUN_PERF_TESTS') != '1':
        pytest.skip(
            "set XRSPATIAL_RUN_PERF_TESTS=1 to run wall-clock perf tests")

    import time

    arr = _smooth_float((1024, 1024), np.float32)
    da = _da(arr)
    p2 = tmp_path / 'pred2_timing.tif'
    p3 = tmp_path / 'pred3_timing.tif'

    # Warm up numba
    to_geotiff(da, str(p2), compression='deflate', predictor=2)
    to_geotiff(da, str(p3), compression='deflate', predictor=3)

    t0 = time.perf_counter()
    to_geotiff(da, str(p2), compression='deflate', predictor=2)
    t_p2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    to_geotiff(da, str(p3), compression='deflate', predictor=3)
    t_p3 = time.perf_counter() - t0

    assert t_p3 < 2.0 * t_p2, (
        f'predictor=3 ({t_p3*1000:.1f} ms) is more than 2x slower than '
        f'predictor=2 ({t_p2*1000:.1f} ms); ngjit row loop may have regressed'
    )
