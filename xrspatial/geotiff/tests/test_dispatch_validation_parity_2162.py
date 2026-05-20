"""Dispatcher kwarg parity across GeoTIFF read entry points (issue #2175).

``open_geotiff`` used to validate ``overview_level``,
``on_gpu_failure``, ``missing_sources``, ``band_nodata``,
``max_cloud_bytes``, and the file-like source restrictions inline at
the top of its body. The three direct backends -- ``read_geotiff_dask``,
``read_geotiff_gpu``, and ``read_vrt`` -- each ran their own
``_validate_overview_level_arg`` call but skipped the rest. A caller
who passed an invalid ``band_nodata`` to dask or ``max_cloud_bytes``
to the GPU reader got no error at all, or got an unrelated
``TypeError`` from the signature.

This module pins parity. ``_validate_dispatch_kwargs`` lives in
``xrspatial/geotiff/_validation.py`` and is called from the top of
every public read entry point. The matrix below walks each kwarg
through every entry point and asserts that the exception type and
message match.

See issue #2175 (parent #2162).
"""
from __future__ import annotations

import importlib.util
import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
    to_geotiff,
    write_vrt,
)


# --------------------------------------------------------------------------
# Skip helpers + small fixtures
# --------------------------------------------------------------------------


def _gpu_available() -> bool:
    """Return True iff cupy imports cleanly AND a CUDA device is up."""
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


def _build_local_tif(tmp_path, name='src_2175.tif'):
    """Write a small valid GeoTIFF used as the dispatcher's source."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'transform': (1.0, 0, 0.0, 0, -1.0, 8.0),
        },
    )
    path = str(tmp_path / name)
    to_geotiff(da, path)
    return path


def _build_vrt(tmp_path):
    """Build a 1-source VRT mosaic referencing a small local GeoTIFF."""
    src = _build_local_tif(tmp_path, name='vrt_src_2175.tif')
    vrt = str(tmp_path / 'mosaic_2175.vrt')
    write_vrt(vrt, [src])
    return vrt, src


# --------------------------------------------------------------------------
# overview_level type rejection through every entry point
# --------------------------------------------------------------------------
#
# ``_validate_overview_level_arg`` rejects bool / str / float with a
# ``TypeError`` whose message names the offending type. The helper runs
# this check first across all four entry points (issue #2074, #2160).
# The cupy gate skips GPU cases when CUDA is unavailable. The GPU and
# VRT entry points also reach the validator before any GPU / source
# parse, so the bad-input path raises on CPU-only hosts.


@pytest.mark.parametrize("value", [True, False])
def test_open_geotiff_overview_level_bool(tmp_path, value):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        open_geotiff(path, overview_level=value)


def test_open_geotiff_overview_level_str(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="str"):
        open_geotiff(path, overview_level="0")


def test_open_geotiff_overview_level_float(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="float"):
        open_geotiff(path, overview_level=1.0)


@pytest.mark.parametrize("value", [True, False])
def test_dask_overview_level_bool(tmp_path, value):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask(path, overview_level=value)


def test_dask_overview_level_str(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="str"):
        read_geotiff_dask(path, overview_level="0")


def test_dask_overview_level_float(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="float"):
        read_geotiff_dask(path, overview_level=1.0)


@pytest.mark.parametrize("value", [True, False])
def test_gpu_overview_level_bool(tmp_path, value):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, overview_level=value)


def test_gpu_overview_level_str(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="str"):
        read_geotiff_gpu(path, overview_level="0")


def test_gpu_overview_level_float(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="float"):
        read_geotiff_gpu(path, overview_level=1.0)


@pytest.mark.parametrize("value", [True, False])
def test_vrt_overview_level_bool(tmp_path, value):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        read_vrt(vrt, overview_level=value)


def test_vrt_overview_level_str(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(TypeError, match="str"):
        read_vrt(vrt, overview_level="0")


def test_vrt_overview_level_float(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(TypeError, match="float"):
        read_vrt(vrt, overview_level=1.0)


# --------------------------------------------------------------------------
# max_cloud_bytes incompatibility through every applicable backend
# --------------------------------------------------------------------------
#
# Only the eager non-VRT non-GPU non-dask branch in ``open_geotiff``
# consumes ``max_cloud_bytes``. The three direct backends never look
# at it and would silently drop the budget before #2175. Parity means
# each backend rejects an explicit value with a ``ValueError`` whose
# message names the kwarg.


def test_open_geotiff_dask_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, chunks=4, max_cloud_bytes=8)


def test_open_geotiff_gpu_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, gpu=True, max_cloud_bytes=8)


def test_open_geotiff_vrt_rejects_max_cloud_bytes(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=8)


def test_dask_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_geotiff_dask(path, max_cloud_bytes=8)


def test_gpu_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_geotiff_gpu(path, max_cloud_bytes=8)


def test_vrt_rejects_max_cloud_bytes(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_vrt(vrt, max_cloud_bytes=8)


def test_explicit_none_max_cloud_bytes_rejected_on_dask_direct(tmp_path):
    """``max_cloud_bytes=None`` is the documented "disable budget" value
    on the eager path. On the dask path it has no consumer, so an
    explicit ``None`` is still rejected -- the sentinel default is the
    only way to pass through without setting an opinion.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_geotiff_dask(path, max_cloud_bytes=None)


def test_explicit_none_max_cloud_bytes_rejected_on_gpu_direct(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_geotiff_gpu(path, max_cloud_bytes=None)


def test_explicit_none_max_cloud_bytes_rejected_on_vrt_direct(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        read_vrt(vrt, max_cloud_bytes=None)


# --------------------------------------------------------------------------
# missing_sources on non-VRT sources
# --------------------------------------------------------------------------
#
# ``missing_sources`` controls the VRT mosaic loop. On a plain GeoTIFF
# the kwarg has no meaning. Every non-VRT entry point rejects an
# explicit value with a ``ValueError`` whose message names the kwarg.


def test_open_geotiff_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        open_geotiff(path, missing_sources='raise')


def test_dask_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        read_geotiff_dask(path, missing_sources='raise')


def test_gpu_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        read_geotiff_gpu(path, missing_sources='raise')


# --------------------------------------------------------------------------
# band_nodata on non-VRT sources
# --------------------------------------------------------------------------
#
# ``band_nodata`` is the #1987 PR 5 opt-out for the VRT mixed-band
# metadata check. On a plain GeoTIFF the kwarg has no meaning -- each
# non-VRT entry point rejects an explicit value.


def test_open_geotiff_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        open_geotiff(path, band_nodata='first')


def test_dask_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        read_geotiff_dask(path, band_nodata='first')


def test_gpu_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        read_geotiff_gpu(path, band_nodata='first')


# --------------------------------------------------------------------------
# on_gpu_failure when GPU is disabled
# --------------------------------------------------------------------------
#
# ``on_gpu_failure`` is the GPU pipeline's strict/auto fallback policy.
# The CPU and dask paths have no such concept, and ``read_vrt`` does
# not route through the GPU decoder. Each non-GPU entry point rejects
# an explicit value with a ``ValueError`` whose message names the kwarg.


def test_open_geotiff_rejects_on_gpu_failure_when_gpu_false(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        open_geotiff(path, on_gpu_failure='strict')


def test_dask_rejects_on_gpu_failure(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        read_geotiff_dask(path, on_gpu_failure='strict')


def test_vrt_rejects_on_gpu_failure(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        read_vrt(vrt, on_gpu_failure='strict')


# --------------------------------------------------------------------------
# File-like sources reject gpu=True / chunks=...
# --------------------------------------------------------------------------
#
# The GPU and dask paths re-open the source by path from worker tasks,
# so a file-like buffer cannot survive the trip. ``open_geotiff`` and
# the direct backends share the rejection now via the helper.


def test_open_geotiff_rejects_file_like_with_chunks(tmp_path):
    path = _build_local_tif(tmp_path)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    with pytest.raises(
            ValueError,
            match=r"chunks=\.\.\. \(dask\) is not supported for file-like"):
        open_geotiff(buf, chunks=4)


def test_open_geotiff_rejects_file_like_with_gpu(tmp_path):
    path = _build_local_tif(tmp_path)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    with pytest.raises(
            ValueError,
            match=r"gpu=True is not supported for file-like"):
        open_geotiff(buf, gpu=True)


def test_dask_rejects_file_like(tmp_path):
    path = _build_local_tif(tmp_path)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    with pytest.raises(
            ValueError,
            match=r"chunks=\.\.\. \(dask\) is not supported for file-like"):
        read_geotiff_dask(buf)


def test_gpu_rejects_file_like(tmp_path):
    path = _build_local_tif(tmp_path)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    with pytest.raises(
            ValueError,
            match=r"gpu=True is not supported for file-like"):
        read_geotiff_gpu(buf)


# --------------------------------------------------------------------------
# Default sentinel pins (no regressions on the happy path)
# --------------------------------------------------------------------------
#
# Every entry point must accept its sentinel defaults without raising.
# A regression on a default would break every call site that omits the
# kwarg.


def test_open_geotiff_defaults_round_trip(tmp_path):
    path = _build_local_tif(tmp_path)
    out = open_geotiff(path)
    assert out.shape == (8, 8)


def test_dask_defaults_round_trip(tmp_path):
    path = _build_local_tif(tmp_path)
    out = read_geotiff_dask(path)
    assert out.shape == (8, 8)


def test_vrt_defaults_round_trip(tmp_path):
    vrt, _src = _build_vrt(tmp_path)
    out = read_vrt(vrt)
    assert out.shape == (8, 8)


# --------------------------------------------------------------------------
# Cross-entry-point message parity. The same invalid input through
# different entry points should produce the same exception text so
# callers can match on a single regex regardless of which backend they
# hit.
# --------------------------------------------------------------------------


def _get_error_message(callable_, *args, **kwargs) -> str:
    """Invoke ``callable_`` and return the exception message it raises."""
    try:
        callable_(*args, **kwargs)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    raise AssertionError("expected an exception, none raised")


def test_max_cloud_bytes_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _build_vrt(tmp_path)
    open_dask_msg = _get_error_message(
        open_geotiff, path, chunks=4, max_cloud_bytes=8)
    direct_dask_msg = _get_error_message(
        read_geotiff_dask, path, max_cloud_bytes=8)
    # Both messages mention max_cloud_bytes and the dask incompatibility.
    assert "max_cloud_bytes" in open_dask_msg
    assert "max_cloud_bytes" in direct_dask_msg
    assert "dask" in open_dask_msg
    assert "dask" in direct_dask_msg

    open_gpu_msg = _get_error_message(
        open_geotiff, path, gpu=True, max_cloud_bytes=8)
    direct_gpu_msg = _get_error_message(
        read_geotiff_gpu, path, max_cloud_bytes=8)
    assert "max_cloud_bytes" in open_gpu_msg
    assert "max_cloud_bytes" in direct_gpu_msg
    assert "gpu" in open_gpu_msg.lower()
    assert "gpu" in direct_gpu_msg.lower()

    open_vrt_msg = _get_error_message(
        open_geotiff, vrt, max_cloud_bytes=8)
    direct_vrt_msg = _get_error_message(
        read_vrt, vrt, max_cloud_bytes=8)
    assert "max_cloud_bytes" in open_vrt_msg
    assert "max_cloud_bytes" in direct_vrt_msg
    assert "vrt" in open_vrt_msg.lower()
    assert "vrt" in direct_vrt_msg.lower()


def test_band_nodata_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    open_msg = _get_error_message(
        open_geotiff, path, band_nodata='first')
    dask_msg = _get_error_message(
        read_geotiff_dask, path, band_nodata='first')
    gpu_msg = _get_error_message(
        read_geotiff_gpu, path, band_nodata='first')
    for msg in (open_msg, dask_msg, gpu_msg):
        assert "band_nodata only applies" in msg


def test_missing_sources_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    open_msg = _get_error_message(
        open_geotiff, path, missing_sources='raise')
    dask_msg = _get_error_message(
        read_geotiff_dask, path, missing_sources='raise')
    gpu_msg = _get_error_message(
        read_geotiff_gpu, path, missing_sources='raise')
    for msg in (open_msg, dask_msg, gpu_msg):
        assert "missing_sources only applies" in msg


def test_on_gpu_failure_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _build_vrt(tmp_path)
    open_msg = _get_error_message(
        open_geotiff, path, on_gpu_failure='strict')
    dask_msg = _get_error_message(
        read_geotiff_dask, path, on_gpu_failure='strict')
    vrt_msg = _get_error_message(
        read_vrt, vrt, on_gpu_failure='strict')
    for msg in (open_msg, dask_msg, vrt_msg):
        assert "on_gpu_failure only applies" in msg


def test_overview_level_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _build_vrt(tmp_path)
    open_msg = _get_error_message(open_geotiff, path, overview_level="bad")
    dask_msg = _get_error_message(read_geotiff_dask, path, overview_level="bad")
    gpu_msg = _get_error_message(read_geotiff_gpu, path, overview_level="bad")
    vrt_msg = _get_error_message(read_vrt, vrt, overview_level="bad")
    for msg in (open_msg, dask_msg, gpu_msg, vrt_msg):
        assert "overview_level must be an int or None" in msg
        assert "str" in msg
