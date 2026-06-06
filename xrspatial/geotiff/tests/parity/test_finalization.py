"""Cross-backend parity for the read finalization pipeline.

Sibling to ``parity/test_backend_matrix.py``. Where ``test_backend_matrix``
asserts pixel/coord/attrs parity over a fixture matrix, this file pins the
shared read-finalization plumbing centralised into helpers in ``_attrs`` /
``_validation``. Three sections:

Section 1 -- Dispatcher kwarg parity
    ``_validate_dispatch_kwargs`` runs at the top of every public read
    entry point so ``overview_level``, ``max_cloud_bytes``,
    ``missing_sources``, ``band_nodata``, ``on_gpu_failure``, and the
    file-like-source guard reject identically across ``open_geotiff`` /
    ``_read_geotiff_dask`` / ``_read_geotiff_gpu`` / ``_read_vrt``.

Section 2 -- Eager finalization parity
    ``_finalize_eager_read`` stamps the same nodata / georef attrs on the
    eager numpy and eager GPU paths. The matrix walks float / int /
    out-of-range sentinels, ``mask_nodata=False``, no-sentinel,
    explicit ``dtype=``, windowed reads, MinIsWhite, and multi-band.

Section 3 -- Lazy finalization parity
    ``_finalize_lazy_read_attrs`` stamps the same attrs on the two dask
    backends (``_read_geotiff_dask`` and the dask branch of
    ``_read_geotiff_gpu``). Covers the five georef states plus the
    ``nodata_pixels_present`` / ``nodata_dtype_cast`` lazy contract.

GPU and dask+GPU rows skip when cupy + CUDA are absent via the shared
``requires_gpu`` marker from ``_helpers/markers.py``.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (_build_vrt, _read_geotiff_dask, _read_geotiff_gpu, _read_vrt,
                               open_geotiff, to_geotiff)
from xrspatial.geotiff._attrs import (GEOREF_STATUS_CRS_ONLY, GEOREF_STATUS_FULL,
                                      GEOREF_STATUS_NONE, GEOREF_STATUS_ROTATED_DROPPED,
                                      GEOREF_STATUS_TRANSFORM_ONLY)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY

from .._helpers.markers import requires_gpu
# Rotated-TIFF writer lives alongside the CRS read tests.
from ..read.test_crs import _write_rotated_tiff

# ===========================================================================
# Section 1 -- Dispatcher kwarg parity
# ===========================================================================
#
# ``open_geotiff`` used to validate dispatcher kwargs inline; the three
# direct backends skipped most of the checks. ``_validate_dispatch_kwargs``
# now runs at the top of every public read entry point so the exception
# type and message match across backends for the same invalid input.


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


def _make_one_source_vrt(tmp_path):
    """Build a 1-source VRT mosaic referencing a small local GeoTIFF."""
    src = _build_local_tif(tmp_path, name='vrt_src_2175.tif')
    vrt = str(tmp_path / 'mosaic_2175.vrt')
    _build_vrt(vrt, [src])
    return vrt, src


# --- overview_level type rejection through every entry point ---


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
        _read_geotiff_dask(path, overview_level=value)


def test_dask_overview_level_str(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="str"):
        _read_geotiff_dask(path, overview_level="0")


def test_dask_overview_level_float(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="float"):
        _read_geotiff_dask(path, overview_level=1.0)


@pytest.mark.parametrize("value", [True, False])
def test_gpu_overview_level_bool(tmp_path, value):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        _read_geotiff_gpu(path, overview_level=value)


def test_gpu_overview_level_str(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="str"):
        _read_geotiff_gpu(path, overview_level="0")


def test_gpu_overview_level_float(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(TypeError, match="float"):
        _read_geotiff_gpu(path, overview_level=1.0)


@pytest.mark.parametrize("value", [True, False])
def test_vrt_overview_level_bool(tmp_path, value):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        _read_vrt(vrt, overview_level=value)


def test_vrt_overview_level_str(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(TypeError, match="str"):
        _read_vrt(vrt, overview_level="0")


def test_vrt_overview_level_float(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(TypeError, match="float"):
        _read_vrt(vrt, overview_level=1.0)


# --- max_cloud_bytes incompatibility through every applicable backend ---


def test_open_geotiff_dask_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, chunks=4, max_cloud_bytes=8)


def test_open_geotiff_gpu_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, gpu=True, max_cloud_bytes=8)


def test_open_geotiff_vrt_rejects_max_cloud_bytes(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=8)


def test_dask_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_geotiff_dask(path, max_cloud_bytes=8)


def test_gpu_rejects_max_cloud_bytes(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_geotiff_gpu(path, max_cloud_bytes=8)


def test_vrt_rejects_max_cloud_bytes(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_vrt(vrt, max_cloud_bytes=8)


def test_explicit_none_max_cloud_bytes_rejected_on_dask_direct(tmp_path):
    """``max_cloud_bytes=None`` is the documented "disable budget" value
    on the eager path. On the dask path it has no consumer, so an
    explicit ``None`` is still rejected -- the sentinel default is the
    only way to pass through without setting an opinion.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_geotiff_dask(path, max_cloud_bytes=None)


def test_explicit_none_max_cloud_bytes_rejected_on_gpu_direct(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_geotiff_gpu(path, max_cloud_bytes=None)


def test_explicit_none_max_cloud_bytes_rejected_on_vrt_direct(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        _read_vrt(vrt, max_cloud_bytes=None)


# --- missing_sources on non-VRT sources ---


def test_open_geotiff_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        open_geotiff(path, missing_sources='raise')


def test_dask_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        _read_geotiff_dask(path, missing_sources='raise')


def test_gpu_rejects_missing_sources_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"missing_sources only applies"):
        _read_geotiff_gpu(path, missing_sources='raise')


# --- band_nodata on non-VRT sources ---


def test_open_geotiff_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        open_geotiff(path, band_nodata='first')


def test_dask_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        _read_geotiff_dask(path, band_nodata='first')


def test_gpu_rejects_band_nodata_on_tif(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"band_nodata only applies"):
        _read_geotiff_gpu(path, band_nodata='first')


# --- on_gpu_failure when GPU is disabled ---


def test_open_geotiff_rejects_on_gpu_failure_when_gpu_false(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        open_geotiff(path, on_gpu_failure='strict')


def test_dask_rejects_on_gpu_failure(tmp_path):
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        _read_geotiff_dask(path, on_gpu_failure='strict')


def test_vrt_rejects_on_gpu_failure(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"on_gpu_failure only applies"):
        _read_vrt(vrt, on_gpu_failure='strict')


# --- File-like sources reject gpu=True / chunks=... ---


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
        _read_geotiff_dask(buf)


def test_gpu_rejects_file_like(tmp_path):
    path = _build_local_tif(tmp_path)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    with pytest.raises(
            ValueError,
            match=r"gpu=True is not supported for file-like"):
        _read_geotiff_gpu(buf)


# --- Path-object sources survive the helper's file-like guard ---


def test_open_geotiff_accepts_path_object(tmp_path):
    from pathlib import Path
    path = _build_local_tif(tmp_path)
    out = open_geotiff(Path(path))
    assert out.shape == (8, 8)


def test_dask_accepts_path_object(tmp_path):
    from pathlib import Path
    path = _build_local_tif(tmp_path)
    out = _read_geotiff_dask(Path(path), chunks=4)
    assert out.shape == (8, 8)


def test_vrt_accepts_path_object(tmp_path):
    from pathlib import Path
    vrt, _src = _make_one_source_vrt(tmp_path)
    out = _read_vrt(Path(vrt))
    assert out.shape == (8, 8)


@requires_gpu
def test_gpu_accepts_path_object(tmp_path):
    from pathlib import Path
    path = _build_local_tif(tmp_path)
    out = _read_geotiff_gpu(Path(path))
    assert out.shape == (8, 8)


def test_gpu_path_object_does_not_raise_file_like_error(tmp_path):
    """Even on a CPU-only host the validator must accept a Path object.

    The dispatch validator runs before any cupy import, so the bad
    behaviour on `main` (treating Path as file-like) raises before any
    GPU code executes. With the fix the validator coerces Path to str
    first and the error only surfaces (if at all) from the GPU stack.
    """
    from pathlib import Path
    path = _build_local_tif(tmp_path)
    # Either the call succeeds (GPU available) or it fails for a real
    # GPU reason. The one thing it must NOT raise is the file-like
    # ValueError introduced by the validator misclassifying Path.
    try:
        _read_geotiff_gpu(Path(path))
    except ValueError as e:
        assert "file-like" not in str(e), (
            f"validator misclassified Path as file-like: {e}"
        )
    except (ImportError, RuntimeError):
        # ImportError: cupy not installed.
        # RuntimeError: CUDA preflight failed.
        # Both are unrelated to the Path-coercion regression.
        pass


# --- Default sentinel pins (no regressions on the happy path) ---


def test_open_geotiff_defaults_round_trip(tmp_path):
    path = _build_local_tif(tmp_path)
    out = open_geotiff(path)
    assert out.shape == (8, 8)


def test_dask_defaults_round_trip(tmp_path):
    path = _build_local_tif(tmp_path)
    out = _read_geotiff_dask(path)
    assert out.shape == (8, 8)


def test_vrt_defaults_round_trip(tmp_path):
    vrt, _src = _make_one_source_vrt(tmp_path)
    out = _read_vrt(vrt)
    assert out.shape == (8, 8)


# --- Cross-entry-point message parity ---


def _get_error(callable_, *args, **kwargs):
    """Invoke ``callable_`` and return the (type_name, message) of the
    exception it raises. Asserting on the type and message separately
    catches a regression where the exception type changes silently
    while the message stays the same.
    """
    try:
        callable_(*args, **kwargs)
    except Exception as e:
        return type(e).__name__, str(e)
    raise AssertionError("expected an exception, none raised")


def test_max_cloud_bytes_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _make_one_source_vrt(tmp_path)
    open_dask = _get_error(open_geotiff, path, chunks=4, max_cloud_bytes=8)
    direct_dask = _get_error(_read_geotiff_dask, path, max_cloud_bytes=8)
    # Both raise ValueError with the same dask-incompatibility message.
    assert open_dask[0] == "ValueError"
    assert direct_dask[0] == "ValueError"
    for _, msg in (open_dask, direct_dask):
        assert "max_cloud_bytes" in msg
        assert "dask" in msg

    open_gpu = _get_error(open_geotiff, path, gpu=True, max_cloud_bytes=8)
    direct_gpu = _get_error(_read_geotiff_gpu, path, max_cloud_bytes=8)
    assert open_gpu[0] == "ValueError"
    assert direct_gpu[0] == "ValueError"
    for _, msg in (open_gpu, direct_gpu):
        assert "max_cloud_bytes" in msg
        assert "gpu" in msg.lower()

    open_vrt = _get_error(open_geotiff, vrt, max_cloud_bytes=8)
    direct_vrt = _get_error(_read_vrt, vrt, max_cloud_bytes=8)
    assert open_vrt[0] == "ValueError"
    assert direct_vrt[0] == "ValueError"
    for _, msg in (open_vrt, direct_vrt):
        assert "max_cloud_bytes" in msg
        assert "vrt" in msg.lower()


def test_band_nodata_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    results = [
        _get_error(open_geotiff, path, band_nodata='first'),
        _get_error(_read_geotiff_dask, path, band_nodata='first'),
        _get_error(_read_geotiff_gpu, path, band_nodata='first'),
    ]
    for kind, msg in results:
        assert kind == "ValueError"
        assert "band_nodata only applies" in msg


def test_missing_sources_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    results = [
        _get_error(open_geotiff, path, missing_sources='raise'),
        _get_error(_read_geotiff_dask, path, missing_sources='raise'),
        _get_error(_read_geotiff_gpu, path, missing_sources='raise'),
    ]
    for kind, msg in results:
        assert kind == "ValueError"
        assert "missing_sources only applies" in msg


def test_on_gpu_failure_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _make_one_source_vrt(tmp_path)
    results = [
        _get_error(open_geotiff, path, on_gpu_failure='strict'),
        _get_error(_read_geotiff_dask, path, on_gpu_failure='strict'),
        _get_error(_read_vrt, vrt, on_gpu_failure='strict'),
    ]
    for kind, msg in results:
        assert kind == "ValueError"
        assert "on_gpu_failure only applies" in msg


def test_overview_level_message_parity(tmp_path):
    path = _build_local_tif(tmp_path)
    vrt, _ = _make_one_source_vrt(tmp_path)
    results = [
        _get_error(open_geotiff, path, overview_level="bad"),
        _get_error(_read_geotiff_dask, path, overview_level="bad"),
        _get_error(_read_geotiff_gpu, path, overview_level="bad"),
        _get_error(_read_vrt, vrt, overview_level="bad"),
    ]
    for kind, msg in results:
        assert kind == "TypeError"
        assert "overview_level must be an int or None" in msg
        assert "str" in msg


# ===========================================================================
# Section 2 -- Eager finalization parity
# ===========================================================================
#
# ``_finalize_eager_read`` stamps nodata / georef attrs on the eager numpy
# path and the three eager GPU paths. Each case reads the same file via
# ``open_geotiff(path)`` and ``open_geotiff(path, gpu=True)`` and compares
# the helper-stamped attrs across the two reads.


def _write_with_nodata(arr, path, *, nodata=None):
    """Helper: write a 2-D array to a tiled GeoTIFF with an optional sentinel."""
    from xrspatial.geotiff._writer import write
    write(arr, path, nodata=nodata, compression='deflate',
          tiled=True, tile_size=16)


def _read_both(path, **kwargs):
    """Read the same file via the eager numpy and eager GPU backends.

    Returns ``(cpu_da, gpu_da)``. ``kwargs`` are forwarded to both
    ``open_geotiff`` calls so each backend sees the same caller
    contract.
    """
    cpu = open_geotiff(path, **kwargs)
    gpu = open_geotiff(path, gpu=True, **kwargs)
    return cpu, gpu


# Subset of attrs ``_finalize_eager_read`` is responsible for.
_LIFECYCLE_ATTRS = (
    'nodata',
    'nodata_pixels_present',
    'nodata_dtype_cast',
    'georef_status',
)


def _assert_lifecycle_attrs_match(cpu_da, gpu_da):
    """Assert the four lifecycle attrs match across backends.

    ``masked_nodata`` is checked separately because the test suite
    asserts on its boolean value when a sentinel is declared.
    """
    for key in _LIFECYCLE_ATTRS:
        cpu_v = cpu_da.attrs.get(key)
        gpu_v = gpu_da.attrs.get(key)
        assert cpu_v == gpu_v, (
            f"attrs[{key!r}] divergence: cpu={cpu_v!r} gpu={gpu_v!r}"
        )


@requires_gpu
def test_float_sentinel_match_and_mask(tmp_path):
    """Float source + sentinel: both backends mask in place, attrs match."""
    arr = np.array(
        [[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'eager_parity_2179_float_sentinel.tif')
    _write_with_nodata(arr, path, nodata=-9999.0)

    cpu, gpu = _read_both(path, masked=True)

    # dtype + masked_nodata first: float source stays at its declared
    # dtype on both backends; the mask substitutes NaN.
    assert cpu.dtype == gpu.dtype
    assert cpu.attrs.get('masked_nodata') is True
    assert gpu.attrs.get('masked_nodata') is True

    # Lifecycle attrs proper. ``nodata_pixels_present`` must surface
    # as a real bool on both backends.
    _assert_lifecycle_attrs_match(cpu, gpu)
    assert isinstance(cpu.attrs.get('nodata_pixels_present'), bool)
    assert isinstance(gpu.attrs.get('nodata_pixels_present'), bool)
    assert cpu.attrs.get('nodata_pixels_present') is True

    # And the NaN locations agree pixel-for-pixel.
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))


@requires_gpu
def test_int_in_range_sentinel_promotes_to_float(tmp_path):
    """uint16 + 65535 sentinel: both backends promote to float64 with NaN."""
    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'eager_parity_2179_int_sentinel.tif')
    _write_with_nodata(arr, path, nodata=65535)

    cpu, gpu = _read_both(path, masked=True)

    # Integer promotion fires on both backends.
    assert cpu.dtype == np.float64
    assert gpu.dtype == np.float64
    assert cpu.attrs.get('masked_nodata') is True
    assert gpu.attrs.get('masked_nodata') is True

    _assert_lifecycle_attrs_match(cpu, gpu)
    assert cpu.attrs.get('nodata_pixels_present') is True

    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))


@requires_gpu
def test_int_out_of_range_sentinel_is_no_op(tmp_path):
    """uint8 + 9999 sentinel: out-of-range, no promotion, presence=False."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    path = str(tmp_path / 'eager_parity_2179_int_oor.tif')
    # 9999 cannot match any uint8 pixel. ``_writer.write`` accepts an
    # int sentinel here without complaining (the writer only refuses
    # bool / NaN values, not out-of-range ints), so we get a file with
    # the literal nodata tag set to 9999 and no pixel matching it.
    _write_with_nodata(arr, path, nodata=9999)

    cpu, gpu = _read_both(path)

    # No promotion when the sentinel is out of range. Both backends
    # leave the uint8 buffer alone.
    assert cpu.dtype == np.uint8
    assert gpu.dtype == np.uint8
    # ``masked_nodata`` is False because the mask did not run; the
    # final dtype is still int.
    assert cpu.attrs.get('masked_nodata') is False
    assert gpu.attrs.get('masked_nodata') is False

    _assert_lifecycle_attrs_match(cpu, gpu)
    assert cpu.attrs.get('nodata_pixels_present') is False


@requires_gpu
def test_mask_nodata_false_keeps_literal_sentinel(tmp_path):
    """mask_nodata=False leaves the buffer untouched on both backends."""
    arr = np.array(
        [[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'eager_parity_2179_mask_false.tif')
    _write_with_nodata(arr, path, nodata=-9999.0)

    cpu, gpu = _read_both(path, mask_nodata=False)

    # No NaN substitution; the literal sentinel survives on both
    # backends with ``masked_nodata=False``.
    assert cpu.dtype == np.float32
    assert gpu.dtype == np.float32
    assert cpu.attrs.get('masked_nodata') is False
    assert gpu.attrs.get('masked_nodata') is False

    _assert_lifecycle_attrs_match(cpu, gpu)
    # The no-mask scan branch still surfaces presence.
    assert cpu.attrs.get('nodata_pixels_present') is True

    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(cpu_arr, gpu_arr)


@requires_gpu
def test_no_declared_sentinel_omits_nodata_attrs(tmp_path):
    """Source without nodata declaration: no lifecycle attrs on either side."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    path = str(tmp_path / 'eager_parity_2179_no_sentinel.tif')
    _write_with_nodata(arr, path, nodata=None)

    cpu, gpu = _read_both(path)

    assert cpu.dtype == np.uint8
    assert gpu.dtype == np.uint8

    # The helper's ``_set_nodata_attrs`` early-returns when there is no
    # declared sentinel, so neither ``nodata`` nor ``masked_nodata``
    # appear on either backend.
    assert 'nodata' not in cpu.attrs
    assert 'nodata' not in gpu.attrs
    assert 'masked_nodata' not in cpu.attrs
    assert 'masked_nodata' not in gpu.attrs
    assert 'nodata_pixels_present' not in cpu.attrs
    assert 'nodata_pixels_present' not in gpu.attrs

    # ``georef_status`` still rides on the helper regardless of nodata
    # state, so the parity assertion exercises that branch too.
    _assert_lifecycle_attrs_match(cpu, gpu)


@requires_gpu
def test_dtype_kwarg_records_post_mask_cast(tmp_path):
    """Explicit dtype= records ``nodata_dtype_cast`` on both backends."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'eager_parity_2179_dtype_cast.tif')
    # Out-of-range sentinel keeps the mask a no-op so the cast attr is
    # the only signal that the user asked for a dtype change; this
    # isolates the ``nodata_dtype_cast`` branch from the mask-driven
    # promotion exercised in ``test_int_in_range_sentinel_promotes_to_float``.
    _write_with_nodata(arr, path, nodata=9999)

    cpu, gpu = _read_both(path, dtype=np.float32)

    assert cpu.dtype == np.float32
    assert gpu.dtype == np.float32
    assert cpu.attrs.get('nodata_dtype_cast') == 'float32'
    assert gpu.attrs.get('nodata_dtype_cast') == 'float32'

    _assert_lifecycle_attrs_match(cpu, gpu)


@requires_gpu
def test_windowed_read_presence_matches_window_contents(tmp_path):
    """Windowed read: nodata_pixels_present reflects the window, not the IFD.

    Pins the slice-before-mask behaviour on the GPU local-eager path.
    Masking the full IFD then slicing would report sentinel presence
    anywhere in the file; the contract is to report presence within the
    requested window. The CPU path has always behaved this way, so the
    two agree.
    """
    # 4x4 raster with the sentinel only in the bottom half so the two
    # windows below land on opposite sides of the presence bool.
    arr = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, -9999.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    path = str(tmp_path / 'eager_parity_2179_windowed.tif')
    _write_with_nodata(arr, path, nodata=-9999.0)

    # Top-left 2x2 window: no sentinel in scope.
    cpu, gpu = _read_both(path, window=(0, 0, 2, 2))
    _assert_lifecycle_attrs_match(cpu, gpu)
    assert cpu.attrs.get('nodata_pixels_present') is False
    assert gpu.attrs.get('nodata_pixels_present') is False

    # Bottom 2x4 window: covers the sentinel.
    cpu, gpu = _read_both(path, window=(2, 0, 4, 4))
    _assert_lifecycle_attrs_match(cpu, gpu)
    assert cpu.attrs.get('nodata_pixels_present') is True
    assert gpu.attrs.get('nodata_pixels_present') is True


@requires_gpu
def test_miniswhite_post_inversion_sentinel_parity(tmp_path):
    """MinIsWhite raster: post-inversion sentinel resolves identically on both backends.

    Exercises the ``_mw_mask_nodata`` branch in the GPU local-eager
    path. The reader inverts the buffer and the post-MinIsWhite
    sentinel is what the helper's mask block compares against on the
    GPU side; the eager numpy path takes the same sentinel off
    ``geo_info._mask_nodata`` through ``read_to_array``. Both should
    land on the same NaN positions and the same lifecycle attrs.
    """
    import tifffile

    # uint8 + nodata=0; MinIsWhite inverts the stored value to 255
    # before masking, and 255 is the post-inversion sentinel.
    stored = np.array([[0, 100, 200], [50, 0, 255]], dtype=np.uint8)
    path = str(tmp_path / 'eager_parity_2179_miniswhite.tif')
    extratags = [("GDAL_NODATA", "s", 0, "0\0", True)]
    tifffile.imwrite(
        path, stored, photometric="miniswhite",
        extratags=extratags, tile=(16, 16),
    )

    cpu, gpu = _read_both(path)

    _assert_lifecycle_attrs_match(cpu, gpu)
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    # NaN positions must agree pixel-for-pixel; the MinIsWhite
    # sentinel resolution drives this.
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))


@requires_gpu
def test_multiband_stripped_parity(tmp_path):
    """3-band stripped read: helper builds (y, x, band) DataArray on both backends.

    The GPU CPU-fallback path lands on stripped files. Multi-band
    output goes through the helper's ``arr.ndim == 3`` branch on
    both backends; the parity assertion covers ``georef_status`` and
    sentinel-related attrs for the multi-band shape so a future
    change to the 3-D coord build cannot silently diverge.
    """
    rng = np.random.RandomState(20260520)
    data = rng.randint(0, 200, size=(32, 48, 3)).astype(np.uint8)
    da_in = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / 'eager_parity_2179_multiband.tif')

    # Stripped (tiled=False) routes the GPU read through the
    # CPU-fallback eager site.
    to_geotiff(da_in, path, tiled=False)

    cpu, gpu = _read_both(path)

    # Shape and dims line up across backends.
    assert cpu.dims == gpu.dims
    assert cpu.shape == gpu.shape == (32, 48, 3)

    _assert_lifecycle_attrs_match(cpu, gpu)
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(cpu_arr, gpu_arr)


# ===========================================================================
# Section 3 -- Lazy finalization parity
# ===========================================================================
#
# ``_finalize_lazy_read_attrs`` centralises the validate-then-populate-then-
# stamp logic shared by ``_read_geotiff_dask`` (CPU+dask) and the dask branch
# of ``_read_geotiff_gpu`` (GPU+dask). Each test opens the same fixture
# through both backends and compares the attrs.

tifffile = pytest.importorskip("tifffile")


def _open_cpu_dask(path, **kwargs):
    return _read_geotiff_dask(path, chunks=2, **kwargs)


def _open_gpu_dask(path, **kwargs):
    return _read_geotiff_gpu(path, chunks=2, **kwargs)


_BACKENDS = [
    pytest.param(_open_cpu_dask, id="dask+numpy"),
    pytest.param(_open_gpu_dask, id="dask+cupy", marks=requires_gpu),
]


def _gpu_dask_available() -> bool:
    """Runtime GPU probe for the conditional cross-backend assertions.

    The ``requires_gpu`` marker handles the skip on the parametrised GPU
    rows; this helper gates the inline ``if`` branches that compare CPU
    against GPU inside an otherwise CPU-only test.
    """
    from .._helpers.markers import gpu_available
    return gpu_available()


# --- Fixture builders, one per georef-status state ---


def _make_full_tiff(path):
    """Float coords + CRS -> ``full``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, path)


def _make_transform_only_tiff(path):
    """Float coords, no CRS -> ``transform_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, path)


def _make_crs_only_tiff(path):
    """No-georef marker + CRS -> ``crs_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, path)


def _make_none_tiff(path):
    """Bare TIFF with no GeoTIFF tags at all -> ``none``."""
    arr = np.zeros((4, 4), dtype=np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )


def _make_rotated_tiff(path):
    """Rotated ``ModelTransformationTag`` (opened with ``allow_rotated``)
    -> ``rotated_dropped``. The data is uint16 because the rotated-TIFF
    writer only emits integer pixels; that's fine for a metadata pin."""
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    _write_rotated_tiff(path, arr)


def _make_float_with_nodata_tiff(path, sentinel=-9999.0):
    """Float raster carrying a GDAL_NODATA tag. Used to exercise the
    nodata lifecycle attrs without forcing the int->float promotion
    branch."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr[0, 0] = sentinel
    da = xr.DataArray(
        arr,
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326, 'nodata': sentinel},
    )
    to_geotiff(da, path)


def _make_int_with_nodata_tiff(path, sentinel=30):
    """Integer raster carrying a sentinel. Lets the dtype-cast tests
    distinguish "graph dtype auto-promoted by masking" from
    "caller asked for an explicit cast"."""
    arr = np.array([[10, 20, 25], [30, 40, 50]], dtype=np.int16)
    da = xr.DataArray(
        arr,
        coords={
            'y': np.array([200.0, 199.0]),
            'x': np.array([100.0, 101.0, 102.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326, 'nodata': sentinel},
    )
    to_geotiff(da, path)


_GEOREF_FIXTURES = [
    pytest.param(_make_full_tiff, GEOREF_STATUS_FULL, False,
                 id="full"),
    pytest.param(_make_transform_only_tiff, GEOREF_STATUS_TRANSFORM_ONLY,
                 False, id="transform_only"),
    pytest.param(_make_crs_only_tiff, GEOREF_STATUS_CRS_ONLY, False,
                 id="crs_only"),
    pytest.param(_make_none_tiff, GEOREF_STATUS_NONE, False,
                 id="none"),
    pytest.param(_make_rotated_tiff, GEOREF_STATUS_ROTATED_DROPPED,
                 True, id="rotated_dropped"),
]


@pytest.mark.parametrize("fixture,expected_status,allow_rotated",
                         _GEOREF_FIXTURES)
def test_georef_status_parity(tmp_path, fixture, expected_status,
                              allow_rotated):
    """Both dask backends emit the same ``georef_status`` for each
    of the five reader states."""
    path = str(tmp_path / f"tmp_2178_status_{expected_status}.tif")
    fixture(path)

    kwargs = {'allow_rotated': True} if allow_rotated else {}
    cpu = _open_cpu_dask(path, **kwargs)
    assert cpu.attrs.get('georef_status') == expected_status

    if _gpu_dask_available():
        gpu = _open_gpu_dask(path, **kwargs)
        assert gpu.attrs.get('georef_status') == expected_status
        assert cpu.attrs['georef_status'] == gpu.attrs['georef_status']


@pytest.mark.parametrize("fixture,expected_status,allow_rotated",
                         _GEOREF_FIXTURES)
def test_attrs_dict_parity(tmp_path, fixture, expected_status,
                           allow_rotated):
    """Both dask backends emit the same attrs dict for each fixture."""
    if not _gpu_dask_available():
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / f"tmp_2178_parity_{expected_status}.tif")
    fixture(path)

    kwargs = {'allow_rotated': True} if allow_rotated else {}
    cpu = _open_cpu_dask(path, **kwargs)
    gpu = _open_gpu_dask(path, **kwargs)

    cpu_attrs = dict(cpu.attrs)
    gpu_attrs = dict(gpu.attrs)
    assert cpu_attrs == gpu_attrs, (
        f"attrs dicts diverged for fixture={expected_status}:\n"
        f"  cpu only: {set(cpu_attrs) - set(gpu_attrs)}\n"
        f"  gpu only: {set(gpu_attrs) - set(cpu_attrs)}\n"
        f"  shared keys with different values: "
        f"{[k for k in set(cpu_attrs) & set(gpu_attrs) if cpu_attrs[k] != gpu_attrs[k]]}"
    )


@pytest.mark.parametrize("opener", _BACKENDS)
def test_nodata_pixels_present_absent_on_lazy(tmp_path, opener):
    """Lazy contract: ``nodata_pixels_present`` stays unset on both
    dask backends."""
    path = str(tmp_path / "tmp_2178_pixels_absent.tif")
    _make_float_with_nodata_tiff(path)
    out = opener(path)
    assert 'nodata_pixels_present' not in out.attrs


def test_nodata_pixels_present_cross_backend(tmp_path):
    """Both backends agree on the absence of ``nodata_pixels_present``
    when reading the same fixture."""
    if not _gpu_dask_available():
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_pixels_cross.tif")
    _make_float_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path)
    gpu = _open_gpu_dask(path)
    assert 'nodata_pixels_present' not in cpu.attrs
    assert 'nodata_pixels_present' not in gpu.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_absent_without_caller_dtype(tmp_path, opener):
    """No ``dtype=`` kwarg: ``nodata_dtype_cast`` stays unset, even
    when masking auto-promotes the graph dtype to float64."""
    path = str(tmp_path / "tmp_2178_no_cast.tif")
    _make_int_with_nodata_tiff(path)
    out = opener(path, mask_nodata=True)
    # Masking promoted the int source to float64 on the graph dtype,
    # but the caller did not ask for a cast.
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert 'nodata_dtype_cast' not in out.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_records_target(tmp_path, opener):
    """Explicit ``dtype=`` kwarg: ``nodata_dtype_cast`` records the
    requested dtype on both backends."""
    path = str(tmp_path / "tmp_2178_with_cast.tif")
    _make_int_with_nodata_tiff(path)
    out = opener(path, mask_nodata=False, dtype=np.float64)
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert 'nodata_pixels_present' not in out.attrs


def test_dtype_cast_parity_cross_backend(tmp_path):
    """Cross-backend: same input + same ``dtype=`` kwarg yields the
    same ``nodata_dtype_cast`` value."""
    if not _gpu_dask_available():
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_cast_cross.tif")
    _make_int_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path, mask_nodata=False, dtype=np.float64)
    gpu = _open_gpu_dask(path, mask_nodata=False, dtype=np.float64)
    assert cpu.attrs.get('nodata_dtype_cast') == gpu.attrs.get('nodata_dtype_cast')
    assert cpu.attrs.get('nodata_dtype_cast') == 'float64'


def test_dtype_cast_absent_parity_cross_backend(tmp_path):
    """Cross-backend: same int input without an explicit ``dtype=``
    leaves ``nodata_dtype_cast`` absent on both backends (the auto-
    promoted graph dtype must not leak as a caller cast)."""
    if not _gpu_dask_available():
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_no_cast_cross.tif")
    _make_int_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path)
    gpu = _open_gpu_dask(path)
    assert 'nodata_dtype_cast' not in cpu.attrs
    assert 'nodata_dtype_cast' not in gpu.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_records_integer_target(tmp_path, opener):
    """Caller-supplied integer ``dtype=`` kwarg: ``nodata_dtype_cast``
    records the integer dtype on both backends. Pins the
    ``dtype.kind != 'f'`` branch of the call-site fixup."""
    path = str(tmp_path / "tmp_2178_int_cast.tif")
    _make_int_with_nodata_tiff(path)
    # ``mask_nodata=False`` keeps the integer dtype; the caller cast
    # then routes the graph dtype to ``int32`` without the masking
    # auto-promotion firing. The pre-helper contract emits
    # ``nodata_dtype_cast='int32'`` and ``masked_nodata=False`` here.
    out = opener(path, mask_nodata=False, dtype=np.int32)
    assert out.dtype == np.int32
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'int32'
    assert 'nodata_pixels_present' not in out.attrs
