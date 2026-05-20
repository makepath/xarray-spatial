"""Cross-backend parity for the eager finalization pipeline (issue #2179).

Wave 2 of #2162 routed the eager numpy path and the three eager GPU
paths in ``_backends/gpu.py`` through the shared
``_finalize_eager_read`` helper introduced in #2177. The four sites
previously inlined the same validate / populate-attrs / mask / cast /
``_set_nodata_attrs`` block; this file pins parity for the attrs the
helper now stamps on both backends so a future change in one branch
cannot silently diverge from the other.

The matrix walks:

* Float source with a sentinel value (mask promotes via NaN).
* Integer source with an in-range sentinel (mask promotes int -> float64).
* Integer source with an out-of-range sentinel (mask is a no-op).
* ``mask_nodata=False`` left-alone semantics.
* Source with no declared sentinel (helper short-circuits both
  ``nodata`` and ``masked_nodata`` attrs).

For each case the test reads the file via the eager numpy backend
(``open_geotiff(path)``) and the eager GPU backend
(``open_geotiff(path, gpu=True)``) and compares the four
``_finalize_eager_read``-stamped attrs across the two reads:
``nodata``, ``nodata_pixels_present``, ``nodata_dtype_cast``, and
``georef_status``. The masked-pixel locations are also compared so a
divergence in the mask step would surface here.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


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
    from xrspatial.geotiff import open_geotiff
    cpu = open_geotiff(path, **kwargs)
    gpu = open_geotiff(path, gpu=True, **kwargs)
    return cpu, gpu


# Subset of attrs ``_finalize_eager_read`` is responsible for; mirrors
# the issue body's parity claim list.
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


# ---------------------------------------------------------------------------
# Float source with in-buffer sentinel
# ---------------------------------------------------------------------------


@_gpu_only
def test_float_sentinel_match_and_mask(tmp_path):
    """Float source + sentinel: both backends mask in place, attrs match."""
    arr = np.array(
        [[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'eager_parity_2179_float_sentinel.tif')
    _write_with_nodata(arr, path, nodata=-9999.0)

    cpu, gpu = _read_both(path)

    # dtype + masked_nodata first: float source stays at its declared
    # dtype on both backends; the mask substitutes NaN.
    assert cpu.dtype == gpu.dtype
    assert cpu.attrs.get('masked_nodata') is True
    assert gpu.attrs.get('masked_nodata') is True

    # Lifecycle attrs proper. ``nodata_pixels_present`` must surface
    # as a real bool on both backends (the issue body calls this out
    # explicitly).
    _assert_lifecycle_attrs_match(cpu, gpu)
    assert isinstance(cpu.attrs.get('nodata_pixels_present'), bool)
    assert isinstance(gpu.attrs.get('nodata_pixels_present'), bool)
    assert cpu.attrs.get('nodata_pixels_present') is True

    # And the NaN locations agree pixel-for-pixel.
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))


# ---------------------------------------------------------------------------
# Integer source with in-range sentinel
# ---------------------------------------------------------------------------


@_gpu_only
def test_int_in_range_sentinel_promotes_to_float(tmp_path):
    """uint16 + 65535 sentinel: both backends promote to float64 with NaN."""
    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'eager_parity_2179_int_sentinel.tif')
    _write_with_nodata(arr, path, nodata=65535)

    cpu, gpu = _read_both(path)

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


# ---------------------------------------------------------------------------
# Integer source with out-of-range sentinel
# ---------------------------------------------------------------------------


@_gpu_only
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


# ---------------------------------------------------------------------------
# mask_nodata=False
# ---------------------------------------------------------------------------


@_gpu_only
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


# ---------------------------------------------------------------------------
# No declared sentinel
# ---------------------------------------------------------------------------


@_gpu_only
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


# ---------------------------------------------------------------------------
# nodata_dtype_cast on explicit dtype=
# ---------------------------------------------------------------------------


@_gpu_only
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
