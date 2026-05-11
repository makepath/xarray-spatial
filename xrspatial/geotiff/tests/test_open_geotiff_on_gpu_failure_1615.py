"""Regression tests for issue #1615.

``open_geotiff(gpu=True)`` used to silently drop the ``on_gpu_failure``
kwarg: the dispatcher did not declare it and the GPU branch did not
forward it to ``read_geotiff_gpu``. Callers that wanted strict GPU
failure semantics had to bypass ``open_geotiff`` entirely and call
``read_geotiff_gpu`` directly, defeating the dispatcher.

The fix:

* ``open_geotiff`` accepts ``on_gpu_failure`` and forwards it to
  ``read_geotiff_gpu`` when ``gpu=True``.
* ``on_gpu_failure`` paired with ``gpu=False`` (or unset) raises
  ``ValueError`` so the kwarg cannot be silently ignored.
* The default sentinel keeps the dispatcher's behavior bit-stable for
  callers that never set ``on_gpu_failure``: ``read_geotiff_gpu`` is
  called without the kwarg, taking its own ``'auto'`` default.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialized."""
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


@pytest.fixture
def small_tiff_path(tmp_path):
    """4x6 single-band tiled tiff usable by both CPU and GPU readers."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'on_gpu_failure_1615.tif'
    to_geotiff(da, str(p), tile_size=4)
    return str(p), arr


# --------------------------------------------------------------------
# Signature: ``open_geotiff`` accepts ``on_gpu_failure``.
# --------------------------------------------------------------------


def test_open_geotiff_signature_includes_on_gpu_failure():
    """Direct ``inspect.signature`` check on the public dispatcher."""
    import inspect

    from xrspatial.geotiff import open_geotiff

    sig = inspect.signature(open_geotiff)
    assert 'on_gpu_failure' in sig.parameters, (
        "open_geotiff must declare on_gpu_failure so the GPU policy is "
        "addressable through the dispatcher entry point"
    )


# --------------------------------------------------------------------
# ``on_gpu_failure`` with ``gpu=False`` rejects up front.
# --------------------------------------------------------------------


def test_on_gpu_failure_with_gpu_false_raises_value_error(small_tiff_path):
    """Refuse rather than silently dropping the kwarg on the CPU path."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, on_gpu_failure='strict')


def test_on_gpu_failure_with_explicit_gpu_false_raises(small_tiff_path):
    """``gpu=False`` explicitly is rejected just like the default."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, gpu=False, on_gpu_failure='auto')


def test_on_gpu_failure_with_chunks_only_raises(small_tiff_path):
    """Dask CPU path is not GPU and should refuse the kwarg."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, chunks=2, on_gpu_failure='auto')


# --------------------------------------------------------------------
# Default sentinel does not change CPU/dask behavior.
# --------------------------------------------------------------------


def test_default_dispatch_unchanged_cpu(small_tiff_path):
    """Not passing ``on_gpu_failure`` keeps CPU dispatch byte-stable."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path
    da = open_geotiff(path)
    np.testing.assert_array_equal(da.values, arr)


def test_default_dispatch_unchanged_dask(small_tiff_path):
    """Not passing ``on_gpu_failure`` keeps Dask CPU dispatch byte-stable."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path
    da = open_geotiff(path, chunks=2)
    np.testing.assert_array_equal(da.values, arr)


# --------------------------------------------------------------------
# GPU forwarding: real behavior parity with ``read_geotiff_gpu``.
# --------------------------------------------------------------------


@_gpu_only
def test_open_geotiff_gpu_forwards_on_gpu_failure_auto(small_tiff_path):
    """``open_geotiff(gpu=True, on_gpu_failure='auto')`` works end-to-end."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, on_gpu_failure='auto')
    # CuPy-backed DataArray -- .data.get() pulls back to host for comparison.
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_forwards_on_gpu_failure_strict(small_tiff_path):
    """``on_gpu_failure='strict'`` also reaches the GPU pipeline."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, on_gpu_failure='strict')
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_rejects_invalid_on_gpu_failure(small_tiff_path):
    """An invalid value still surfaces from the underlying validator."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        open_geotiff(path, gpu=True, on_gpu_failure='loose')


# --------------------------------------------------------------------
# Static-only check that works on CPU CI: passing an invalid value
# with gpu=True still routes through to read_geotiff_gpu's validator.
# --------------------------------------------------------------------


def test_invalid_on_gpu_failure_reaches_gpu_validator_on_cpu(small_tiff_path):
    """Even on a CPU-only host, the kwarg should reach ``read_geotiff_gpu``.

    Without GPU hardware, ``read_geotiff_gpu`` raises ``ImportError`` (no
    cupy) or runs through to the actual decode. The kwarg validator runs
    *before* the cupy import, so an invalid value surfaces deterministically
    in both environments. This pins the forwarding wire even on CPU-only CI.
    """
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path
    # gpu=True + invalid value: validation fires before any cupy import,
    # so the ValueError reaches us on every host.
    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        open_geotiff(path, gpu=True, on_gpu_failure='loose')
