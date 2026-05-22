"""Regression test for #1705: writer-trio nodata / streaming_buffer_bytes annotations.

Follow-up to #1654. The api-consistency sweep on 2026-05-12 found two
remaining annotation gaps across the public writer trio (``to_geotiff``,
``write_geotiff_gpu``, ``write_vrt``):

* ``nodata`` -- annotated as ``float | int | None`` on ``write_vrt``
  (added by #1684) but bare ``=None`` on ``to_geotiff`` and
  ``write_geotiff_gpu``. The three docstrings all describe the same
  accepted-type set ("float, int, or None"), so the annotation should
  match across siblings.

* ``streaming_buffer_bytes`` -- ``int`` (default 256 MB) on
  ``to_geotiff`` versus ``int | None`` (default None) on
  ``write_geotiff_gpu``. The GPU writer no-ops this kwarg
  (``del streaming_buffer_bytes`` in the body) so the type signature
  was the only consistency dimension; pin both to ``int`` so callers
  passing the same kwargs to either entry point see the same hint.

This module pins both annotations against future drift.
"""
from __future__ import annotations

import inspect

from xrspatial.geotiff import to_geotiff, write_geotiff_gpu, write_vrt


def _annotation(fn, param_name):
    """Return the stringified annotation for ``fn``'s ``param_name``."""
    sig = inspect.signature(fn)
    p = sig.parameters[param_name]
    assert p.annotation is not inspect.Parameter.empty, (
        f"{fn.__name__}({param_name}=...) is missing a type annotation"
    )
    return str(p.annotation)


# --- nodata: float | int | None on every writer entry point ---


def test_to_geotiff_nodata_annotated():
    assert _annotation(to_geotiff, 'nodata') == 'float | int | None'


def test_write_geotiff_gpu_nodata_annotated():
    assert _annotation(write_geotiff_gpu, 'nodata') == 'float | int | None'


def test_write_vrt_nodata_annotated():
    """Pre-existing annotation from #1684 -- keep it pinned."""
    assert _annotation(write_vrt, 'nodata') == 'float | int | None'


# --- streaming_buffer_bytes: int on both writer entry points ---


def test_to_geotiff_streaming_buffer_bytes_annotated():
    """Pre-existing -- ``int`` with a 256 MB default."""
    assert _annotation(to_geotiff, 'streaming_buffer_bytes') == 'int'
    assert (
        inspect.signature(to_geotiff)
        .parameters['streaming_buffer_bytes']
        .default
        == 256 * 1024 * 1024
    )


def test_write_geotiff_gpu_streaming_buffer_bytes_annotated():
    """GPU writer must agree with ``to_geotiff`` on type and default so a
    caller forwarding the same kwargs to either entry point sees the same
    hint. The kwarg is a runtime no-op on the GPU writer (deleted on
    entry); the annotation parity is the only consistency dimension."""
    assert _annotation(
        write_geotiff_gpu, 'streaming_buffer_bytes'
    ) == 'int'
    assert (
        inspect.signature(write_geotiff_gpu)
        .parameters['streaming_buffer_bytes']
        .default
        == 256 * 1024 * 1024
    )


# --- Smoke: the new annotations do not break runtime call semantics ---


def test_to_geotiff_nodata_int_runtime(tmp_path):
    """``nodata=<int>`` still round-trips through ``to_geotiff`` and the
    sentinel survives into the read-back attrs."""
    import numpy as np
    import xarray as xr

    from xrspatial.geotiff import open_geotiff

    arr = np.full((8, 8), -9999, dtype=np.int32)
    arr[2:6, 2:6] = 42
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    path = str(tmp_path / 'nodata_int.tif')
    to_geotiff(da, path, nodata=-9999)
    r = open_geotiff(path)
    assert r.attrs.get('nodata') == -9999


def test_write_geotiff_gpu_streaming_buffer_bytes_runtime_noop(tmp_path):
    """Passing an explicit ``streaming_buffer_bytes`` to the GPU writer
    must remain a no-op. The body still does ``del streaming_buffer_bytes``
    so the value has no effect on the produced file."""
    import pytest

    from .conftest import gpu_available

    if not gpu_available():
        pytest.skip("cupy + CUDA required for write_geotiff_gpu")

    import cupy
    import numpy as np
    import xarray as xr

    arr_cpu = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    arr_gpu = cupy.asarray(arr_cpu)
    da_gpu = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(64.0, 0, -1), 'x': np.arange(64.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 64.0)},
    )
    p1 = str(tmp_path / 'default.tif')
    p2 = str(tmp_path / 'override.tif')
    write_geotiff_gpu(da_gpu, p1)
    write_geotiff_gpu(da_gpu, p2, streaming_buffer_bytes=8 * 1024 * 1024)
    # Both files have identical sizes -- the buffer kwarg is a no-op.
    import os

    assert os.path.getsize(p1) == os.path.getsize(p2)
