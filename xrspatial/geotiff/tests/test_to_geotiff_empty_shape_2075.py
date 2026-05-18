"""Regression tests for issue #2075.

``to_geotiff`` used to accept arrays with a zero-height or zero-width
spatial dim and write a TIFF whose IFD claimed shape ``(0, N)`` or
``(N, 0)``. The reader then rejected the file with the generic
"Invalid image dimensions" message that never named the writer as the
source.

The fix raises ``ValueError`` at the write entry point. The failure
happens before any bytes hit disk, and the message names the offending
dimension so callers know which axis went empty (a clip / window
operation is the common cause).
"""
from __future__ import annotations

import importlib.util

import dask.array as dsk
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


def _cupy_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _cupy_available()


_EMPTY_SHAPES = [
    pytest.param((0, 5), id="zero-height"),
    pytest.param((5, 0), id="zero-width"),
    pytest.param((0, 0), id="both-zero"),
]


@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test_to_geotiff_rejects_empty_numpy(tmp_path, shape):
    h, w = shape
    da = xr.DataArray(
        np.zeros(shape, dtype=np.float32),
        dims=("y", "x"),
    )
    out = tmp_path / f"tmp_2075_empty_{h}x{w}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value)
    # The message must name the writer that the user called so the
    # traceback names the right entry point.
    assert "to_geotiff" in msg
    assert "empty" in msg.lower()
    if h == 0:
        assert "height=0" in msg
    if w == 0:
        assert "width=0" in msg
    # Nothing should have been written.
    assert not out.exists()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_write_geotiff_gpu_rejects_empty(tmp_path):
    """``write_geotiff_gpu`` is a public entry point and does not go
    through ``to_geotiff``; make sure the empty-shape guard fires there
    too (the suggestion from PR #2078 review)."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    arr = cp.zeros((0, 5), dtype=cp.float32)
    out = tmp_path / "tmp_2075_empty_gpu_0x5.tif"
    with pytest.raises(ValueError) as excinfo:
        write_geotiff_gpu(arr, str(out))
    msg = str(excinfo.value)
    assert "write_geotiff_gpu" in msg
    assert "height=0" in msg
    assert not out.exists()


def test_to_geotiff_rejects_empty_dask(tmp_path):
    # One dask variant is enough to exercise the streaming entry point.
    shape = (0, 5)
    da = xr.DataArray(
        dsk.zeros(shape, dtype=np.float32, chunks=shape if 0 not in shape
                  else (1, 1)),
        dims=("y", "x"),
    )
    out = tmp_path / "tmp_2075_empty_dask_0x5.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value).lower()
    assert "height" in msg or "empty" in msg or "(0, 5)" in msg
    assert not out.exists()
