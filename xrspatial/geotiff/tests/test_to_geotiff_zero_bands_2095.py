"""Regression tests for issue #2095.

``to_geotiff`` validated the spatial axes of a 3D writer input but not
the band/sample axis. A DataArray of shape ``(0, y, x)`` band-first or
``(y, x, 0)`` band-last passed every guard and reached the IFD assembly
with ``samples_per_pixel == 0``. The resulting TIFF read back as a 2D
single-band raster, masking the upstream collapse of the band axis --
silent data fabrication.

The fix raises ``ValueError`` at the writer entry point on both layouts
and on every public writer surface (``to_geotiff``, ``write``,
``write_streaming``, and ``write_geotiff_gpu``). The message names the
offending axis so callers know what went empty.
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


_ZERO_BAND_LAYOUTS = [
    pytest.param(
        (0, 5, 5),
        ("band", "y", "x"),
        id="band-first",
    ),
    pytest.param(
        (5, 5, 0),
        ("y", "x", "band"),
        id="band-last",
    ),
]


@pytest.mark.parametrize("shape,dims", _ZERO_BAND_LAYOUTS)
def test_to_geotiff_rejects_zero_bands_numpy(tmp_path, shape, dims):
    da = xr.DataArray(np.zeros(shape, dtype=np.uint8), dims=dims)
    out = tmp_path / f"tmp_2095_zerobands_{'_'.join(map(str, shape))}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value)
    assert "to_geotiff" in msg
    assert "no bands" in msg.lower() or "0 bands" in msg
    # Nothing should have been written.
    assert not out.exists()


@pytest.mark.parametrize("shape,dims", _ZERO_BAND_LAYOUTS)
def test_to_geotiff_rejects_zero_bands_dask(tmp_path, shape, dims):
    # Dask cannot construct an array with a zero-length chunk along a
    # zero-length dim, so build the dask array with chunks of 1 on the
    # spatial axes and 1 on the band axis if non-zero. We only need the
    # validator to fire before any compute happens.
    chunks = tuple(1 if s == 0 else s for s in shape)
    arr = dsk.zeros(shape, dtype=np.uint8, chunks=chunks)
    da = xr.DataArray(arr, dims=dims)
    out = tmp_path / f"tmp_2095_zerobands_dask_{'_'.join(map(str, shape))}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value).lower()
    assert "band" in msg
    assert not out.exists()


def test_write_band_last_zero_bands_direct(tmp_path):
    """``write`` is a public entry point. Direct callers (no DataArray
    wrapper, no dims) pass raw numpy arrays through the band-last
    convention, so a ``(y, x, 0)`` array must fail closed here too."""
    from xrspatial.geotiff._writer import write

    arr = np.zeros((5, 5, 0), dtype=np.uint8)
    out = tmp_path / "tmp_2095_write_zerobands.tif"
    with pytest.raises(ValueError) as excinfo:
        write(arr, str(out))
    msg = str(excinfo.value)
    # The error template starts with ``"<entry_point> cannot write a
    # raster with no bands"``. Anchor to that exact prefix so the
    # assertion fails if the wrong entry point fires (every message
    # also contains the substring "write" further on, so an `in`
    # check would not distinguish ``write`` from ``write_streaming``
    # or ``write_geotiff_gpu``).
    assert msg.startswith("write cannot write")
    assert "0 bands" in msg or "no bands" in msg.lower()
    assert not out.exists()


def test_write_streaming_zero_bands_direct(tmp_path):
    """``write_streaming`` is the dask-aware entry point. Direct callers
    pass band-last dask arrays, so a ``(y, x, 0)`` chunked array must
    fail closed before any tile-row math runs."""
    from xrspatial.geotiff._writer import write_streaming

    arr = dsk.zeros((5, 5, 0), dtype=np.uint8, chunks=(5, 5, 1))
    out = tmp_path / "tmp_2095_write_streaming_zerobands.tif"
    with pytest.raises(ValueError) as excinfo:
        write_streaming(arr, str(out))
    msg = str(excinfo.value)
    assert msg.startswith("write_streaming cannot write")
    assert "0 bands" in msg or "no bands" in msg.lower()
    assert not out.exists()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_write_geotiff_gpu_rejects_zero_bands(tmp_path):
    """The GPU writer is a separate public entry point. The zero-band
    guard must fire there too without dispatching any GPU work."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    arr = xr.DataArray(
        cp.zeros((0, 5, 5), dtype=cp.uint8),
        dims=("band", "y", "x"),
    )
    out = tmp_path / "tmp_2095_zerobands_gpu.tif"
    with pytest.raises(ValueError) as excinfo:
        write_geotiff_gpu(arr, str(out))
    msg = str(excinfo.value)
    assert msg.startswith("write_geotiff_gpu cannot write")
    assert "0 bands" in msg or "no bands" in msg.lower()
    assert not out.exists()
