"""Regression tests for issue #2094.

Before this fix, the GeoTIFF writer entry points validated empty
spatial dims but left the band/sample axis unchecked. A 3D DataArray
with zero bands (band-first ``(0, 5, 5)`` or band-last ``(5, 5, 0)``)
slipped past the pre-layout guard, hit the band-first ``moveaxis``,
and was written as a TIFF whose IFD advertised
``SamplesPerPixel=1`` and ``height=5, width=5``. The file then read
back as a non-empty single-band raster -- silent data fabrication
from an empty input.

The fix runs a shared post-layout shape validator after each writer
normalises layout to its internal band-last convention. The check
fires at every public writer entry point (eager ``to_geotiff``,
direct ``_writer.write`` / ``write_streaming``, and
``write_geotiff_gpu``).
"""
from __future__ import annotations

import importlib.util

import dask.array as dsk
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._writer import write, write_streaming


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
    pytest.param((0, 5, 5), ("band", "y", "x"), id="band-first"),
    pytest.param((5, 5, 0), ("y", "x", "band"), id="band-last"),
]


@pytest.mark.parametrize("shape,dims", _ZERO_BAND_LAYOUTS)
def test_to_geotiff_rejects_zero_band_numpy(tmp_path, shape, dims):
    da = xr.DataArray(np.empty(shape, dtype=np.uint8), dims=dims)
    out = tmp_path / f"tmp_2094_zero_band_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out), compression="none")
    msg = str(excinfo.value)
    # Message must name the public entry point so the traceback is
    # useful, and must call out the offending samples_per_pixel axis
    # so the caller knows which dim went empty.
    assert "to_geotiff" in msg
    assert "zero-band" in msg or "samples_per_pixel" in msg
    assert "samples_per_pixel=0" in msg
    assert not out.exists()


def test_direct_write_rejects_zero_band(tmp_path):
    """``_writer.write`` is the lowest writer in the stack; bare
    callers (tests, the GPU fallback path) must hit the same guard
    so a zero-band IFD cannot reach disk."""
    arr = np.empty((5, 5, 0), dtype=np.uint8)
    out = tmp_path / "tmp_2094_direct_write.tif"
    with pytest.raises(ValueError) as excinfo:
        write(arr, str(out), compression="none")
    msg = str(excinfo.value)
    assert "write" in msg
    assert "samples_per_pixel=0" in msg
    assert not out.exists()


def test_write_streaming_rejects_zero_band(tmp_path):
    """``write_streaming`` derives ``samples`` from ``shape[2]``; a
    zero-band input would otherwise produce an IFD whose
    SamplesPerPixel field is 0."""
    arr = dsk.from_array(np.empty((5, 5, 0), dtype=np.uint8))
    out = tmp_path / "tmp_2094_write_streaming.tif"
    with pytest.raises(ValueError) as excinfo:
        write_streaming(arr, str(out), compression="none")
    msg = str(excinfo.value)
    assert "write_streaming" in msg
    assert "samples_per_pixel=0" in msg
    assert not out.exists()


@pytest.mark.parametrize("shape,dims", _ZERO_BAND_LAYOUTS)
def test_to_geotiff_rejects_zero_band_dask(tmp_path, shape, dims):
    """The streaming dask path runs the post-layout check after
    ``da.moveaxis(raw, 0, -1)``, so a band-first zero-band input is
    caught at the same boundary as a band-last one."""
    chunks = tuple(max(1, s) for s in shape)
    arr = dsk.from_array(np.empty(shape, dtype=np.uint8), chunks=chunks)
    da = xr.DataArray(arr, dims=dims)
    out = tmp_path / f"tmp_2094_zero_band_dask_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out), compression="none")
    msg = str(excinfo.value)
    assert "to_geotiff" in msg
    assert "samples_per_pixel=0" in msg
    assert not out.exists()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
@pytest.mark.parametrize("shape,dims", _ZERO_BAND_LAYOUTS)
def test_write_geotiff_gpu_rejects_zero_band(tmp_path, shape, dims):
    """``write_geotiff_gpu`` is a public entry point that does not
    funnel through ``to_geotiff``; verify the post-layout guard
    fires there too."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    da = xr.DataArray(cp.empty(shape, dtype=cp.uint8), dims=dims)
    out = tmp_path / f"tmp_2094_zero_band_gpu_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError) as excinfo:
        write_geotiff_gpu(da, str(out), compression="none")
    msg = str(excinfo.value)
    assert "write_geotiff_gpu" in msg
    assert "samples_per_pixel=0" in msg
    assert not out.exists()


def test_to_geotiff_still_accepts_valid_3d(tmp_path):
    """Positive control: a non-empty 3D DataArray still writes
    successfully through both layout conventions."""
    for shape, dims in [
        ((3, 5, 5), ("band", "y", "x")),
        ((5, 5, 3), ("y", "x", "band")),
    ]:
        da = xr.DataArray(
            np.zeros(shape, dtype=np.uint8), dims=dims,
        )
        out = tmp_path / f"tmp_2094_valid_{'_'.join(dims)}.tif"
        to_geotiff(da, str(out), compression="none")
        assert out.exists() and out.stat().st_size > 0
