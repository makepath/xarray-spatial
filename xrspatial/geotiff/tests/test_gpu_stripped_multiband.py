"""Regression tests for ``read_geotiff_gpu`` on stripped multi-band TIFFs.

The stripped fallback inside ``read_geotiff_gpu`` previously hardcoded
``dims=['y', 'x']`` even when the underlying CPU read returned a 3-D
``(y, x, band)`` array for multi-band stripped files. That raised::

    ValueError: dimensions ('y', 'x') must have the same length as the
    number of data dimensions, ndim=3

This module covers the multi-band stripped path and a single-band
sanity check that the 2-D path still works.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest
import xarray as xr


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


@_gpu_only
def test_stripped_3band_uint8():
    """3-band uint8 stripped TIFF reads as (y, x, band)."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 200, size=(64, 96, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt.tif')
        to_geotiff(da, p, tiled=False)
        out = read_geotiff_gpu(p)
        assert out.dims == ('y', 'x', 'band')
        assert out.shape == (64, 96, 3)
        np.testing.assert_array_equal(out.data.get(), data)


@_gpu_only
def test_stripped_2band_uint16():
    """2-band uint16 stripped TIFF reads as (y, x, band)."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 60000, size=(48, 80, 2)).astype(np.uint16)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt2.tif')
        to_geotiff(da, p, tiled=False)
        out = read_geotiff_gpu(p)
        assert out.dims == ('y', 'x', 'band')
        assert out.shape == (48, 80, 2)
        assert out.data.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(out.data.get(), data)


@_gpu_only
def test_stripped_singleband_still_2d():
    """Single-band stripped TIFF still produces a 2-D (y, x) DataArray."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 200, size=(40, 60)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt1.tif')
        to_geotiff(da, p, tiled=False)
        out = read_geotiff_gpu(p)
        assert out.dims == ('y', 'x')
        assert out.shape == (40, 60)
        np.testing.assert_array_equal(out.data.get(), data)
