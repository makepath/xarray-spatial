"""Regression tests for issue #1732.

The stripped-TIFF fallback inside ``read_geotiff_gpu`` previously called
``read_to_array(source, overview_level=overview_level)`` and threw away
the caller's ``max_pixels``, ``window``, and ``band`` arguments. That
meant:

- a user-supplied ``max_pixels`` safety cap was silently ignored on
  stripped files (the default ~1B pixel cap applied instead),
- windowed reads decoded the entire image before slicing on the GPU, and
- single-band selection on a multi-band stripped file still decoded
  every band on the CPU.

These tests assert that all three kwargs are now forwarded to
``read_to_array`` so the stripped GPU path matches the contract of the
tiled GPU path.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
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
def test_stripped_max_pixels_cap_is_enforced():
    """max_pixels smaller than the file must raise before full decode."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(64, 96)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_cap.tif')
        to_geotiff(da, p, tiled=False)
        # 64 * 96 = 6144 pixels; cap at 1000 must reject.
        with pytest.raises(ValueError, match="max_pixels|pixel"):
            read_geotiff_gpu(p, max_pixels=1000)


@_gpu_only
def test_stripped_window_returns_only_window():
    """Windowed read on a stripped file returns the window-sized array
    with coords and transform that match the window origin.

    The post-decode ``_gpu_apply_window_band`` call was replaced with a
    coord-only computation in #1732. Compare against the CPU eager path
    (which is the parity reference for this exact fixture) so a
    regression in the coord-only branch -- or a drift in the windowed
    ``attrs['transform']`` -- shows up here.
    """
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(64, 96)).astype(np.uint8)
    # Explicit y/x coords give the file a real georef so the coord-only
    # path computes a non-trivial windowed transform / origin -- a plain
    # ``dims=['y','x']`` array writes a no-georef TIFF where the coord
    # branch degenerates to integer arange and would not catch a
    # transform-math regression.
    da = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={
            'y': np.arange(64, dtype=np.float64) * 0.5 + 100.0,
            'x': np.arange(96, dtype=np.float64) * 0.5 + 200.0,
        },
        attrs={'crs': 4326},
    )

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_win.tif')
        to_geotiff(da, p, tiled=False)
        win = (8, 16, 40, 80)  # 32x64 window
        out = read_geotiff_gpu(p, window=win)
        assert out.shape == (32, 64)
        np.testing.assert_array_equal(out.data.get(), data[8:40, 16:80])

        # Coords + transform parity vs the CPU eager path. CPU runs
        # through ``open_geotiff``'s own windowed-coord branch (line
        # ~833), so any drift between the two coord computations is
        # caught here.
        cpu = open_geotiff(p, window=win)
        np.testing.assert_array_equal(out.coords['y'].values,
                                      cpu.coords['y'].values)
        np.testing.assert_array_equal(out.coords['x'].values,
                                      cpu.coords['x'].values)
        # ``attrs['transform']`` carries the windowed origin (origin_x
        # shifted by c0 * pixel_width, origin_y by r0 * pixel_height).
        # Pin the exact tuple as well as parity with CPU so a regression
        # in either ``_populate_attrs_from_geo_info`` or the coord-only
        # branch is visible.
        assert out.attrs['transform'] == cpu.attrs['transform']
        # pixel_width=0.5, origin_x=200, c0=16  -> 200 + 16*0.5 = 208
        # pixel_height=0.5, origin_y=100-0.5*0.5=99.75 (PixelIsArea),
        # r0=8 -> 99.75 + 8*0.5 = 103.75
        # to_geotiff writes the raw geo-transform (edge origin), so:
        #   origin_x_raw = 200 - 0.25 = 199.75; +16*0.5 = 207.75
        #   origin_y_raw = 100 - 0.25 =  99.75; +8*0.5  = 103.75
        assert out.attrs['transform'] == (0.5, 0.0, 207.75,
                                          0.0, 0.5, 103.75)


@_gpu_only
def test_stripped_band_selection_returns_2d():
    """Selecting band=1 on a 3-band stripped file returns a 2D array
    matching the requested band."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(48, 80, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_band.tif')
        to_geotiff(da, p, tiled=False)
        out = read_geotiff_gpu(p, band=1)
        assert out.dims == ('y', 'x')
        assert out.shape == (48, 80)
        np.testing.assert_array_equal(out.data.get(), data[:, :, 1])


@_gpu_only
def test_stripped_window_plus_band():
    """Windowed read with band selection composes correctly."""
    from xrspatial.geotiff import read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(48, 80, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_wb.tif')
        to_geotiff(da, p, tiled=False)
        win = (4, 8, 36, 72)  # 32x64
        out = read_geotiff_gpu(p, window=win, band=2)
        assert out.dims == ('y', 'x')
        assert out.shape == (32, 64)
        np.testing.assert_array_equal(
            out.data.get(), data[4:36, 8:72, 2])
