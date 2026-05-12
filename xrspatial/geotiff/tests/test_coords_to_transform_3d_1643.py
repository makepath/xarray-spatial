"""Regression test for issue #1643.

``_coords_to_transform`` previously used ``dims[-2]`` and ``dims[-1]`` to
look up y/x coords. On a 3D ``(y, x, band)`` DataArray that picked
``x`` and ``band``, so ``to_geotiff`` silently wrote a wrong
GeoTransform when ``attrs['transform']`` was absent. The helper now
detects the band-like trailing/leading dim and uses the two spatial
dims regardless of position.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _coords_to_transform, open_geotiff, to_geotiff

try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _make_geo_da_3d(dims):
    """3D DataArray with georeferenced y/x coords and a band axis."""
    shape = []
    for d in dims:
        if d in ('y',):
            shape.append(10)
        elif d in ('x',):
            shape.append(20)
        else:
            shape.append(3)
    arr = np.arange(int(np.prod(shape)), dtype=np.uint8).reshape(shape)
    coords = {
        'y': np.linspace(100.0, 200.0, 10),
        'x': np.linspace(500.0, 700.0, 20),
        'band': np.arange(3),
    }
    return xr.DataArray(arr, dims=list(dims), coords=coords)


def test_coords_to_transform_yxband_returns_yx_spacing():
    """3D (y, x, band) picks y/x spacing rather than (x, band) spacing."""
    da = _make_geo_da_3d(('y', 'x', 'band'))
    gt = _coords_to_transform(da)
    # y spacing = (200 - 100) / 9, x spacing = (700 - 500) / 19
    assert gt is not None
    np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
    np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)


def test_coords_to_transform_bandyx_returns_yx_spacing():
    """3D (band, y, x) also returns the y/x transform."""
    da = _make_geo_da_3d(('band', 'y', 'x'))
    gt = _coords_to_transform(da)
    assert gt is not None
    np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
    np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)


def test_coords_to_transform_2d_unchanged():
    """2D (y, x) keeps its original behaviour."""
    da = xr.DataArray(
        np.zeros((10, 20), dtype=np.uint8),
        dims=['y', 'x'],
        coords={
            'y': np.linspace(100.0, 200.0, 10),
            'x': np.linspace(500.0, 700.0, 20),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
    np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)


def test_to_geotiff_roundtrip_3d_yxband_no_transform_attr(tmp_path):
    """to_geotiff -> open_geotiff round-trip on 3D arrays preserves coords.

    Before the fix the on-disk transform was derived from (x, band)
    spacing, so the round-tripped y/x coords had wrong pixel size and
    origin. After the fix the 3D output matches the 2D output.
    """
    da_3d = _make_geo_da_3d(('y', 'x', 'band'))
    da_2d = xr.DataArray(
        np.zeros((10, 20), dtype=np.uint8),
        dims=['y', 'x'],
        coords={
            'y': np.linspace(100.0, 200.0, 10),
            'x': np.linspace(500.0, 700.0, 20),
        },
    )

    p2 = str(tmp_path / 'roundtrip_1643_2d.tif')
    p3 = str(tmp_path / 'roundtrip_1643_3d.tif')
    to_geotiff(da_2d, p2)
    to_geotiff(da_3d, p3)

    rt2 = open_geotiff(p2)
    rt3 = open_geotiff(p3)
    np.testing.assert_allclose(rt3.y.values, rt2.y.values)
    np.testing.assert_allclose(rt3.x.values, rt2.x.values)
    assert rt3.attrs.get('transform') == rt2.attrs.get('transform')


def test_to_geotiff_roundtrip_3d_bandyx_no_transform_attr(tmp_path):
    """(band, y, x) input round-trips with the correct transform.

    ``to_geotiff`` remaps a (band, y, x) input to (y, x, band) before
    writing, but ``_coords_to_transform`` runs against the original
    dim order. The fix handles both 3D layouts.
    """
    da_3d = _make_geo_da_3d(('band', 'y', 'x'))
    da_2d = xr.DataArray(
        np.zeros((10, 20), dtype=np.uint8),
        dims=['y', 'x'],
        coords={
            'y': np.linspace(100.0, 200.0, 10),
            'x': np.linspace(500.0, 700.0, 20),
        },
    )

    p2 = str(tmp_path / 'roundtrip_1643_2d_b.tif')
    p3 = str(tmp_path / 'roundtrip_1643_3d_bandfirst.tif')
    to_geotiff(da_2d, p2)
    to_geotiff(da_3d, p3)

    rt2 = open_geotiff(p2)
    rt3 = open_geotiff(p3)
    np.testing.assert_allclose(rt3.y.values, rt2.y.values)
    np.testing.assert_allclose(rt3.x.values, rt2.x.values)


def test_to_geotiff_3d_without_transform_attr_does_not_invent_unit_pixels(
        tmp_path):
    """Regression sanity: the bad transform was pixel_width=1.0 (band
    axis spacing). Assert the round-tripped pixel_width is finite,
    non-unit, and matches the source x spacing.
    """
    da = _make_geo_da_3d(('y', 'x', 'band'))
    p = str(tmp_path / 'roundtrip_1643_3d_not_unit.tif')
    to_geotiff(da, p)
    rt = open_geotiff(p)
    pw = abs(float(rt.x.values[1] - rt.x.values[0]))
    # Source x spacing is (700-500)/19 = ~10.526. The buggy path would
    # have produced pw=1.0 (the band axis spacing).
    assert pw > 1.5, (
        f"round-tripped pixel_width={pw} suggests the band-axis spacing "
        f"leaked into the GeoTransform; expected ~10.526")


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_write_geotiff_gpu_roundtrip_3d_no_transform_attr(tmp_path):
    """GPU writer shares ``_coords_to_transform`` with the CPU writer.

    Same regression on the GPU path: a 3D ``(y, x, band)`` cupy
    DataArray without ``attrs['transform']`` would previously round-trip
    through a unit pixel-width transform.
    """
    import cupy as cp

    from xrspatial.geotiff import write_geotiff_gpu

    np_arr = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    da = xr.DataArray(
        cp.asarray(np_arr),
        dims=['y', 'x', 'band'],
        coords={
            'y': np.linspace(100.0, 200.0, 10),
            'x': np.linspace(500.0, 700.0, 20),
            'band': np.arange(3),
        },
    )
    p = str(tmp_path / 'roundtrip_1643_3d_gpu.tif')
    write_geotiff_gpu(da, p)
    rt = open_geotiff(p)
    pw = abs(float(rt.x.values[1] - rt.x.values[0]))
    assert pw > 1.5, (
        f"GPU writer round-tripped pixel_width={pw}; expected ~10.526")
    ph = abs(float(rt.y.values[1] - rt.y.values[0]))
    assert ph > 1.5, (
        f"GPU writer round-tripped pixel_height={ph}; expected ~11.111")
