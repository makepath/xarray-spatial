"""Regression test for issue #1720.

``_coords_to_transform`` previously read ``x[1] - x[0]`` and
``y[1] - y[0]`` as the pixel sizes without checking that the rest of
the spacing matched. GeoTIFF supports only an affine transform, so
non-uniform coords cannot be expressed faithfully; the writer would
silently use the first-step spacing and produce a wrong transform.

The fix validates ``np.diff(x)`` and ``np.diff(y)`` against a relative
tolerance of ``1e-6`` of the median step and raises ``ValueError`` if
spacing is non-uniform.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _coords_to_transform, to_geotiff


def _make_da(x_coords, y_coords):
    arr = np.zeros((len(y_coords), len(x_coords)), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.asarray(y_coords), 'x': np.asarray(x_coords)},
    )


def test_uniform_coords_ok():
    """Uniform coords write successfully (no regression)."""
    da = _make_da(np.linspace(500.0, 700.0, 20), np.linspace(100.0, 200.0, 10))
    gt = _coords_to_transform(da)
    assert gt is not None
    np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
    np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)


def test_uniform_coords_roundtrip_to_geotiff_1720(tmp_path):
    """End-to-end write succeeds on uniform coords."""
    da = _make_da(np.linspace(500.0, 700.0, 20), np.linspace(100.0, 200.0, 10))
    p = str(tmp_path / 'uniform_1720.tif')
    to_geotiff(da, p)


def test_non_uniform_x_raises_1720():
    """Non-uniform x coords raise ValueError naming x."""
    # Mostly-uniform x with one stretched gap to expose the bug.
    x = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    y = np.linspace(0.0, 10.0, 11)
    da = _make_da(x, y)
    with pytest.raises(ValueError, match=r"\bx coords are not uniformly spaced"):
        _coords_to_transform(da)


def test_non_uniform_y_raises_1720():
    """Non-uniform y coords raise ValueError naming y."""
    x = np.linspace(0.0, 10.0, 11)
    y = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    da = _make_da(x, y)
    with pytest.raises(ValueError, match=r"\by coords are not uniformly spaced"):
        _coords_to_transform(da)


def test_jitter_within_tolerance_ok_1720():
    """Float jitter within 1e-6 relative tolerance writes successfully."""
    x = np.linspace(0.0, 100.0, 11)
    # Add jitter at ~1e-8 relative scale — well below the 1e-6 threshold.
    rng = np.random.default_rng(0)
    x = x + rng.uniform(-1e-7, 1e-7, size=x.shape)
    y = np.linspace(0.0, 50.0, 6)
    da = _make_da(x, y)
    gt = _coords_to_transform(da)
    assert gt is not None


def test_jitter_just_above_tolerance_raises_1720():
    """Jitter just above the 1e-6 relative tolerance raises ValueError."""
    # Step size is 10; one diff is 10 + 1e-4 -> relative deviation ~1e-5,
    # which exceeds the 1e-6 threshold.
    x = np.array([0.0, 10.0, 20.0001, 30.0001, 40.0001])
    y = np.linspace(0.0, 50.0, 6)
    da = _make_da(x, y)
    with pytest.raises(ValueError, match=r"max relative deviation"):
        _coords_to_transform(da)


def test_two_sample_coords_ok_1720():
    """Two-sample coords (one diff) trivially pass the regularity check."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 5.0])
    da = _make_da(x, y)
    gt = _coords_to_transform(da)
    assert gt is not None
    np.testing.assert_allclose(gt.pixel_width, 10.0)
    np.testing.assert_allclose(gt.pixel_height, 5.0)


def test_constant_coords_raises_1720():
    """Constant coords (step == 0) raise ValueError."""
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y = np.linspace(0.0, 10.0, 4)
    da = _make_da(x, y)
    with pytest.raises(ValueError, match=r"x coords are constant"):
        _coords_to_transform(da)
