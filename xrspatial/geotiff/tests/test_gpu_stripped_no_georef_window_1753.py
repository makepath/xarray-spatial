"""Regression tests for issue #1753.

The GPU stripped fallback in ``read_geotiff_gpu`` -- introduced in PR
#1738 (issue #1732) so the GPU path forwards ``window=`` /
``max_pixels`` / ``band=`` to the CPU helper -- did not honour the
``has_georef=False`` shortcut for windowed reads. Before this fix, a
non-georef TIFF read via ``open_geotiff(path, gpu=True, window=...)``
returned float64 coords synthesised from the default unit
``GeoTransform`` (e.g. ``[-0.5, -1.5, ...]``) while the matching CPU
eager + dask paths returned integer pixel coords
``np.arange(c0, c1, dtype=np.int64)``.

The fix routes the ``has_georef=False`` case through the same integer
coord shortcut on the GPU stripped fallback. PixelIsArea and
PixelIsPoint georef files still get float64 coords with the correct
shift.

The original PR #1722 (issue #1710) added the ``has_georef`` guard to
the eager numpy path, the dask path, and the tiled GPU path. PR #1738
re-introduced the bug on the stripped GPU fallback only.

Mirrors ``test_no_georef_windowed_coords_1710.py`` (which already
covered the CPU and tiled GPU paths) and adds an explicit
offset-window check for the stripped GPU fallback the prior test did
not exercise.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._writer import write


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


@pytest.fixture
def no_georef_stripped_path_1753(tmp_path):
    """8x8 float32 stripped TIFF with NO GeoTIFF tags.

    Stripped layout (``tiled=False``) routes ``read_geotiff_gpu``
    through the CPU fallback at line 2696 of ``geotiff/__init__.py``,
    which is the code path #1753 regresses.
    """
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "no_georef_stripped_1753.tif")
    write(arr, path, compression='none', tiled=False)
    return path


@pytest.fixture
def no_georef_stripped_uint16_path_1753(tmp_path):
    """8x8 uint16 stripped TIFF with NO GeoTIFF tags.

    Used to confirm the fix does not depend on dtype and to exercise
    the multi-band stripped GPU fallback path.
    """
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = str(tmp_path / "no_georef_stripped_uint16_1753.tif")
    write(arr, path, compression='none', tiled=False)
    return path


@_gpu_only
class TestStrippedGpuWindowedNoGeoref:
    """GPU stripped fallback must emit integer pixel coords on no-georef
    windowed reads, matching CPU eager + dask + tiled GPU."""

    def test_window_at_origin(self, no_georef_stripped_path_1753):
        """Window at (0, 0) -> coords ``np.arange(0, 4)``."""
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64, (
            f"y.dtype should be int64 for no-georef; got {da.y.dtype}"
        )
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_offset_window(self, no_georef_stripped_path_1753):
        """Offset window at (2, 3) -> coords ``np.arange(2, 6) / (3, 7)``."""
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        # Integer pixel coords must reflect the window offset -- a
        # window at (2, 3) returns coords [2, 3, 4, 5] / [3, 4, 5, 6],
        # not [0, 1, 2, 3] / [0, 1, 2, 3].
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_dask_cupy_window(self, no_georef_stripped_path_1753):
        """``gpu=True, chunks=...`` shares the same fallback coord path."""
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          chunks=4, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_dask_cupy_offset_window(self, no_georef_stripped_path_1753):
        """``gpu=True, chunks=...`` with an offset window."""
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          chunks=4, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_no_transform_attr(self, no_georef_stripped_path_1753):
        """No fabricated ``attrs['transform']`` for non-georef windowed
        reads on GPU. ``_populate_attrs_from_geo_info`` already skipped
        this for ``has_georef=False``; verify the fix preserved that
        invariant on the stripped GPU fallback."""
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(0, 0, 4, 4))
        assert 'transform' not in da.attrs, (
            f"non-georef stripped GPU windowed read should not carry a "
            f"transform attr; got {dict(da.attrs)}"
        )

    def test_uint16_window(self, no_georef_stripped_uint16_path_1753):
        """Fix is dtype-independent; uint16 source still gets int64 coords."""
        da = open_geotiff(no_georef_stripped_uint16_path_1753, gpu=True,
                          window=(1, 2, 5, 6))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(1, 5))
        np.testing.assert_array_equal(da.x.values, np.arange(2, 6))


@_gpu_only
class TestStrippedGpuWindowedBackendParity:
    """Every backend must agree on coord dtype and values for the same
    windowed read of a non-georef stripped TIFF.

    Catches the bug at the boundary the test_no_georef_windowed_coords
    parity check almost covered: that test exercised parity for the
    (0, 0)-origin window. The offset-window case here would catch the
    issue earlier because the wrong-coord output is more obviously
    wrong than a zero-origin one (``[-0.5, -1.5]`` vs ``[0, 1]``).
    """

    @pytest.mark.parametrize("win", [
        (0, 0, 4, 4),
        (2, 3, 6, 7),
        (1, 1, 7, 7),
    ])
    def test_dtype_parity(self, no_georef_stripped_path_1753, win):
        eager = open_geotiff(no_georef_stripped_path_1753, window=win)
        dask = open_geotiff(no_georef_stripped_path_1753, chunks=4, window=win)
        gpu = open_geotiff(no_georef_stripped_path_1753, gpu=True, window=win)
        gpu_dk = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                              chunks=4, window=win)
        assert eager.y.dtype == dask.y.dtype == gpu.y.dtype == gpu_dk.y.dtype
        assert eager.y.dtype == np.int64
        # The #1753 regression affected both axes (float64 coords from
        # the default unit GeoTransform on x as well as y); assert
        # x-axis parity + int64 so a future change can't silently
        # return int-valued float coords on x while passing the value
        # equality check below.
        assert eager.x.dtype == dask.x.dtype == gpu.x.dtype == gpu_dk.x.dtype
        assert eager.x.dtype == np.int64

    @pytest.mark.parametrize("win", [
        (0, 0, 4, 4),
        (2, 3, 6, 7),
        (1, 1, 7, 7),
    ])
    def test_value_parity(self, no_georef_stripped_path_1753, win):
        eager = open_geotiff(no_georef_stripped_path_1753, window=win)
        gpu = open_geotiff(no_georef_stripped_path_1753, gpu=True, window=win)
        np.testing.assert_array_equal(eager.y.values, gpu.y.values)
        np.testing.assert_array_equal(eager.x.values, gpu.x.values)


@_gpu_only
class TestStrippedGpuWindowedGeorefStillWorks:
    """The fix must not regress georeferenced stripped GPU reads.

    The pre-fix code already handled PixelIsArea (with the half-pixel
    shift) and PixelIsPoint correctly; reaffirm that here so a future
    refactor cannot reintroduce a coord regression on georef files via
    the same code path that #1753 fixed.
    """

    def test_georef_pixel_is_area_window(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_pixel_is_area_1753.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, gpu=True, window=(0, 0, 4, 4))
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        # PixelIsArea: x[0] = origin_x + 0.5 * pixel_width = 500015.0
        assert da.x.values[0] == pytest.approx(500015.0)
        assert da.y.values[0] == pytest.approx(3999985.0)
        # transform attr SHOULD still be present
        assert 'transform' in da.attrs

    def test_georef_offset_window(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_offset_1753.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, gpu=True, window=(2, 3, 6, 7))
        # PixelIsArea: x[0] = origin_x + (c0 + 0.5) * pixel_width
        # = 500000 + (3 + 0.5) * 30 = 500105.0
        assert da.x.values[0] == pytest.approx(500105.0)
        # y[0] = origin_y + (r0 + 0.5) * pixel_height
        # = 4000000 + (2 + 0.5) * -30 = 3999925.0
        assert da.y.values[0] == pytest.approx(3999925.0)
