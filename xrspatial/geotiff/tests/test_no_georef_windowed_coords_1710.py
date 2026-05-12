"""Regression tests for issue #1710.

A TIFF with no GeoTIFF tags (no ModelPixelScale / ModelTiepoint /
GeoKeyDirectory) used to produce different `y`/`x` coordinates between
full reads and windowed reads. Full reads emitted integer pixel coords
``[0, 1, 2, ...]`` via the ``has_georef=False`` shortcut in
``_geo_to_coords``; windowed reads bypassed that shortcut and synthesised
float64 coords like ``[-0.5, -1.5, ...]`` from the default unit
``GeoTransform``.

The fix routes every windowed-read path through the same shortcut, and
suppresses the fabricated ``attrs['transform']`` for non-georef files.
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
def no_georef_path_1710(tmp_path):
    """8x8 float32 TIFF with NO GeoTIFF tags (no ModelPixelScale etc.)."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "no_georef_1710.tif")
    write(arr, path, compression='none', tiled=False)
    return path


class TestEagerWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710)
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))
        np.testing.assert_array_equal(da.x.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        """Windowed read at (2, 3) origin should give coords [2,3,4,5] / [3,4,5,6]."""
        da = open_geotiff(no_georef_path_1710, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_no_transform_attr_emitted(self, no_georef_path_1710):
        """Non-georef files should not advertise a fabricated transform."""
        da = open_geotiff(no_georef_path_1710)
        assert 'transform' not in da.attrs, (
            f"non-georef file should not carry a fabricated transform; "
            f"got attrs={dict(da.attrs)}"
        )
        # Same for windowed read
        da_w = open_geotiff(no_georef_path_1710, window=(0, 0, 4, 4))
        assert 'transform' not in da_w.attrs


class TestDaskWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4)
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))


@_gpu_only
class TestGpuWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True)
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_dask_cupy_windowed_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True, chunks=4,
                          window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        """GPU windowed read at a non-zero origin: the stripped-GPU
        fallback in ``read_geotiff_gpu`` checked ``t is None`` instead
        of ``has_georef`` (issue #1753 / regression of #1710), so a
        non-georef TIFF emitted ``[-0.5, -1.5, ...]`` via the unit
        ``GeoTransform`` placeholder. Pin the contract that the offset
        window produces file-relative integer coords identical to the
        eager numpy path.
        """
        da = open_geotiff(no_georef_path_1710, gpu=True,
                          window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_offset_window_no_transform_attr(self, no_georef_path_1710):
        """Non-georef GPU windowed read still must not advertise a
        fabricated ``attrs['transform']`` -- ``_populate_attrs_from_geo_info``
        gates on ``has_georef`` too, so flag the round-trip here.
        """
        da = open_geotiff(no_georef_path_1710, gpu=True,
                          window=(2, 3, 6, 7))
        assert 'transform' not in da.attrs, (
            f"non-georef GPU windowed read should not carry a "
            f"fabricated transform; got attrs={dict(da.attrs)}"
        )


class TestBackendParity:
    """Full read and windowed read must agree on coord dtype and values
    across every backend pair."""

    def test_dtype_parity_full(self, no_georef_path_1710):
        np_da = open_geotiff(no_georef_path_1710)
        dk_da = open_geotiff(no_georef_path_1710, chunks=4)
        assert np_da.y.dtype == dk_da.y.dtype == np.int64
        if _HAS_GPU:
            gpu_da = open_geotiff(no_georef_path_1710, gpu=True)
            assert gpu_da.y.dtype == np.int64

    def test_dtype_parity_windowed(self, no_georef_path_1710):
        win = (0, 0, 4, 4)
        np_da = open_geotiff(no_georef_path_1710, window=win)
        dk_da = open_geotiff(no_georef_path_1710, chunks=4, window=win)
        assert np_da.y.dtype == dk_da.y.dtype == np.int64
        if _HAS_GPU:
            gpu_da = open_geotiff(no_georef_path_1710, gpu=True, window=win)
            assert gpu_da.y.dtype == np.int64

    def test_full_and_windowed_first_n_match(self, no_georef_path_1710):
        """Coords for a window starting at (0, 0) should equal the first N
        coords of the full read."""
        win = (0, 0, 4, 4)
        full = open_geotiff(no_georef_path_1710)
        win_da = open_geotiff(no_georef_path_1710, window=win)
        np.testing.assert_array_equal(full.y.values[:4], win_da.y.values)
        np.testing.assert_array_equal(full.x.values[:4], win_da.x.values)


class TestGeorefStillWorks:
    """The fix must not regress georeferenced reads.

    Files with real GeoTIFF tags still get float64 coords with the
    half-pixel shift for PixelIsArea.
    """

    def test_georef_full_read_keeps_float64(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_1710.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path)
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        assert da.x.values[0] == pytest.approx(500015.0)
        # transform attr SHOULD be present for georef files
        assert 'transform' in da.attrs

    def test_georef_windowed_read_keeps_float64(self, tmp_path):
        from xrspatial.geotiff._geotags import GeoTransform
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_win_1710.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, window=(0, 0, 4, 4))
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        assert 'transform' in da.attrs
