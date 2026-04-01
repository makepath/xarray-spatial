import numpy as np
import pytest
import xarray as xr

from xrspatial.visibility import _bresenham_line, _extract_transect


class TestBresenhamLine:
    def test_horizontal(self):
        cells = _bresenham_line(0, 0, 0, 4)
        assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_vertical(self):
        cells = _bresenham_line(0, 0, 4, 0)
        assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_diagonal(self):
        cells = _bresenham_line(0, 0, 3, 3)
        assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_single_cell(self):
        cells = _bresenham_line(2, 3, 2, 3)
        assert cells == [(2, 3)]

    def test_steep_negative(self):
        cells = _bresenham_line(4, 2, 0, 0)
        # Must start at (4, 2) and end at (0, 0)
        assert cells[0] == (4, 2)
        assert cells[-1] == (0, 0)
        assert len(cells) == 5

    def test_includes_endpoints(self):
        cells = _bresenham_line(1, 1, 5, 8)
        assert cells[0] == (1, 1)
        assert cells[-1] == (5, 8)


def _make_raster(data):
    """Module-level helper for creating test rasters."""
    h, w = data.shape
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.arange(h, dtype=float),
                'x': np.arange(w, dtype=float)},
    )


class TestExtractTransect:
    def test_numpy_diagonal(self):
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster = _make_raster(data)
        cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        elev, xs, ys = _extract_transect(raster, cells)
        np.testing.assert_array_equal(elev, [0, 6, 12, 18, 24])
        np.testing.assert_array_equal(xs, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(ys, [0, 1, 2, 3, 4])

    def test_dask_matches_numpy(self):
        import dask.array as da
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster_np = _make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(3, 3))
        cells = [(0, 0), (2, 3), (4, 4)]
        elev_np, _, _ = _extract_transect(raster_np, cells)
        elev_da, _, _ = _extract_transect(raster_dask, cells)
        np.testing.assert_array_equal(elev_np, elev_da)
