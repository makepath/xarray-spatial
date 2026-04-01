import numpy as np
import pytest
import xarray as xr

from xrspatial.visibility import _bresenham_line


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
