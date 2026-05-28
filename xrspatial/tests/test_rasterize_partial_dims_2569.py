"""Tests for issue #2569: partial width/height overrides with ``like=``
must raise rather than be silently ignored.

The previous behaviour silently fell through to the ``like`` branch when
only one of ``width`` or ``height`` was provided, so
``rasterize(..., like=template, width=W)`` returned the template's
width with no warning.  See the docstring change on ``rasterize`` for
the documented contract: width and height are overridden together, or
not at all.
"""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from xrspatial.rasterize import rasterize


def _make_like(width=10, height=10):
    """2D template DataArray with georeferenced descending-y coords."""
    x = np.linspace(0.5, width - 0.5, width)
    y = np.linspace(height - 0.5, 0.5, height)
    data = np.zeros((height, width), dtype=np.float64)
    return xr.DataArray(
        data, dims=['y', 'x'], coords={'y': y, 'x': x},
    )


class TestPartialDimsWithLike:
    """``like=`` + exactly one of ``width`` / ``height`` must raise."""

    def test_like_plus_both_dims_honored(self):
        # Sanity: explicit width AND height override the template size.
        like = _make_like(width=10, height=10)
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, width=5, height=4,
            fill=0,
        )
        assert result.sizes == {'y': 4, 'x': 5}

    def test_like_plus_only_width_raises(self):
        like = _make_like()
        with pytest.raises(ValueError, match="height was not"):
            rasterize(
                [(box(2, 2, 8, 8), 1.0)], like=like, width=2, fill=0,
            )

    def test_like_plus_only_height_raises(self):
        like = _make_like()
        with pytest.raises(ValueError, match="width was not"):
            rasterize(
                [(box(2, 2, 8, 8), 1.0)], like=like, height=2, fill=0,
            )

    def test_like_plus_neither_uses_like_grid(self):
        like = _make_like(width=10, height=10)
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        assert result.sizes == {'y': 10, 'x': 10}
        # Output coords reuse the template's coords bit-exactly.
        np.testing.assert_array_equal(
            result.coords['x'].values, like.coords['x'].values,
        )
        np.testing.assert_array_equal(
            result.coords['y'].values, like.coords['y'].values,
        )


class TestPartialDimsNoLike:
    """Partial width/height without ``like`` must also raise -- the
    fall-through used to land on ``resolution`` or the final "must
    specify ..." error, both of which dropped the explicit dimension
    without a clear message.
    """

    def test_only_width_with_resolution_raises(self):
        # Even with resolution providing the size, a half-specified
        # width/height is still a contract violation.
        with pytest.raises(ValueError, match="height was not"):
            rasterize(
                [(box(2, 2, 8, 8), 1.0)],
                bounds=(0, 0, 10, 10), width=5, resolution=1.0, fill=0,
            )

    def test_only_height_alone_raises(self):
        with pytest.raises(ValueError, match="width was not"):
            rasterize(
                [(box(2, 2, 8, 8), 1.0)],
                bounds=(0, 0, 10, 10), height=5, fill=0,
            )
