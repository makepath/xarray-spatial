"""Issue #2568 -- rasterize(like=...) descending-x orientation.

The rasterizer always burns with column 0 = xmin (standard image
convention).  When the template's x axis is descending (high-to-low),
the burned array has to be flipped along axis 1 so result.sel(x=...)
lines up with the geometry in world coordinates, and result.x still
equals like.x.

Mirror of the y-axis tests added for #2170.
"""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from xrspatial.rasterize import rasterize


try:
    import dask.array  # noqa: F401
    has_dask = True
except Exception:
    has_dask = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason='CUDA / CuPy not available')
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason='dask not installed')


def _like_2568(x_descending, y_descending=True, width=4, height=4):
    """Build a 4x4 like-grid with the requested x and y orientations.

    Default y is descending (the standard image convention used
    throughout the project).
    """
    if x_descending:
        x = np.linspace(width - 0.5, 0.5, width)
    else:
        x = np.linspace(0.5, width - 0.5, width)
    if y_descending:
        y = np.linspace(height - 0.5, 0.5, height)
    else:
        y = np.linspace(0.5, height - 0.5, height)
    return xr.DataArray(
        np.zeros((height, width), dtype=np.float64),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )


class TestLikeXOrientation2568:
    """Burning into a descending-x like must agree with ascending-x by
    world coordinate, and output.x must round-trip like.x exactly."""

    def test_numpy_descending_x_matches_ascending_by_world_x(self):
        # Box in lower-left corner of the world grid -- with ascending x
        # this lands in column 0, with descending x it lands in the last
        # column, but result.sel(x=0.5) must return 1.0 in both cases.
        geom = [(box(0, 0, 1, 1), 1.0)]
        r_asc = rasterize(geom, like=_like_2568(x_descending=False), fill=0)
        r_desc = rasterize(geom, like=_like_2568(x_descending=True), fill=0)

        # like.x is preserved verbatim
        np.testing.assert_array_equal(r_asc.x.values, [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(r_desc.x.values, [3.5, 2.5, 1.5, 0.5])

        # World-coord selection agrees
        for yw in [0.5, 1.5, 2.5, 3.5]:
            for xw in [0.5, 1.5, 2.5, 3.5]:
                a = float(r_asc.sel(y=yw, x=xw).item())
                b = float(r_desc.sel(y=yw, x=xw).item())
                assert a == b, (
                    f"mismatch at world (y={yw}, x={xw}): "
                    f"asc={a}, desc={b}"
                )

        # The burned cell is at the lower-left corner of the world grid
        assert float(r_asc.sel(y=0.5, x=0.5).item()) == 1.0
        assert float(r_desc.sel(y=0.5, x=0.5).item()) == 1.0
        # And nowhere near the right column (world x=3.5)
        assert float(r_asc.sel(y=0.5, x=3.5).item()) == 0.0
        assert float(r_desc.sel(y=0.5, x=3.5).item()) == 0.0

    def test_numpy_output_array_matches_like_orientation(self):
        """Column 0 of the output must correspond to like.x[0] in world
        coords, no matter which way x points."""
        geom = [(box(0, 0, 1, 1), 1.0)]
        r_asc = rasterize(geom, like=_like_2568(x_descending=False), fill=0)
        r_desc = rasterize(geom, like=_like_2568(x_descending=True), fill=0)
        # Ascending: column 0 is leftmost (x=0.5), so the burned cell sits
        # there in the bottom row.
        assert r_asc.values[-1, 0] == 1.0
        assert r_asc.values[-1, -1] == 0.0
        # Descending: column 0 is rightmost (x=3.5), so the burned cell
        # sits in the last column instead.
        assert r_desc.values[-1, -1] == 1.0
        assert r_desc.values[-1, 0] == 0.0

    def test_numpy_round_trip_with_xr_align(self):
        """output.x must equal like.x exactly so xr.align still works."""
        geom = [(box(0, 0, 1, 1), 1.0)]
        for x_descending in (True, False):
            like = _like_2568(x_descending=x_descending)
            result = rasterize(geom, like=like, fill=0)
            np.testing.assert_array_equal(result.x.values, like.x.values)
            np.testing.assert_array_equal(result.y.values, like.y.values)
            aligned_result, aligned_like = xr.align(result, like)
            assert aligned_result.sizes == result.sizes

    def test_numpy_points_respect_orientation(self):
        """Same check with a point geometry rather than a polygon."""
        from shapely.geometry import Point
        geom = [(Point(0.5, 0.5), 7.0)]
        r_asc = rasterize(geom, like=_like_2568(x_descending=False), fill=0)
        r_desc = rasterize(geom, like=_like_2568(x_descending=True), fill=0)
        assert float(r_asc.sel(y=0.5, x=0.5).item()) == 7.0
        assert float(r_desc.sel(y=0.5, x=0.5).item()) == 7.0

    def test_numpy_lines_respect_orientation(self):
        """Line along the left edge -- world x=0.5, all y."""
        from shapely.geometry import LineString
        geom = [(LineString([(0.5, 0.5), (0.5, 3.5)]), 5.0)]
        r_asc = rasterize(geom, like=_like_2568(x_descending=False), fill=0)
        r_desc = rasterize(geom, like=_like_2568(x_descending=True), fill=0)
        for yw in [0.5, 1.5, 2.5, 3.5]:
            assert float(r_asc.sel(y=yw, x=0.5).item()) == 5.0
            assert float(r_desc.sel(y=yw, x=0.5).item()) == 5.0

    def test_numpy_descending_x_and_ascending_y(self):
        """Both axes flipped relative to image convention.  Should still
        agree by world coords."""
        geom = [(box(0, 0, 1, 1), 1.0)]
        like_flipped = _like_2568(x_descending=True, y_descending=False)
        like_normal = _like_2568(x_descending=False, y_descending=True)
        r_flipped = rasterize(geom, like=like_flipped, fill=0)
        r_normal = rasterize(geom, like=like_normal, fill=0)
        for yw in [0.5, 1.5, 2.5, 3.5]:
            for xw in [0.5, 1.5, 2.5, 3.5]:
                a = float(r_flipped.sel(y=yw, x=xw).item())
                b = float(r_normal.sel(y=yw, x=xw).item())
                assert a == b, (
                    f"mismatch at world (y={yw}, x={xw}): "
                    f"flipped={a}, normal={b}"
                )
        assert float(r_flipped.sel(y=0.5, x=0.5).item()) == 1.0

    @skip_no_dask
    def test_dask_numpy_descending_x_matches_ascending(self):
        geom = [(box(0, 0, 1, 1), 1.0)]
        r_asc = rasterize(
            geom, like=_like_2568(x_descending=False), fill=0, chunks=2,
        ).compute()
        r_desc = rasterize(
            geom, like=_like_2568(x_descending=True), fill=0, chunks=2,
        ).compute()
        np.testing.assert_array_equal(r_asc.x.values, [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(r_desc.x.values, [3.5, 2.5, 1.5, 0.5])
        for xw in [0.5, 1.5, 2.5, 3.5]:
            a = float(r_asc.sel(y=0.5, x=xw).item())
            b = float(r_desc.sel(y=0.5, x=xw).item())
            assert a == b
        assert float(r_asc.sel(y=0.5, x=0.5).item()) == 1.0
        assert float(r_desc.sel(y=0.5, x=0.5).item()) == 1.0

    @skip_no_cuda
    def test_cupy_descending_x_matches_ascending(self):
        geom = [(box(0, 0, 1, 1), 1.0)]
        r_asc = rasterize(
            geom, like=_like_2568(x_descending=False), fill=0, use_cuda=True,
        )
        r_desc = rasterize(
            geom, like=_like_2568(x_descending=True), fill=0, use_cuda=True,
        )
        asc_vals = r_asc.data.get() if hasattr(r_asc.data, 'get') \
            else r_asc.values
        desc_vals = r_desc.data.get() if hasattr(r_desc.data, 'get') \
            else r_desc.values
        np.testing.assert_array_equal(r_asc.x.values, [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(r_desc.x.values, [3.5, 2.5, 1.5, 0.5])
        # ascending: burned column is first; descending: burned column
        # is last.
        assert asc_vals[-1, 0] == 1.0
        assert desc_vals[-1, -1] == 1.0

    @skip_no_cuda
    @skip_no_dask
    def test_dask_cupy_descending_x_matches_ascending(self):
        geom = [(box(0, 0, 1, 1), 1.0)]
        r_asc = rasterize(
            geom, like=_like_2568(x_descending=False), fill=0,
            use_cuda=True, chunks=2,
        ).compute()
        r_desc = rasterize(
            geom, like=_like_2568(x_descending=True), fill=0,
            use_cuda=True, chunks=2,
        ).compute()
        asc_vals = r_asc.data.get() if hasattr(r_asc.data, 'get') \
            else r_asc.values
        desc_vals = r_desc.data.get() if hasattr(r_desc.data, 'get') \
            else r_desc.values
        assert asc_vals[-1, 0] == 1.0
        assert desc_vals[-1, -1] == 1.0

    def test_numpy_explicit_bounds_skips_flip(self):
        """When bounds are passed explicitly, the orientation flip path
        is bypassed (caller has full control of the output grid)."""
        like = _like_2568(x_descending=True)
        # With explicit bounds + the same width/height, the output is
        # rebuilt with linspace coords (ascending x), not reused from
        # like.  This is the existing behaviour for ascending-y and
        # should match for descending-x.
        result = rasterize(
            [(box(0, 0, 1, 1), 1.0)], like=like,
            bounds=(0, 0, 4, 4), fill=0,
        )
        # Coords were rebuilt ascending
        np.testing.assert_array_equal(result.x.values, [0.5, 1.5, 2.5, 3.5])
        assert float(result.sel(y=0.5, x=0.5).item()) == 1.0
