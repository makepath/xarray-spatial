"""Regression tests for issue #2566.

``_check_uniform_axis`` in ``xrspatial.rasterize`` previously compared
``abs(np.diff(coords))`` against ``abs(expected_step)``.  That accepted
zig-zag / duplicate-coord patterns like ``x = [0.5, 1.5, 0.5, 1.5]``
as "uniform" because the magnitudes of the diffs were all 1.0.  The
rasterizer then reused those duplicate coords on the output, so
``.sel(x=0.5)`` returned two columns whose data came from different
burn cells.

The fix switches to comparing the *signed* ``np.diff`` against the
signed first interval.  Descending axes still pass (a descending y
has a negative signed first interval and all-negative signed diffs),
but zig-zag patterns are rejected with a clear ``ValueError`` naming
the offending axis.

These tests pin:

1. Ascending uniform x passes.
2. Descending uniform y passes (the orientation flip in ``rasterize``
   still has to work, so the validator must not gate on sign).
3. Zig-zag duplicate-valued x is rejected with a message naming x.
4. Zig-zag duplicate-valued y is rejected with a message naming y.
5. Strictly increasing but non-uniform spacing is still rejected
   (the original #2168 case).
6. Single-cell axes still short-circuit cleanly.
7. The zig-zag symptom from the issue (``.sel(x=0.5)`` returning two
   columns) cannot occur because the call raises before producing a
   DataArray with duplicate coords.
"""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from xrspatial.rasterize import rasterize


def _make_like(x, y):
    return xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.float64),
        dims=('y', 'x'),
        coords={'y': np.asarray(y, dtype=np.float64),
                'x': np.asarray(x, dtype=np.float64)},
    )


class TestSignedStepValidation_2566:
    """``_check_uniform_axis`` must require signed diffs to match."""

    def test_ascending_uniform_passes(self):
        # Both axes ascending and uniform -- the basic happy path.
        x_2566 = np.linspace(0.5, 9.5, 10)
        y_2566 = np.linspace(0.5, 9.5, 10)
        like_2566 = _make_like(x_2566, y_2566)
        # Should not raise.
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            like=like_2566, fill=0,
        )
        assert result.shape == (10, 10)

    def test_descending_uniform_y_passes(self):
        # Descending y is the standard image-orientation case.  The
        # signed first interval is negative; all signed diffs are
        # also negative and equal, so the check should accept it.
        x_2566 = np.linspace(0.5, 9.5, 10)
        y_2566 = np.linspace(9.5, 0.5, 10)
        like_2566 = _make_like(x_2566, y_2566)
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            like=like_2566, fill=0,
        )
        assert result.shape == (10, 10)
        assert np.any(result.values == 1.0)

    def test_zigzag_x_rejected(self):
        # ``x = [0.5, 1.5, 0.5, 1.5]`` -- abs(diff) == [1, 1, 1] but
        # signs alternate.  Previously accepted; now rejected.
        x_2566 = np.array([0.5, 1.5, 0.5, 1.5])
        y_2566 = np.linspace(3.5, 0.5, 4)
        like_2566 = _make_like(x_2566, y_2566)
        with pytest.raises(ValueError) as excinfo:
            rasterize(
                [(box(0, 0, 2, 4), 1.0)],
                like=like_2566, fill=0,
            )
        msg = str(excinfo.value)
        assert "'x'" in msg
        assert "non-uniform" in msg.lower()

    def test_zigzag_y_rejected(self):
        # Same pattern on y.
        x_2566 = np.linspace(0.5, 3.5, 4)
        y_2566 = np.array([0.5, 1.5, 0.5, 1.5])
        like_2566 = _make_like(x_2566, y_2566)
        with pytest.raises(ValueError) as excinfo:
            rasterize(
                [(box(0, 0, 4, 2), 1.0)],
                like=like_2566, fill=0,
            )
        msg = str(excinfo.value)
        assert "'y'" in msg
        assert "non-uniform" in msg.lower()

    def test_strictly_increasing_non_uniform_still_rejected(self):
        # The original #2168 case: monotonic but irregular spacing.
        # The signed-diff fix must not regress this.
        x_2566 = np.array([0.0, 1.0, 2.5, 3.5])
        y_2566 = np.linspace(3.5, 0.5, 4)
        like_2566 = _make_like(x_2566, y_2566)
        with pytest.raises(ValueError, match=r"'x'"):
            rasterize(
                [(box(0, 0, 3.5, 4), 1.0)],
                like=like_2566, fill=0,
            )

    def test_single_cell_axis_passes(self):
        # size < 3 short-circuits; a 2-row template along y should
        # still validate without raising.  Pair with a uniform x.
        x_2566 = np.linspace(0.5, 9.5, 10)
        y_2566 = np.array([1.5, 0.5])
        like_2566 = _make_like(x_2566, y_2566)
        # Should not raise.
        rasterize(
            [(box(0, 0, 10, 2), 1.0)],
            like=like_2566, fill=0,
        )

    def test_zigzag_does_not_silently_duplicate_coords(self):
        # Belt-and-braces: confirm the symptom described in the issue
        # (``.sel(x=0.5)`` returning two columns) is impossible now
        # because the rasterize call raises before it can produce a
        # DataArray with duplicate coords.
        x_2566 = np.array([0.5, 1.5, 0.5, 1.5])
        y_2566 = np.linspace(3.5, 0.5, 4)
        like_2566 = _make_like(x_2566, y_2566)
        with pytest.raises(ValueError):
            rasterize(
                [(box(0, 0, 2, 4), 1.0)],
                like=like_2566, fill=0,
            )
