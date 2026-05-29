"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 4).

Closes two groups of validation error paths in ``rasterize()`` that no test
exercised after the pass-1 (2026-05-17), pass-2 (2026-05-21), and pass-3
(2026-05-27) audits.  Both are pure-Python guards that run before any backend
dispatch, so they need no CUDA.

- Cat 4 MEDIUM -- partial width/height.  The guard at rasterize.py
  (``if (width is None) != (height is None)``) raises ValueError when exactly
  one of ``width`` / ``height`` is passed.  The docstring documents this as
  intended behaviour, but neither the width-only nor the height-only branch
  has a test.  A regression that dropped the guard would silently fill the
  missing dimension from ``resolution`` / ``like`` (or fail deeper inside a
  helper with an opaque message) instead of raising the documented error.

- Cat 4 MEDIUM -- resolution input type and shape.  ``resolution=`` has
  several input-validation branches that each raise a distinct ValueError:
  non-number / non-sequence (string, dict), wrong-ndim numpy array,
  wrong-length sequence (len 1 or 3+), and non-numeric sequence elements.
  The existing ``test_invalid_resolution_scalar`` / ``test_invalid_resolution_tuple``
  tests in test_rasterize.py cover only non-finite or non-positive *values*,
  not these type / shape branches, so a regression that loosened or reordered
  them would ship silently.

The "fix" in this sweep is *adding tests*.  No source changes.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)


def _one_box():
    """A single unit-square (geometry, value) pair for validation calls."""
    return [(box(0, 0, 1, 1), 1.0)]


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- partial width/height
# ---------------------------------------------------------------------------

class TestPartialWidthHeight:
    """Passing only one of ``width`` / ``height`` must raise ValueError.

    The guard names both the dimension that was given and the one that was
    missing, so the message differs by direction.  Pin both directions.
    """

    def test_width_only_raises(self):
        with pytest.raises(
            ValueError,
            match=r"width was provided but height was not",
        ):
            rasterize(_one_box(), width=10, bounds=(0, 0, 1, 1))

    def test_height_only_raises(self):
        with pytest.raises(
            ValueError,
            match=r"height was provided but width was not",
        ):
            rasterize(_one_box(), height=10, bounds=(0, 0, 1, 1))

    def test_width_only_raises_even_with_resolution(self):
        # The guard fires before the resolution branch, so a partial
        # width/height is rejected even when resolution could have sized
        # the output -- the documented "pass both or neither" contract.
        with pytest.raises(
            ValueError,
            match=r"width was provided but height was not",
        ):
            rasterize(_one_box(), width=10, resolution=1.0,
                      bounds=(0, 0, 1, 1))


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- resolution input type and shape validation
# ---------------------------------------------------------------------------

class TestResolutionInputValidation:
    """``resolution=`` type / shape guards, distinct from the value guards
    (non-finite / non-positive) already covered in test_rasterize.py.
    """

    @pytest.mark.parametrize("bad", ["abc", {"x": 1}])
    def test_non_number_non_sequence_raises(self, bad):
        with pytest.raises(
            ValueError,
            match=r"resolution must be a number or a length-2 sequence",
        ):
            rasterize(_one_box(), resolution=bad, bounds=(0, 0, 1, 1))

    @pytest.mark.parametrize("bad", [(1,), (1, 2, 3), [1, 2, 3, 4]])
    def test_wrong_length_sequence_raises(self, bad):
        with pytest.raises(
            ValueError,
            match=r"resolution sequence must have length 2",
        ):
            rasterize(_one_box(), resolution=bad, bounds=(0, 0, 1, 1))

    def test_wrong_ndim_numpy_array_raises(self):
        # A 2-D numpy array must be rejected before the length-2 check so a
        # (2, 2) array does not slip past as "length 2".
        with pytest.raises(
            ValueError,
            match=r"resolution array must be 1-D with length 2",
        ):
            rasterize(_one_box(), resolution=np.ones((2, 2)),
                      bounds=(0, 0, 1, 1))

    def test_non_numeric_sequence_elements_raise(self):
        with pytest.raises(
            ValueError,
            match=r"resolution sequence elements must be numbers",
        ):
            rasterize(_one_box(), resolution=("a", "b"),
                      bounds=(0, 0, 1, 1))

    def test_length_2_numpy_array_accepted(self):
        # Positive control: a 1-D length-2 numpy array is a valid resolution
        # and must NOT be rejected by the shape guards above.
        result = rasterize(_one_box(), resolution=np.array([1.0, 1.0]),
                           bounds=(0, 0, 4, 4), fill=0)
        assert result.shape == (4, 4)
