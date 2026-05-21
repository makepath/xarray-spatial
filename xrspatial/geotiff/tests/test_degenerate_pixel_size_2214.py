"""Fail-closed default for 1xN / Nx1 writes without an explicit transform.

Issue #2214: ``coords_to_transform`` used to borrow the non-degenerate
axis's pixel size for the degenerate axis whenever one spatial dim had
length 1. That assumption is unsafe -- a 30 m by 10 m source raster
served as a 1xN strip would silently write out with 30 m by 30 m pixels.

The fix:
* Default behaviour is fail-closed. A 1xN / Nx1 DataArray with spatial
  coords but no ``attrs['transform']`` and no opt-in flag now raises
  ``ValueError``.
* ``attrs['transform']`` (rasterio 6-tuple) supplies the true pixel
  geometry and round-trips bit-exactly.
* ``attrs['assume_square_pixels_for_degenerate_axis'] = True`` opts in
  to the #1945 borrow-from-other-axis path for callers who know their
  source is square.
* Multi-row / multi-column writes are untouched.

These tests pin the new contract across the eager numpy writer
(the writer everyone hits first) and the helper itself, which all
other backends share via ``_coords_to_transform``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._coords import coords_to_transform


# Source raster the bug reporter described: 30 m x pixels, 10 m y pixels.
PIXEL_X_TRUE = 30.0
PIXEL_Y_TRUE = 10.0
X0 = -120.0
Y0 = 45.0


def _strip_1xN_nonsquare() -> xr.DataArray:
    """A 1xN strip whose source raster has non-square pixels.

    The x coord spacing is 30 (readable from coords). The y axis is
    length 1, so the y pixel size of 10 cannot be recovered from coords.
    This is the case the writer must fail closed on.
    """
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(1, 8),
        dims=("y", "x"),
        coords={
            "x": X0 + np.arange(8, dtype="float64") * PIXEL_X_TRUE,
            "y": np.array([Y0], dtype="float64"),
        },
        attrs={"crs": 4326},
    )


def _strip_Nx1_nonsquare() -> xr.DataArray:
    """An Nx1 profile whose source raster has non-square pixels."""
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(8, 1),
        dims=("y", "x"),
        coords={
            "x": np.array([X0], dtype="float64"),
            "y": Y0 - np.arange(8, dtype="float64") * PIXEL_Y_TRUE,
        },
        attrs={"crs": 4326},
    )


# ---------------------------------------------------------------------------
# 1xN / Nx1 raise without an opt-in or attrs['transform']
# ---------------------------------------------------------------------------

class TestDegenerateWritesFailClosed:
    """A 1xN / Nx1 write with spatial coords must raise without opt-in."""

    def test_1xN_without_transform_or_optin_raises(self, tmp_path):
        da = _strip_1xN_nonsquare()
        p = str(tmp_path / "fail_1xN_2214.tif")
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, p)
        msg = str(excinfo.value)
        # The error must name both escape hatches.
        assert "transform" in msg
        assert "assume_square_pixels_for_degenerate_axis" in msg

    def test_Nx1_without_transform_or_optin_raises(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        p = str(tmp_path / "fail_Nx1_2214.tif")
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, p)
        msg = str(excinfo.value)
        assert "transform" in msg
        assert "assume_square_pixels_for_degenerate_axis" in msg


# ---------------------------------------------------------------------------
# Explicit transform path: caller supplies the true pixel geometry
# ---------------------------------------------------------------------------

class TestDegenerateWritesWithExplicitTransform:
    """``attrs['transform']`` round-trips the supplied pixel size exactly."""

    def test_1xN_with_attrs_transform_round_trips_true_pixel_size(self, tmp_path):
        da = _strip_1xN_nonsquare()
        # rasterio 6-tuple: (a, b, c, d, e, f) = (px, 0, ox, 0, py, oy)
        true_transform = (
            PIXEL_X_TRUE, 0.0, X0 - PIXEL_X_TRUE * 0.5,
            0.0, -PIXEL_Y_TRUE, Y0 + PIXEL_Y_TRUE * 0.5,
        )
        da = da.copy()
        da.attrs = {**da.attrs, "transform": true_transform}

        p = str(tmp_path / "explicit_1xN_2214.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # The non-degenerate axis (x) keeps its true 30 m step.
        x_step = float(r.coords["x"][1] - r.coords["x"][0])
        assert x_step == pytest.approx(PIXEL_X_TRUE)
        # And the readback transform records the true 10 m y pixel,
        # not the borrowed 30 m. attrs['transform'] is a rasterio
        # 6-tuple; element 4 is pixel_height (negative by convention).
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_Nx1_with_attrs_transform_round_trips_true_pixel_size(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        true_transform = (
            PIXEL_X_TRUE, 0.0, X0 - PIXEL_X_TRUE * 0.5,
            0.0, -PIXEL_Y_TRUE, Y0 + PIXEL_Y_TRUE * 0.5,
        )
        da = da.copy()
        da.attrs = {**da.attrs, "transform": true_transform}

        p = str(tmp_path / "explicit_Nx1_2214.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        y_step = float(r.coords["y"][1] - r.coords["y"][0])
        # y decreases top-to-bottom by convention.
        assert y_step == pytest.approx(-PIXEL_Y_TRUE)
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)


# ---------------------------------------------------------------------------
# Opt-in flag: caller explicitly accepts the borrow-from-other-axis fallback
# ---------------------------------------------------------------------------

class TestDegenerateWritesWithOptIn:
    """``attrs['assume_square_pixels_for_degenerate_axis'] = True`` opts in.

    Behaviour matches the pre-#2214 #1945 borrow path: the writer assumes
    the source raster is square and copies the non-degenerate axis's
    pixel size onto the degenerate axis. We pin both the resulting
    transform and the fact that the opt-in *must* be the boolean ``True``
    -- a stray string like ``'no'`` must not enable the borrow.
    """

    def test_1xN_optin_borrows_from_x_axis(self, tmp_path):
        da = _strip_1xN_nonsquare()
        da = da.copy()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}

        p = str(tmp_path / "optin_1xN_2214.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # The borrow path copies the magnitude of the x step onto the
        # y axis with the y-down sign convention. With the bug
        # reporter's source (true x=30, true y=10) the file now records
        # y=-30. That is the documented opt-in cost.
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_X_TRUE)

    def test_Nx1_optin_borrows_from_y_axis(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        da = da.copy()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}

        p = str(tmp_path / "optin_Nx1_2214.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # Borrow runs the other direction: x picks up |y step|.
        tx = r.attrs["transform"]
        # True y step is -10 (top-down), so |pixel_height| = 10 is what
        # gets copied to pixel_width.
        assert tx[0] == pytest.approx(PIXEL_Y_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_optin_must_be_boolean_True_not_truthy_string(self, tmp_path):
        """A stray ``'yes'`` value must not silently enable the borrow path."""
        da = _strip_1xN_nonsquare()
        da = da.copy()
        # 'yes' is truthy in Python but is NOT the boolean True. The
        # identity check on ``_assume_square_for_degenerate`` rejects
        # everything that isn't ``is True`` so an accidental attrs
        # value can't accidentally re-enable the silent-invent path.
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": "yes"}

        p = str(tmp_path / "optin_bad_2214.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


# ---------------------------------------------------------------------------
# Non-degenerate writes are unchanged
# ---------------------------------------------------------------------------

class TestMultiRowMultiColumnUnchanged:
    """The fix must not touch the regular (non-degenerate) write path."""

    def test_2x2_writes_without_optin(self, tmp_path):
        """A 2x2 raster reads its pixel size off the coords; no opt-in needed."""
        da = xr.DataArray(
            np.arange(4, dtype="float32").reshape(2, 2),
            dims=("y", "x"),
            coords={
                "x": np.array([X0, X0 + PIXEL_X_TRUE], dtype="float64"),
                "y": np.array([Y0, Y0 - PIXEL_Y_TRUE], dtype="float64"),
            },
            attrs={"crs": 4326},
        )
        p = str(tmp_path / "multi_2x2_2214.tif")
        # No fail-closed: both axes have length >= 2.
        to_geotiff(da, p)

        r = open_geotiff(p)
        tx = r.attrs["transform"]
        # True (non-borrowed) pixel sizes on both axes.
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_3x5_writes_without_optin(self, tmp_path):
        rng = np.random.RandomState(0)
        arr = rng.random((3, 5)).astype("float32")
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "x": X0 + np.arange(5, dtype="float64") * PIXEL_X_TRUE,
                "y": Y0 - np.arange(3, dtype="float64") * PIXEL_Y_TRUE,
            },
            attrs={"crs": 4326},
        )
        p = str(tmp_path / "multi_3x5_2214.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)
        np.testing.assert_array_equal(np.asarray(r.values), arr)


# ---------------------------------------------------------------------------
# Helper-level tests: coords_to_transform contract
# ---------------------------------------------------------------------------

class TestCoordsToTransformHelperContract:
    """Direct tests for ``coords_to_transform`` so the contract is pinned
    independent of any writer wrapping."""

    def test_degenerate_without_optin_returns_None(self):
        """The writer relies on ``None`` to trigger the fail-closed branch
        via ``require_transform_for_georeferenced``."""
        da = _strip_1xN_nonsquare()
        assert coords_to_transform(da) is None

    def test_degenerate_with_optin_returns_borrowed_transform(self):
        da = _strip_1xN_nonsquare()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}
        t = coords_to_transform(da)
        assert t is not None
        assert t.pixel_width == pytest.approx(PIXEL_X_TRUE)
        # Borrowed -- not the true 10.0.
        assert t.pixel_height == pytest.approx(-PIXEL_X_TRUE)

    def test_multi_axis_ignores_optin_flag(self):
        """The opt-in flag is only consulted for the degenerate branch.
        A regular 2x2 write doesn't trip the borrow path even if the
        flag is set, so the writer can't accidentally start borrowing."""
        da = xr.DataArray(
            np.arange(4, dtype="float32").reshape(2, 2),
            dims=("y", "x"),
            coords={
                "x": np.array([X0, X0 + PIXEL_X_TRUE], dtype="float64"),
                "y": np.array([Y0, Y0 - PIXEL_Y_TRUE], dtype="float64"),
            },
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        assert t.pixel_width == pytest.approx(PIXEL_X_TRUE)
        assert t.pixel_height == pytest.approx(-PIXEL_Y_TRUE)
