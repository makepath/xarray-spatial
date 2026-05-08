"""TIFF Orientation tag (274) decode tests for issue #1503.

Before the fix, the reader silently ignored tag 274 and returned the file's
stored pixel order regardless of which corner the data was supposed to
start from. Files written with orientation 2-8 decoded to flipped or
rotated arrays, a silent geometric error.

Each test writes a small array with a specific orientation tag, then
checks that ``open_geotiff`` produces the array remapped per TIFF 6.0
spec. tifffile's ``imwrite`` is used to embed the Orientation tag via
``extratags``; tifffile's ``imread`` does *not* itself apply orientation
(it returns the stored buffer as-is) so the comparison target here is
computed from the source array rather than read back through tifffile.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._reader import read_to_array

tifffile = pytest.importorskip("tifffile")


# Eight orientation values defined by TIFF 6.0.
_ORIENTATIONS = [1, 2, 3, 4, 5, 6, 7, 8]


def _write_with_orientation(path, arr, orientation):
    """Write *arr* to *path* with the given Orientation tag value.

    tifffile's ``imwrite`` does not expose Orientation as a kwarg, but
    the ``extratags`` parameter accepts (tag_id, dtype_code, count, value,
    write_once) tuples that get emitted into the IFD verbatim. ``H`` is
    the unsigned short (TIFF type 3) struct code.
    """
    tifffile.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


def _expected_for_orientation(stored, orientation):
    """Return what *stored* should look like after applying *orientation*.

    Mirrors the spec table in :func:`xrspatial.geotiff._reader._apply_orientation`.
    """
    if orientation == 1:
        return stored
    if orientation == 2:
        return stored[:, ::-1]
    if orientation == 3:
        return stored[::-1, ::-1]
    if orientation == 4:
        return stored[::-1, :]
    if orientation == 5:
        return stored.T
    if orientation == 6:
        return stored.T[:, ::-1]
    if orientation == 7:
        return stored.T[::-1, ::-1]
    if orientation == 8:
        return stored.T[::-1, :]
    raise AssertionError(orientation)


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_orientation_matches_spec(tmp_path, orientation):
    """open_geotiff applies the spec-defined transform for each orientation."""
    # Asymmetric data (different height and width, distinct row/column
    # values) so any axis swap or flip shows up as a clear mismatch.
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    expected = _expected_for_orientation(arr, orientation)
    got = open_geotiff(str(path))

    assert got.values.shape == expected.shape, (
        f"orientation={orientation}: shape mismatch "
        f"got={got.values.shape} expected={expected.shape}"
    )
    np.testing.assert_array_equal(got.values, expected)


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_orientation_coords_match_post_orientation_shape(
    tmp_path, orientation
):
    """y/x coordinate arrays size matches the post-orientation array shape."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_coords_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))

    h, w = da.values.shape
    assert da.coords['y'].shape == (h,)
    assert da.coords['x'].shape == (w,)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_swap_dims(tmp_path, orientation):
    """Orientations 5-8 swap rows and columns relative to the stored shape."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)  # h=4, w=6
    path = tmp_path / f"orient_swap_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))

    # File stores h=4, w=6. After orientation 5-8 the displayed shape is
    # (6, 4) -- width and height swap.
    assert da.values.shape == (6, 4)


def test_orientation_default_unchanged(tmp_path):
    """A file without an Orientation tag defaults to 1 (no transform)."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "no_orient.tif"
    tifffile.imwrite(str(path), arr)

    da = open_geotiff(str(path))
    np.testing.assert_array_equal(da.values, arr)


def test_orientation_with_window_raises(tmp_path):
    """Windowed read on a non-default orientation raises ValueError."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "orient2_window.tif"
    _write_with_orientation(path, arr, 2)

    with pytest.raises(ValueError, match="orientation"):
        read_to_array(str(path), window=(0, 0, 2, 2))

    with pytest.raises(ValueError, match="orientation"):
        open_geotiff(str(path), window=(0, 0, 2, 2))


def test_orientation_1_with_window_still_works(tmp_path):
    """Default orientation (1) with window= keeps working as before."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "orient1_window.tif"
    _write_with_orientation(path, arr, 1)

    da = open_geotiff(str(path), window=(0, 0, 2, 3))
    assert da.values.shape == (2, 3)
    np.testing.assert_array_equal(da.values, arr[:2, :3])
