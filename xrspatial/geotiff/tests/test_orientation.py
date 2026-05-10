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


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_orientation_tag_not_passed_through_extra_tags(tmp_path, orientation):
    """Tag 274 must not survive on the returned DataArray.

    Without this, a read+write round-trip would re-emit the original
    Orientation tag even though the pixel buffer is already remapped --
    downstream readers would apply the orientation a second time.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_passthrough_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    da = open_geotiff(str(path))
    extra = da.attrs.get('extra_tags') or []
    tag_ids = [t[0] if isinstance(t, tuple) else t for t in extra]
    assert 274 not in tag_ids, (
        f"orientation={orientation}: tag 274 leaked into extra_tags={extra}"
    )


def test_orientation_round_trip_does_not_double_apply(tmp_path):
    """open_geotiff -> to_geotiff -> open_geotiff returns the same array.

    Concretely: a file written with orientation=4 reads as flipped
    (correct), the writer emits a normal file (no orientation tag), and
    a second read returns the same array. If tag 274 leaked through
    extra_tags, the second read would apply orientation=4 again.
    """
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path1 = tmp_path / "orient4_in.tif"
    _write_with_orientation(path1, arr, 4)

    da1 = open_geotiff(str(path1))

    path2 = tmp_path / "orient4_out.tif"
    to_geotiff(da1, str(path2))
    da2 = open_geotiff(str(path2))

    np.testing.assert_array_equal(da2.values, da1.values)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientation_5_to_8_warn_on_georef(tmp_path, orientation):
    """Axis-swap orientations on georef'd files emit a warning.

    The transform swap is shape-correct but geometrically approximate
    for orientations 6/7/8; the user should be told.
    """
    import warnings as _warnings

    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f"orient_georef_{orientation}.tif"
    # tifffile lets us tag a CRS via the resolution + metadata; the
    # simplest path that triggers our georef branch is to embed a
    # ModelPixelScale + GeoKeyDirectory pair pointing at EPSG:4326.
    tifffile.imwrite(
        str(path), arr,
        extratags=[
            (274, 'H', 1, orientation, True),
            # ModelPixelScaleTag (33550): scale_x, scale_y, scale_z
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            # ModelTiepointTag (33922): I, J, K, X, Y, Z
            (33922, 'd', 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), True),
            # GeoKeyDirectoryTag (34735): version 1.1.0, 1 key,
            # GTModelType=2 (Geographic), GeographicType=4326
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,   # GTModelTypeGeoKey = 2
                2048, 0, 1, 4326,  # GeographicTypeGeoKey = 4326
            ), True),
        ],
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        da = open_geotiff(str(path))

    assert da is not None
    msgs = [str(w.message) for w in caught if 'Orientation' in str(w.message)]
    assert msgs, (
        f"orientation={orientation}: no warning emitted on georef file"
    )


# ---------------------------------------------------------------------------
# Geographic coordinate updates for mirror-flip orientations (issue #1537)
# ---------------------------------------------------------------------------
#
# Orientations 2/3/4 flip the array horizontally, both axes, or vertically.
# The reader used to apply the buffer flip but leave the y/x coord arrays
# computed from the original transform, so xarray label-based lookups
# returned the wrong pixel for georeferenced files.

_GEOREF_EXTRA_AREA = [
    (33550, 'd', 3, (1.0, 1.0, 0.0)),
    (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0)),
    (34735, 'H', 12, (
        1, 1, 0, 2,
        1024, 0, 1, 2,
        2048, 0, 1, 4326,
    )),
]
_GEOREF_EXTRA_POINT = [
    (33550, 'd', 3, (1.0, 1.0, 0.0)),
    (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0)),
    (34735, 'H', 16, (
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 2,    # PixelIsPoint
        2048, 0, 1, 4326,
    )),
]


def _write_with_orient_and_georef(path, arr, orientation, raster_type='area'):
    """Write *arr* with Orientation tag + EPSG:4326 georef."""
    extras = (_GEOREF_EXTRA_POINT if raster_type == 'point'
              else _GEOREF_EXTRA_AREA)
    tifffile.imwrite(
        str(path), arr,
        extratags=[(274, 'H', 1, orientation, True)] + [
            (tag, dt, count, val, True) for (tag, dt, count, val) in extras
        ],
    )


@pytest.mark.parametrize('orientation', [2, 3, 4])
def test_orient_2_3_4_coords_track_pixel_flip_area(tmp_path, orientation):
    """Mirror-flip orientations: a label-based lookup at a fixed (x, y)
    must return the same pixel value regardless of the file's orientation.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)  # h=4, w=6

    # Reference: orientation=1 (no transform) tells us the geographic
    # coords for each file pixel under PixelIsArea. Pick three corners
    # so x-flip, y-flip, and combined flips each have a target.
    ref_path = tmp_path / 'orient_ref.tif'
    _write_with_orient_and_georef(ref_path, arr, 1)
    ref = open_geotiff(str(ref_path))
    # Pixel (0, 0): top-left, value 0, at x=100.5, y=49.5
    # Pixel (0, 5): top-right, value 5, at x=105.5, y=49.5
    # Pixel (3, 0): bottom-left, value 18, at x=100.5, y=46.5
    # Pixel (3, 5): bottom-right, value 23, at x=105.5, y=46.5
    targets = [
        (100.5, 49.5, 0),
        (105.5, 49.5, 5),
        (100.5, 46.5, 18),
        (105.5, 46.5, 23),
    ]
    for x, y, expected in targets:
        assert int(ref.sel(x=x, y=y).item()) == expected

    # Now check the same coords under the flipped orientation.
    path = tmp_path / f'orient_{orientation}.tif'
    _write_with_orient_and_georef(path, arr, orientation)
    da = open_geotiff(str(path))
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f'orientation={orientation}: sel(x={x}, y={y}) returned {got}, '
            f'expected {expected} (mirror-flip lost the pixel-to-coord '
            f'binding)'
        )


@pytest.mark.parametrize('orientation', [2, 3, 4])
def test_orient_2_3_4_coords_track_pixel_flip_point(tmp_path, orientation):
    """Same coord-fidelity check for PixelIsPoint files."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)

    ref_path = tmp_path / 'orient_ref_pt.tif'
    _write_with_orient_and_georef(ref_path, arr, 1, raster_type='point')
    ref = open_geotiff(str(ref_path))
    # PixelIsPoint: pixel (0, 0) center = (100, 50), pixel (0, 5) = (105, 50)
    # pixel (3, 0) = (100, 47), pixel (3, 5) = (105, 47)
    targets = [
        (100.0, 50.0, 0),
        (105.0, 50.0, 5),
        (100.0, 47.0, 18),
        (105.0, 47.0, 23),
    ]
    for x, y, expected in targets:
        assert int(ref.sel(x=x, y=y).item()) == expected

    path = tmp_path / f'orient_{orientation}_pt.tif'
    _write_with_orient_and_georef(path, arr, orientation, raster_type='point')
    da = open_geotiff(str(path))
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f'PixelIsPoint orient={orientation}: sel(x={x}, y={y})={got}, '
            f'expected {expected}'
        )


@pytest.mark.parametrize(
    'orientation,expected_first_x,expected_first_y',
    [
        # x_first=100.5 means the leftmost displayed column carries the
        # original left-edge geographic coordinate; 105.5 means it carries
        # the original right-edge coordinate (i.e. the coord array got
        # flipped along x).
        (2, 105.5, 49.5),  # x flipped, y unchanged
        (3, 105.5, 46.5),  # both flipped
        (4, 100.5, 46.5),  # x unchanged, y flipped
    ],
)
def test_orient_2_3_4_coord_arrays(
    tmp_path, orientation, expected_first_x, expected_first_y,
):
    """The first y/x coord lands on the right edge after a flip."""
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / f'orient_arr_{orientation}.tif'
    _write_with_orient_and_georef(path, arr, orientation)

    da = open_geotiff(str(path))
    np.testing.assert_allclose(da.x[0].item(), expected_first_x)
    np.testing.assert_allclose(da.y[0].item(), expected_first_y)


def test_orient_2_3_4_no_geo_still_uses_pixel_coords(tmp_path):
    """Without georef, the legacy integer pixel coord behaviour is preserved.

    A file without ModelPixelScale / ModelTiepoint stays on integer
    coords regardless of orientation; the fix must not start fabricating
    coordinates for un-georeferenced files.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    for orientation in (2, 3, 4):
        path = tmp_path / f'orient_no_geo_{orientation}.tif'
        tifffile.imwrite(
            str(path), arr,
            extratags=[(274, 'H', 1, orientation, True)],
        )
        da = open_geotiff(str(path))
        # Coords default to 0..N-1 when has_georef is False; the
        # orientation transform fix must not change that.
        assert da.x.values.dtype.kind in ('i', 'u'), (
            f'orient={orientation}: x coords drifted off integer '
            f'(dtype={da.x.values.dtype})'
        )


def test_orientation_with_band_selection_returns_2d(tmp_path):
    """band= followed by an orientation transpose returns a 2D array.

    Regression: an earlier draft applied orientation before slicing the
    band, which wasted memory and produced confusing intermediates.
    Now band slicing happens first.
    """
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    path = tmp_path / "orient5_rgb.tif"
    tifffile.imwrite(
        str(path), rgb, photometric='rgb',
        extratags=[(274, 'H', 1, 5, True)],
    )

    # Orientation 5 transposes spatial axes, so output spatial shape is
    # (6, 4). Band 1 returns just that channel.
    da = open_geotiff(str(path), band=1)
    assert da.values.shape == (6, 4)
    expected = rgb[..., 1].T  # band 1 then transpose
    np.testing.assert_array_equal(da.values, expected)
