"""User-authored integer spatial coords must not silently drop georef (#2087, #2120).

Pre-#2087, ``coords_to_transform`` returned ``None`` whenever either x
or y had an integer dtype, which silently stripped georef from any
user-authored integer-coord grid. Issue #2087 tightened the shape
check to the exact reader pattern (int64 ascending step-1 on both
axes), but the same trade-off bit a smaller niche: user grids that
happened to be int64 ascending step-1 starting at non-zero offsets
(e.g. ``x=[500,501,502], y=[1000,1001]``) still lost their georef.

Issue #2120 moved the placeholder signal off coord shape entirely:
the reader stamps ``attrs[_NO_GEOREF_KEY] = True`` together with the
placeholder coords, and the writer checks that marker instead of
guessing from shape. These tests pin the contract:

1. The legitimate no-georef round-trip still works when the marker
   is carried forward.
2. User-authored integer-coord grids are not silently stripped, even
   if they match the placeholder shape. Without the marker the writer
   synthesises a real unit transform and the array round-trips with
   float coords.
3. Non-uniform int coords raise ``NonUniformCoordsError`` rather than
   the silent strip.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._coords import _has_no_georef_marker
from xrspatial.geotiff._geotags import _NO_GEOREF_KEY

# --- Unit checks on the no-georef marker predicate ----------------------
#
# Pre-#2133, ``xrspatial.geotiff._coords`` exported an
# ``_is_no_georef_sentinel`` helper that inspected coord shape (int64,
# ``np.arange``-style). The writer no longer consults that predicate;
# the only signal is ``attrs[_NO_GEOREF_KEY]``. These tests pin the
# marker-based predicate ``_has_no_georef_marker`` that replaced it.


def _arange_int64_shape(coord: np.ndarray) -> bool:
    """Test-local predicate matching the read-side placeholder shape.

    ``coords_from_pixel_geometry`` emits ``np.arange(start, stop,
    dtype=np.int64)`` for the y/x coords whenever the source file
    carries no transform tags -- both for full reads (``start=0``) and
    windowed reads (``start=window_offset``). This helper exists only
    so a few legacy round-trip assertions can verify the on-disk shape
    came back unchanged; it is not the production no-georef signal.
    """
    if coord.dtype != np.int64:
        return False
    n = len(coord)
    if n < 1:
        return False
    return bool(np.array_equal(
        coord, np.arange(coord[0], coord[0] + n, dtype=np.int64)
    ))


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({_NO_GEOREF_KEY: True}, True),
        ({}, False),
        ({_NO_GEOREF_KEY: False}, False),
        ({_NO_GEOREF_KEY: 'yes'}, False),     # not identity-True
        ({_NO_GEOREF_KEY: 1}, False),         # truthy int, not True
        ({'other': True}, False),
    ],
)
def test_marker_predicate_identity_check(attrs, expected):
    da = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        coords={'y': np.arange(2, dtype=np.int64), 'x': np.arange(2, dtype=np.int64)},
        dims=('y', 'x'),
        attrs=attrs,
    )
    assert _has_no_georef_marker(da) is expected


@pytest.mark.parametrize(
    "coord",
    [
        np.arange(5, dtype=np.int64),         # full read
        np.arange(3, 8, dtype=np.int64),      # windowed read
        np.arange(0, 1, dtype=np.int64),      # degenerate 1-element
        np.array([10, 11, 12], dtype=np.int64),
    ],
)
def test_arange_int64_shape_helper_accepts(coord):
    assert _arange_int64_shape(coord)


@pytest.mark.parametrize(
    "coord",
    [
        np.array([100, 101, 102], dtype=np.int32),     # int32, not int64
        np.array([100, 101, 102], dtype=np.float64),   # float
        np.array([200, 199], dtype=np.int64),          # descending
        np.array([0, 2, 4], dtype=np.int64),           # step != 1
        np.array([1, 2, 5], dtype=np.int64),           # non-uniform
        np.array([], dtype=np.int64),                  # empty
    ],
)
def test_arange_int64_shape_helper_rejects(coord):
    assert not _arange_int64_shape(coord)


# --- Round-trip behaviour ------------------------------------------------


def _make_georef_int_grid():
    # User-authored projected grid with integer-spaced coords. ``y``
    # decreases top-to-bottom by convention, so it does not match the
    # ascending sentinel even before any other check.
    return xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        coords={'y': np.array([200, 199]), 'x': np.array([100, 101, 102])},
        dims=('y', 'x'),
    )


def test_user_authored_int_grid_writes_real_transform(tmp_path):
    da = _make_georef_int_grid()
    path = str(tmp_path / "tmp_2087_int_grid.tif")
    to_geotiff(da, path)

    out = open_geotiff(path)
    # Coord values round-trip exactly; dtype flips int -> float because
    # the file now carries a real transform and the reader emits float
    # pixel-center coords.
    assert out.coords['x'].dtype.kind == 'f'
    assert out.coords['y'].dtype.kind == 'f'
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(out.coords['y'].values, [200.0, 199.0])
    # Transform attr is present (the bug was that it wasn't).
    assert out.attrs.get('transform') is not None


def test_both_axes_ascending_int64_step1_round_trips_with_georef(tmp_path):
    # Pre-#2120 the writer treated any int64 ascending step-1 grid as
    # the no-georef placeholder (because the reader emits coords of that
    # shape) and silently stripped the georef. That trade-off bit real
    # users whose projected grids happened to start at integer offsets
    # like ``x=[500, 501, 502], y=[1000, 1001]``. Issue #2120 moved the
    # placeholder signal to ``attrs[_NO_GEOREF_KEY]`` so the writer no
    # longer guesses from coord shape alone. A user-authored grid that
    # matches the arange pattern but lacks the marker now writes a
    # real transform and round-trips with float coords.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_both_arange.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.coords['x'].dtype.kind == 'f'
    assert out.coords['y'].dtype.kind == 'f'
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(out.coords['y'].values, [200.0, 201.0, 202.0])
    assert out.attrs.get('transform') is not None


def test_user_authored_int_grid_with_explicit_transform(tmp_path):
    # Caller in the ambiguous-trade-off corner who wants georef sets
    # attrs['transform'] explicitly. The writer must use that
    # transform rather than the sentinel inference.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={'transform': (1.0, 0.0, 99.5, 0.0, 1.0, 199.5)},
    )
    path = str(tmp_path / "tmp_2087_explicit_transform.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])


def test_non_uniform_int_coords_raise(tmp_path):
    # Non-uniform integer spacing under the old sentinel silently
    # stripped georef. The pre-#2133 fallback caught this via the
    # lower-level ``coords_to_transform`` ("not uniformly spaced"
    # message). Post-#2133, the write-metadata validator catches it
    # first with a different message because the integer-dtype
    # exemption has been replaced with a marker-based one. Either
    # message satisfies the contract: a non-uniform write must
    # raise rather than silently misrepresent the grid.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_non_uniform.tif")
    with pytest.raises(ValueError, match="non.?uniform"):
        to_geotiff(da, path)


def test_int_x_float_y_writes_transform(tmp_path):
    # One axis integer, the other float: under the old sentinel any
    # integer axis defeated the transform-inference. Under the
    # tightened sentinel, the float y axis means the int x axis falls
    # through to ``coords_to_transform`` (which handles int math
    # fine) and a transform is written.
    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        coords={
            'y': np.array([50.5, 49.5], dtype=np.float64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_mixed_dtypes.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])


# --- The legitimate no-georef round-trip must keep working ---------------


def test_no_georef_roundtrip_preserved(tmp_path):
    # The no-georef round-trip starts from a real no-georef file: the
    # reader stamps ``attrs[_NO_GEOREF_KEY] = True`` together with the
    # int64 ``np.arange``-shaped coords, and the writer carries the
    # marker forward so the next ``to_geotiff`` does not invent a
    # transform. Issue #2120 made the marker the only signal -- a
    # user constructing the same coord arrays from scratch without the
    # marker now writes a real unit transform instead (see
    # ``test_both_axes_ascending_int64_step1_round_trips_with_georef``).
    from xrspatial.geotiff._coords import _NO_GEOREF_KEY
    src = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.arange(4, dtype=np.int64), 'x': np.arange(4, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2087_no_georef.tif")
    to_geotiff(src, path)

    out = open_geotiff(path)
    # Verify the read came back as no-georef.
    assert out.coords['x'].dtype == np.int64
    assert out.coords['y'].dtype == np.int64
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True

    # Round-trip: write again. No transform should be invented.
    path2 = str(tmp_path / "tmp_2087_no_georef_rt.tif")
    to_geotiff(out, path2)
    out2 = open_geotiff(path2)
    assert out2.coords['x'].dtype == np.int64
    assert out2.attrs.get('transform') is None
    assert out2.attrs.get(_NO_GEOREF_KEY) is True


def test_windowed_no_georef_roundtrip_with_marker(tmp_path):
    # When a caller explicitly opts into no-georef writes via
    # ``attrs[_NO_GEOREF_KEY] = True``, the windowed-offset arange
    # pattern that windowed reads return round-trips cleanly. Without
    # the marker, the same coord values write a real transform
    # (covered by ``test_both_axes_ascending_int64_step1_round_trips``).
    from xrspatial.geotiff._coords import _NO_GEOREF_KEY
    da = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        coords={
            'y': np.arange(10, 13, dtype=np.int64),
            'x': np.arange(20, 24, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2087_windowed.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.coords['x'].dtype == np.int64
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True
