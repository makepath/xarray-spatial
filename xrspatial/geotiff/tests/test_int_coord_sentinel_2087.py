"""User-authored integer spatial coords must not silently drop georef (#2087).

The pre-fix sentinel at ``_coords.py:272`` and ``:353`` returned
``None`` from ``coords_to_transform`` (and waved the validator
through) whenever either x or y had an integer dtype. The intent
was to round-trip the read-side ``np.arange(N, dtype=int64)``
placeholder that ``coords_from_pixel_geometry`` emits for files
with no GeoTIFF transform tags. The sentinel was too broad: it
also caught user-authored projected grids with integer-spaced
coords, which lost their georef silently on write.

The tightened sentinel matches only the exact reader pattern:
``int64`` dtype, ascending, contiguous step ``+1`` on both axes.
These tests pin both directions:

1. The legitimate no-georef round-trip still works -- a file
   with no transform tags reads back with int64 coords and
   writes back without inventing a transform.
2. User-authored integer-coord grids are no longer silently
   stripped. ``x=[100,101,102], y=[200,199]`` now writes a real
   transform and round-trips with float coords.
3. Subsets and windowed reads of the no-georef placeholder
   (which still produce ``arange``-shaped int64 arrays) continue
   to round-trip cleanly.
4. Non-uniform int coords raise ``NonUniformCoordsError`` instead
   of silently stripping.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._coords import _is_no_georef_sentinel


# --- Unit checks on the sentinel helper itself --------------------------


@pytest.mark.parametrize(
    "coord",
    [
        np.arange(5, dtype=np.int64),         # full read
        np.arange(3, 8, dtype=np.int64),      # windowed read
        np.arange(0, 1, dtype=np.int64),      # degenerate 1-element
        np.array([10, 11, 12], dtype=np.int64),
    ],
)
def test_sentinel_accepts_arange_int64(coord):
    assert _is_no_georef_sentinel(coord)


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
def test_sentinel_rejects_non_arange(coord):
    assert not _is_no_georef_sentinel(coord)


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


def test_both_axes_ascending_int64_step1_trade_off(tmp_path):
    # Documented trade-off corner: when both axes are int64 ascending
    # step-1 (i.e. both match the read-side arange pattern exactly),
    # the sentinel cannot distinguish a user-authored grid from the
    # read-side no-georef placeholder. The writer resolves to
    # no-georef. A caller wanting georef on this pattern must set
    # ``attrs['transform']`` explicitly (see the next test).
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    # Both axes match the sentinel exactly (int64 ascending step 1).
    # This is the documented trade-off: ambiguous between
    # "read-side placeholder" and "user-authored arange-shaped grid".
    # The sentinel resolves to no-georef. A caller wanting georef on
    # this pattern must set attrs['transform'] explicitly.
    path = str(tmp_path / "tmp_2087_both_arange.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    # No-georef round-trip: coords come back as int64 0..N-1
    # placeholders, not the original values.
    assert out.coords['x'].dtype == np.int64
    assert out.coords['y'].dtype == np.int64


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
    # stripped georef. Under the tightened sentinel it falls through
    # to ``coords_to_transform`` and trips the uniform-spacing check.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_non_uniform.tif")
    with pytest.raises(ValueError, match="not uniformly spaced"):
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
    # Build a no-georef file via the read-side path: write a file with
    # ``attrs['transform']`` explicitly cleared, then re-read it.
    # ``open_geotiff`` returns int64 ``np.arange``-shaped coords; the
    # next write must not invent a transform.
    src = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.arange(4, dtype=np.int64), 'x': np.arange(4, dtype=np.int64)},
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_no_georef.tif")
    to_geotiff(src, path)

    out = open_geotiff(path)
    # Verify the read came back as no-georef.
    assert out.coords['x'].dtype == np.int64
    assert out.coords['y'].dtype == np.int64
    assert out.attrs.get('transform') is None

    # Round-trip: write again. No transform should be invented.
    path2 = str(tmp_path / "tmp_2087_no_georef_rt.tif")
    to_geotiff(out, path2)
    out2 = open_geotiff(path2)
    assert out2.coords['x'].dtype == np.int64
    assert out2.attrs.get('transform') is None


def test_windowed_no_georef_roundtrip(tmp_path):
    # The read-side emits ``np.arange(c0, c1, dtype=int64)`` for
    # windowed reads, so the sentinel must accept arange starting at
    # any integer, not just 0. Test by writing an array whose coords
    # mimic a windowed no-georef read.
    da = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        coords={
            'y': np.arange(10, 13, dtype=np.int64),
            'x': np.arange(20, 24, dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2087_windowed.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    # Under the documented trade-off this is still treated as
    # no-georef (both axes match the arange pattern). Coords come
    # back as 0..N-1 placeholders.
    assert out.coords['x'].dtype == np.int64
    assert out.attrs.get('transform') is None
