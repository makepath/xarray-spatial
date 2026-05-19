"""User int64 step-1 grids must keep their georef on round-trip (#2120).

#2087 tightened the no-georef detection from "any integer dtype" to
"int64 ascending step-1 on both axes", which fixed the original silent
strip for non-arange grids. A smaller niche still tripped over the
shape-based check: user grids with int64 step-1 coords starting at a
non-zero offset (e.g. ``x=[500,501,502], y=[1000,1001]``) still matched
the reader's placeholder pattern and were silently emitted as no-georef.

The fix moves the placeholder signal to ``attrs[_NO_GEOREF_KEY]``
(stamped by the reader, checked by the writer). These tests pin the
new contract:

1. A user-authored int64 step-1 grid with no marker keeps CRS and gets
   a synthesised unit transform.
2. Round-tripping a real no-georef file (where the reader stamps the
   marker) still produces a no-transform output.
3. Manually opting in to no-georef via the marker on a user-built
   DataArray writes without a transform, even with integer coords.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._coords import _NO_GEOREF_KEY


def test_int64_step1_user_grid_keeps_crs_on_round_trip(tmp_path):
    x = np.array([500, 501, 502], dtype=np.int64)
    y = np.array([1000, 1001], dtype=np.int64)
    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        coords={'y': y, 'x': x}, dims=('y', 'x'),
        attrs={'crs': 4326},
    )

    path = str(tmp_path / "tmp_2120_user_int64_grid.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)

    # Coord values round-trip exactly, dtype flips int -> float because
    # the file now carries a real transform.
    np.testing.assert_array_equal(out.coords['x'].values, [500.0, 501.0, 502.0])
    np.testing.assert_array_equal(out.coords['y'].values, [1000.0, 1001.0])
    assert out.attrs.get('crs') == 4326
    assert out.attrs.get('transform') is not None
    # The marker must not be set on a georef read.
    assert _NO_GEOREF_KEY not in out.attrs


def test_no_georef_file_carries_marker_and_round_trips(tmp_path):
    # Build a no-georef file via the explicit-marker write path.
    src = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.arange(4, dtype=np.int64),
                'x': np.arange(4, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2120_no_georef.tif")
    to_geotiff(src, path)

    out = open_geotiff(path)
    assert out.attrs.get(_NO_GEOREF_KEY) is True
    assert out.attrs.get('transform') is None
    assert out.coords['x'].dtype == np.int64

    # Round-trip preserves the marker and the absence of a transform.
    path2 = str(tmp_path / "tmp_2120_no_georef_rt.tif")
    to_geotiff(out, path2)
    out2 = open_geotiff(path2)
    assert out2.attrs.get(_NO_GEOREF_KEY) is True
    assert out2.attrs.get('transform') is None


def test_int64_step1_user_grid_without_marker_writes_transform(tmp_path):
    # Direct repro of the silent-strip the pre-fix code would emit.
    # x and y both match the reader's arange placeholder shape exactly,
    # but the marker is absent, so the writer treats this as a normal
    # georef grid and synthesises a unit transform.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2120_step1_no_marker.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(out.coords['y'].values, [200.0, 201.0, 202.0])


def test_marker_on_user_grid_skips_transform_synthesis(tmp_path):
    # A caller can opt into a no-georef write on an int-coord array by
    # setting the marker explicitly. The writer trusts the marker
    # rather than re-deriving a transform from the coords.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.arange(3, dtype=np.int64),
            'x': np.arange(3, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2120_explicit_marker.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True
