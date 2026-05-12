"""Regression tests for issue #1733.

``write_vrt`` previously trusted the first source for resolution,
sample format + bps (dtype), band count, and CRS. A mismatched source
would silently produce a VRT that placed pixels incorrectly or
re-interpreted bytes as the wrong dtype downstream.

These tests assert that ``write_vrt`` now rejects mismatched sources
with a clear ``ValueError`` covering each of those properties, and
still accepts sources that match within a small float tolerance on
pixel size.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import write_vrt


def _unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f"vrt_1733_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_tif(path: str, *, h: int, w: int, dtype, bands: int = 1,
               px: float = 1.0, py: float = -1.0,
               origin_x: float = 0.0, origin_y: float = 100.0,
               crs: int | None = 4326) -> None:
    if bands == 1:
        arr = np.arange(h * w, dtype=dtype).reshape(h, w)
        dims = ['y', 'x']
    else:
        arr = np.arange(h * w * bands, dtype=dtype).reshape(h, w, bands)
        dims = ['y', 'x', 'band']
    y = origin_y + (np.arange(h) + 0.5) * py
    x = origin_x + (np.arange(w) + 0.5) * px
    coords = {'y': y, 'x': x}
    attrs = {}
    if crs is not None:
        attrs['crs'] = crs
    da = xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)
    to_geotiff(da, path, compression='none')


def test_mismatched_pixel_size_raises(tmp_path):
    d = _unique_dir(tmp_path, "px")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, px=1.0, py=-1.0)
    # Place b adjacent so the geometry would otherwise work, but the
    # pixel size disagrees.
    _write_tif(b, h=4, w=4, dtype=np.float32, px=2.0, py=-2.0,
               origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="pixel size"):
        write_vrt(vrt, [a, b])


def test_mismatched_dtype_raises(tmp_path):
    d = _unique_dir(tmp_path, "dtype")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    _write_tif(b, h=4, w=4, dtype=np.int16, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="dtype|sample_format|bps"):
        write_vrt(vrt, [a, b])


def test_mismatched_band_count_raises(tmp_path):
    d = _unique_dir(tmp_path, "bands")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, bands=1)
    _write_tif(b, h=4, w=4, dtype=np.float32, bands=3, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="band count"):
        write_vrt(vrt, [a, b])


def test_compatible_sources_succeed(tmp_path):
    d = _unique_dir(tmp_path, "ok")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_pixel_size_within_tolerance_accepted(tmp_path):
    d = _unique_dir(tmp_path, "tol")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, px=1.0, py=-1.0)
    # Drift well below the 1e-6 relative tolerance.
    _write_tif(b, h=4, w=4, dtype=np.float32,
               px=1.0 + 1e-10, py=-1.0, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_single_source_still_works(tmp_path):
    d = _unique_dir(tmp_path, "one")
    a = os.path.join(d, "a.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a])
    assert os.path.exists(vrt)


def test_mismatched_crs_raises(tmp_path):
    # Two sources with different non-empty CRS values must be rejected,
    # otherwise the VRT would inherit the first source's CRS and silently
    # misproject the second.
    d = _unique_dir(tmp_path, "crs_diff")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=3857)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        write_vrt(vrt, [a, b])


def test_asymmetric_crs_raises_first_set_second_missing(tmp_path):
    # First source has a CRS, second is written without one. The VRT
    # would otherwise be tagged with the first source's CRS, which can
    # misplace data when the second source actually came from a
    # different (or unknown) projection.
    d = _unique_dir(tmp_path, "crs_first")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=None)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        write_vrt(vrt, [a, b])


def test_asymmetric_crs_raises_first_missing_second_set(tmp_path):
    # Symmetric case: first source missing a CRS, second has one. The
    # earlier guard only triggered when both sides were set, so this
    # would have silently produced an untagged VRT despite one source
    # carrying a known projection.
    d = _unique_dir(tmp_path, "crs_second")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=None)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=4326)
    vrt = os.path.join(d, "out.vrt")
    with pytest.raises(ValueError, match="CRS"):
        write_vrt(vrt, [a, b])


def test_matching_crs_succeeds(tmp_path):
    # Sanity check: two sources with the same CRS should still be
    # accepted (defends against an overly aggressive equality check).
    d = _unique_dir(tmp_path, "crs_match")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=4326)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=4326)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)


def test_both_missing_crs_succeeds(tmp_path):
    # If neither source has a CRS, the VRT just won't be tagged with one
    # and there's nothing to mis-tag. This must not raise.
    d = _unique_dir(tmp_path, "crs_both_missing")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_tif(a, h=4, w=4, dtype=np.float32, crs=None)
    _write_tif(b, h=4, w=4, dtype=np.float32, origin_x=4.0, crs=None)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b])
    assert os.path.exists(vrt)
