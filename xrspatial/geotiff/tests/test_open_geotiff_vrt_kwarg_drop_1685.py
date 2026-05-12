"""Regression test for #1685: ``open_geotiff`` silently dropped
``overview_level`` and ``on_gpu_failure`` when the source was a VRT.

The api-consistency sweep on 2026-05-12 flagged that ``open_geotiff``
documents both kwargs as supported, but the VRT dispatch branch routes
to ``read_vrt`` whose signature accepts neither. Calls like
``open_geotiff('mosaic.vrt', overview_level=2)`` returned full-resolution
data with no warning. Issue #1561 fixed the same class of bug for the
dask and GPU dispatch branches; this one closes the remaining gap by
refusing the unsupported combinations up front.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff, write_vrt


@pytest.fixture
def small_vrt(tmp_path):
    """Two-tile uint16 VRT we can hand to ``open_geotiff``."""
    arr_a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da_a = xr.DataArray(
        arr_a,
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([0.5, 1.5, 2.5, 3.5]),
        },
        attrs={"crs": 4326},
    )
    tile_a = tmp_path / "tile_a.tif"
    to_geotiff(da_a, str(tile_a))

    arr_b = np.arange(16, 32, dtype=np.uint16).reshape(4, 4)
    da_b = xr.DataArray(
        arr_b,
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([4.5, 5.5, 6.5, 7.5]),
        },
        attrs={"crs": 4326},
    )
    tile_b = tmp_path / "tile_b.tif"
    to_geotiff(da_b, str(tile_b))

    vrt_path = tmp_path / "mosaic.vrt"
    write_vrt(str(vrt_path), [str(tile_a), str(tile_b)])
    return str(vrt_path)


def test_open_geotiff_vrt_rejects_overview_level(small_vrt):
    """``overview_level`` plus ``.vrt`` raises ValueError, not a silent drop."""
    with pytest.raises(ValueError, match="overview_level is not supported"):
        open_geotiff(small_vrt, overview_level=1)


def test_open_geotiff_vrt_accepts_overview_level_zero(small_vrt):
    """``overview_level=0`` is documented as full resolution (the default),
    so passing it on a VRT is semantically equivalent to omitting the kwarg
    and must not raise. Only non-zero overview levels are rejected.
    """
    da = open_geotiff(small_vrt, overview_level=0)
    # Same shape as the no-kwarg case: two 4x4 tiles side-by-side.
    assert da.shape == (4, 8)


def test_open_geotiff_vrt_rejects_on_gpu_failure_with_gpu_true(small_vrt):
    """``on_gpu_failure='strict'`` plus ``.vrt`` (gpu=True) is refused."""
    # The check fires before any GPU code runs; no CUDA needed.
    with pytest.raises(ValueError, match="on_gpu_failure is not supported"):
        open_geotiff(small_vrt, gpu=True, on_gpu_failure="strict")


def test_open_geotiff_vrt_without_unsupported_kwargs_still_works(small_vrt):
    """The previously-accepted kwargs still flow through to ``read_vrt``."""
    da = open_geotiff(small_vrt)
    # Two 4x4 tiles side-by-side; result is 4x8.
    assert da.shape == (4, 8)


def test_open_geotiff_vrt_with_window_still_works(small_vrt):
    """``window`` was already forwarded; this regression should not break it."""
    da = open_geotiff(small_vrt, window=(0, 1, 4, 5))
    assert da.shape == (4, 4)


def test_open_geotiff_non_vrt_still_accepts_overview_level(tmp_path):
    """The fix is VRT-specific; ``.tif`` sources keep accepting overview_level."""
    # Build a single COG with one overview so overview_level=0 round-trips.
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "y": np.arange(8, dtype=np.float64),
            "x": np.arange(8, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )
    tif_path = tmp_path / "with_ovr.tif"
    to_geotiff(da, str(tif_path), cog=True, tile_size=4, overview_levels=[2])
    # Either overview_level value must be accepted without raising.
    open_geotiff(str(tif_path), overview_level=0)
    open_geotiff(str(tif_path), overview_level=1)
