"""Dask read of a TIFF with overview levels (COG pyramid).

``read_geotiff_dask`` accepts an ``overview_level`` kwarg that forwards
to ``_read_geo_info`` for IFD selection and to ``_delayed_read_window``
for per-chunk decoding. Before this module landed, no test exercised
that combination, so a regression where the dask path silently ignored
the overview level (returning full-resolution chunks) or sampled the
wrong IFD would not be caught.

This module writes a small COG-style file with two overview levels
(half- and quarter-resolution) and asserts that:

  * The returned ``DataArray`` shape matches the overview level's
    dimensions, not the full-resolution dimensions.
  * The per-chunk windowed reader pulls bytes from the correct IFD
    (the computed values agree with a non-dask reference at the same
    overview level).
  * ``overview_level=None`` (default) still returns the full-resolution
    image, so the new code path does not change default behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest


tifffile = pytest.importorskip("tifffile")
dask_array = pytest.importorskip("dask.array")


def _write_cog_with_overviews(path: str, data: np.ndarray) -> None:
    """Write *data* as a tiled TIFF with two precomputed overview IFDs.

    Writes the primary IFD followed by half- and quarter-resolution
    overview IFDs, each tagged ``subfiletype=1`` so the reader treats
    them as a pyramid (matching how ``_write_normal_cog`` in
    ``test_overview_filter.py`` builds COG fixtures). This mirrors what
    GDAL's ``gdaladdo`` emits.
    """
    half = data[::2, ::2]
    quart = data[::4, ::4]
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
        tw.write(quart, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)


def test_dask_overview_level_zero_matches_full_res(tmp_path):
    """``overview_level=0`` returns full resolution (the base IFD)."""
    from xrspatial.geotiff import read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov.tif")
    _write_cog_with_overviews(path, arr)

    da_arr = read_geotiff_dask(path, chunks=32, overview_level=0)
    assert da_arr.shape == arr.shape
    np.testing.assert_array_equal(da_arr.compute().values, arr)


def test_dask_overview_level_one_returns_half_res(tmp_path):
    """``overview_level=1`` materialises the half-resolution overview."""
    from xrspatial.geotiff import read_geotiff_dask
    from xrspatial.geotiff import open_geotiff

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov1.tif")
    _write_cog_with_overviews(path, arr)

    # Eager reference at the same overview level -- the dask path should
    # pull the same bytes from the same IFD.
    eager = open_geotiff(path, overview_level=1)

    da_arr = read_geotiff_dask(path, chunks=16, overview_level=1)
    assert da_arr.shape == eager.shape, (
        f"dask returned {da_arr.shape} but eager returned {eager.shape} "
        "at overview_level=1"
    )
    assert isinstance(da_arr.data, dask_array.Array)
    np.testing.assert_array_equal(da_arr.compute().values, eager.values)


def test_dask_overview_level_two_returns_quarter_res(tmp_path):
    """``overview_level=2`` materialises the quarter-resolution overview."""
    from xrspatial.geotiff import read_geotiff_dask
    from xrspatial.geotiff import open_geotiff

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov2.tif")
    _write_cog_with_overviews(path, arr)

    eager = open_geotiff(path, overview_level=2)

    da_arr = read_geotiff_dask(path, chunks=8, overview_level=2)
    assert da_arr.shape == eager.shape
    np.testing.assert_array_equal(da_arr.compute().values, eager.values)


def test_dask_overview_level_none_returns_full_res(tmp_path):
    """``overview_level=None`` keeps default behaviour: full resolution."""
    from xrspatial.geotiff import read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov_none.tif")
    _write_cog_with_overviews(path, arr)

    da_arr = read_geotiff_dask(path, chunks=32, overview_level=None)
    assert da_arr.shape == arr.shape
    np.testing.assert_array_equal(da_arr.compute().values, arr)
