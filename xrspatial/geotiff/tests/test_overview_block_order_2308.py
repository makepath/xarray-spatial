"""COG overview tile-block ordering invariant (issue #2308).

The COG spec requires the on-disk pixel-data layout to run from the
smallest overview through progressively larger overviews and end with
the main-resolution image. External readers (rio-cogeo, GDAL's
``validate_cloud_optimized_geotiff``) flag a file when the byte
ordering reverses or interleaves these blocks even though the IFD
chain walks main -> ov1 -> ov2 the conventional way. These tests lock
the byte order in as a regression gate so the writer cannot drift back
to the old layout.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._header import parse_all_ifds, parse_header


def _min_block_offset(ifd) -> int:
    """Return the smallest tile-offset (or strip-offset) for an IFD."""
    offsets = ifd.tile_offsets
    if offsets is None:
        offsets = ifd.strip_offsets
    assert offsets is not None and len(offsets) > 0, (
        "IFD has neither tile_offsets nor strip_offsets")
    return min(offsets)


def _read_block_order(path: str) -> list[int]:
    """Return ``min_block_offset`` for each IFD in walk order."""
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return [_min_block_offset(ifd) for ifd in ifds]


def _make_da(shape, bands: int | None = None) -> xr.DataArray:
    """Build a synthetic DataArray with a sane CRS / coordinate grid."""
    rng = np.random.RandomState(17)
    if bands is None:
        arr = rng.rand(*shape).astype("float32")
        dims = ("y", "x")
    else:
        arr = rng.rand(bands, *shape).astype("float32")
        dims = ("band", "y", "x")
    h, w = shape
    coords = {
        "y": np.linspace(45, 44, h),
        "x": np.linspace(-120, -119, w),
    }
    return xr.DataArray(arr, dims=dims, coords=coords, attrs={"crs": 4326})


@pytest.mark.parametrize("bands", [None, 3])
def test_cog_overview_block_order_invariant_2308(bands):
    """Pixel blocks must run smallest-overview -> larger -> main.

    The IFD walk order is ``[main, ov_factor_2, ov_factor_4]`` (full
    resolution first). The on-disk pixel-block order must be the
    reverse: factor-4 overview blocks first, then factor-2 overview
    blocks, with the main-resolution blocks last (issue #2308).
    """
    da = _make_da((256, 256), bands=bands)
    with tempfile.TemporaryDirectory() as td:
        suffix = "rgb" if bands else "mono"
        path = os.path.join(td, f"order_2308_{suffix}.tif")
        to_geotiff(
            da, path, compression="deflate", cog=True,
            tile_size=64, overview_levels=[2, 4],
        )

        block_offsets = _read_block_order(path)
        # IFD walk: [main, ov_factor_2, ov_factor_4]
        main_min, ov2_min, ov4_min = block_offsets
        # COG layout: factor-4 (smallest overview) -> factor-2 -> main.
        assert ov4_min < ov2_min, (
            f"smallest overview blocks should sit before larger "
            f"overview blocks: ov4_min={ov4_min}, ov2_min={ov2_min}")
        assert ov2_min < main_min, (
            f"overview blocks should sit before main-resolution "
            f"blocks: ov2_min={ov2_min}, main_min={main_min}")


def test_cog_overview_block_order_three_levels_2308():
    """Same invariant with three overview levels (factor 2/4/8)."""
    da = _make_da((512, 512))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "order_2308_three.tif")
        to_geotiff(
            da, path, compression="deflate", cog=True,
            tile_size=64, overview_levels=[2, 4, 8],
        )

        block_offsets = _read_block_order(path)
        # IFD walk: [main, ov2, ov4, ov8]
        main_min, ov2_min, ov4_min, ov8_min = block_offsets
        # On-disk: ov8 -> ov4 -> ov2 -> main
        assert ov8_min < ov4_min < ov2_min < main_min, (
            f"COG block order broken: ov8={ov8_min} ov4={ov4_min} "
            f"ov2={ov2_min} main={main_min}")


def _rio_cogeo_or_skip():
    """Skip the rio-cogeo gate when the dependency isn't installed.

    Mirrors the skip semantics used in ``test_cog_writer_compliance``:
    contributor laptops without rio-cogeo see a skip, CI with rio-cogeo
    runs the strict check.
    """
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        pytest.skip("rio-cogeo not installed")
    return cog_validate


@pytest.mark.parametrize("bands", [None, 3])
def test_cog_overview_block_order_rio_cogeo_2308(bands):
    """``rio-cogeo cog_validate`` returns valid=True with no block-order errors."""
    cog_validate = _rio_cogeo_or_skip()
    da = _make_da((256, 256), bands=bands)
    with tempfile.TemporaryDirectory() as td:
        suffix = "rgb" if bands else "mono"
        path = os.path.join(td, f"order_2308_rio_{suffix}.tif")
        to_geotiff(
            da, path, compression="deflate", cog=True,
            tile_size=64, overview_levels=[2, 4],
        )
        valid, errors, _warnings = cog_validate(path, strict=False)
        assert valid, f"rio_cogeo cog_validate failed: {errors}"
        # Defensive secondary assertion: the two block-order messages
        # from #2308 must not reappear even if some future writer
        # change keeps the validator happy on the headline check.
        joined = " ".join(errors).lower()
        assert "offset of the first block" not in joined, (
            f"block-order errors regressed: {errors}")
