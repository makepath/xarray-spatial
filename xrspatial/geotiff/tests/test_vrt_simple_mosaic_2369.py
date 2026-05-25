"""Positive coverage for simple VRT mosaics over GeoTIFF sources (#2369).

Part of epic #2342. The release contract lists "simple GDAL VRT mosaics
backed by GeoTIFF sources" as a supported subset on both the eager
numpy and dask read paths. Other VRT tests in this directory hit that
subset indirectly while exercising codecs, dtypes, or chunk policies,
but nothing pins the contract down on its own.

This module covers the supported mosaic shape explicitly:

* 2x1 horizontal mosaic of two compatible tiles
* 2x2 mosaic of four compatible tiles
* Windowed read where the request window lines up with source pixels
* Dask read of both mosaics with multiple chunks
* One multi-band 2x1 mosaic

Each test asserts values, coords, and key attrs (``crs``, ``transform``,
``nodata``). Shape-only assertions are not enough -- the point of these
tests is that the contract holds at the pixel level.

Implementation uses ``to_geotiff`` to build the source tiles and the
internal ``write_vrt`` to wire up the mosaic XML, mirroring the pattern
already used in ``test_vrt_lazy_chunks_1814.py``.
"""
from __future__ import annotations

import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_vrt, to_geotiff
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


# ---------------------------------------------------------------------------
# Tile builders
# ---------------------------------------------------------------------------

# Per-tile pixel size and CRS shared by every mosaic in this module.
# Picking constants once keeps the assertion blocks below short and makes
# transform comparisons unambiguous.
_PIXEL_W = 0.001
_PIXEL_H = -0.001
_CRS = 4326
_NODATA = -9999.0


def _make_tile(
    tmp_dir,
    name: str,
    data: np.ndarray,
    origin_x: float,
    origin_y: float,
    *,
    nodata: float | None = _NODATA,
) -> str:
    """Write ``data`` as a single-band GeoTIFF anchored at the given origin.

    Returns the on-disk path. ``data`` shape is ``(H, W)``.
    """
    height, width = data.shape
    y = np.array([origin_y + _PIXEL_H * (i + 0.5) for i in range(height)])
    x = np.array([origin_x + _PIXEL_W * (j + 0.5) for j in range(width)])
    attrs = {'crs': _CRS}
    if nodata is not None:
        attrs['nodata'] = nodata
    raster = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs=attrs,
    )
    path = os.path.join(tmp_dir, name)
    to_geotiff(raster, path, nodata=nodata)
    return path


def _make_multiband_tile(
    tmp_dir,
    name: str,
    data: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> str:
    """Write a multi-band GeoTIFF anchored at the given origin.

    ``data`` shape is ``(H, W, B)``.
    """
    height, width, nbands = data.shape
    y = np.array([origin_y + _PIXEL_H * (i + 0.5) for i in range(height)])
    x = np.array([origin_x + _PIXEL_W * (j + 0.5) for j in range(width)])
    raster = xr.DataArray(
        data,
        dims=['y', 'x', 'band'],
        coords={'y': y, 'x': x, 'band': np.arange(nbands)},
        attrs={'crs': _CRS},
    )
    path = os.path.join(tmp_dir, name)
    to_geotiff(raster, path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mosaic_2x1(tmp_path):
    """Two 32x32 float32 tiles side-by-side, west and east.

    Yields ``(vrt_path, expected_array, origin_x, origin_y)``. The
    expected array is the horizontal concatenation of the two source
    arrays.
    """
    td = tmp_path / 'tmp_2369_2x1'
    td.mkdir()
    td = str(td)
    height, width = 32, 32
    left_data = np.arange(height * width, dtype=np.float32).reshape(height, width)
    right_data = (
        left_data + (height * width)
    ).astype(np.float32)

    origin_x, origin_y = -120.0, 45.0
    left_path = _make_tile(td, 'left.tif', left_data, origin_x, origin_y)
    right_path = _make_tile(
        td, 'right.tif', right_data,
        origin_x + _PIXEL_W * width, origin_y,
    )

    vrt_path = os.path.join(td, 'mosaic_2x1.vrt')
    _write_vrt_internal(vrt_path, [left_path, right_path])

    expected = np.concatenate([left_data, right_data], axis=1)
    yield vrt_path, expected, origin_x, origin_y


@pytest.fixture
def mosaic_2x2(tmp_path):
    """Four 32x32 float32 tiles arranged 2 rows by 2 cols.

    Yields ``(vrt_path, expected_array, origin_x, origin_y)`` with the
    expected array stitched in (row, col) order.
    """
    td = tmp_path / 'tmp_2369_2x2'
    td.mkdir()
    td = str(td)
    h, w = 32, 32
    # Use distinct constant values per tile so any swap shows up loudly
    # in the assertion diff.
    tile_nw = np.full((h, w), 1.0, dtype=np.float32)
    tile_ne = np.full((h, w), 2.0, dtype=np.float32)
    tile_sw = np.full((h, w), 3.0, dtype=np.float32)
    tile_se = np.full((h, w), 4.0, dtype=np.float32)

    origin_x, origin_y = -120.0, 45.0
    nw_path = _make_tile(td, 'nw.tif', tile_nw, origin_x, origin_y)
    ne_path = _make_tile(
        td, 'ne.tif', tile_ne,
        origin_x + _PIXEL_W * w, origin_y,
    )
    sw_path = _make_tile(
        td, 'sw.tif', tile_sw,
        origin_x, origin_y + _PIXEL_H * h,
    )
    se_path = _make_tile(
        td, 'se.tif', tile_se,
        origin_x + _PIXEL_W * w, origin_y + _PIXEL_H * h,
    )

    vrt_path = os.path.join(td, 'mosaic_2x2.vrt')
    _write_vrt_internal(
        vrt_path, [nw_path, ne_path, sw_path, se_path],
    )

    top = np.concatenate([tile_nw, tile_ne], axis=1)
    bottom = np.concatenate([tile_sw, tile_se], axis=1)
    expected = np.concatenate([top, bottom], axis=0)
    yield vrt_path, expected, origin_x, origin_y


@pytest.fixture
def mosaic_multiband_2x1(tmp_path):
    """Two 3-band 32x32 float32 tiles side-by-side."""
    td = tmp_path / 'tmp_2369_mb_2x1'
    td.mkdir()
    td = str(td)
    h, w, b = 32, 32, 3
    rng = np.random.default_rng(2369)
    left_data = rng.random((h, w, b), dtype=np.float32)
    right_data = rng.random((h, w, b), dtype=np.float32)

    origin_x, origin_y = -120.0, 45.0
    left_path = _make_multiband_tile(
        td, 'left_mb.tif', left_data, origin_x, origin_y,
    )
    right_path = _make_multiband_tile(
        td, 'right_mb.tif', right_data,
        origin_x + _PIXEL_W * w, origin_y,
    )

    vrt_path = os.path.join(td, 'mosaic_mb.vrt')
    _write_vrt_internal(vrt_path, [left_path, right_path])

    # read_vrt mirrors the on-disk layout: per-band plane stacked on
    # the trailing axis to match what ``to_geotiff`` wrote, so the
    # expected mosaic is (H, W, B).
    expected = np.stack(
        [
            np.concatenate([left_data[..., k], right_data[..., k]], axis=1)
            for k in range(b)
        ],
        axis=-1,
    )
    yield vrt_path, expected, origin_x, origin_y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_attrs_ok(
    result,
    *,
    expected_nodata=None,
    expected_origin_x=None,
    expected_origin_y=None,
):
    """Common attr assertions for VRT reads in this module.

    Checks that ``crs`` and ``transform`` are present and consistent
    with the fixture constants, and optionally that ``nodata`` matches.
    When ``expected_origin_x`` / ``expected_origin_y`` are passed, the
    transform's origin entries are checked too -- pixel size alone is
    not enough to catch a translation bug.
    """
    assert 'crs' in result.attrs, (
        f"crs missing from attrs; have {sorted(result.attrs)}"
    )
    # write_vrt may normalise the CRS to a WKT string; accept either
    # form as long as it is non-empty and references EPSG:4326.
    crs_val = result.attrs['crs']
    if isinstance(crs_val, int):
        assert crs_val == _CRS
    else:
        assert crs_val, "crs attr is present but empty"
        # Best-effort string check: WKT for 4326 mentions WGS 84
        assert 'WGS' in str(crs_val) or '4326' in str(crs_val), (
            f"crs attr does not look like EPSG:4326: {crs_val!r}"
        )

    assert 'transform' in result.attrs, (
        f"transform missing from attrs; have {sorted(result.attrs)}"
    )
    transform = result.attrs['transform']
    # transform format is (pixel_w, 0, origin_x, 0, pixel_h, origin_y)
    # in the affine 6-tuple convention this module uses elsewhere.
    assert len(transform) == 6, (
        f"transform should be a 6-tuple, got {transform!r}"
    )
    assert transform[0] == pytest.approx(_PIXEL_W), (
        f"transform pixel width = {transform[0]}, expected {_PIXEL_W}"
    )
    assert transform[4] == pytest.approx(_PIXEL_H), (
        f"transform pixel height = {transform[4]}, expected {_PIXEL_H}"
    )
    if expected_origin_x is not None:
        assert transform[2] == pytest.approx(expected_origin_x), (
            f"transform origin_x = {transform[2]}, "
            f"expected {expected_origin_x}"
        )
    if expected_origin_y is not None:
        assert transform[5] == pytest.approx(expected_origin_y), (
            f"transform origin_y = {transform[5]}, "
            f"expected {expected_origin_y}"
        )

    if expected_nodata is not None:
        assert 'nodata' in result.attrs, (
            f"nodata missing from attrs; have {sorted(result.attrs)}"
        )
        assert result.attrs['nodata'] == pytest.approx(expected_nodata)


def _assert_coords_monotonic(result, *, expected_origin_x, expected_origin_y):
    """Check that x/y coords are monotonic and start at the expected origin
    (within half a pixel: TIFF coords are pixel centers, not corners).
    """
    x = np.asarray(result['x'].values)
    y = np.asarray(result['y'].values)

    # x increases west to east, y decreases north to south.
    assert np.all(np.diff(x) > 0), "x coord is not strictly increasing"
    assert np.all(np.diff(y) < 0), "y coord is not strictly decreasing"

    # First pixel center: origin + half pixel.
    assert x[0] == pytest.approx(expected_origin_x + _PIXEL_W * 0.5)
    assert y[0] == pytest.approx(expected_origin_y + _PIXEL_H * 0.5)


# ---------------------------------------------------------------------------
# 1. Eager numpy reads of horizontal and 2x2 mosaics
# ---------------------------------------------------------------------------

def test_eager_2x1_mosaic_values_coords_attrs(mosaic_2x1):
    """Eager read of a 2x1 horizontal mosaic returns the concatenated
    pixel block, with monotonic coords and the fixture's crs / transform
    / nodata on attrs.
    """
    vrt_path, expected, ox, oy = mosaic_2x1

    result = read_vrt(vrt_path)

    assert result.shape == expected.shape, (
        f"eager 2x1 shape {result.shape}, expected {expected.shape}"
    )
    np.testing.assert_array_equal(result.values, expected)
    _assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _assert_attrs_ok(
        result,
        expected_nodata=_NODATA,
        expected_origin_x=ox,
        expected_origin_y=oy,
    )


def test_eager_2x2_mosaic_values_coords_attrs(mosaic_2x2):
    """Eager read of a 2x2 mosaic stitches tiles in the right order.

    Each tile has a distinct constant value, so a misordered placement
    surfaces immediately in the value assertion rather than appearing
    only as a numeric diff.
    """
    vrt_path, expected, ox, oy = mosaic_2x2

    result = read_vrt(vrt_path)

    assert result.shape == expected.shape, (
        f"eager 2x2 shape {result.shape}, expected {expected.shape}"
    )
    np.testing.assert_array_equal(result.values, expected)
    _assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _assert_attrs_ok(
        result,
        expected_nodata=_NODATA,
        expected_origin_x=ox,
        expected_origin_y=oy,
    )


# ---------------------------------------------------------------------------
# 2. Windowed read that maps cleanly into source windows
# ---------------------------------------------------------------------------

def test_windowed_read_aligned_with_source_boundary(mosaic_2x1):
    """A window crossing the seam between the two source tiles returns
    the same pixels as slicing the full mosaic.

    The window picked here covers the right half of the left tile and
    the left half of the right tile: both halves land on whole-pixel
    boundaries inside their respective sources, so this is the
    "request lines up with source pixels" case from the issue.
    """
    vrt_path, expected, ox, oy = mosaic_2x1
    h = expected.shape[0]  # 32

    # Window: full height, last 16 cols of left tile through first
    # 16 cols of right tile. 32 / 2 = 16 keeps each side aligned to
    # the source's own pixel grid.
    r0, c0, r1, c1 = 0, 16, h, 48

    result = read_vrt(vrt_path, window=(r0, c0, r1, c1))

    np.testing.assert_array_equal(result.values, expected[r0:r1, c0:c1])

    # Coords on the windowed result should correspond to the windowed
    # slice of the full mosaic's coords.
    full = read_vrt(vrt_path)
    np.testing.assert_array_equal(
        np.asarray(result['x'].values),
        np.asarray(full['x'].values)[c0:c1],
    )
    np.testing.assert_array_equal(
        np.asarray(result['y'].values),
        np.asarray(full['y'].values)[r0:r1],
    )
    # Windowed transform origin shifts by the row / col offset times
    # pixel size. (r0=0 keeps origin_y at the fixture's oy.)
    expected_window_ox = ox + _PIXEL_W * c0
    expected_window_oy = oy + _PIXEL_H * r0
    _assert_attrs_ok(
        result,
        expected_nodata=_NODATA,
        expected_origin_x=expected_window_ox,
        expected_origin_y=expected_window_oy,
    )


# ---------------------------------------------------------------------------
# 3. Dask reads with multiple chunks
# ---------------------------------------------------------------------------

def test_dask_2x1_mosaic_multi_chunk_matches_eager(mosaic_2x1):
    """Dask read with chunks smaller than the mosaic returns the same
    pixels as the eager read, and uses a real multi-block dask graph.
    """
    vrt_path, expected, ox, oy = mosaic_2x1

    chunked = read_vrt(vrt_path, chunks=(16, 16))

    # Real dask graph with the expected number of blocks: 32/16 = 2
    # rows, 64/16 = 4 cols.
    assert isinstance(chunked.data, da.Array), (
        f"expected dask Array, got {type(chunked.data).__name__}"
    )
    assert chunked.data.numblocks == (2, 4), (
        f"expected 2x4 blocks, got {chunked.data.numblocks}"
    )

    computed = chunked.compute()
    np.testing.assert_array_equal(computed.values, expected)
    _assert_coords_monotonic(computed, expected_origin_x=ox, expected_origin_y=oy)
    _assert_attrs_ok(
        computed,
        expected_nodata=_NODATA,
        expected_origin_x=ox,
        expected_origin_y=oy,
    )


def test_dask_2x2_mosaic_multi_chunk_matches_eager(mosaic_2x2):
    """Dask read of the 2x2 mosaic with chunk size below tile size.

    Chunks of 16 split each 32x32 tile into 2x2 blocks. The full
    mosaic is 64x64 so the resulting dask array is 4x4 blocks.
    """
    vrt_path, expected, ox, oy = mosaic_2x2

    chunked = read_vrt(vrt_path, chunks=(16, 16))

    assert isinstance(chunked.data, da.Array)
    assert chunked.data.numblocks == (4, 4), (
        f"expected 4x4 blocks, got {chunked.data.numblocks}"
    )

    computed = chunked.compute()
    np.testing.assert_array_equal(computed.values, expected)
    _assert_coords_monotonic(computed, expected_origin_x=ox, expected_origin_y=oy)
    _assert_attrs_ok(
        computed,
        expected_nodata=_NODATA,
        expected_origin_x=ox,
        expected_origin_y=oy,
    )


# ---------------------------------------------------------------------------
# 4. Multi-band mosaic
# ---------------------------------------------------------------------------

def test_eager_multiband_2x1_mosaic(mosaic_multiband_2x1):
    """Eager read of a multi-band 2x1 mosaic returns one stitched plane
    per band.

    Multi-band VRT reads return shape ``(H, W, B)`` to match the
    on-disk layout; assert per-band values against the stack built in
    the fixture.
    """
    vrt_path, expected, ox, oy = mosaic_multiband_2x1

    result = read_vrt(vrt_path)

    # Multi-band path: trailing axis is band.
    assert result.shape == expected.shape, (
        f"multiband 2x1 shape {result.shape}, expected {expected.shape}"
    )
    np.testing.assert_array_equal(result.values, expected)

    # Coords use the same x/y dims as the single-band case; only the
    # number of bands changed.
    _assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _assert_attrs_ok(
        result, expected_origin_x=ox, expected_origin_y=oy,
    )


def test_dask_multiband_2x1_mosaic_matches_eager(mosaic_multiband_2x1):
    """Dask read of the multi-band 2x1 mosaic with sub-tile chunks must
    match the eager read pixel-for-pixel across every band.

    Chunking exercises per-block band handling: a bug that loses a
    band on one chunk but not another would not appear in the eager
    test above.
    """
    vrt_path, expected, ox, oy = mosaic_multiband_2x1

    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(16, 16))

    # The chunked array should be dask-backed; the band axis may or
    # may not be split depending on the implementation, but the
    # spatial axes must be.
    assert isinstance(chunked.data, da.Array), (
        f"expected dask Array, got {type(chunked.data).__name__}"
    )

    computed = chunked.compute()
    assert computed.shape == eager.shape
    np.testing.assert_array_equal(computed.values, eager.values)
    np.testing.assert_array_equal(computed.values, expected)
    _assert_coords_monotonic(
        computed, expected_origin_x=ox, expected_origin_y=oy,
    )
    _assert_attrs_ok(
        computed, expected_origin_x=ox, expected_origin_y=oy,
    )
