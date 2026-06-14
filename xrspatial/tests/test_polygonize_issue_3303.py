"""Tests for issue #3303: jitted _region_ranges_scan in the float dask
chunk path.

The fix replaced the pure-Python per-pixel loop in
``_compute_region_value_ranges`` with the jitted ``_region_ranges_scan``,
reused the region labels from ``_polygonize_numpy_regions`` instead of
re-running ``_calculate_regions``, and corrected the recorded ``w_match``
/ ``s_match`` flags (the old loop called the numba overload generator
``_is_close`` from pure Python, which returns the implementation function
-- always truthy -- instead of a bool).

These tests pin three things:

1. ``_compute_region_value_ranges`` gives identical results whether the
   caller passes precomputed ``regions`` or lets it recompute them.
2. The recorded boundary-cell flags equal a naive ``_values_close``
   reference (this fails on the pre-#3303 code).
3. Public dask float output still matches numpy across chunkings and
   connectivities.
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.polygonize import (
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _compute_region_value_ranges,
    _polygonize_numpy,
    _polygonize_numpy_regions,
    _values_close,
    polygonize,
)

dask = pytest.importorskip("dask")
import dask.array as da  # noqa: E402


def _reference_ranges(block, mask_block, connectivity_8, atol, rtol, column,
                      row_offset, col_offset, ny_total, nx_total):
    """Naive Python port of the per-pixel aggregation, using
    ``_values_close`` for the W / S flags (the documented semantics)."""
    _, _, regions = _polygonize_numpy_regions(
        block, mask_block, connectivity_8, None, atol=atol, rtol=rtol)

    ny, nx = block.shape
    if np.issubdtype(block.dtype, np.floating):
        nan_mask = ~np.isnan(block)
        if mask_block is not None:
            full_mask = mask_block.astype(bool) & nan_mask
        else:
            full_mask = nan_mask
    else:
        full_mask = mask_block
    if nx == 1:
        nx_eff = 2
        values_flat = np.hstack((block, block)).ravel()
        if full_mask is not None:
            full_mask = np.hstack((full_mask, np.zeros_like(full_mask)))
        else:
            full_mask = np.zeros((ny, 2), dtype=bool)
            full_mask[:, 0] = True
    else:
        nx_eff = nx
        values_flat = block.ravel()
    mask_flat = full_mask.ravel() if full_mask is not None else None

    track = ny_total is not None and nx_total is not None
    left = track and col_offset > 0
    right = track and col_offset + nx < nx_total
    bottom = track and row_offset > 0
    top = track and row_offset + ny < ny_total

    n_regions = len(column)
    val_min = [None] * n_regions
    val_max = [None] * n_regions
    cells_out = [{} for _ in range(n_regions)]
    for ij in range(len(regions)):
        r = regions[ij]
        if r == 0:
            continue
        if mask_flat is not None and not mask_flat[ij]:
            continue
        idx = r - 1
        if idx >= n_regions:
            continue
        v = values_flat[ij]
        if val_min[idx] is None or v < val_min[idx]:
            val_min[idx] = v
        if val_max[idx] is None or v > val_max[idx]:
            val_max[idx] = v
        lc = ij % nx_eff
        lr = ij // nx_eff
        if nx == 1 and lc != 0:
            continue
        on_internal = ((left and lc == 0) or (right and lc == nx - 1) or
                       (bottom and lr == 0) or (top and lr == ny - 1))
        if on_internal:
            w = (lc > 0 and
                 (mask_flat is None or mask_flat[ij - 1]) and
                 bool(_values_close(v, values_flat[ij - 1], atol, rtol)))
            s = (lr > 0 and
                 (mask_flat is None or mask_flat[ij - nx_eff]) and
                 bool(_values_close(v, values_flat[ij - nx_eff], atol, rtol)))
            cells_out[idx][(lc + col_offset, lr + row_offset)] = (
                v, bool(w), bool(s))
    ranges = []
    for i in range(n_regions):
        if val_min[i] is None:
            ranges.append((column[i], column[i]))
        else:
            ranges.append((val_min[i], val_max[i]))
    return ranges, cells_out


def _cases():
    rng = np.random.default_rng(3303)
    a = rng.random((13, 17)) * 1e-4 + 1.0
    a[3:5, 4:9] = np.nan
    m = rng.random((13, 17)) > 0.2
    b = rng.integers(0, 3, (9, 1)).astype(np.float64)
    c = rng.integers(0, 4, (12, 12)).astype(np.float32)
    return [
        ("float-nan", a, None),
        ("float-mask", a, m),
        ("nx1", b, None),
        ("float32", c, None),
    ]


@pytest.mark.parametrize("name,block,mask", _cases())
@pytest.mark.parametrize("connectivity_8", [False, True])
@pytest.mark.parametrize("offsets", [
    (0, 0, None, None),    # single chunk: no internal edges
    (4, 6, 40, 40),        # all four edges internal
    (0, 6, 40, 40),        # bottom edge on the raster boundary
])
def test_precomputed_regions_match_recompute(name, block, mask,
                                             connectivity_8, offsets):
    ro, co, nyt, nxt = offsets
    column, _, regions = _polygonize_numpy_regions(
        block, mask, connectivity_8, None)
    with_regions = _compute_region_value_ranges(
        block, mask, connectivity_8, _DEFAULT_ATOL, _DEFAULT_RTOL, column,
        row_offset=ro, col_offset=co, ny_total=nyt, nx_total=nxt,
        regions=regions)
    without_regions = _compute_region_value_ranges(
        block, mask, connectivity_8, _DEFAULT_ATOL, _DEFAULT_RTOL, column,
        row_offset=ro, col_offset=co, ny_total=nyt, nx_total=nxt)
    assert with_regions == without_regions


@pytest.mark.parametrize("name,block,mask", _cases())
@pytest.mark.parametrize("connectivity_8", [False, True])
def test_ranges_and_flags_match_reference(name, block, mask, connectivity_8):
    ro, co, nyt, nxt = 4, 6, 40, 40
    column, _, regions = _polygonize_numpy_regions(
        block, mask, connectivity_8, None)
    got_ranges, got_cells = _compute_region_value_ranges(
        block, mask, connectivity_8, _DEFAULT_ATOL, _DEFAULT_RTOL, column,
        row_offset=ro, col_offset=co, ny_total=nyt, nx_total=nxt,
        regions=regions)
    ref_ranges, ref_cells = _reference_ranges(
        block, mask, connectivity_8, _DEFAULT_ATOL, _DEFAULT_RTOL, column,
        ro, co, nyt, nxt)
    assert len(got_ranges) == len(ref_ranges)
    for g, r in zip(got_ranges, ref_ranges):
        np.testing.assert_allclose(g, r, rtol=0, atol=0)
    assert got_cells == ref_cells


def test_w_s_flags_record_real_closeness():
    # Pre-#3303 the flags were truthy whenever the position / mask
    # conditions held, because _is_close called from pure Python returns
    # numba's implementation function instead of a bool.
    block = np.array([[1.0, 5.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0]])
    column, _, regions = _polygonize_numpy_regions(block, None, False, None)
    _, cells = _compute_region_value_ranges(
        block, None, False, 1e-8, 1e-5, column,
        row_offset=0, col_offset=0, ny_total=8, nx_total=4, regions=regions)
    merged = {}
    for d in cells:
        merged.update(d)
    # Only the top edge (local_row == ny-1 == 1) is internal here
    # (col_offset == row_offset == 0 and the block spans the full width).
    # Pixel (1, 1) has value 1.0; W neighbour (0, 1) is 1.0 (close) and
    # S neighbour (1, 0) is 5.0 (not close).  The pre-#3303 code records
    # s_match=True here because the position/mask conditions hold.
    v, w_match, s_match = merged[(1, 1)]
    assert v == 1.0
    assert w_match is True
    assert s_match is False
    # Pixel (0, 1) has value 1.0; no W neighbour, S neighbour (0, 0)
    # is 1.0 (close).
    v, w_match, s_match = merged[(0, 1)]
    assert v == 1.0
    assert w_match is False
    assert s_match is True


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("chunks", [(5, 7), (4, 4), (13, 5)])
def test_dask_float_matches_numpy(connectivity, chunks):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 4, (13, 17)).astype(np.float64)
    data += rng.random((13, 17)) * 1e-7  # tolerance-level noise
    data[2:4, 5:9] = np.nan
    raster = xr.DataArray(data, dims=["y", "x"])
    dask_raster = xr.DataArray(
        da.from_array(data, chunks=chunks), dims=["y", "x"])

    col_np, pp_np = polygonize(raster, connectivity=connectivity)
    col_da, pp_da = polygonize(dask_raster, connectivity=connectivity)

    assert col_np == col_da
    assert len(pp_np) == len(pp_da)
    # Boundary-merged rings can start at a different vertex than numpy's
    # trace (pre-existing behaviour, also on the code before #3303), so
    # compare geometric equality rather than raw vertex arrays.
    shapely_geom = pytest.importorskip("shapely.geometry")
    for rings_np, rings_da in zip(pp_np, pp_da):
        poly_np = shapely_geom.Polygon(rings_np[0], rings_np[1:])
        poly_da = shapely_geom.Polygon(rings_da[0], rings_da[1:])
        assert poly_np.equals(poly_da)


def test_polygonize_numpy_unchanged_by_regions_split():
    # _polygonize_numpy must stay a thin wrapper with identical output.
    rng = np.random.default_rng(7)
    block = rng.integers(0, 3, (8, 9)).astype(np.float64)
    col_a, pp_a = _polygonize_numpy(block, None, False, None)
    col_b, pp_b, regions = _polygonize_numpy_regions(block, None, False, None)
    assert col_a == col_b
    assert len(pp_a) == len(pp_b)
    for ra, rb in zip(pp_a, pp_b):
        for x, y in zip(ra, rb):
            np.testing.assert_array_equal(x, y)
    # Region labels cover exactly the polygons _scan emitted.
    assert regions.max() == len(col_b)
