"""Cross-backend parity tests for issue #2172.

Dask 8-connectivity used to split polygons at chunk corners where the only
adjacency between two cells of the same value was diagonal.  These tests
pin the fixed behaviour by comparing polygon count and per-region area
between the NumPy and Dask backends on inputs where diagonal-only
adjacency crosses chunk boundaries.
"""

from collections import Counter

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from ..polygonize import (
    _merge_polygon_rings,
    _signed_ring_area,
    polygonize,
)

dask_required = pytest.mark.skipif(da is None, reason="dask not installed")


def _per_value_area(values, polygons):
    """Sum absolute exterior-ring area for each pixel value."""
    by_val = {}
    for val, rings in zip(values, polygons):
        area = abs(_signed_ring_area(rings[0]))
        by_val[val] = by_val.get(val, 0.0) + area
    return by_val


def _assert_parity(arr, chunks, connectivity, name=""):
    """Polygonize via NumPy and via Dask and assert they agree.

    Comparison covers polygon count, per-value polygon count, and
    per-value total area.  The Dask backend is allowed to return rings
    that revisit a vertex (figure-8 shape) because the NumPy backend
    does too — only counts and areas need to match.
    """
    raster_np = xr.DataArray(arr)
    raster_dk = xr.DataArray(da.from_array(arr, chunks=chunks))

    vals_np, polys_np = polygonize(raster_np, connectivity=connectivity)
    vals_dk, polys_dk = polygonize(raster_dk, connectivity=connectivity)

    assert len(vals_np) == len(vals_dk), (
        f"{name}: polygon count mismatch "
        f"(numpy={len(vals_np)}, dask={len(vals_dk)})"
    )
    assert Counter(vals_np) == Counter(vals_dk), (
        f"{name}: per-value polygon count mismatch "
        f"(numpy={Counter(vals_np)}, dask={Counter(vals_dk)})"
    )

    area_np = _per_value_area(vals_np, polys_np)
    area_dk = _per_value_area(vals_dk, polys_dk)
    assert set(area_np) == set(area_dk), (
        f"{name}: value-set mismatch"
    )
    for val in area_np:
        # Polygonize areas come back exact for integer inputs and exact
        # within float ULP for float inputs.  Use a very small tolerance
        # to avoid being thrown off by add-order drift in float sums.
        assert area_np[val] == pytest.approx(area_dk[val], abs=1e-12), (
            f"{name}: per-value area mismatch for {val}: "
            f"numpy={area_np[val]}, dask={area_dk[val]}"
        )


@dask_required
class TestDask8ConnDiagonalRepro:
    """Direct reproduction from the bug report."""

    def test_2x2_diagonal_1x1_chunks(self):
        # Original repro: two value=1 cells diagonal, two value=0 cells
        # diagonal.  NumPy yields 2 polygons; Dask (1, 1) used to yield 4.
        arr = np.array([[1, 0], [0, 1]], dtype=np.int32)
        _assert_parity(arr, (1, 1), connectivity=8,
                       name="2x2 diagonal 1x1 chunks")

    def test_2x2_diagonal_exact_count(self):
        arr = np.array([[1, 0], [0, 1]], dtype=np.int32)
        vals_dk, _ = polygonize(
            xr.DataArray(da.from_array(arr, chunks=(1, 1))),
            connectivity=8,
        )
        assert len(vals_dk) == 2
        assert Counter(vals_dk) == Counter([0, 1])


@dask_required
class TestDask8ConnChunkCornerParity:
    """Inputs designed to put diagonal-only adjacency on a chunk corner."""

    @pytest.mark.parametrize("chunks", [(1, 1), (2, 2), (1, 2), (2, 1)])
    def test_4x4_checkerboard(self, chunks):
        arr = (np.indices((4, 4)).sum(axis=0) % 2).astype(np.int32)
        # NumPy 8-conn merges everything diagonally — 2 polygons total.
        _assert_parity(arr, chunks, connectivity=8,
                       name=f"4x4 checker chunks={chunks}")

    @pytest.mark.parametrize("chunks", [(2, 2), (3, 3), (2, 3)])
    def test_6x6_checkerboard(self, chunks):
        arr = (np.indices((6, 6)).sum(axis=0) % 2).astype(np.int32)
        _assert_parity(arr, chunks, connectivity=8,
                       name=f"6x6 checker chunks={chunks}")

    @pytest.mark.parametrize("chunks", [(2, 2), (3, 3)])
    def test_diagonal_stripe(self, chunks):
        # A diagonal stripe that crosses several chunk corners.
        arr = np.eye(6, dtype=np.int32)
        _assert_parity(arr, chunks, connectivity=8,
                       name=f"diagonal stripe chunks={chunks}")

    def test_diagonal_stripe_with_offset(self):
        # Wider diagonal band of 1s on a 0 background, crossing both
        # types of chunk corners (interior 4-chunk corner and chunk-edge
        # midpoint).
        arr = (np.eye(8, dtype=np.int32)
               + np.eye(8, k=1, dtype=np.int32))
        _assert_parity(arr, (3, 3), connectivity=8,
                       name="thick diagonal chunks=(3,3)")

    def test_x_shape(self):
        # Two crossing diagonals: forces several diagonal-merge corners.
        arr = (np.eye(7, dtype=np.int32)
               + np.eye(7, dtype=np.int32)[::-1])
        # Don't normalise — the polygonize cares about value identity,
        # not magnitude.
        _assert_parity(arr, (2, 2), connectivity=8,
                       name="X shape chunks=(2,2)")


@dask_required
class TestDask8ConnDoesNotRegress4Conn:
    """4-connectivity behaviour must not change."""

    def test_checker_4conn_keeps_every_cell_separate(self):
        arr = (np.indices((4, 4)).sum(axis=0) % 2).astype(np.int32)
        vals_np, _ = polygonize(xr.DataArray(arr), connectivity=4)
        vals_dk, _ = polygonize(
            xr.DataArray(da.from_array(arr, chunks=(2, 2))),
            connectivity=4,
        )
        # 4-conn never merges diagonals; every cell is its own polygon.
        assert len(vals_np) == 16
        assert len(vals_dk) == 16

    def test_diagonal_4conn_random_counts(self):
        # Polygon counts must still match under 4-connectivity after the
        # 8-connectivity fix.  (Per-value areas have a separate
        # chunk-size-dependent quirk on 4-connectivity that is tracked
        # by a different issue; counts are unaffected.)
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 3, (20, 20), dtype=np.int32)
        vals_np, _ = polygonize(xr.DataArray(arr), connectivity=4)
        vals_dk, _ = polygonize(
            xr.DataArray(da.from_array(arr, chunks=(5, 5))),
            connectivity=4,
        )
        assert Counter(vals_np) == Counter(vals_dk)


@dask_required
class TestDask8ConnFloatAndMaskedInputs:
    """Cover the float / NaN tolerance paths and large random rasters."""

    def test_float_checkerboard(self):
        arr = ((np.indices((4, 4)).sum(axis=0) % 2)
               .astype(np.float64))
        _assert_parity(arr, (2, 2), connectivity=8,
                       name="float checker (2,2)")

    def test_float_with_nan(self):
        arr = ((np.indices((4, 4)).sum(axis=0) % 2)
               .astype(np.float64))
        arr[1, 1] = np.nan
        _assert_parity(arr, (2, 2), connectivity=8,
                       name="float NaN (2,2)")

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7), (1, 1), (4, 4)])
    def test_random_int_20x20(self, chunks):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 3, (20, 20), dtype=np.int32)
        _assert_parity(arr, chunks, connectivity=8,
                       name=f"random 20x20 chunks={chunks}")


class TestMergePolygonRingsDirect:
    """Direct unit tests on ``_merge_polygon_rings`` covering the
    8-conn vertex-pairing rule.  These pin the trace priority order
    (straight-through > right-turn > left-turn > u-turn) so a future
    refactor cannot reintroduce the chunk-corner bug by accident.
    """

    def test_two_diagonally_adjacent_squares_merge_into_figure_8(self):
        # Two 1x1 squares sharing only a corner vertex.  Under
        # 8-conn the merge must produce a single self-touching ring
        # (a figure-8), not two separate squares.
        sq_a = [np.array([
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        ])]
        sq_b = [np.array([
            [1.0, 1.0], [2.0, 1.0],
            [2.0, 2.0], [1.0, 2.0], [1.0, 1.0],
        ])]
        merged = _merge_polygon_rings([sq_a, sq_b], connectivity_8=True)
        assert len(merged) == 1
        # Combined area = 1 + 1 = 2.
        assert abs(_signed_ring_area(merged[0][0])) == pytest.approx(2.0)
        # The ring revisits the shared corner (1, 1).  Count occurrences.
        ext = merged[0][0]
        revisits = sum(
            1 for k in range(len(ext) - 1)
            if ext[k, 0] == 1.0 and ext[k, 1] == 1.0
        )
        assert revisits == 2, f"expected figure-8 ring through (1,1), got {revisits} visits"

    def test_4conn_keeps_diagonal_squares_separate(self):
        # Same input, but 4-conn must keep them as two rings.
        sq_a = [np.array([
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        ])]
        sq_b = [np.array([
            [1.0, 1.0], [2.0, 1.0],
            [2.0, 2.0], [1.0, 2.0], [1.0, 1.0],
        ])]
        merged = _merge_polygon_rings([sq_a, sq_b], connectivity_8=False)
        assert len(merged) == 2

    def test_edge_cancellation_still_merges_under_8conn(self):
        # Two 1x1 squares sharing an EDGE.  Edge cancellation must
        # produce one 1x2 rectangle regardless of the rel==0 priority
        # rule.
        sq_a = [np.array([
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        ])]
        sq_b = [np.array([
            [1.0, 0.0], [2.0, 0.0],
            [2.0, 1.0], [1.0, 1.0], [1.0, 0.0],
        ])]
        merged = _merge_polygon_rings([sq_a, sq_b], connectivity_8=True)
        assert len(merged) == 1
        assert abs(_signed_ring_area(merged[0][0])) == pytest.approx(2.0)
