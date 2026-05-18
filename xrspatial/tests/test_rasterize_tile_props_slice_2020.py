"""Per-tile props slicing in dask backends (issue #2020).

Locks in that ``_run_dask_numpy`` and ``_run_dask_cupy`` no longer embed
the full ``line_props`` / ``point_props`` table into every delayed tile
task.  Only the geometries referenced by each tile end up in that tile's
closure, mirroring the polygon path's ``poly_props[pmask]`` slicing.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

try:
    from shapely.geometry import LineString, Point, box  # noqa: F401
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

if has_shapely:
    from xrspatial.rasterize import (
        _slice_props_for_tile,
        rasterize,
    )

pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_dask, reason="dask not installed"),
    pytest.mark.skipif(not has_geopandas, reason="geopandas not installed"),
]


# ---------------------------------------------------------------------------
# _slice_props_for_tile helper
# ---------------------------------------------------------------------------

class TestSlicePropsForTile:
    def test_empty_geom_idx_returns_empty_props(self):
        props = np.arange(50, dtype=np.float64).reshape(10, 5)
        empty_idx = np.empty(0, dtype=np.int32)
        local_idx, sliced = _slice_props_for_tile(empty_idx, props)
        assert len(local_idx) == 0
        assert sliced.shape == (0, 5)
        assert sliced.dtype == props.dtype

    def test_remap_round_trips_props(self):
        """``props[geom_idx]`` and ``sliced[local_idx]`` agree row-wise."""
        props = np.arange(20, dtype=np.float64).reshape(10, 2)
        geom_idx = np.array([7, 2, 7, 9, 2, 3], dtype=np.int32)
        local_idx, sliced = _slice_props_for_tile(geom_idx, props)
        assert local_idx.dtype == np.int32
        # local_idx values must index into sliced
        assert local_idx.max() < len(sliced)
        np.testing.assert_array_equal(sliced[local_idx], props[geom_idx])

    def test_single_geom_compacts_to_one_row(self):
        props = np.arange(50, dtype=np.float64).reshape(10, 5)
        geom_idx = np.array([4, 4, 4], dtype=np.int32)
        local_idx, sliced = _slice_props_for_tile(geom_idx, props)
        assert sliced.shape == (1, 5)
        assert np.all(local_idx == 0)
        np.testing.assert_array_equal(sliced[0], props[4])

    def test_preserves_column_count_on_empty(self):
        """Empty input must keep the column dimension so downstream
        indexing into ``[:, j]`` does not raise."""
        props = np.empty((0, 3), dtype=np.float64)
        empty_idx = np.empty(0, dtype=np.int32)
        _, sliced = _slice_props_for_tile(empty_idx, props)
        assert sliced.shape == (0, 3)


# ---------------------------------------------------------------------------
# Graph-size regression: localized points should not embed full point_props
# ---------------------------------------------------------------------------

class TestDaskGraphPayloadBounded:
    """Per-tile delayed args must not embed the full per-type props table."""

    def test_point_graph_scales_with_per_tile_points_not_total(self):
        rng = np.random.default_rng(0)
        n_points = 5_000
        # Multi-column props amplify any per-task embedding waste
        n_cols = 8
        xs = rng.uniform(-180, 180, n_points)
        ys = rng.uniform(-90, 90, n_points)
        data = {f"c{j}": rng.random(n_points) for j in range(n_cols)}
        data["geometry"] = [Point(x, y) for x, y in zip(xs, ys)]
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        result = rasterize(
            gdf,
            width=2560,
            height=2560,
            bounds=(-180, -90, 180, 90),
            chunks=(256, 256),
            columns=[f"c{j}" for j in range(n_cols)],
        )
        graph = result.data.__dask_graph__()
        graph_bytes = len(pickle.dumps(dict(graph)))

        # Without the fix: 100 tiles * 5000 points * 8 cols * 8 bytes
        # = 32 MB just for point_props duplicates, on top of segment data.
        # With the fix: each tile sees ~50 points so each task embeds
        # ~3 KB of point_props; total stays well under 4 MB end-to-end.
        assert graph_bytes < 4_000_000, (
            f"graph pickled to {graph_bytes:,} bytes -- the per-tile "
            f"point_props slice may have regressed and the full table is "
            f"being embedded into every delayed task."
        )

    def test_localized_line_graph_scales_with_per_tile_lines(self):
        rng = np.random.default_rng(1)
        n_lines = 5_000
        n_cols = 8
        geoms = []
        for _ in range(n_lines):
            # Each line spans a single ~1-degree cell, so most tiles
            # see only a handful of segments.
            cx = rng.uniform(-170, 170)
            cy = rng.uniform(-80, 80)
            geoms.append(
                LineString([(cx, cy),
                            (cx + rng.uniform(-1, 1),
                             cy + rng.uniform(-1, 1))])
            )
        data = {f"c{j}": rng.random(n_lines) for j in range(n_cols)}
        data["geometry"] = geoms
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        result = rasterize(
            gdf,
            width=2560,
            height=2560,
            bounds=(-180, -90, 180, 90),
            chunks=(256, 256),
            columns=[f"c{j}" for j in range(n_cols)],
        )
        graph = result.data.__dask_graph__()
        graph_bytes = len(pickle.dumps(dict(graph)))

        # Without the fix: ~32 MB.  With the fix: ~1-2 MB for localized
        # lines.  Bound the regression check at 5 MB to leave headroom
        # for pickle overhead.
        assert graph_bytes < 5_000_000, (
            f"graph pickled to {graph_bytes:,} bytes -- localized lines "
            f"should not embed the full line_props in every tile."
        )


# ---------------------------------------------------------------------------
# Correctness: filtered props slice produces identical pixels to baseline
# ---------------------------------------------------------------------------

class TestDaskBackendOutputUnchanged:
    """Slicing per tile must not change a single rasterized pixel."""

    def _make_gdf(self, seed=42, n_lines=30, n_points=30):
        rng = np.random.default_rng(seed)
        geoms = []
        for _ in range(n_lines):
            cx = rng.uniform(-150, 150)
            cy = rng.uniform(-70, 70)
            geoms.append(
                LineString([(cx, cy),
                            (cx + rng.uniform(-5, 5),
                             cy + rng.uniform(-5, 5))])
            )
        for _ in range(n_points):
            geoms.append(
                Point(rng.uniform(-150, 150), rng.uniform(-70, 70))
            )
        vals = list(range(1, len(geoms) + 1))
        return gpd.GeoDataFrame(
            {"value": vals, "geometry": geoms}, crs="EPSG:4326"
        )

    def test_numpy_vs_dask_lines_and_points(self):
        gdf = self._make_gdf()
        np_res = rasterize(
            gdf, column="value",
            width=512, height=512, bounds=(-180, -90, 180, 90),
            fill=0.0,
        )
        dk_res = rasterize(
            gdf, column="value",
            width=512, height=512, bounds=(-180, -90, 180, 90),
            fill=0.0, chunks=(64, 64),
        )
        np.testing.assert_array_equal(np_res.values, dk_res.values)

    def test_numpy_vs_dask_sum_merge(self):
        """Slicing must preserve overlap-sensitive merges (sum)."""
        gdf = self._make_gdf(seed=7, n_lines=80, n_points=80)
        np_res = rasterize(
            gdf, column="value",
            width=256, height=256, bounds=(-180, -90, 180, 90),
            fill=0.0, merge="sum",
        )
        dk_res = rasterize(
            gdf, column="value",
            width=256, height=256, bounds=(-180, -90, 180, 90),
            fill=0.0, merge="sum", chunks=(32, 32),
        )
        np.testing.assert_array_equal(np_res.values, dk_res.values)

    def test_numpy_vs_dask_no_lines_no_points(self):
        """Polygon-only input must still produce identical results."""
        gdf = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]},
            geometry=[box(0, 0, 4, 4), box(6, 6, 10, 10)],
            crs="EPSG:4326",
        )
        np_res = rasterize(
            gdf, column="value",
            width=64, height=64, bounds=(0, 0, 10, 10), fill=0.0,
        )
        dk_res = rasterize(
            gdf, column="value",
            width=64, height=64, bounds=(0, 0, 10, 10),
            fill=0.0, chunks=(16, 16),
        )
        np.testing.assert_array_equal(np_res.values, dk_res.values)

    def test_line_straddling_tile_boundary_renders_contiguously(self):
        """A line whose pixel bbox overlaps two tiles must be present in
        both tile closures.  If the props slice ever drops the geometry
        from one of the two tiles, the dask output will diverge from
        numpy along the seam (gap pixels) or carry the fill value where
        the line should be."""
        # Single horizontal line crossing the vertical seam at x=0 on a
        # bounds=(-10,-10,10,10) raster with width=80, chunks=(40,40).
        # Tile seam in pixel space sits at column 40; the line spans
        # x=-5..5 (pixel columns ~20..60), straddling the seam.
        gdf = gpd.GeoDataFrame(
            {"value": [7.0]},
            geometry=[LineString([(-5.0, 0.0), (5.0, 0.0)])],
            crs="EPSG:4326",
        )
        np_res = rasterize(
            gdf, column="value",
            width=80, height=80, bounds=(-10, -10, 10, 10), fill=0.0,
        )
        dk_res = rasterize(
            gdf, column="value",
            width=80, height=80, bounds=(-10, -10, 10, 10),
            fill=0.0, chunks=(40, 40),
        )
        # Whole-array equality: any dropped pixel along the seam fails.
        np.testing.assert_array_equal(np_res.values, dk_res.values)
        # Structural assertion: the value must appear on both sides of
        # the column-40 seam so a future bug that only renders one half
        # is caught even if the numpy reference also regressed.
        burned = (dk_res.values == 7.0)
        assert burned[:, :40].any(), "line missing from left tile"
        assert burned[:, 40:].any(), "line missing from right tile"
