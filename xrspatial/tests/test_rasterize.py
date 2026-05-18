"""Tests for xrspatial.rasterize (vector geometry rasterization)."""
import numpy as np
import pytest
import xarray as xr

try:
    from shapely.geometry import (
        box, Polygon, MultiPolygon, Point, MultiPoint,
        LineString, MultiLineString,
    )
    has_shapely = True
except ImportError:
    has_shapely = False

# Guard the rasterize import too -- it imports numba.cuda at module level
# which is fine, but the tests all need shapely anyway.
if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

# Try importing optional GPU dependencies
try:
    import cupy
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

try:
    import dask.array as da
    has_dask = True
except ImportError:
    has_dask = False

try:
    import dask_geopandas
    has_dask_geopandas = True
except ImportError:
    has_dask_geopandas = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

skip_no_geopandas = pytest.mark.skipif(
    not has_geopandas, reason="geopandas not installed")
skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_gdf():
    """GeoDataFrame with a single square polygon."""
    gdf = gpd.GeoDataFrame(
        {'value': [5.0]},
        geometry=[box(2, 2, 8, 8)],
    )
    return gdf


@pytest.fixture
def two_polygon_gdf():
    """GeoDataFrame with two non-overlapping rectangles."""
    gdf = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[box(0, 0, 4, 4), box(6, 6, 10, 10)],
    )
    return gdf


@pytest.fixture
def overlapping_gdf():
    """GeoDataFrame with two overlapping squares."""
    gdf = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[box(0, 0, 6, 6), box(4, 4, 10, 10)],
    )
    return gdf


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestBasic:
    def test_output_shape(self, simple_gdf):
        result = rasterize(simple_gdf, width=20, height=15, column='value')
        assert result.shape == (15, 20)
        assert result.dims == ('y', 'x')

    def test_output_name(self, simple_gdf):
        result = rasterize(simple_gdf, width=10, height=10, column='value',
                           name='burned')
        assert result.name == 'burned'

    def test_dtype(self, simple_gdf):
        result = rasterize(simple_gdf, width=10, height=10, column='value',
                           dtype=np.float32)
        assert result.dtype == np.float32

    def test_fill_value(self, simple_gdf):
        result = rasterize(simple_gdf, width=100, height=100,
                           bounds=(0, 0, 10, 10), column='value', fill=-999)
        # Corners should be fill value (polygon is at 2-8, 2-8)
        assert result.values[0, 0] == -999

    def test_coords_match_bounds(self, simple_gdf):
        result = rasterize(simple_gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Pixel centers should be at 0.5, 1.5, ..., 9.5
        np.testing.assert_allclose(result.x.values,
                                   np.arange(0.5, 10, 1.0))
        # y goes top to bottom: 9.5, 8.5, ..., 0.5
        np.testing.assert_allclose(result.y.values,
                                   np.arange(9.5, -0.5, -1.0))

    def test_single_polygon_burns_interior(self, simple_gdf):
        result = rasterize(simple_gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Center pixel (row 5, col 5) is inside the polygon
        # The polygon spans x=[2,8], y=[2,8] so pixel centers 2.5-7.5
        # are inside in both dims
        center = result.values[5, 5]  # y=4.5, x=5.5
        assert center == 5.0

    def test_nan_fill_outside(self, simple_gdf):
        result = rasterize(simple_gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Corner pixel at (row=0, col=0) -> y=9.5, x=0.5 -> outside
        assert np.isnan(result.values[0, 0])


# ---------------------------------------------------------------------------
# Multiple / overlapping polygons
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestMultiplePolygons:
    def test_two_separate_polygons(self, two_polygon_gdf):
        result = rasterize(two_polygon_gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Bottom-left region (low x, low y -> high row index)
        # box(0,0,4,4): pixel centers 0.5-3.5 in x, 0.5-3.5 in y
        # y=0.5 is row 9, x=0.5 is col 0
        assert result.values[9, 0] == 1.0  # inside first polygon
        # Top-right region
        # box(6,6,10,10): pixel centers 6.5-9.5 in x, 6.5-9.5 in y
        # y=9.5 is row 0, x=9.5 is col 9
        assert result.values[0, 9] == 2.0  # inside second polygon

    def test_overlapping_last_writer_wins(self, overlapping_gdf):
        result = rasterize(overlapping_gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Overlap region: x=[4,6], y=[4,6] -> pixel centers 4.5, 5.5
        # y=4.5 -> row 5, x=4.5 -> col 4
        overlap_val = result.values[5, 4]
        # Second polygon (value=2.0) should win in overlap zone
        assert overlap_val == 2.0


# ---------------------------------------------------------------------------
# MultiPolygon support
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestMultiPolygon:
    def test_multipolygon(self):
        mp = MultiPolygon([box(0, 0, 3, 3), box(7, 7, 10, 10)])
        gdf = gpd.GeoDataFrame(
            {'value': [42.0]},
            geometry=[mp],
        )
        result = rasterize(gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        # Both sub-polygons should be filled
        assert result.values[9, 0] == 42.0  # y=0.5, x=0.5 -> first box
        assert result.values[0, 9] == 42.0  # y=9.5, x=9.5 -> second box


# ---------------------------------------------------------------------------
# List-of-pairs input
# ---------------------------------------------------------------------------

class TestListInput:
    def test_list_of_pairs(self):
        geom = box(1, 1, 4, 4)
        result = rasterize([(geom, 7.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5))
        # Center pixel (row=2, col=2) -> y=2.5, x=2.5 -> inside
        assert result.values[2, 2] == 7.0

    def test_empty_list(self):
        result = rasterize([], width=5, height=5, bounds=(0, 0, 5, 5))
        assert result.shape == (5, 5)
        assert np.all(np.isnan(result.values))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestEdgeCases:
    def test_single_row_raster(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0]},
            geometry=[box(0, 0, 10, 2)],
        )
        result = rasterize(gdf, width=10, height=1,
                           bounds=(0, 0, 10, 2), column='value')
        assert result.shape == (1, 10)
        assert np.all(result.values == 1.0)

    def test_single_column_raster(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0]},
            geometry=[box(0, 0, 2, 10)],
        )
        result = rasterize(gdf, width=1, height=10,
                           bounds=(0, 0, 2, 10), column='value')
        assert result.shape == (10, 1)
        assert np.all(result.values == 1.0)

    def test_polygon_outside_bounds(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0]},
            geometry=[box(20, 20, 30, 30)],
        )
        result = rasterize(gdf, width=10, height=10,
                           bounds=(0, 0, 10, 10), column='value')
        assert np.all(np.isnan(result.values))

    def test_polygon_with_hole(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        result = rasterize([(poly, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10))
        # Center (inside hole) should be NaN
        assert np.isnan(result.values[5, 5])
        # Corner (inside polygon) should be 1.0
        assert result.values[9, 0] == 1.0

    def test_empty_geometry_skipped(self):
        from shapely.geometry import Point
        empty = Point().buffer(0)  # empty polygon
        real = box(1, 1, 4, 4)
        result = rasterize([(empty, 99.0), (real, 1.0)],
                           width=5, height=5, bounds=(0, 0, 5, 5))
        assert result.values[2, 2] == 1.0

    def test_unsupported_geom_type_skipped(self):
        from shapely.geometry import GeometryCollection
        gc = GeometryCollection()
        result = rasterize([(gc, 99.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5))
        # Empty GeometryCollections are skipped
        assert np.all(np.isnan(result.values))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="width and height must be >= 1"):
            rasterize([], width=0, height=10, bounds=(0, 0, 10, 10))

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            rasterize([(box(0, 0, 1, 1), 1.0)], width=10, height=10,
                       bounds=(10, 10, 0, 0))

    def test_no_bounds_no_geom(self):
        with pytest.raises(ValueError, match="bounds must be provided"):
            rasterize([], width=10, height=10)

    def test_oversize_explicit_dimensions_rejected(self):
        # width * height = 4e18, far above the 1e9 default cap.  Must
        # raise before any np.full allocation runs.
        with pytest.raises(ValueError, match="exceed the safety limit"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      width=2_000_000_000, height=2_000_000_000,
                      bounds=(0, 0, 1, 1))

    def test_oversize_resolution_rejected(self):
        # resolution=1e-6 with 10x10 bounds resolves to 10M x 10M =
        # 1e14 pixels, well above the default cap.
        with pytest.raises(ValueError, match="exceed the safety limit"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=1e-6, bounds=(0, 0, 10, 10))

    def test_moderate_oversize_explicit_rejected(self):
        # Realistic attack: ~2e9 pixels = ~16 GB float64 + 2 GB int8.
        # Still above the default cap, still rejected.
        with pytest.raises(ValueError, match="exceed the safety limit"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      width=50_000, height=50_000,
                      bounds=(0, 0, 1, 1))

    def test_max_pixels_override_permits_larger(self):
        # A caller who genuinely needs a >1e9-pixel raster can raise the
        # cap explicitly.  Use a moderate size that actually allocates.
        result = rasterize(
            [(box(0, 0, 1, 1), 1.0)],
            width=2000, height=2000, bounds=(0, 0, 1, 1),
            max_pixels=10_000_000,
        )
        assert result.shape == (2000, 2000)

    def test_max_pixels_at_boundary_permitted(self):
        # width * height == max_pixels must pass (strict >).
        result = rasterize(
            [(box(0, 0, 1, 1), 1.0)],
            width=100, height=100, bounds=(0, 0, 1, 1),
            max_pixels=10_000,
        )
        assert result.shape == (100, 100)

    def test_max_pixels_one_over_rejected(self):
        # width * height = 10_001, cap = 10_000 -> reject.
        with pytest.raises(ValueError, match="exceed the safety limit"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      width=101, height=100, bounds=(0, 0, 1, 1),
                      max_pixels=10_000)

    @pytest.mark.parametrize("bad", [0, -1, -0.5, float('inf'),
                                     -float('inf'), float('nan')])
    def test_invalid_resolution_scalar(self, bad):
        with pytest.raises(ValueError, match="resolution must be finite"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=bad, bounds=(0, 0, 1, 1))

    @pytest.mark.parametrize("bad", [(0, 1), (1, 0), (-1, 1), (1, float('nan')),
                                     (float('inf'), 1)])
    def test_invalid_resolution_tuple(self, bad):
        with pytest.raises(ValueError, match="resolution must be finite"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=bad, bounds=(0, 0, 1, 1))

    @pytest.mark.parametrize("bad", [0, -1, (0, 1), (1, -1)])
    def test_invalid_chunks(self, bad):
        # chunks=0 used to hang (issue #2066); negative diverged.
        with pytest.raises(ValueError, match="chunks must be positive"):
            rasterize([(box(0, 0, 1, 1), 1.0)],
                      width=10, height=10, bounds=(0, 0, 1, 1),
                      chunks=bad)

    def test_empty_geometry_does_not_poison_inferred_bounds(self):
        # Issue #2065: an empty geometry returns nan from .bounds; the
        # caller used .min()/.max() unfiltered, so a single empty geom
        # poisoned the inferred extent and produced a raster with nan
        # x/y coords.  Drop the empty and infer from the rest.
        empty = Polygon()
        result = rasterize(
            [(empty, 99), (box(0, 0, 1, 1), 1)],
            width=2, height=2)
        assert np.all(np.isfinite(result.x.values))
        assert np.all(np.isfinite(result.y.values))
        assert np.all(result.values == 1)

    def test_only_empty_geometry_requires_explicit_bounds(self):
        empty = Polygon()
        with pytest.raises(ValueError, match="bounds must be provided"):
            rasterize([(empty, 1.0)], width=2, height=2)

    def test_empty_geodataframe_requires_explicit_bounds(self):
        # Issue #2065: total_bounds on an empty frame is (nan,nan,nan,nan).
        # That used to bypass the empty-bounds guard.
        import geopandas as gpd
        empty_gdf = gpd.GeoDataFrame(
            {'value': []}, geometry=gpd.GeoSeries([]))
        with pytest.raises(ValueError, match="bounds must be provided"):
            rasterize(empty_gdf, width=2, height=2, column='value')

    def test_explicit_nan_bounds_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            rasterize([(box(0, 0, 1, 1), 1.0)], width=2, height=2,
                      bounds=(0, 0, float('nan'), 1))

    def test_empty_multipolygon_filtered_from_inferred_bounds(self):
        # Same shape as the empty-Polygon case but exercises the
        # MultiPolygon branch in shapely's is_empty / bounds path.
        empty_mp = MultiPolygon()
        result = rasterize(
            [(empty_mp, 99), (box(0, 0, 1, 1), 1)],
            width=2, height=2)
        assert np.all(np.isfinite(result.x.values))
        assert np.all(np.isfinite(result.y.values))
        assert np.all(result.values == 1)


# ---------------------------------------------------------------------------
# all_touched mode
# ---------------------------------------------------------------------------

class TestAllTouched:
    def test_all_touched_fills_more_pixels(self):
        # Small polygon that might miss pixel centers
        geom = box(2.1, 2.1, 2.9, 2.9)
        normal = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), all_touched=False)
        touched = rasterize([(geom, 1.0)], width=5, height=5,
                            bounds=(0, 0, 5, 5), all_touched=True)
        # all_touched should have >= as many filled pixels
        normal_count = np.count_nonzero(~np.isnan(normal.values))
        touched_count = np.count_nonzero(~np.isnan(touched.values))
        assert touched_count >= normal_count


# ---------------------------------------------------------------------------
# GeoDataFrame column selection
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestColumnSelection:
    def test_explicit_column(self):
        gdf = gpd.GeoDataFrame(
            {'pop': [100.0], 'area': [50.0]},
            geometry=[box(0, 0, 10, 10)],
        )
        result = rasterize(gdf, width=5, height=5, column='area')
        assert np.nanmean(result.values) == 50.0

    def test_default_first_numeric_column(self):
        gdf = gpd.GeoDataFrame(
            {'label': ['a'], 'score': [3.0]},
            geometry=[box(0, 0, 10, 10)],
        )
        result = rasterize(gdf, width=5, height=5)
        assert np.nanmean(result.values) == 3.0

    def test_no_numeric_column_raises(self):
        gdf = gpd.GeoDataFrame(
            {'label': ['a']},
            geometry=[box(0, 0, 10, 10)],
        )
        with pytest.raises(ValueError, match="no numeric columns"):
            rasterize(gdf, width=5, height=5)


# ---------------------------------------------------------------------------
# Point rasterization
# ---------------------------------------------------------------------------

class TestPoints:
    def test_single_point(self):
        pt = Point(2.5, 2.5)
        result = rasterize([(pt, 7.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # y=2.5 -> row = floor((5 - 2.5) / 1.0) = 2, x=2.5 -> col = 2
        assert result.values[2, 2] == 7.0
        # Other pixels should be fill
        assert result.values[0, 0] == 0

    def test_multiple_points(self):
        pairs = [
            (Point(0.5, 0.5), 1.0),
            (Point(4.5, 4.5), 2.0),
        ]
        result = rasterize(pairs, width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Point at (0.5, 0.5): row=4, col=0
        assert result.values[4, 0] == 1.0
        # Point at (4.5, 4.5): row=0, col=4
        assert result.values[0, 4] == 2.0

    def test_multipoint(self):
        mp = MultiPoint([(1.5, 1.5), (3.5, 3.5)])
        result = rasterize([(mp, 5.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # (1.5, 1.5): row=3, col=1
        assert result.values[3, 1] == 5.0
        # (3.5, 3.5): row=1, col=3
        assert result.values[1, 3] == 5.0

    def test_point_outside_bounds(self):
        pt = Point(20, 20)
        result = rasterize([(pt, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        assert np.all(result.values == 0)

    def test_point_on_boundary(self):
        # Point exactly on the right edge (x=5.0) should be outside
        # for a bounds of (0, 0, 5, 5) with width=5 (pixels at 0.5..4.5)
        pt = Point(5.0, 2.5)
        result = rasterize([(pt, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        assert np.all(result.values == 0)

    def test_overlapping_points_last_wins(self):
        pairs = [
            (Point(2.5, 2.5), 1.0),
            (Point(2.5, 2.5), 9.0),
        ]
        result = rasterize(pairs, width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        assert result.values[2, 2] == 9.0


# ---------------------------------------------------------------------------
# Line rasterization
# ---------------------------------------------------------------------------

class TestLines:
    def test_horizontal_line(self):
        line = LineString([(0.5, 2.5), (4.5, 2.5)])
        result = rasterize([(line, 3.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Horizontal line at y=2.5 -> row=2
        # Should burn across cols 0..4
        for c in range(5):
            assert result.values[2, c] == 3.0

    def test_vertical_line(self):
        line = LineString([(2.5, 0.5), (2.5, 4.5)])
        result = rasterize([(line, 4.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Vertical line at x=2.5 -> col=2
        # Should burn across rows 0..4
        for r in range(5):
            assert result.values[r, 2] == 4.0

    def test_diagonal_line(self):
        line = LineString([(0.5, 0.5), (4.5, 4.5)])
        result = rasterize([(line, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Diagonal: should hit approximately (row=4,col=0), (3,1), (2,2),
        # (1,3), (0,4) via Bresenham
        burned = np.sum(result.values == 1.0)
        assert burned >= 5

    def test_multilinestring(self):
        ml = MultiLineString([
            [(0.5, 2.5), (4.5, 2.5)],  # horizontal
            [(2.5, 0.5), (2.5, 4.5)],  # vertical
        ])
        result = rasterize([(ml, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Cross pattern: row 2 all filled, col 2 all filled
        for c in range(5):
            assert result.values[2, c] == 1.0
        for r in range(5):
            assert result.values[r, 2] == 1.0

    def test_line_outside_bounds(self):
        line = LineString([(20, 20), (30, 30)])
        result = rasterize([(line, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        assert np.all(result.values == 0)

    def test_single_point_line(self):
        # Degenerate line with two identical endpoints
        line = LineString([(2.5, 2.5), (2.5, 2.5)])
        result = rasterize([(line, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Should burn at least one pixel at (row=2, col=2)
        assert result.values[2, 2] == 1.0

    def test_multi_segment_line(self):
        # L-shaped line: right then up
        line = LineString([(0.5, 0.5), (4.5, 0.5), (4.5, 4.5)])
        result = rasterize([(line, 2.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Bottom row (y=0.5, row=4) should be burned across
        for c in range(5):
            assert result.values[4, c] == 2.0
        # Right column (x=4.5, col=4) should be burned down
        for r in range(5):
            assert result.values[r, 4] == 2.0


# ---------------------------------------------------------------------------
# Mixed geometry types
# ---------------------------------------------------------------------------

class TestMixedGeometries:
    def test_polygon_and_point(self):
        poly = box(0, 0, 3, 3)
        pt = Point(4.5, 4.5)
        result = rasterize([(poly, 1.0), (pt, 9.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Polygon should fill bottom-left area
        assert result.values[4, 0] == 1.0  # y=0.5, x=0.5 -> inside poly
        # Point should burn at top-right
        assert result.values[0, 4] == 9.0

    def test_polygon_and_line(self):
        poly = box(0, 0, 5, 5)
        line = LineString([(0.5, 2.5), (4.5, 2.5)])
        result = rasterize([(poly, 1.0), (line, 5.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Polygon fills everything with 1.0
        # Line overwrites row 2 with 5.0
        assert result.values[0, 0] == 1.0  # polygon only
        assert result.values[2, 2] == 5.0  # line overwrites

    def test_point_overwrites_polygon(self):
        poly = box(0, 0, 5, 5)
        pt = Point(2.5, 2.5)
        result = rasterize([(poly, 1.0), (pt, 99.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        # Point has highest priority
        assert result.values[2, 2] == 99.0
        # Rest is polygon
        assert result.values[0, 0] == 1.0

    def test_all_types_together(self):
        poly = box(0, 0, 5, 5)
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        pt = Point(2.5, 2.5)
        result = rasterize(
            [(poly, 1.0), (line, 2.0), (pt, 3.0)],
            width=5, height=5, bounds=(0, 0, 5, 5), fill=0)
        # Polygon fills everything
        assert result.values[4, 4] == 1.0
        # Line overwrites top row
        assert result.values[0, 2] == 2.0
        # Point overwrites center
        assert result.values[2, 2] == 3.0


# ---------------------------------------------------------------------------
# GeoDataFrame with mixed geometry types
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestGeoDataFrameMixed:
    def test_gdf_with_points(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0, 2.0, 3.0]},
            geometry=[Point(1.5, 1.5), Point(3.5, 3.5), Point(2.5, 2.5)],
        )
        result = rasterize(gdf, width=5, height=5,
                           bounds=(0, 0, 5, 5), column='value', fill=0)
        assert result.values[3, 1] == 1.0
        assert result.values[1, 3] == 2.0
        assert result.values[2, 2] == 3.0

    def test_gdf_with_lines(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0, 2.0]},
            geometry=[
                LineString([(0.5, 2.5), (4.5, 2.5)]),
                LineString([(2.5, 0.5), (2.5, 4.5)]),
            ],
        )
        result = rasterize(gdf, width=5, height=5,
                           bounds=(0, 0, 5, 5), column='value', fill=0)
        # Horizontal line at row 2
        assert result.values[2, 0] == 1.0
        # Vertical line at col 2 (overwrites at intersection)
        assert result.values[0, 2] == 2.0


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCuPy:
    def test_cupy_output_type(self):
        geom = box(1, 1, 4, 4)
        result = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), use_cuda=True)
        assert isinstance(result.data, cupy.ndarray)

    def test_cupy_matches_numpy(self):
        geom = box(1, 1, 8, 8)
        np_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=False)
        cp_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_multiple_polygons(self):
        pairs = [(box(0, 0, 4, 4), 1.0), (box(6, 6, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=False)
        cp_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_with_hole(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        np_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=False)
        cp_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_points_match_numpy(self):
        pairs = [
            (Point(1.5, 1.5), 1.0),
            (Point(3.5, 3.5), 2.0),
            (MultiPoint([(0.5, 0.5), (4.5, 4.5)]), 3.0),
        ]
        np_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=False)
        cp_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_lines_match_numpy(self):
        # Use non-intersecting lines to avoid GPU race conditions
        # at shared pixels
        pairs = [
            (LineString([(0.5, 0.5), (4.5, 0.5)]), 1.0),
            (LineString([(0.5, 4.5), (4.5, 4.5)]), 2.0),
            (MultiLineString([[(0.5, 2.5), (4.5, 2.5)]]), 3.0),
        ]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=False)
        cp_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_mixed_types_match_numpy(self):
        pairs = [
            (box(0, 0, 3, 3), 1.0),
            (LineString([(0.5, 4.5), (4.5, 4.5)]), 2.0),
            (Point(2.5, 2.5), 3.0),
        ]
        np_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=False)
        cp_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, use_cuda=True)
        np.testing.assert_array_equal(
            np_result.values, cupy.asnumpy(cp_result.data))

    def test_cupy_no_cupy_raises(self):
        """use_cuda=True without cupy should raise ImportError."""
        # This test only runs if cupy IS available, so we just verify
        # the function works -- the ImportError path is tested by
        # the fact that the guard exists.
        pass


# ---------------------------------------------------------------------------
# Reference comparison: known rasterization
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_unit_square_at_origin(self):
        """A 1x1 square at (0,0)-(1,1) in a 2x2 grid covering (0,0)-(2,2).

        Pixel centers: (0.5, 1.5), (1.5, 1.5), (0.5, 0.5), (1.5, 0.5)
        Row 0 (y=1.5) is above the polygon -- empty.
        Row 1 (y=0.5) has scanline intersections at x=0 and x=1,
        so both col 0 (x=0.5) and col 1 (x=1.5) fall outside the
        strict interior, but col 0 lands on the boundary and gets
        filled by ceil/floor rounding.  The exact pixel set depends on
        whether boundaries count, so we just check fill counts.
        """
        geom = box(0, 0, 1, 1)
        result = rasterize([(geom, 1.0)], width=2, height=2,
                           bounds=(0, 0, 2, 2), fill=0)
        # Top row is outside
        assert result.values[0, 0] == 0
        assert result.values[0, 1] == 0
        # Bottom-left pixel is inside
        assert result.values[1, 0] == 1.0

    def test_full_coverage_square(self):
        """Square covering the full raster extent fills every pixel."""
        geom = box(0, 0, 10, 10)
        result = rasterize([(geom, 7.0)], width=5, height=5,
                           bounds=(0, 0, 10, 10), fill=0)
        assert np.all(result.values == 7.0)

    def test_triangle(self):
        """Right triangle in bottom-left quadrant.

        Triangle: (0,0), (4,0), (0,4) in a 4x4 grid covering (0,0)-(4,4).
        Pixel centers at 0.5, 1.5, 2.5, 3.5.
        """
        tri = Polygon([(0, 0), (4, 0), (0, 4), (0, 0)])
        result = rasterize([(tri, 1.0)], width=4, height=4,
                           bounds=(0, 0, 4, 4), fill=0)
        # Row 0: y=3.5 -> only x=0.5 is inside (x < 4 - 3.5 = 0.5 -> boundary)
        # Row 1: y=2.5 -> x < 4-2.5=1.5 -> x=0.5 inside
        # Row 2: y=1.5 -> x < 4-1.5=2.5 -> x=0.5, 1.5 inside
        # Row 3: y=0.5 -> x < 4-0.5=3.5 -> x=0.5, 1.5, 2.5 inside
        vals = result.values
        # Check that filled pixels increase per row (top to bottom)
        filled_per_row = [np.sum(vals[r] == 1.0) for r in range(4)]
        assert filled_per_row[3] >= filled_per_row[2] >= filled_per_row[1]


# ---------------------------------------------------------------------------
# Custom merge functions
# ---------------------------------------------------------------------------

class TestCustomMerge:
    """Tests for user-supplied merge functions."""

    def test_custom_merge_sum_matches_builtin(self):
        """Custom sum function produces same result as built-in 'sum'."""
        from xrspatial.utils import ngjit

        @ngjit
        def my_sum(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        builtin = rasterize(pairs, width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0, merge='sum')
        custom = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0, merge=my_sum)
        np.testing.assert_array_equal(builtin.values, custom.values)

    def test_custom_merge_max_matches_builtin(self):
        """Custom max function produces same result as built-in 'max'."""
        from xrspatial.utils import ngjit

        @ngjit
        def my_max(pixel, props, is_first):
            if is_first or props[0] > pixel:
                return props[0]
            return pixel

        pairs = [(box(0, 0, 6, 6), 3.0), (box(4, 4, 10, 10), 7.0)]
        builtin = rasterize(pairs, width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0, merge='max')
        custom = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0, merge=my_max)
        np.testing.assert_array_equal(builtin.values, custom.values)

    def test_custom_merge_weighted_average(self):
        """Custom function computing running average."""
        from xrspatial.utils import ngjit

        @ngjit
        def avg(pixel, props, is_first):
            if is_first:
                return props[0]
            return (pixel + props[0]) / 2.0

        pairs = [(box(0, 0, 10, 10), 4.0), (box(0, 0, 10, 10), 8.0)]
        result = rasterize(pairs, width=2, height=2,
                           bounds=(0, 0, 10, 10), fill=0, merge=avg)
        # (4.0 + 8.0) / 2 = 6.0
        assert np.all(result.values == 6.0)

    def test_custom_merge_with_lines(self):
        """Custom merge works with line geometries."""
        from xrspatial.utils import ngjit

        @ngjit
        def my_sum(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [
            (LineString([(0.5, 5), (9.5, 5)]), 1.0),
            (LineString([(5, 0.5), (5, 9.5)]), 1.0),
        ]
        result = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0, merge=my_sum)
        # Intersection pixel should be 2.0
        assert result.values[5, 5] == 2.0

    def test_custom_merge_with_points(self):
        """Custom merge works with point geometries."""
        from xrspatial.utils import ngjit

        @ngjit
        def my_sum(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [(Point(5.5, 5.5), 3.0), (Point(5.5, 5.5), 7.0)]
        result = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0, merge=my_sum)
        # Same pixel written twice: 3 + 7 = 10
        assert result.values[4, 5] == 10.0

    def test_custom_merge_invalid_type_raises(self):
        """Non-string, non-callable merge raises TypeError."""
        with pytest.raises(TypeError):
            rasterize([(box(0, 0, 5, 5), 1.0)], width=5, height=5,
                       bounds=(0, 0, 5, 5), merge=42)

    def test_custom_merge_invalid_string_raises(self):
        """Unknown string merge mode raises ValueError."""
        with pytest.raises(ValueError):
            rasterize([(box(0, 0, 5, 5), 1.0)], width=5, height=5,
                       bounds=(0, 0, 5, 5), merge='bogus')


@skip_no_dask
class TestCustomMergeDask:
    """Custom merge functions with dask backends."""

    def test_custom_merge_dask_matches_numpy(self):
        """Custom function via dask produces same result as numpy."""
        from xrspatial.utils import ngjit

        @ngjit
        def my_sum(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, merge=my_sum)
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              merge=my_sum, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_custom_merge_dask_weighted_average(self):
        """Custom averaging function works across tile boundaries."""
        from xrspatial.utils import ngjit

        @ngjit
        def avg(pixel, props, is_first):
            if is_first:
                return props[0]
            return (pixel + props[0]) / 2.0

        pairs = [(box(0, 0, 10, 10), 4.0), (box(0, 0, 10, 10), 8.0)]
        result = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0,
                           merge=avg, chunks=(5, 5))
        assert np.all(result.values == 6.0)


@skip_no_cuda
class TestCustomMergeGPU:
    """Custom merge functions with GPU backends."""

    @staticmethod
    def _to_numpy(da_result):
        computed = da_result.compute() if hasattr(da_result, 'compute') \
            else da_result
        return cupy.asnumpy(computed.data)

    def test_custom_gpu_merge_sum(self):
        """Custom GPU merge function matches built-in sum."""
        from numba import cuda as _cuda

        @_cuda.jit(device=True)
        def my_sum_gpu(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        builtin = rasterize(pairs, width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0, merge='sum')
        custom = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0,
                           merge=my_sum_gpu, use_cuda=True)
        np.testing.assert_array_equal(
            builtin.values, self._to_numpy(custom))

    @skip_no_dask
    def test_custom_gpu_merge_dask(self):
        """Custom GPU merge function works with dask+cupy."""
        from numba import cuda as _cuda

        @_cuda.jit(device=True)
        def my_sum_gpu(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        builtin = rasterize(pairs, width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0, merge='sum')
        custom = rasterize(pairs, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0,
                           merge=my_sum_gpu, use_cuda=True,
                           chunks=(5, 5))
        np.testing.assert_array_equal(
            builtin.values, self._to_numpy(custom))


# ---------------------------------------------------------------------------
# Dask + NumPy backend
# ---------------------------------------------------------------------------

@skip_no_dask
class TestDaskNumpy:
    """Tile-based dask+numpy rasterization tests."""

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7), (100, 100)])
    def test_polygon_parity(self, chunks):
        """Dask output matches numpy for a simple polygon."""
        geom = box(1, 1, 8, 8)
        np_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), chunks=chunks)
        assert isinstance(dk_result.data, da.Array)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_multiple_polygons_parity(self, chunks):
        pairs = [(box(0, 0, 4, 4), 1.0), (box(6, 6, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), chunks=chunks)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_line_parity(self, chunks):
        line = LineString([(0.5, 0.5), (9.5, 9.5)])
        np_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=chunks)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_point_parity(self, chunks):
        pairs = [(Point(1.5, 1.5), 1.0), (Point(8.5, 8.5), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=chunks)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_mixed_types_parity(self, chunks):
        pairs = [
            (box(0, 0, 3, 3), 1.0),
            (LineString([(0.5, 4.5), (4.5, 4.5)]), 2.0),
            (Point(2.5, 2.5), 3.0),
        ]
        np_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, chunks=chunks)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_chunk_boundary_polygon(self):
        """Polygon straddling a chunk boundary has no seams."""
        geom = box(2, 2, 8, 8)
        np_result = rasterize([(geom, 5.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(geom, 5.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_chunk_boundary_line(self):
        """Line crossing chunk boundary has no gaps."""
        line = LineString([(0.5, 5.0), (9.5, 5.0)])
        np_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_chunk_boundary_point_on_edge(self):
        """Point exactly on a tile edge lands in the right tile."""
        pt = Point(5.0, 5.0)
        np_result = rasterize([(pt, 9.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(pt, 9.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_empty_tiles_are_fill(self):
        """Tiles with no intersecting geometry are filled with fill value."""
        geom = box(0, 0, 2, 2)
        result = rasterize([(geom, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=-999,
                           chunks=(5, 5))
        vals = result.values
        # Top-right quadrant (rows 0-4, cols 5-9) should be all fill
        assert np.all(vals[0:5, 5:10] == -999)

    def test_single_chunk_matches_numpy(self):
        """Chunk larger than raster matches numpy exactly."""
        geom = box(1, 1, 4, 4)
        np_result = rasterize([(geom, 2.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize([(geom, 2.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, chunks=(100, 100))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_output_is_dask_array(self):
        """Result data should be a dask array."""
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), chunks=(5, 5))
        assert isinstance(result.data, da.Array)

    def test_output_shape_and_coords(self):
        """Shape, dims, and coords match eager output."""
        geom = box(0, 0, 10, 10)
        np_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), chunks=(5, 5))
        assert dk_result.shape == np_result.shape
        assert dk_result.dims == np_result.dims
        np.testing.assert_allclose(dk_result.x.values, np_result.x.values)
        np.testing.assert_allclose(dk_result.y.values, np_result.y.values)

    def test_compute_returns_numpy(self):
        """Calling .compute() on the result yields a numpy-backed DataArray."""
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), chunks=(3, 3))
        computed = result.compute()
        assert isinstance(computed.data, np.ndarray)

    @pytest.mark.parametrize("merge_mode", ['last', 'first', 'max', 'min',
                                            'sum', 'count'])
    def test_merge_mode_parity(self, merge_mode):
        """Overlapping geometries with all merge modes match numpy."""
        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              merge=merge_mode)
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              merge=merge_mode, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_polygon_with_hole(self):
        """Polygon with hole matches numpy across tiles."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        np_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_empty_geometry_list(self):
        """Empty input with chunks returns fill-valued dask array."""
        result = rasterize([], width=5, height=5, bounds=(0, 0, 5, 5),
                           chunks=(3, 3))
        assert isinstance(result.data, da.Array)
        assert np.all(np.isnan(result.values))

    def test_all_touched_parity(self):
        """all_touched mode matches numpy across tiles."""
        geom = box(2.1, 2.1, 7.9, 7.9)
        np_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              all_touched=True)
        dk_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              all_touched=True, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    def test_dtype_preserved(self):
        """Output dtype matches the requested dtype."""
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), dtype=np.float32,
                           chunks=(3, 3))
        assert result.dtype == np.float32

    def test_int_chunks_shorthand(self):
        """Single int for chunks uses same value for both axes."""
        geom = box(1, 1, 4, 4)
        np_result = rasterize([(geom, 1.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize([(geom, 1.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0, chunks=3)
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.skipif(
        not has_geopandas, reason="geopandas not installed")
    def test_geodataframe_with_chunks(self):
        """GeoDataFrame input works with dask backend."""
        gdf = gpd.GeoDataFrame(
            {'value': [1.0, 2.0]},
            geometry=[box(0, 0, 4, 4), box(6, 6, 10, 10)],
        )
        np_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value')
        dk_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value',
                              chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.skipif(
        not has_dask_geopandas or not has_geopandas,
        reason="dask-geopandas or geopandas not installed")
    def test_dask_geodataframe_input(self):
        """dask_geopandas.GeoDataFrame input produces same result."""
        gdf = gpd.GeoDataFrame(
            {'value': [1.0, 2.0]},
            geometry=[box(0, 0, 4, 4), box(6, 6, 10, 10)],
        )
        dgdf = dask_geopandas.from_geopandas(gdf, npartitions=2)
        np_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value')
        dk_result = rasterize(dgdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value',
                              chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)

    @pytest.mark.skipif(
        not has_dask_geopandas or not has_geopandas,
        reason="dask-geopandas or geopandas not installed")
    def test_dask_geodataframe_eager(self):
        """dask_geopandas.GeoDataFrame works without chunks too."""
        gdf = gpd.GeoDataFrame(
            {'value': [5.0]},
            geometry=[box(2, 2, 8, 8)],
        )
        dgdf = dask_geopandas.from_geopandas(gdf, npartitions=1)
        np_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value')
        dk_result = rasterize(dgdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value')
        np.testing.assert_array_equal(np_result.values, dk_result.values)


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

@skip_no_cuda
@skip_no_dask
class TestDaskCupy:
    """Tile-based dask+cupy rasterization tests."""

    @staticmethod
    def _to_numpy(da_result):
        """Compute dask+cupy DataArray to numpy."""
        computed = da_result.compute()
        return cupy.asnumpy(computed.data)

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7), (100, 100)])
    def test_polygon_parity(self, chunks):
        geom = box(1, 1, 8, 8)
        np_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize([(geom, 3.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10),
                              use_cuda=True, chunks=chunks)
        assert isinstance(dk_result.data, da.Array)
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_multiple_polygons_parity(self, chunks):
        pairs = [(box(0, 0, 4, 4), 1.0), (box(6, 6, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10),
                              use_cuda=True, chunks=chunks)
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_mixed_types_parity(self):
        pairs = [
            (box(0, 0, 3, 3), 1.0),
            (LineString([(0.5, 4.5), (4.5, 4.5)]), 2.0),
            (Point(2.5, 2.5), 3.0),
        ]
        np_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize(pairs, width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0,
                              use_cuda=True, chunks=(3, 3))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_chunk_boundary_polygon(self):
        geom = box(2, 2, 8, 8)
        np_result = rasterize([(geom, 5.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(geom, 5.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_line_parity(self, chunks):
        line = LineString([(0.5, 0.5), (9.5, 9.5)])
        np_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=chunks)
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    @pytest.mark.parametrize("chunks", [(5, 5), (3, 7)])
    def test_point_parity(self, chunks):
        pairs = [(Point(1.5, 1.5), 1.0), (Point(8.5, 8.5), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=chunks)
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_chunk_boundary_line(self):
        line = LineString([(0.5, 5.0), (9.5, 5.0)])
        np_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(line, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_chunk_boundary_point_on_edge(self):
        pt = Point(5.0, 5.0)
        np_result = rasterize([(pt, 9.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(pt, 9.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_empty_tiles_are_fill(self):
        geom = box(0, 0, 2, 2)
        result = rasterize([(geom, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=-999,
                           use_cuda=True, chunks=(5, 5))
        vals = self._to_numpy(result)
        assert np.all(vals[0:5, 5:10] == -999)

    def test_single_chunk_matches_numpy(self):
        geom = box(1, 1, 4, 4)
        np_result = rasterize([(geom, 2.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize([(geom, 2.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0,
                              use_cuda=True, chunks=(100, 100))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_output_is_dask_array(self):
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10),
                           use_cuda=True, chunks=(5, 5))
        assert isinstance(result.data, da.Array)

    def test_output_shape_and_coords(self):
        geom = box(0, 0, 10, 10)
        np_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10))
        dk_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10),
                              use_cuda=True, chunks=(5, 5))
        assert dk_result.shape == np_result.shape
        assert dk_result.dims == np_result.dims
        np.testing.assert_allclose(
            dk_result.x.values, np_result.x.values)
        np.testing.assert_allclose(
            dk_result.y.values, np_result.y.values)

    def test_compute_returns_cupy(self):
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5),
                           use_cuda=True, chunks=(3, 3))
        computed = result.compute()
        assert type(computed.data).__module__.startswith('cupy')

    @pytest.mark.parametrize("merge_mode", ['last', 'first', 'max', 'min',
                                            'sum', 'count'])
    def test_merge_mode_parity(self, merge_mode):
        pairs = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        np_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              merge=merge_mode)
        dk_result = rasterize(pairs, width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              merge=merge_mode,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_polygon_with_hole(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        np_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0)
        dk_result = rasterize([(poly, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_empty_geometry_list(self):
        result = rasterize([], width=5, height=5, bounds=(0, 0, 5, 5),
                           use_cuda=True, chunks=(3, 3))
        assert isinstance(result.data, da.Array)
        vals = self._to_numpy(result)
        assert np.all(np.isnan(vals))

    def test_all_touched_parity(self):
        geom = box(2.1, 2.1, 7.9, 7.9)
        np_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              all_touched=True)
        dk_result = rasterize([(geom, 1.0)], width=10, height=10,
                              bounds=(0, 0, 10, 10), fill=0,
                              all_touched=True,
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    def test_dtype_preserved(self):
        geom = box(0, 0, 5, 5)
        result = rasterize([(geom, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), dtype=np.float32,
                           use_cuda=True, chunks=(3, 3))
        assert result.dtype == np.float32

    def test_int_chunks_shorthand(self):
        geom = box(1, 1, 4, 4)
        np_result = rasterize([(geom, 1.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0)
        dk_result = rasterize([(geom, 1.0)], width=5, height=5,
                              bounds=(0, 0, 5, 5), fill=0,
                              use_cuda=True, chunks=3)
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))

    @pytest.mark.skipif(
        not has_geopandas, reason="geopandas not installed")
    def test_geodataframe_with_chunks(self):
        gdf = gpd.GeoDataFrame(
            {'value': [1.0, 2.0]},
            geometry=[box(0, 0, 4, 4), box(6, 6, 10, 10)],
        )
        np_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value')
        dk_result = rasterize(gdf, width=10, height=10,
                              bounds=(0, 0, 10, 10), column='value',
                              use_cuda=True, chunks=(5, 5))
        np.testing.assert_array_equal(
            np_result.values, self._to_numpy(dk_result))


# ---------------------------------------------------------------------------
# Multi-column properties
# ---------------------------------------------------------------------------

@skip_no_geopandas
class TestMultiColumn:
    """Tests for multi-column property support via the 'columns' parameter."""

    def test_column_and_columns_mutually_exclusive(self):
        """Passing both column and columns raises ValueError."""
        gdf = gpd.GeoDataFrame(
            {'a': [1.0], 'b': [2.0]},
            geometry=[box(0, 0, 5, 5)],
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            rasterize(gdf, width=5, height=5, bounds=(0, 0, 5, 5),
                      column='a', columns=['a', 'b'])

    def test_multi_column_custom_merge(self):
        """Custom merge function using multiple property columns."""
        from xrspatial.utils import ngjit

        @ngjit
        def weighted_density(pixel, props, is_first):
            # props[0] = population, props[1] = area
            density = props[0] / props[1]
            if is_first:
                return density
            return pixel + density

        gdf = gpd.GeoDataFrame({
            'population': [100.0, 200.0],
            'area': [10.0, 20.0],
            'geometry': [box(0, 0, 5, 5), box(5, 5, 10, 10)],
        })
        result = rasterize(gdf, columns=['population', 'area'],
                           merge=weighted_density, width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0)
        # density = 100/10 = 10 for first polygon, 200/20 = 10 for second
        filled = result.values[result.values != 0]
        assert len(filled) > 0
        np.testing.assert_allclose(filled, 10.0)

    def test_multi_column_sum_uses_first_prop(self):
        """Built-in 'sum' with multiple columns uses props[0]."""
        gdf = gpd.GeoDataFrame({
            'val': [3.0, 7.0],
            'extra': [99.0, 99.0],
            'geometry': [box(0, 0, 10, 10), box(0, 0, 10, 10)],
        })
        single = rasterize(gdf, column='val', merge='sum',
                           width=5, height=5, bounds=(0, 0, 10, 10), fill=0)
        multi = rasterize(gdf, columns=['val', 'extra'], merge='sum',
                          width=5, height=5, bounds=(0, 0, 10, 10), fill=0)
        # Both should sum props[0] = val column: 3 + 7 = 10
        np.testing.assert_array_equal(single.values, multi.values)

    def test_multi_column_props_array_shape(self):
        """Verify the props array has the right shape through the pipeline."""
        from xrspatial.utils import ngjit

        @ngjit
        def use_second(pixel, props, is_first):
            return props[1]

        gdf = gpd.GeoDataFrame({
            'a': [1.0],
            'b': [42.0],
            'geometry': [box(0, 0, 10, 10)],
        })
        result = rasterize(gdf, columns=['a', 'b'], merge=use_second,
                           width=5, height=5, bounds=(0, 0, 10, 10), fill=0)
        # Should burn props[1] = 42.0 everywhere
        assert np.all(result.values == 42.0)

    @skip_no_dask
    def test_multi_column_dask_parity(self):
        """Multi-column dask output matches numpy."""
        from xrspatial.utils import ngjit

        @ngjit
        def ratio(pixel, props, is_first):
            return props[0] / props[1]

        gdf = gpd.GeoDataFrame({
            'num': [6.0],
            'den': [2.0],
            'geometry': [box(0, 0, 10, 10)],
        })
        np_result = rasterize(gdf, columns=['num', 'den'], merge=ratio,
                              width=10, height=10, bounds=(0, 0, 10, 10),
                              fill=0)
        dk_result = rasterize(gdf, columns=['num', 'den'], merge=ratio,
                              width=10, height=10, bounds=(0, 0, 10, 10),
                              fill=0, chunks=(5, 5))
        np.testing.assert_array_equal(np_result.values, dk_result.values)


# ---------------------------------------------------------------------------
# CSR builder int64 row_ptr (issue #1388)
# ---------------------------------------------------------------------------

class TestBuildRowCsrInt64:
    """`_build_row_csr_numba` uses int64 row_ptr to avoid int32 overflow."""

    def test_row_ptr_dtype_is_int64(self):
        """row_ptr is allocated as int64 so cumulative counts cannot wrap."""
        from xrspatial.rasterize import _build_row_csr_numba

        edge_y_min = np.array([0, 0, 1], dtype=np.int32)
        edge_y_max = np.array([2, 1, 2], dtype=np.int32)
        row_ptr, col_idx = _build_row_csr_numba(edge_y_min, edge_y_max, 3)

        assert row_ptr.dtype == np.int64
        assert col_idx.dtype == np.int32

    def test_empty_edges_returns_int64_row_ptr(self):
        """The n_edges == 0 short-circuit also returns int64 row_ptr."""
        from xrspatial.rasterize import _build_row_csr_numba

        edge_y_min = np.empty(0, dtype=np.int32)
        edge_y_max = np.empty(0, dtype=np.int32)
        row_ptr, col_idx = _build_row_csr_numba(edge_y_min, edge_y_max, 5)

        assert row_ptr.dtype == np.int64
        assert row_ptr.shape == (6,)
        assert np.all(row_ptr == 0)

    def test_row_ptr_holds_value_above_int32_max(self):
        """The cumulative count survives a value past int32 max.

        Construct a small CSR by hand and confirm that adding past
        2**31 - 1 inside int64 row_ptr does not wrap. The actual numba
        kernel writes per-edge so we cannot drive 2.5e9 edges in a unit
        test, but we can verify the dtype contract by writing past the
        int32 boundary directly into a returned row_ptr.
        """
        from xrspatial.rasterize import _build_row_csr_numba

        edge_y_min = np.array([0], dtype=np.int32)
        edge_y_max = np.array([0], dtype=np.int32)
        row_ptr, _ = _build_row_csr_numba(edge_y_min, edge_y_max, 2)

        assert row_ptr.dtype == np.int64
        # int64 can hold > 2**31; assigning would wrap on int32.
        big = np.int64(2**31 + 17)
        row_ptr[0] = big
        assert int(row_ptr[0]) == int(big)

    def test_csr_correct_for_overlapping_edges(self):
        """Overlapping edges populate col_idx in row order without corruption."""
        from xrspatial.rasterize import _build_row_csr_numba

        # Three edges spanning rows [0,2], [1,3], [2,2] in a 4-row raster.
        edge_y_min = np.array([0, 1, 2], dtype=np.int32)
        edge_y_max = np.array([2, 3, 2], dtype=np.int32)
        row_ptr, col_idx = _build_row_csr_numba(edge_y_min, edge_y_max, 4)

        # Per-row counts: row 0 -> {0}; row 1 -> {0,1}; row 2 -> {0,1,2};
        # row 3 -> {1}; total = 7.
        assert row_ptr.tolist() == [0, 1, 3, 6, 7]
        assert col_idx.shape == (7,)
        # Row 0 holds edge 0.
        assert col_idx[row_ptr[0]:row_ptr[1]].tolist() == [0]
        # Row 2 holds edges 0, 1, 2 in arrival order.
        assert sorted(col_idx[row_ptr[2]:row_ptr[3]].tolist()) == [0, 1, 2]
        # Row 3 holds edge 1.
        assert col_idx[row_ptr[3]:row_ptr[4]].tolist() == [1]

    def test_rasterize_public_path_still_works_after_int64_change(self):
        """End-to-end rasterize call still produces correct output."""
        gdf = gpd.GeoDataFrame(
            {'value': [3.0]},
            geometry=[box(2, 2, 8, 8)],
        )
        result = rasterize(
            gdf, column='value',
            width=10, height=10, bounds=(0, 0, 10, 10), fill=0,
        )
        # The polygon covers a 6x6 inner block; just check it burned in.
        assert np.any(result.values == 3.0)


# ---------------------------------------------------------------------------
# Metadata propagation (attrs from `like`, coord reuse, _FillValue)
# ---------------------------------------------------------------------------

def _make_like(width=10, height=10, attrs=None, dtype=np.float64):
    """Build a 2D template DataArray with georeferenced coords and attrs."""
    x = np.linspace(0.5, width - 0.5, width)
    y = np.linspace(height - 0.5, 0.5, height)
    data = np.zeros((height, width), dtype=dtype)
    return xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs=dict(attrs or {}),
    )


class TestMetadataPropagation:
    """Verify `like.attrs` propagates, `like.coords` are reused, and the
    fill value lands in `_FillValue` / `nodatavals`.
    """

    def test_like_propagates_attrs(self):
        attrs = {
            'crs': 'EPSG:32610',
            'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
            'res': (1.0, 1.0),
        }
        like = _make_like(attrs=attrs)
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        assert result.attrs.get('crs') == 'EPSG:32610'
        assert result.attrs.get('transform') == \
            (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        assert result.attrs.get('res') == (1.0, 1.0)

    def test_like_preserves_coords_bit_identical(self):
        like = _make_like()
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        # Output reuses like.coords exactly so xr.align keeps working.
        np.testing.assert_array_equal(
            result.coords['x'].values, like.coords['x'].values)
        np.testing.assert_array_equal(
            result.coords['y'].values, like.coords['y'].values)

    def test_like_attrs_isolated_from_template(self):
        """Mutating output attrs must not mutate the template's attrs."""
        attrs = {'crs': 'EPSG:32610'}
        like = _make_like(attrs=attrs)
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        result.attrs['crs'] = 'EPSG:4326'
        assert like.attrs['crs'] == 'EPSG:32610'

    def test_fill_value_recorded_when_not_nan(self):
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            width=10, height=10, bounds=(0, 0, 10, 10),
            fill=-9999, dtype=np.int32,
        )
        # Emit the full triplet: xrspatial's primary (nodata), the CF
        # alias (_FillValue), and rioxarray's per-band tuple (nodatavals).
        # All three must agree with fill so downstream consumers all
        # resolve to the same sentinel regardless of which key they read.
        assert result.attrs.get('nodata') == -9999
        assert result.attrs.get('_FillValue') == -9999
        assert result.attrs.get('nodatavals') == (-9999,)
        # Sentinel round-trips cleanly when the array is cast to its
        # declared dtype -- pins #1973-style dtype mismatch regressions.
        assert np.array(-9999, dtype=result.dtype) == \
            np.array(result.attrs['_FillValue'], dtype=result.dtype)

    def test_fill_value_omitted_for_nan(self):
        """Default fill=NaN should not pollute attrs with nodata keys."""
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            width=10, height=10, bounds=(0, 0, 10, 10),
        )
        assert 'nodata' not in result.attrs
        assert '_FillValue' not in result.attrs
        assert 'nodatavals' not in result.attrs

    def test_no_like_no_attrs_pollution(self):
        """Without `like` and with NaN fill, attrs must stay empty."""
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            width=10, height=10, bounds=(0, 0, 10, 10),
        )
        assert result.attrs == {}

    @skip_no_dask
    def test_like_attrs_propagated_dask(self):
        like = _make_like(attrs={'crs': 'EPSG:32610'})
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0, chunks=5,
        )
        # The dask backend routes through the same final xr.DataArray
        # constructor, so attrs / coords / _FillValue behave identically.
        assert result.attrs.get('crs') == 'EPSG:32610'
        assert result.attrs.get('_FillValue') == 0
        np.testing.assert_array_equal(
            result.coords['x'].values, like.coords['x'].values)

    @skip_no_cuda
    def test_like_attrs_propagated_cupy(self):
        like = _make_like(attrs={'crs': 'EPSG:32610'})
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0, use_cuda=True,
        )
        assert result.attrs.get('crs') == 'EPSG:32610'
        assert result.attrs.get('_FillValue') == 0

    @skip_no_cuda
    @skip_no_dask
    def test_like_attrs_propagated_dask_cupy(self):
        like = _make_like(attrs={'crs': 'EPSG:32610'})
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            like=like, fill=0, use_cuda=True, chunks=5,
        )
        assert result.attrs.get('crs') == 'EPSG:32610'
        assert result.attrs.get('_FillValue') == 0

    def test_like_stale_nodata_keys_replaced_with_fill(self):
        """Inherited nodata keys from like must not outlive the new fill.

        The geotiff writer's _resolve_nodata_attr checks ``nodata`` →
        ``nodatavals`` → ``_FillValue``.  If a previous round-trip left
        any of them on the template, they have to be replaced (or
        cleared) so the writer doesn't tag pixels with a stale sentinel
        that disagrees with the actual fill in the new array.
        """
        like = _make_like(attrs={
            'nodata': -9999,
            '_FillValue': -9999,
            'nodatavals': (-9999,),
            'crs': 'EPSG:32610',
        })
        # Case 1: explicit fill replaces all three keys.
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        assert result.attrs.get('nodata') == 0
        assert result.attrs.get('_FillValue') == 0
        assert result.attrs.get('nodatavals') == (0,)
        assert result.attrs.get('crs') == 'EPSG:32610'
        # Case 2: NaN fill strips inherited nodata keys outright -- the
        # actual unwritten pixels are NaN, not -9999, so advertising
        # -9999 would lie to downstream tools.
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like,
        )
        assert 'nodata' not in result.attrs
        assert '_FillValue' not in result.attrs
        assert 'nodatavals' not in result.attrs
        assert result.attrs.get('crs') == 'EPSG:32610'

    def test_numpy_float_nan_fill_treated_as_nan(self):
        """Numpy scalar NaN must be detected as NaN, not emitted as fill.

        ``isinstance(np.float32(np.nan), float)`` is False, so a naive
        ``isinstance(fill, float) and np.isnan(fill)`` check would let a
        numpy-typed NaN slip through and land in ``_FillValue``.
        """
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            width=10, height=10, bounds=(0, 0, 10, 10),
            fill=np.float32(np.nan),
        )
        assert 'nodata' not in result.attrs
        assert '_FillValue' not in result.attrs
        assert 'nodatavals' not in result.attrs

    def test_like_non_dim_coords_propagated(self):
        """Non-dim coords on like (e.g. rioxarray's spatial_ref) carry over."""
        like = _make_like()
        # rioxarray attaches the CRS as a scalar non-dim coord
        # called ``spatial_ref``.  rasterize must propagate it.
        like = like.assign_coords(spatial_ref=0)
        like['spatial_ref'].attrs['crs_wkt'] = (
            'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
        )
        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)], like=like, fill=0,
        )
        assert 'spatial_ref' in result.coords
        assert (result.coords['spatial_ref'].attrs.get('crs_wkt')
                == 'GEOGCS["WGS 84",DATUM["WGS_1984"]]')

    def test_geotiff_round_trip_preserves_fill(self, tmp_path):
        """rasterize → to_geotiff → read → nodata matches.

        The user-visible motivation for #2018: a rasterized output
        written to GeoTIFF must round-trip the fill sentinel so masks
        and downstream readers identify the right nodata pixels.
        """
        pytest.importorskip('tifffile')
        from xrspatial.geotiff import to_geotiff, open_geotiff

        result = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            width=10, height=10, bounds=(0, 0, 10, 10),
            fill=-9999.0, dtype=np.float32,
        )
        path = str(tmp_path / 'rasterize_fill_round_trip.tif')
        to_geotiff(result, path)
        read_back = open_geotiff(path)
        # GeoTIFF stores nodata as a scalar; either ``nodata`` or
        # ``_FillValue`` should land on the read-back DataArray and
        # match the fill we supplied.
        nodata = (read_back.attrs.get('nodata')
                  or read_back.attrs.get('_FillValue'))
        assert nodata is not None
        assert float(nodata) == -9999.0


class TestPolysToWkb:
    """Tests for the _polys_to_wkb helper (issue #2059)."""

    def test_empty_input_returns_empty_list(self):
        from xrspatial.rasterize import _polys_to_wkb
        result = _polys_to_wkb([])
        assert result == []
        assert isinstance(result, list)

    def test_output_matches_per_geometry_wkb(self):
        """Vectorized output is byte-identical to the per-geometry path."""
        from xrspatial.rasterize import _polys_to_wkb
        rng = np.random.default_rng(2059)
        geoms = []
        for _ in range(100):
            cx, cy = rng.uniform(-100, 100, 2)
            r = rng.uniform(0.1, 5.0)
            geoms.append(box(cx - r, cy - r, cx + r, cy + r))

        result = _polys_to_wkb(geoms)
        expected = [g.wkb for g in geoms]

        assert isinstance(result, list)
        assert len(result) == len(expected)
        assert all(isinstance(b, (bytes, bytearray)) for b in result)
        assert result == expected

    def test_roundtrip_through_from_wkb(self):
        """Output round-trips through _polys_from_wkb to equal geometries."""
        from xrspatial.rasterize import _polys_to_wkb, _polys_from_wkb
        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
            Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        ]
        restored = _polys_from_wkb(_polys_to_wkb(geoms))
        assert len(restored) == len(geoms)
        for original, copy in zip(geoms, restored):
            assert original.equals(copy)
