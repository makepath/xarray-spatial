"""Accuracy and invariant tests for xrspatial.rasterize (#995).

These tests exercise mathematical properties of correct rasterization
rather than spot-checking individual pixel values.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import (
        box, Polygon, MultiPolygon, Point, MultiPoint,
        LineString, MultiLineString,
    )
    from shapely import contains
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import rasterio
    import rasterio.features
    has_rasterio = True
except ImportError:
    has_rasterio = False

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_polygon(cx, cy, r, n=128):
    """Create an approximate circle polygon with *n* vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    coords.append(coords[0])  # close ring
    return Polygon(coords)


def _star_polygon(cx, cy, r_outer, r_inner, n_points=5):
    """Create a star polygon with alternating outer/inner vertices."""
    coords = []
    for i in range(n_points * 2):
        angle = np.pi / 2 + i * np.pi / n_points
        r = r_outer if i % 2 == 0 else r_inner
        coords.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    coords.append(coords[0])
    return Polygon(coords)


def _l_shape_polygon():
    """Concave L-shaped polygon."""
    return Polygon([
        (0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6), (0, 0)
    ])


def _filled_count(result):
    """Count non-NaN pixels in a rasterize result."""
    vals = result.values
    return int(np.count_nonzero(~np.isnan(vals)))


def _filled_mask(result):
    """Boolean mask of filled pixels."""
    return ~np.isnan(result.values)


# ---------------------------------------------------------------------------
# 1. Watertight tiling (checkerboard)
# ---------------------------------------------------------------------------

class TestWatertightTiling:
    """Non-overlapping polygons that tile a region should fill every pixel
    exactly once when rasterized with merge='count'."""

    def test_checkerboard_count_equals_one(self):
        """4x4 grid of unit squares tiling a 4x4 region."""
        pairs = []
        for row in range(4):
            for col in range(4):
                sq = box(col, row, col + 1, row + 1)
                pairs.append((sq, 1.0))

        result = rasterize(pairs, width=40, height=40,
                           bounds=(0, 0, 4, 4), merge='count', fill=0)
        vals = result.values
        assert np.all(vals == 1.0), (
            f"Expected all pixels == 1, got min={vals.min()}, max={vals.max()}, "
            f"zeros={np.sum(vals == 0)}, twos={np.sum(vals == 2)}"
        )

    def test_two_adjacent_rectangles_no_gap(self):
        """Two rectangles sharing an edge: no gaps, no overlaps."""
        left = box(0, 0, 5, 10)
        right = box(5, 0, 10, 10)
        result = rasterize([(left, 1.0), (right, 1.0)],
                           width=100, height=100,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        vals = result.values
        assert np.all(vals == 1.0), (
            f"Gap or overlap: zeros={np.sum(vals == 0)}, "
            f"twos={np.sum(vals == 2)}"
        )

    def test_four_quadrant_tiling(self):
        """Four squares meeting at the center of the raster."""
        quads = [
            box(0, 0, 5, 5),
            box(5, 0, 10, 5),
            box(0, 5, 10, 10),  # intentionally one wide rect
        ]
        # Split top into two
        quads[2] = box(0, 5, 5, 10)
        quads.append(box(5, 5, 10, 10))

        pairs = [(q, 1.0) for q in quads]
        result = rasterize(pairs, width=50, height=50,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        vals = result.values
        assert np.all(vals == 1.0)

    def test_vertical_strips(self):
        """Vertical strip tiling -- tests shared vertical edges."""
        pairs = [(box(i, 0, i + 1, 10), 1.0) for i in range(10)]
        result = rasterize(pairs, width=100, height=100,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        assert np.all(result.values == 1.0)

    def test_horizontal_strips(self):
        """Horizontal strip tiling -- tests shared horizontal edges."""
        pairs = [(box(0, i, 10, i + 1), 1.0) for i in range(10)]
        result = rasterize(pairs, width=100, height=100,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        assert np.all(result.values == 1.0)


# ---------------------------------------------------------------------------
# 2. Area conservation
# ---------------------------------------------------------------------------

class TestAreaConservation:
    """Rasterized area should converge toward analytic area as resolution
    increases."""

    def test_circle_area_convergence(self):
        """Circle area should converge to pi*r^2."""
        cx, cy, r = 5.0, 5.0, 3.0
        expected_area = np.pi * r ** 2

        errors = []
        for res in [50, 100, 200]:
            circle = _circle_polygon(cx, cy, r, n=256)
            result = rasterize([(circle, 1.0)], width=res, height=res,
                               bounds=(0, 0, 10, 10), fill=np.nan)
            pixel_area = (10.0 / res) ** 2
            raster_area = _filled_count(result) * pixel_area
            rel_error = abs(raster_area - expected_area) / expected_area
            errors.append(rel_error)

        # Error should decrease with resolution
        assert errors[-1] < errors[0], (
            f"Area error should decrease with resolution: {errors}")
        # At 200x200 (pixel = 0.05), error should be small
        assert errors[-1] < 0.02, (
            f"Circle area error too large at 200x200: {errors[-1]:.4f}")

    def test_rectangle_exact_area(self):
        """Rectangle aligned to pixel grid should have exact area."""
        # 4x6 rectangle in a 10x10 grid at 100x100 resolution
        # Pixel size = 0.1, so 4/0.1 = 40 cols, 6/0.1 = 60 rows = 2400 pixels
        rect = box(2, 1, 6, 7)
        result = rasterize([(rect, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=np.nan)
        pixel_area = 0.1 * 0.1
        raster_area = _filled_count(result) * pixel_area
        expected = 4.0 * 6.0
        assert abs(raster_area - expected) < 1e-10, (
            f"Rectangle area mismatch: got {raster_area}, expected {expected}")

    def test_triangle_area_convergence(self):
        """Triangle area should converge to base*height/2."""
        tri = Polygon([(1, 1), (9, 1), (5, 9), (1, 1)])
        expected_area = 0.5 * 8 * 8  # base=8, height=8

        errors = []
        for res in [50, 100, 200]:
            result = rasterize([(tri, 1.0)], width=res, height=res,
                               bounds=(0, 0, 10, 10), fill=np.nan)
            pixel_area = (10.0 / res) ** 2
            raster_area = _filled_count(result) * pixel_area
            rel_error = abs(raster_area - expected_area) / expected_area
            errors.append(rel_error)

        # Error should be small and non-increasing with resolution
        assert errors[-1] <= errors[0] + 1e-10
        assert errors[-1] < 0.02


# ---------------------------------------------------------------------------
# 3. Point-in-polygon cross-check
# ---------------------------------------------------------------------------

class TestPointInPolygonCrossCheck:
    """For each pixel center, shapely's contains() should agree with the
    rasterized fill/no-fill decision."""

    def _check_pip(self, polygon, width, height, bounds, min_bdist=None):
        """Cross-check rasterized result against shapely contains().

        Pixels within *min_bdist* of the polygon boundary are skipped
        because the scanline evaluates edge crossings at pixel-boundary
        y-coordinates (not pixel-center y), which can produce ±1 pixel
        disagreement at slanted edges.  Default tolerance is one pixel
        diagonal.
        """
        result = rasterize([(polygon, 1.0)], width=width, height=height,
                           bounds=bounds, fill=np.nan)
        xmin, ymin, xmax, ymax = bounds
        px = (xmax - xmin) / width
        py = (ymax - ymin) / height
        if min_bdist is None:
            min_bdist = np.sqrt(px ** 2 + py ** 2)  # one pixel diagonal

        filled = _filled_mask(result)
        mismatches = 0

        boundary = polygon.boundary
        for r in range(height):
            y = ymax - (r + 0.5) * py
            for c in range(width):
                x = xmin + (c + 0.5) * px
                pt = Point(x, y)
                if boundary.distance(pt) < min_bdist:
                    continue
                shapely_inside = polygon.contains(pt)
                if shapely_inside != filled[r, c]:
                    mismatches += 1

        return mismatches

    def test_star_polygon(self):
        """Star (concave) polygon: interior/exterior agree with shapely."""
        star = _star_polygon(5, 5, 4, 1.5, n_points=5)
        mismatches = self._check_pip(star, 50, 50, (0, 0, 10, 10))
        assert mismatches == 0, f"{mismatches} pixel mismatches for star"

    def test_l_shape(self):
        """L-shaped (concave) polygon: interior/exterior agree."""
        L = _l_shape_polygon()
        mismatches = self._check_pip(L, 60, 60, (0, 0, 6, 6))
        assert mismatches == 0, f"{mismatches} pixel mismatches for L-shape"

    def test_rectangle(self):
        """Simple rectangle cross-check."""
        rect = box(2, 3, 7, 8)
        mismatches = self._check_pip(rect, 40, 40, (0, 0, 10, 10))
        assert mismatches == 0, f"{mismatches} pixel mismatches for rectangle"

    def test_polygon_with_hole(self):
        """Polygon with hole: interior of hole should not be filled."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        mismatches = self._check_pip(poly, 50, 50, (0, 0, 10, 10))
        assert mismatches == 0, (
            f"{mismatches} pixel mismatches for polygon with hole")


# ---------------------------------------------------------------------------
# 4. Annulus (donut) test
# ---------------------------------------------------------------------------

class TestAnnulus:
    """Polygon with a concentric circular hole -- area should match
    pi*(R^2 - r^2)."""

    def test_annulus_area(self):
        cx, cy = 5.0, 5.0
        R, r = 4.0, 2.0
        outer = _circle_polygon(cx, cy, R, n=256)
        inner = _circle_polygon(cx, cy, r, n=256)
        annulus = outer.difference(inner)

        expected_area = np.pi * (R ** 2 - r ** 2)

        result = rasterize([(annulus, 1.0)], width=200, height=200,
                           bounds=(0, 0, 10, 10), fill=np.nan)
        pixel_area = (10.0 / 200) ** 2
        raster_area = _filled_count(result) * pixel_area
        rel_error = abs(raster_area - expected_area) / expected_area
        assert rel_error < 0.02, (
            f"Annulus area error: {rel_error:.4f} "
            f"(got {raster_area:.2f}, expected {expected_area:.2f})")

    def test_hole_interior_is_empty(self):
        """Pixel centers inside the hole should be NaN."""
        cx, cy = 5.0, 5.0
        outer = _circle_polygon(cx, cy, 4.0, n=256)
        inner = _circle_polygon(cx, cy, 2.0, n=256)
        annulus = outer.difference(inner)

        result = rasterize([(annulus, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=np.nan)

        # Center pixel should be NaN (inside hole)
        center_r = 50  # y=5.0
        center_c = 50  # x=5.0
        assert np.isnan(result.values[center_r, center_c]), (
            "Center of annulus (inside hole) should be NaN")

        # A ring of pixels near the center should also be NaN
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                dist = np.sqrt(dr ** 2 + dc ** 2) * (10.0 / 100)
                if dist < 1.5:  # well inside the r=2.0 hole
                    assert np.isnan(
                        result.values[center_r + dr, center_c + dc]
                    ), f"Pixel at offset ({dr},{dc}) should be inside hole"


# ---------------------------------------------------------------------------
# 5. Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    """Symmetric polygons centered on the grid should produce symmetric
    rasters.

    The scanline evaluates edge crossings at pixel-boundary y-coordinates
    (not pixel-center), which can break perfect symmetry at boundary
    pixels where edges are nearly tangent.  We check that:
    1. Axis-aligned shapes (square) are perfectly symmetric.
    2. Non-axis-aligned shapes have at most a small number of boundary
       mismatches.
    """

    def test_square_symmetry(self):
        """Centered square (axis-aligned) should have perfect 4-fold
        symmetry."""
        sq = box(2, 2, 8, 8)
        result = rasterize([(sq, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=0)
        vals = result.values
        # LR
        np.testing.assert_array_equal(vals, vals[:, ::-1])
        # TB
        np.testing.assert_array_equal(vals, vals[::-1, :])
        # 90-degree rotation (only for square grid + centered square)
        np.testing.assert_array_equal(vals, vals.T)

    def test_circle_approximate_symmetry(self):
        """Circle should be approximately symmetric (boundary pixels
        may differ due to scanline coordinate evaluation)."""
        circle = _circle_polygon(5, 5, 3, n=256)
        result = rasterize([(circle, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=0)
        vals = result.values
        total = np.sum(vals > 0)

        # LR symmetry: mismatch should be bounded to boundary pixels
        # (the scanline's pixel-boundary y-evaluation affects all non-
        # axis-aligned edges, so ~60% of boundary pixels can differ)
        lr_diff = int(np.sum(vals != vals[:, ::-1]))
        assert lr_diff < total * 0.08, (
            f"Circle LR asymmetry too large: {lr_diff}/{total}")

        # TB symmetry
        tb_diff = int(np.sum(vals != vals[::-1, :]))
        assert tb_diff < total * 0.08, (
            f"Circle TB asymmetry too large: {tb_diff}/{total}")

    def test_diamond_approximate_symmetry(self):
        """Diamond should be approximately symmetric."""
        diamond = Polygon([(5, 2), (8, 5), (5, 8), (2, 5), (5, 2)])
        result = rasterize([(diamond, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=0)
        vals = result.values
        total = np.sum(vals > 0)

        lr_diff = int(np.sum(vals != vals[:, ::-1]))
        assert lr_diff < total * 0.08, (
            f"Diamond LR asymmetry too large: {lr_diff}/{total}")

        tb_diff = int(np.sum(vals != vals[::-1, :]))
        assert tb_diff < total * 0.08, (
            f"Diamond TB asymmetry too large: {tb_diff}/{total}")


# ---------------------------------------------------------------------------
# 6. Merge mode identities
# ---------------------------------------------------------------------------

class TestMergeModeIdentities:
    """Mathematical identities between merge modes."""

    def test_sum_ones_equals_count(self):
        """merge='sum' with all values=1 should equal merge='count'."""
        polys = [box(0, 0, 6, 6), box(4, 4, 10, 10)]
        pairs_sum = [(p, 1.0) for p in polys]
        pairs_count = [(p, 99.0) for p in polys]  # value ignored for count

        sum_result = rasterize(pairs_sum, width=50, height=50,
                               bounds=(0, 0, 10, 10), merge='sum', fill=0)
        count_result = rasterize(pairs_count, width=50, height=50,
                                 bounds=(0, 0, 10, 10), merge='count', fill=0)
        np.testing.assert_array_equal(
            sum_result.values, count_result.values,
            err_msg="sum with values=1 should equal count")

    def test_max_single_geometry_equals_last(self):
        """With one geometry, max/min/first/last should all agree."""
        geom = box(1, 1, 9, 9)
        val = 42.0
        kwargs = dict(width=30, height=30, bounds=(0, 0, 10, 10), fill=0)

        last = rasterize([(geom, val)], merge='last', **kwargs).values
        first = rasterize([(geom, val)], merge='first', **kwargs).values
        mx = rasterize([(geom, val)], merge='max', **kwargs).values
        mn = rasterize([(geom, val)], merge='min', **kwargs).values

        np.testing.assert_array_equal(last, first)
        np.testing.assert_array_equal(last, mx)
        np.testing.assert_array_equal(last, mn)

    def test_first_last_agree_non_overlapping(self):
        """With non-overlapping geometries, first and last should match."""
        pairs = [(box(0, 0, 4, 4), 1.0), (box(6, 6, 10, 10), 2.0)]
        kwargs = dict(width=50, height=50, bounds=(0, 0, 10, 10), fill=0)

        first = rasterize(pairs, merge='first', **kwargs).values
        last = rasterize(pairs, merge='last', **kwargs).values
        np.testing.assert_array_equal(first, last)

    def test_max_always_gte_individual(self):
        """merge='max' result should be >= any individual geometry's value
        at every pixel."""
        pairs = [(box(0, 0, 6, 6), 3.0), (box(4, 4, 10, 10), 7.0)]
        kwargs = dict(width=50, height=50, bounds=(0, 0, 10, 10), fill=0)

        max_result = rasterize(pairs, merge='max', **kwargs).values

        for geom, val in pairs:
            single = rasterize([(geom, val)], merge='last', **kwargs).values
            filled = single > 0
            assert np.all(max_result[filled] >= single[filled]), (
                f"max result should be >= individual value {val}")

    def test_min_always_lte_individual(self):
        """merge='min' result should be <= any individual geometry's value
        at every pixel."""
        pairs = [(box(0, 0, 6, 6), 3.0), (box(4, 4, 10, 10), 7.0)]
        kwargs = dict(width=50, height=50, bounds=(0, 0, 10, 10), fill=0)

        min_result = rasterize(pairs, merge='min', **kwargs).values

        for geom, val in pairs:
            single = rasterize([(geom, val)], merge='last', **kwargs).values
            filled = single > 0
            assert np.all(min_result[filled] <= single[filled]), (
                f"min result should be <= individual value {val}")

    def test_count_nonneg(self):
        """merge='count' should never produce negative values."""
        pairs = [(box(0, 0, 6, 6), 1.0), (box(3, 3, 8, 8), 1.0),
                 (box(5, 5, 10, 10), 1.0)]
        result = rasterize(pairs, width=50, height=50,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        assert np.all(result.values >= 0)


# ---------------------------------------------------------------------------
# 7. all_touched monotonicity
# ---------------------------------------------------------------------------

class TestAllTouchedMonotonicity:
    """all_touched=True should fill a superset of all_touched=False pixels."""

    def _check_superset(self, pairs, width, height, bounds):
        normal = rasterize(pairs, width=width, height=height,
                           bounds=bounds, all_touched=False, fill=np.nan)
        touched = rasterize(pairs, width=width, height=height,
                            bounds=bounds, all_touched=True, fill=np.nan)

        normal_filled = _filled_mask(normal)
        touched_filled = _filled_mask(touched)

        # Every pixel filled by normal must also be filled by touched
        assert np.all(touched_filled | ~normal_filled), (
            "all_touched=True should be a superset of all_touched=False")

        # Touched should fill at least as many pixels
        assert np.sum(touched_filled) >= np.sum(normal_filled)

    def test_small_polygon(self):
        """Small polygon that might miss pixel centers."""
        geom = box(2.1, 2.1, 2.9, 2.9)
        self._check_superset([(geom, 1.0)], 10, 10, (0, 0, 5, 5))

    def test_tilted_rectangle(self):
        """Tilted rectangle -- diagonal edges cross many pixel boundaries."""
        poly = Polygon([(2, 1), (8, 3), (7, 7), (1, 5), (2, 1)])
        self._check_superset([(poly, 1.0)], 50, 50, (0, 0, 10, 10))

    def test_triangle(self):
        tri = Polygon([(1, 1), (9, 1), (5, 9), (1, 1)])
        self._check_superset([(tri, 1.0)], 50, 50, (0, 0, 10, 10))

    def test_circle(self):
        circle = _circle_polygon(5, 5, 3, n=64)
        self._check_superset([(circle, 1.0)], 50, 50, (0, 0, 10, 10))

    def test_multiple_geometries(self):
        """Monotonicity should hold even with multiple geometries."""
        pairs = [
            (box(0, 0, 3, 3), 1.0),
            (Polygon([(5, 2), (8, 5), (5, 8), (2, 5)]), 2.0),
        ]
        self._check_superset(pairs, 50, 50, (0, 0, 10, 10))


# ---------------------------------------------------------------------------
# 8. Reference comparison (rasterio)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
class TestRasterioReference:
    """Compare our rasterize output against rasterio.features.rasterize.

    Our scanline evaluates edge crossings at pixel-boundary y-coordinates
    while GDAL uses a different approach, so boundary pixels can differ.
    We check that:
    1. Axis-aligned rectangles match exactly.
    2. Non-axis-aligned shapes agree on >97% of pixels.
    """

    def _rasterio_rasterize(self, shapes, width, height, bounds):
        """Rasterize using rasterio as a reference."""
        from rasterio.transform import from_bounds
        transform = from_bounds(*bounds, width, height)
        out = rasterio.features.rasterize(
            shapes, out_shape=(height, width), transform=transform,
            fill=0, dtype='float64')
        return out

    def test_simple_rectangle(self):
        """Rectangle should match rasterio exactly."""
        geom = box(2, 2, 8, 8)
        bounds = (0, 0, 10, 10)
        w, h = 50, 50

        ours = rasterize([(geom, 1.0)], width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize([(geom, 1.0)], w, h, bounds)
        np.testing.assert_array_equal(
            ours, ref, err_msg="Rectangle: mismatch vs rasterio")

    def test_multiple_rectangles(self):
        """Multiple axis-aligned rectangles should match exactly."""
        shapes = [(box(0, 0, 4, 4), 1.0), (box(6, 6, 10, 10), 2.0)]
        bounds = (0, 0, 10, 10)
        w, h = 50, 50

        ours = rasterize(shapes, width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize(shapes, w, h, bounds)
        np.testing.assert_array_equal(
            ours, ref, err_msg="Multiple rectangles: mismatch vs rasterio")

    def test_overlapping_rectangles(self):
        """Overlapping rectangles with 'last' merge should match."""
        shapes = [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]
        bounds = (0, 0, 10, 10)
        w, h = 50, 50

        ours = rasterize(shapes, width=w, height=h,
                         bounds=bounds, fill=0, merge='last').values
        ref = self._rasterio_rasterize(shapes, w, h, bounds)
        np.testing.assert_array_equal(
            ours, ref, err_msg="Overlapping: mismatch vs rasterio")

    def test_polygon_with_hole(self):
        """Polygon with axis-aligned hole should match exactly."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
        poly = Polygon(exterior, [hole])
        bounds = (0, 0, 10, 10)
        w, h = 50, 50

        ours = rasterize([(poly, 1.0)], width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize([(poly, 1.0)], w, h, bounds)
        np.testing.assert_array_equal(
            ours, ref, err_msg="Polygon with hole: mismatch vs rasterio")

    def test_concave_l_shape(self):
        """L-shape (axis-aligned edges) should match exactly."""
        L = _l_shape_polygon()
        bounds = (0, 0, 6, 6)
        w, h = 60, 60

        ours = rasterize([(L, 1.0)], width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize([(L, 1.0)], w, h, bounds)
        np.testing.assert_array_equal(
            ours, ref, err_msg="L-shape: mismatch vs rasterio")

    def test_triangle_close_match(self):
        """Triangle (slanted edges) should agree on >97% of pixels."""
        tri = Polygon([(1, 1), (9, 1), (5, 9), (1, 1)])
        bounds = (0, 0, 10, 10)
        w, h = 50, 50

        ours = rasterize([(tri, 1.0)], width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize([(tri, 1.0)], w, h, bounds)
        total = w * h
        diff = int(np.sum(ours != ref))
        assert diff < total * 0.03, (
            f"Triangle: {diff}/{total} pixels differ vs rasterio")

    def test_star_close_match(self):
        """Star (slanted concave edges) should agree on >97% of pixels."""
        star = _star_polygon(5, 5, 4, 1.5, n_points=5)
        bounds = (0, 0, 10, 10)
        w, h = 100, 100

        ours = rasterize([(star, 1.0)], width=w, height=h,
                         bounds=bounds, fill=0).values
        ref = self._rasterio_rasterize([(star, 1.0)], w, h, bounds)
        total = w * h
        diff = int(np.sum(ours != ref))
        assert diff < total * 0.03, (
            f"Star: {diff}/{total} pixels differ vs rasterio")


# ---------------------------------------------------------------------------
# 9. Stress tests / edge cases for accuracy
# ---------------------------------------------------------------------------

class TestAccuracyEdgeCases:
    """Edge cases that can trip up scanline and Bresenham algorithms."""

    def test_very_thin_horizontal_polygon_all_touched(self):
        """Polygon thinner than one pixel height fills nothing under
        center-point rules but should fill with all_touched=True."""
        thin = box(1, 4.3, 9, 4.7)
        normal = rasterize([(thin, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0)
        touched = rasterize([(thin, 1.0)], width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0,
                            all_touched=True)
        # all_touched should burn the boundary
        assert np.sum(touched.values > 0) > 0, (
            "Very thin polygon should fill pixels with all_touched=True")
        # Normal may or may not fill (depends on exact pixel alignment)
        assert np.sum(touched.values > 0) >= np.sum(normal.values > 0)

    def test_very_thin_vertical_polygon_all_touched(self):
        """Vertical thin polygon should fill with all_touched=True."""
        thin = box(4.3, 1, 4.7, 9)
        touched = rasterize([(thin, 1.0)], width=10, height=10,
                            bounds=(0, 0, 10, 10), fill=0,
                            all_touched=True)
        assert np.sum(touched.values > 0) > 0

    def test_polygon_spanning_single_pixel(self):
        """Small polygon covering one pixel center should fill <= 1 pixel."""
        # Pixel center at (2.5, 7.5) -> row=2, col=2 in 10x10 grid [0,10]
        tiny = box(2.1, 7.1, 2.9, 7.9)
        result = rasterize([(tiny, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=np.nan)
        filled = _filled_count(result)
        # At coarse resolution, a tiny polygon may fill 0 or 1 pixels
        assert filled <= 1

    def test_many_small_non_overlapping_polygons(self):
        """Many small polygons should not cause fill errors."""
        pairs = []
        for i in range(10):
            for j in range(10):
                sq = box(i + 0.1, j + 0.1, i + 0.9, j + 0.9)
                pairs.append((sq, float(i * 10 + j)))
        result = rasterize(pairs, width=100, height=100,
                           bounds=(0, 0, 10, 10), merge='count', fill=0)
        # Each small square should count at most 1 (no overlaps)
        assert np.all(result.values <= 1)

    def test_bowtie_concave_polygon(self):
        """Concave polygon with a narrow pinch point."""
        # U-shaped polygon
        u_shape = Polygon([
            (0, 0), (10, 0), (10, 8), (8, 8), (8, 3), (2, 3),
            (2, 8), (0, 8), (0, 0)
        ])
        result = rasterize([(u_shape, 1.0)], width=50, height=40,
                           bounds=(0, 0, 10, 8), fill=np.nan)

        # Interior of the U (should be empty)
        # The open part: x in [2,8], y in [3,8] -- pixel centers at y > 3
        # For this test, check a pixel clearly inside the opening
        # y=5.5 -> row = (8 - 5.5)/0.2 = 12.5 -> row 12
        # x=5.0 -> col = (5.0)/0.2 = 25
        assert np.isnan(result.values[12, 25]), (
            "Interior of U-shape should be empty")

        # Bottom of U should be filled
        # y=1.5 -> row = (8 - 1.5)/0.2 = 32.5 -> row 32
        assert result.values[32, 25] == 1.0, (
            "Bottom of U-shape should be filled")

    def test_nearly_degenerate_triangle(self):
        """Very thin triangle (nearly degenerate) should not crash."""
        thin_tri = Polygon([(0, 0), (10, 0.001), (10, 0), (0, 0)])
        result = rasterize([(thin_tri, 1.0)], width=100, height=100,
                           bounds=(0, 0, 10, 10), fill=0)
        # Should not crash; may fill 0 or a few pixels
        assert result.shape == (100, 100)
