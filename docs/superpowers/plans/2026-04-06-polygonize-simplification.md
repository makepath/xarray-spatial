# Polygonize Geometry Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add topology-preserving Douglas-Peucker simplification to `polygonize()` via shared-edge decomposition.

**Architecture:** New `simplify_tolerance` and `simplify_method` parameters on `polygonize()`. Simplification runs after boundary tracing / chunk merging but before output conversion. A shared-edge approach decomposes all polygon rings into unique edge chains at junction vertices, simplifies each chain once with numba-compiled Douglas-Peucker, then reassembles rings. This guarantees adjacent polygons share identical simplified boundaries (no gaps/overlaps).

**Tech Stack:** Python, numba (`@ngjit`), numpy, xarray, pytest. Optional: shapely (for topology tests only).

---

### Task 1: Douglas-Peucker kernel

**Files:**
- Modify: `xrspatial/polygonize.py` (insert after `_group_rings_into_polygons` ~line 920)
- Test: `xrspatial/tests/test_polygonize.py`

- [ ] **Step 1: Write the failing test for `_douglas_peucker`**

Add at the end of `xrspatial/tests/test_polygonize.py`:

```python
class TestSimplifyHelpers:
    """Tests for internal simplification helper functions."""

    def test_douglas_peucker_straight_line(self):
        """DP on a straight line should reduce to just endpoints."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _douglas_peucker(coords, 0.1)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_douglas_peucker_preserves_bend(self):
        """DP should keep a vertex that exceeds tolerance."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 0.0]],
                          dtype=np.float64)
        # Distance of (2,3) from line (0,0)-(4,0) is 3.0
        result = _douglas_peucker(coords, 2.0)
        assert len(result) == 3  # all points kept

    def test_douglas_peucker_removes_below_tolerance(self):
        """DP should remove a vertex within tolerance."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [2.0, 0.5], [4.0, 0.0]],
                          dtype=np.float64)
        # Distance of (2,0.5) from line (0,0)-(4,0) is 0.5
        result = _douglas_peucker(coords, 1.0)
        expected = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert_allclose(result, expected)

    def test_douglas_peucker_two_points(self):
        """DP on two points should return them unchanged."""
        from ..polygonize import _douglas_peucker
        coords = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = _douglas_peucker(coords, 1.0)
        assert_allclose(result, coords)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest xrspatial/tests/test_polygonize.py::TestSimplifyHelpers -v`
Expected: FAIL with ImportError (`_douglas_peucker` not found)

- [ ] **Step 3: Implement `_douglas_peucker`**

Add to `xrspatial/polygonize.py` after `_group_rings_into_polygons` (~line 920), before `_merge_polygon_rings`:

```python
@ngjit
def _perpendicular_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from point (px,py) to line (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


@ngjit
def _douglas_peucker(coords, tolerance):
    """Douglas-Peucker line simplification on an Nx2 float64 array.

    Endpoints are always preserved. Returns a new Nx2 array with
    only the retained vertices.
    """
    n = len(coords)
    if n <= 2:
        return coords.copy()

    # Iterative DP using an explicit stack to avoid recursion depth issues.
    keep = np.zeros(n, dtype=nb.boolean)
    keep[0] = True
    keep[n - 1] = True

    # Stack of (start, end) index pairs.
    stack = [(np.int64(0), np.int64(n - 1))]
    while len(stack) > 0:
        start, end = stack.pop()
        if end - start < 2:
            continue

        ax, ay = coords[start, 0], coords[start, 1]
        bx, by = coords[end, 0], coords[end, 1]

        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            d = _perpendicular_distance(
                coords[i, 0], coords[i, 1], ax, ay, bx, by)
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > tolerance:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))

    count = 0
    for i in range(n):
        if keep[i]:
            count += 1

    result = np.empty((count, 2), dtype=np.float64)
    j = 0
    for i in range(n):
        if keep[i]:
            result[j, 0] = coords[i, 0]
            result[j, 1] = coords[i, 1]
            j += 1

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest xrspatial/tests/test_polygonize.py::TestSimplifyHelpers -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add xrspatial/polygonize.py xrspatial/tests/test_polygonize.py
git commit -m "Add Douglas-Peucker kernel for polygonize simplification (#1151)"
```

---

### Task 2: Shared-edge simplification orchestrator

**Files:**
- Modify: `xrspatial/polygonize.py` (insert after `_douglas_peucker`)
- Test: `xrspatial/tests/test_polygonize.py`

- [ ] **Step 1: Write the failing test for `_simplify_polygons`**

Add to the `TestSimplifyHelpers` class in `xrspatial/tests/test_polygonize.py`:

```python
    def test_simplify_polygons_reduces_vertices(self):
        """Simplification should reduce vertex count on staircase edges."""
        from ..polygonize import _simplify_polygons

        # L-shaped polygon (exterior only, staircase boundary).
        # 3x3 grid, value in top-left 2x2 block.
        raster = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [0, 0, 0]], dtype=np.int64)
        data = xr.DataArray(raster)
        column_orig, pp_orig = polygonize(data, return_type="numpy")

        from ..polygonize import _simplify_polygons
        pp_simplified = _simplify_polygons(pp_orig, tolerance=0.5)

        # Area must be preserved.
        for orig_rings, simp_rings in zip(pp_orig, pp_simplified):
            orig_area = sum(calc_boundary_area(r) for r in orig_rings)
            simp_area = sum(calc_boundary_area(r) for r in simp_rings)
            assert_allclose(simp_area, orig_area, atol=1e-10)

    def test_simplify_polygons_topology_preserved(self):
        """Adjacent simplified polygons should not create gaps."""
        from ..polygonize import _simplify_polygons

        # Checkerboard-ish: two adjacent rectangles sharing an edge.
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        column, pp = polygonize(data, return_type="numpy")

        pp_simplified = _simplify_polygons(pp, tolerance=0.0)

        # With tolerance=0, no vertices should be removed; output
        # should match input exactly.
        for orig_rings, simp_rings in zip(pp, pp_simplified):
            assert len(orig_rings) == len(simp_rings)
            for orig_ring, simp_ring in zip(orig_rings, simp_rings):
                assert_allclose(simp_ring, orig_ring)

    def test_simplify_polygons_shared_edge_identical(self):
        """Two polygons sharing an edge must have identical simplified edges."""
        from ..polygonize import _simplify_polygons

        # Create a raster where two regions share a staircase boundary.
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        column, pp = polygonize(data, return_type="numpy")

        pp_simplified = _simplify_polygons(pp, tolerance=1.5)

        # Extract edge vertices of each polygon. The shared boundary
        # should appear in both polygons with identical coordinates.
        # Check total area is preserved (which requires no gaps/overlaps).
        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_simplified)
        assert_allclose(total_simp, total_orig, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest xrspatial/tests/test_polygonize.py::TestSimplifyHelpers::test_simplify_polygons_reduces_vertices -v`
Expected: FAIL with ImportError (`_simplify_polygons` not found)

- [ ] **Step 3: Implement shared-edge orchestrator functions**

Add to `xrspatial/polygonize.py` after `_douglas_peucker`:

```python
def _find_junctions(all_rings):
    """Find junction vertices where 3+ ring boundaries meet.

    A junction is any (x, y) coordinate that appears in 3 or more
    distinct rings.  These vertices are pinned during simplification.

    Parameters
    ----------
    all_rings : list of list of np.ndarray
        polygon_points structure: list of polygons, each polygon is
        a list of rings (Nx2 arrays, closed).

    Returns
    -------
    set of (float, float)
    """
    vertex_ring_count = {}  # (x, y) -> set of ring identifiers
    ring_id = 0
    for rings in all_rings:
        for ring in rings:
            for k in range(len(ring) - 1):  # skip closing duplicate
                pt = (ring[k, 0], ring[k, 1])
                if pt not in vertex_ring_count:
                    vertex_ring_count[pt] = set()
                vertex_ring_count[pt].add(ring_id)
            ring_id += 1

    return {pt for pt, ids in vertex_ring_count.items() if len(ids) >= 3}


def _split_ring_at_junctions(ring, junctions):
    """Split a closed ring into chains at junction vertices.

    Each chain starts and ends at a junction vertex (endpoints included
    in the chain).  If the ring contains no junctions, the entire ring
    is returned as a single chain.

    Parameters
    ----------
    ring : np.ndarray, shape (N, 2)
        Closed ring (first == last vertex).
    junctions : set of (float, float)

    Returns
    -------
    list of np.ndarray
        Each array is an Mx2 chain.  Consecutive chains share their
        endpoint/startpoint.
    """
    n = len(ring) - 1  # number of unique vertices

    # Find indices of junction vertices within this ring.
    junction_indices = []
    for k in range(n):
        if (ring[k, 0], ring[k, 1]) in junctions:
            junction_indices.append(k)

    if len(junction_indices) == 0:
        # No junctions: return the whole ring as a single chain.
        return [ring.copy()]

    # Rotate ring so that the first junction is at index 0.
    first = junction_indices[0]
    if first > 0:
        # Rotate unique vertices, then re-close.
        rotated = np.empty_like(ring)
        rotated[:n - first] = ring[first:n]
        rotated[n - first:n] = ring[:first]
        rotated[n] = rotated[0]
        ring = rotated
        junction_indices = [(ji - first) % n for ji in junction_indices]
        junction_indices.sort()

    # Split at each junction.
    chains = []
    for i in range(len(junction_indices)):
        start = junction_indices[i]
        if i + 1 < len(junction_indices):
            end = junction_indices[i + 1]
        else:
            end = n  # wrap back to first junction (index 0 after rotation)
        chains.append(ring[start:end + 1].copy())

    return chains


def _chain_key(chain):
    """Canonical key for deduplicating shared edge chains.

    Two chains that connect the same pair of junctions but are traversed
    in opposite directions should map to the same key.  We use the sorted
    endpoint pair plus the frozenset of interior vertices.
    """
    start = (chain[0, 0], chain[0, 1])
    end = (chain[-1, 0], chain[-1, 1])
    if start > end:
        start, end = end, start
    # Include chain length to disambiguate chains between the same
    # junction pair with different paths.
    interior = tuple(
        (chain[k, 0], chain[k, 1]) for k in range(1, len(chain) - 1))
    # For reversed chains, interior order is reversed.
    interior_rev = interior[::-1]
    interior = min(interior, interior_rev)
    return (start, end, interior)


def _simplify_polygons(polygon_points, tolerance):
    """Topology-preserving simplification of all polygons.

    Uses shared-edge decomposition: finds junction vertices, splits
    rings into chains at junctions, simplifies each unique chain once
    with Douglas-Peucker, then reassembles rings.

    Parameters
    ----------
    polygon_points : list of list of np.ndarray
        Output of polygonize backend: list of polygons, each polygon
        is [exterior_ring, *hole_rings].
    tolerance : float
        Douglas-Peucker tolerance in coordinate units.

    Returns
    -------
    list of list of np.ndarray
        Same structure as input, with simplified coordinates.
    """
    if tolerance <= 0:
        return polygon_points

    # Step 1: Find junctions.
    junctions = _find_junctions(polygon_points)

    # Step 2 & 3: Split rings into chains, deduplicate, simplify.
    simplified_chains = {}  # chain_key -> simplified np.ndarray

    # We also need to track how to reassemble each ring.
    # ring_info[poly_idx][ring_idx] = list of (chain_key, is_reversed)
    ring_info = []

    for poly_idx, rings in enumerate(polygon_points):
        poly_info = []
        for ring in rings:
            chains = _split_ring_at_junctions(ring, junctions)
            chain_refs = []
            for chain in chains:
                key = _chain_key(chain)
                if key not in simplified_chains:
                    simplified_chains[key] = _douglas_peucker(chain, tolerance)
                # Determine if this chain was reversed relative to canonical.
                start = (chain[0, 0], chain[0, 1])
                canonical_start = (simplified_chains[key][0, 0],
                                   simplified_chains[key][0, 1])
                is_reversed = (start != canonical_start)
                chain_refs.append((key, is_reversed))
            poly_info.append(chain_refs)
        ring_info.append(poly_info)

    # Step 4: Reassemble rings.
    result = []
    for poly_idx, rings in enumerate(polygon_points):
        new_rings = []
        for ring_idx, chain_refs in enumerate(ring_info[poly_idx]):
            if len(chain_refs) == 1 and len(chain_refs[0]) == 2:
                key, is_reversed = chain_refs[0]
                simplified = simplified_chains[key]
                if is_reversed:
                    simplified = simplified[::-1].copy()
                # Ensure ring is closed.
                if not (simplified[0, 0] == simplified[-1, 0] and
                        simplified[0, 1] == simplified[-1, 1]):
                    simplified = np.vstack([simplified, simplified[:1]])
                new_rings.append(simplified)
            else:
                # Multiple chains: concatenate (drop duplicate junction points).
                parts = []
                for key, is_reversed in chain_refs:
                    simplified = simplified_chains[key]
                    if is_reversed:
                        simplified = simplified[::-1].copy()
                    if parts:
                        # Skip first point (same as last of previous chain).
                        parts.append(simplified[1:])
                    else:
                        parts.append(simplified)
                assembled = np.vstack(parts)
                # Ensure ring is closed.
                if not (assembled[0, 0] == assembled[-1, 0] and
                        assembled[0, 1] == assembled[-1, 1]):
                    assembled = np.vstack([assembled, assembled[:1]])
                new_rings.append(assembled)

        # Drop degenerate rings (fewer than 4 vertices = triangle minimum).
        filtered = []
        for ring in new_rings:
            if len(ring) >= 4:
                filtered.append(ring)
            elif len(new_rings) > 0 and ring is new_rings[0]:
                # Keep exterior even if degenerate (shouldn't happen with
                # reasonable tolerances, but better than losing the polygon).
                filtered.append(ring)
        if filtered:
            result.append(filtered)
        else:
            result.append(new_rings)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_polygonize.py::TestSimplifyHelpers -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add xrspatial/polygonize.py xrspatial/tests/test_polygonize.py
git commit -m "Add shared-edge simplification orchestrator (#1151)"
```

---

### Task 3: Wire into `polygonize()` public API

**Files:**
- Modify: `xrspatial/polygonize.py` (function signature + body at ~line 1021)
- Test: `xrspatial/tests/test_polygonize.py`

- [ ] **Step 1: Write failing tests for the public API**

Add to `xrspatial/tests/test_polygonize.py`:

```python
class TestPolygonizeSimplify:
    """Tests for simplify_tolerance and simplify_method parameters."""

    def test_tolerance_none_no_change(self):
        """tolerance=None should produce identical output to no-arg call."""
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        col1, pp1 = polygonize(data)
        col2, pp2 = polygonize(data, simplify_tolerance=None)
        assert col1 == col2
        for r1, r2 in zip(pp1, pp2):
            for a, b in zip(r1, r2):
                assert_allclose(a, b)

    def test_tolerance_zero_no_change(self):
        """tolerance=0.0 should produce identical output."""
        raster = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=np.int64)
        data = xr.DataArray(raster)
        col1, pp1 = polygonize(data)
        col2, pp2 = polygonize(data, simplify_tolerance=0.0)
        assert col1 == col2
        for r1, r2 in zip(pp1, pp2):
            for a, b in zip(r1, r2):
                assert_allclose(a, b)

    def test_negative_tolerance_raises(self):
        """Negative tolerance should raise ValueError."""
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(ValueError, match="simplify_tolerance"):
            polygonize(data, simplify_tolerance=-1.0)

    def test_visvalingam_not_implemented(self):
        """Visvalingam-Whyatt should raise NotImplementedError."""
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(NotImplementedError):
            polygonize(data, simplify_tolerance=1.0,
                       simplify_method="visvalingam-whyatt")

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        raster = np.array([[1, 1], [1, 1]], dtype=np.int64)
        data = xr.DataArray(raster)
        with pytest.raises(ValueError, match="simplify_method"):
            polygonize(data, simplify_tolerance=1.0,
                       simplify_method="invalid")

    def test_simplify_reduces_vertices(self):
        """Simplification should reduce total vertex count."""
        # Staircase boundary between two values.
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        _, pp_orig = polygonize(data)
        _, pp_simp = polygonize(data, simplify_tolerance=1.5)

        orig_verts = sum(len(r) for rings in pp_orig for r in rings)
        simp_verts = sum(len(r) for rings in pp_simp for r in rings)
        assert simp_verts < orig_verts

    def test_simplify_preserves_area(self):
        """Total area must be preserved after simplification."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        _, pp_orig = polygonize(data)
        _, pp_simp = polygonize(data, simplify_tolerance=1.5)

        total_orig = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_orig)
        total_simp = sum(
            sum(calc_boundary_area(r) for r in rings) for rings in pp_simp)
        assert_allclose(total_simp, total_orig, atol=1e-10)

    def test_simplify_with_geopandas(self):
        """Simplification should work with geopandas return type."""
        pytest.importorskip("geopandas")
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        gdf = polygonize(data, simplify_tolerance=0.5,
                         return_type="geopandas")
        assert len(gdf) == 2  # two polygons
        assert gdf.geometry.is_valid.all()

    def test_simplify_with_geojson(self):
        """Simplification should work with geojson return type."""
        raster = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int64)
        data = xr.DataArray(raster)
        fc = polygonize(data, simplify_tolerance=0.5,
                        return_type="geojson")
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2


@pytest.mark.skipif(da is None, reason="dask not installed")
class TestPolygonizeSimplifyDask:
    """Simplification with dask backend."""

    def test_simplify_dask_matches_numpy(self):
        """Dask simplification should produce same areas as numpy."""
        raster = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ], dtype=np.int64)

        data_np = xr.DataArray(raster)
        data_dask = xr.DataArray(da.from_array(raster, chunks=(3, 3)))

        _, pp_np = polygonize(data_np, simplify_tolerance=1.5)
        _, pp_dask = polygonize(data_dask, simplify_tolerance=1.5)

        # Compare per-value area sums.
        col_np, _ = polygonize(data_np, simplify_tolerance=1.5)
        col_dask, _ = polygonize(data_dask, simplify_tolerance=1.5)

        areas_np = {}
        for val, rings in zip(col_np, pp_np):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_np[val] = areas_np.get(val, 0.0) + a

        areas_dask = {}
        for val, rings in zip(col_dask, pp_dask):
            a = sum(calc_boundary_area(r) for r in rings)
            areas_dask[val] = areas_dask.get(val, 0.0) + a

        for val in areas_np:
            assert_allclose(areas_dask[val], areas_np[val], atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest xrspatial/tests/test_polygonize.py::TestPolygonizeSimplify::test_negative_tolerance_raises -v`
Expected: FAIL with TypeError (unexpected keyword argument)

- [ ] **Step 3: Modify `polygonize()` signature and body**

In `xrspatial/polygonize.py`, update the `polygonize` function:

**Signature** — add the two new parameters:

```python
def polygonize(
    raster: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
    connectivity: int = 4,
    transform: Optional[np.ndarray] = None,
    column_name: str = "DN",
    return_type: str = "numpy",
    simplify_tolerance: Optional[float] = None,
    simplify_method: str = "douglas-peucker",
):
```

**Docstring** — add parameter descriptions after the `return_type` entry:

```
    simplify_tolerance: float, optional
        Douglas-Peucker simplification tolerance in coordinate units.
        When set, polygon boundaries are simplified using shared-edge
        decomposition to preserve topology between adjacent polygons.
        Default is None (no simplification).

    simplify_method: str, default="douglas-peucker"
        Simplification algorithm.  Currently only "douglas-peucker" is
        supported.  "visvalingam-whyatt" is reserved for future use.
```

**Validation** — add after the transform check block (~line 1112):

```python
    # Check simplification parameters.
    if simplify_tolerance is not None and simplify_tolerance < 0:
        raise ValueError(
            "simplify_tolerance must be non-negative, "
            f"got {simplify_tolerance}")
    if simplify_method not in ("douglas-peucker", "visvalingam-whyatt"):
        raise ValueError(
            f"simplify_method must be 'douglas-peucker' or "
            f"'visvalingam-whyatt', got '{simplify_method}'")
    if (simplify_method == "visvalingam-whyatt"
            and simplify_tolerance is not None
            and simplify_tolerance > 0):
        raise NotImplementedError(
            "Visvalingam-Whyatt simplification is not yet implemented")
```

**Simplification call** — add after the `mapper(raster)(...)` call and before the return-type conversion block:

```python
    # Apply simplification if requested.
    if simplify_tolerance is not None and simplify_tolerance > 0:
        polygon_points = _simplify_polygons(polygon_points, simplify_tolerance)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_polygonize.py::TestPolygonizeSimplify xrspatial/tests/test_polygonize.py::TestPolygonizeSimplifyDask -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest xrspatial/tests/test_polygonize.py -v`
Expected: All existing tests PASS (new parameters have defaults so no breakage)

- [ ] **Step 6: Commit**

```bash
git add xrspatial/polygonize.py xrspatial/tests/test_polygonize.py
git commit -m "Wire simplify_tolerance into polygonize() public API (#1151)"
```

---

### Task 4: Update documentation

**Files:**
- Modify: `docs/source/reference/utilities.rst` (no changes needed — autodoc picks up new params)
- Verify: docstring is complete and renders correctly

- [ ] **Step 1: Verify docstring has the new parameters**

Read `xrspatial/polygonize.py` and confirm `simplify_tolerance` and `simplify_method` are documented in the `polygonize` docstring.

- [ ] **Step 2: Test docs build (if sphinx is available)**

Run: `cd docs && make html 2>&1 | tail -20`
If sphinx is not set up, skip. The autodoc entry in `utilities.rst` already points to `xrspatial.polygonize.polygonize`, so new params will appear automatically.

- [ ] **Step 3: Commit (if any doc changes were needed)**

Only commit if manual changes were required. Autodoc should handle it.

---

### Task 5: Create user guide notebook

**Files:**
- Create: `examples/user_guide/50_Polygonize_Simplification.ipynb`

- [ ] **Step 1: Create the notebook**

Create `examples/user_guide/50_Polygonize_Simplification.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Polygonize with Geometry Simplification

`polygonize()` converts raster regions into vector polygons. On high-resolution rasters, the resulting geometries can have thousands of vertices per polygon, making them unwieldy for rendering, file export, and spatial joins.

The `simplify_tolerance` parameter applies Douglas-Peucker simplification during polygonization. Topology is preserved: adjacent polygons share identical simplified boundaries, so no gaps or overlaps appear between neighbors.
```

**Cell 2 (code):**
```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from xrspatial import polygonize
```

**Cell 3 (markdown):**
```markdown
## Generate a sample classified raster

We'll create a synthetic land-cover raster with irregular region boundaries — the kind of output you'd get from a classification or segmentation step.
```

**Cell 4 (code):**
```python
rng = np.random.default_rng(42)
shape = (80, 120)

# Start with smooth noise, then classify into 4 land-cover types.
from scipy.ndimage import gaussian_filter
noise = rng.standard_normal(shape)
smooth = gaussian_filter(noise, sigma=8)
classified = np.digitize(smooth, bins=[-0.5, 0.0, 0.5]) + 1  # values 1-4

raster = xr.DataArray(classified.astype(np.int32))

fig, ax = plt.subplots(figsize=(10, 6))
raster.plot(ax=ax, cmap="Set2", add_colorbar=True)
ax.set_title("Classified raster (4 land-cover types)")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

**Cell 5 (markdown):**
```markdown
## Polygonize without simplification

First, let's see what the raw pixel-boundary polygons look like.
```

**Cell 6 (code):**
```python
col_raw, pp_raw = polygonize(raster)
total_verts_raw = sum(len(r) for rings in pp_raw for r in rings)
print(f"Polygons: {len(pp_raw)}, Total vertices: {total_verts_raw}")
```

**Cell 7 (code):**
```python
def plot_polygons(polygon_points, column, title, ax):
    """Plot polygons colored by value."""
    cmap = plt.cm.Set2
    vals = sorted(set(column))
    val_to_color = {v: cmap(i / max(len(vals) - 1, 1)) for i, v in enumerate(vals)}

    patches = []
    colors = []
    for val, rings in zip(column, polygon_points):
        ext = rings[0]
        patches.append(MplPolygon(ext[:, :2], closed=True))
        colors.append(val_to_color[val])

    pc = PatchCollection(patches, facecolors=colors, edgecolors="black",
                         linewidths=0.3)
    ax.add_collection(pc)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.invert_yaxis()

fig, ax = plt.subplots(figsize=(10, 6))
plot_polygons(pp_raw, col_raw, f"Raw polygons ({total_verts_raw} vertices)", ax)
plt.tight_layout()
plt.show()
```

**Cell 8 (markdown):**
```markdown
## Polygonize with simplification

Now apply Douglas-Peucker simplification with increasing tolerances. The `simplify_tolerance` is in the raster's coordinate units (pixels here, but would be meters or degrees with a georeferenced raster).

Topology is preserved: adjacent polygons share identical simplified edges, so no gaps appear between them.
```

**Cell 9 (code):**
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, tol in zip(axes, [0.5, 1.5, 3.0]):
    col, pp = polygonize(raster, simplify_tolerance=tol)
    n_verts = sum(len(r) for rings in pp for r in rings)
    reduction = 100 * (1 - n_verts / total_verts_raw)
    plot_polygons(pp, col,
                  f"tolerance={tol}  ({n_verts} verts, {reduction:.0f}% reduction)",
                  ax)

plt.tight_layout()
plt.show()
```

**Cell 10 (markdown):**
```markdown
## GeoDataFrame output

`simplify_tolerance` works with all return types. Here's a GeoDataFrame:
```

**Cell 11 (code):**
```python
gdf = polygonize(raster, simplify_tolerance=1.5, return_type="geopandas",
                 column_name="landcover")
print(gdf.head())
print(f"\nAll geometries valid: {gdf.geometry.is_valid.all()}")
```

- [ ] **Step 2: Commit**

```bash
git add examples/user_guide/50_Polygonize_Simplification.ipynb
git commit -m "Add polygonize simplification user guide notebook (#1151)"
```

---

### Task 6: Update README feature matrix

**Files:**
- Modify: `README.md` (~line 514)

- [ ] **Step 1: Verify no README change is needed**

The simplification feature adds parameters to the existing `polygonize()` function. No new function is created, and backend support does not change. The existing README row:

```
| [Polygonize](xrspatial/polygonize.py) | Converts contiguous regions of equal value into vector polygons | Standard (CCL) | ✅️ | ✅️ | ✅️ | 🔄 |
```

is still accurate. **No change needed.** Skip this task.

---

### Task 7: Final integration check

- [ ] **Step 1: Run the full polygonize test suite**

Run: `pytest xrspatial/tests/test_polygonize.py -v --tb=short`
Expected: All tests pass, including new simplification tests.

- [ ] **Step 2: Verify backward compatibility**

Run: `python -c "from xrspatial import polygonize; import xarray as xr, numpy as np; d = xr.DataArray(np.array([[1,2],[3,4]])); c, p = polygonize(d); print(f'{len(c)} polygons')"`
Expected: `4 polygons` (no error, same behavior as before)
