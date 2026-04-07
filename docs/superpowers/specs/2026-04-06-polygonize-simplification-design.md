# Polygonize Geometry Simplification

**Issue:** #1151
**Date:** 2026-04-06

## Problem

`polygonize()` produces exact pixel-boundary polygons. On high-resolution
rasters this creates dense geometries with thousands of vertices per polygon,
making output slow to render, large on disk, and unwieldy for spatial joins.

The current workaround chains GDAL's `gdal_polygonize.py` with
`ogr2ogr -simplify`, adding an external dependency and intermediate file.

## API

Two new parameters on `polygonize()`:

```python
def polygonize(
    raster, mask=None, connectivity=4, transform=None,
    column_name="DN", return_type="numpy",
    simplify_tolerance=None,            # float, coordinate units
    simplify_method="douglas-peucker",  # str
):
```

- `simplify_tolerance=None` or `0.0`: no simplification (backward compatible).
- `simplify_tolerance > 0`: apply Douglas-Peucker with given tolerance.
- `simplify_method="visvalingam-whyatt"`: raises `NotImplementedError`.
- Negative tolerance raises `ValueError`.

## Algorithm: Shared-Edge Douglas-Peucker

Topology-preserving simplification via shared-edge decomposition, same
approach used by TopoJSON and GRASS `v.generalize`.

### Pipeline position

Simplification runs between boundary tracing and output conversion:

```
CCL -> boundary tracing -> [simplification] -> output conversion
```

For dask backends, simplification runs after chunk merging.

### Steps

1. **Find junctions.** Scan all ring vertices. A junction is any coordinate
   that appears as a vertex in 3 or more distinct rings. These points are
   pinned and never removed by simplification.

2. **Split rings into edge chains.** Walk each ring and split at junction
   vertices. Each resulting chain connects two junctions (or forms a closed
   loop when the ring contains no junctions). Each chain is shared by at
   most 2 adjacent polygons.

3. **Deduplicate chains.** Normalize each chain by its sorted endpoint pair
   so shared edges between adjacent polygons are identified and simplified
   only once.

4. **Simplify each chain.** Apply Douglas-Peucker to each unique chain.
   Junction endpoints are fixed. The DP implementation is numba-compiled
   (`@ngjit`) for performance on large coordinate arrays.

5. **Reassemble rings.** Replace each ring's chain segments with their
   simplified versions and rebuild the ring coordinate arrays.

### Why this preserves topology

Adjacent polygons reference the same physical edge chain. Simplifying
each chain once means both neighbors get identical simplified boundaries.
No gaps or overlaps can arise because there is no independent simplification
of shared geometry.

## Implementation

All new code lives in `xrspatial/polygonize.py` as internal functions.

### New functions

| Function | Decorator | Purpose |
|---|---|---|
| `_find_junctions(all_rings)` | pure Python | Scan rings, return set of junction coords |
| `_split_ring_at_junctions(ring, junctions)` | pure Python | Break one ring into chains at junctions |
| `_normalize_chain(chain)` | pure Python | Canonical key for deduplication |
| `_douglas_peucker(coords, tolerance)` | `@ngjit` | DP simplification on Nx2 array |
| `_simplify_polygons(polygon_points, tolerance)` | pure Python | Orchestrator: junctions -> split -> DP -> reassemble |

### Integration point

In `polygonize()`, after the `mapper(raster)(...)` call returns
`(column, polygon_points)` and before the return-type conversion block:

```python
if simplify_tolerance and simplify_tolerance > 0:
    polygon_points = _simplify_polygons(polygon_points, simplify_tolerance)
```

### Backend behavior

- **NumPy / CuPy:** simplification runs on CPU-side coordinate arrays
  returned by boundary tracing (CuPy already transfers to CPU for tracing).
- **Dask:** simplification runs after `_merge_chunk_polygons()`, on the
  fully merged result.
- No GPU-side simplification. Boundary tracing is already CPU-bound;
  simplification follows the same pattern.

## Constraints

- No Visvalingam-Whyatt yet. The `simplify_method` parameter is present
  in the API for forward compatibility; passing `"visvalingam-whyatt"`
  raises `NotImplementedError`.
- No streaming simplification. The full polygon set must fit in memory,
  same constraint as existing boundary tracing.
- Minimum ring size after simplification: exterior rings keep at least 4
  vertices (3 unique + closing). Degenerate rings (area below tolerance
  squared) are dropped.

## Testing

- Correctness: known 4x4 raster, verify simplified polygon areas match
  originals (simplification must not change topology, only vertex count).
- Vertex reduction: verify output has fewer vertices than unsimplified.
- Topology: verify no gaps between adjacent polygons (union of simplified
  polygons equals union of originals, within floating-point tolerance).
- Edge cases: tolerance=0, tolerance=None, negative tolerance, single-pixel
  raster, raster with one uniform value.
- Backend parity: numpy and dask produce same results.
- Return types: simplification works with all five return types.

## Out of scope

- Visvalingam-Whyatt implementation (future PR).
- GPU-accelerated simplification.
- Per-chunk simplification for dask (simplification is post-merge only).
- Area-weighted simplification or other adaptive tolerance schemes.
