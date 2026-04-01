# Multi-Observer Viewshed and Line-of-Sight Profiles

**Issue:** xarray-contrib/xarray-spatial#1145  
**Date:** 2026-04-01  
**Module:** `xrspatial/visibility.py` (new)

## Overview

Add three public functions for multi-observer visibility analysis and
point-to-point line-of-sight profiling. All functions build on the existing
single-observer `viewshed()` rather than reimplementing the sweep algorithm.

## Public API

### `cumulative_viewshed`

```python
def cumulative_viewshed(
    raster: xarray.DataArray,
    observers: list[dict],
    target_elev: float = 0,
    max_distance: float | None = None,
) -> xarray.DataArray:
```

**Parameters:**

- `raster` -- elevation DataArray (any backend).
- `observers` -- list of dicts. Required keys: `x`, `y`. Optional keys:
  `observer_elev` (default 0), `target_elev` (overrides the function-level
  default), `max_distance` (per-observer override).
- `target_elev` -- default target elevation for observers that don't specify
  their own.
- `max_distance` -- default maximum analysis radius for observers that don't
  specify their own.

**Returns:** integer DataArray where each cell value is the count of observers
with line-of-sight to that cell. Cells visible to zero observers are 0.

**Algorithm:**

1. For each observer, call `viewshed(raster, ...)` with that observer's
   parameters.
2. Convert the result to a binary mask (1 where value != INVISIBLE, else 0).
3. Sum all masks element-wise.

**Backend behaviour:**

| Backend | Strategy |
|---|---|
| NumPy | Loop over observers, accumulate in-place into a numpy int32 array. |
| CuPy | Delegates to `viewshed()` which handles CuPy dispatch. Accumulate on device. |
| Dask+NumPy | Wrap each `viewshed()` call as a `dask.delayed` task, convert each result to a binary dask array, sum lazily. The graph is submitted once at the end. |
| Dask+CuPy | Same as Dask+NumPy -- `viewshed()` handles the backend internally. |

For dask backends, `max_distance` should be set (either globally or
per-observer) to keep each viewshed computation tractable. Without it, each
observer viewshed loads the full raster into memory.

### `visibility_frequency`

```python
def visibility_frequency(
    raster: xarray.DataArray,
    observers: list[dict],
    target_elev: float = 0,
    max_distance: float | None = None,
) -> xarray.DataArray:
```

Thin wrapper: returns `cumulative_viewshed(...) / len(observers)` cast to
float64. Same parameters and backend behaviour.

### `line_of_sight`

```python
def line_of_sight(
    raster: xarray.DataArray,
    x0: float, y0: float,
    x1: float, y1: float,
    observer_elev: float = 0,
    target_elev: float = 0,
    frequency_mhz: float | None = None,
) -> xarray.Dataset:
```

**Parameters:**

- `raster` -- elevation DataArray.
- `x0, y0` -- observer coordinates in data space.
- `x1, y1` -- target coordinates in data space.
- `observer_elev` -- height above terrain at the observer point.
- `target_elev` -- height above terrain at the target point.
- `frequency_mhz` -- if set, compute first Fresnel zone clearance.

**Returns:** `xarray.Dataset` with dimension `sample` (one entry per cell
along the transect) containing:

| Variable | Type | Description |
|---|---|---|
| `distance` | float64 | Distance from observer along the transect |
| `elevation` | float64 | Terrain elevation at the sample point |
| `los_height` | float64 | Height of the line-of-sight ray at that point |
| `visible` | bool | Whether the cell is visible from the observer |
| `x` | float64 | x-coordinate of the sample point |
| `y` | float64 | y-coordinate of the sample point |
| `fresnel_radius` | float64 | First Fresnel zone radius (only if `frequency_mhz` set) |
| `fresnel_clear` | bool | Whether Fresnel zone is clear of terrain (only if `frequency_mhz` set) |

**Algorithm:**

1. Convert (x0, y0) and (x1, y1) to grid indices using the raster's
   coordinate arrays.
2. Walk the line between the two grid cells using Bresenham's algorithm to
   get the sequence of (row, col) pairs.
3. Extract terrain elevation at each cell. For dask/cupy rasters, pull only
   the transect cells to numpy.
4. Compute the straight-line LOS height at each sample point by linear
   interpolation between observer and target heights (terrain + offsets).
5. Walk forward from the observer tracking the maximum elevation angle seen
   so far. A cell is visible if no prior cell has a higher angle.
6. If `frequency_mhz` is set, compute the first Fresnel zone radius at each
   point: `F1 = sqrt(d1 * d2 * c / (f * D))` where d1 and d2 are distances
   from observer and target, D is total distance, f is frequency, and c is
   the speed of light. A point has Fresnel clearance if the terrain is at
   least F1 below the LOS height.

**Backend behaviour:** Always runs on CPU. For CuPy-backed rasters, the
transect elevations are copied to host. For dask-backed rasters, the transect
slice is computed. The transect is at most `max(H, W)` cells long so this is
always cheap.

## Module structure

```
xrspatial/visibility.py
    cumulative_viewshed()      -- public
    visibility_frequency()     -- public
    line_of_sight()            -- public
    _bresenham_line()          -- private, returns list of (row, col) pairs
    _extract_transect()        -- private, pulls elevation values from any backend
    _fresnel_radius()          -- private, first Fresnel zone calculation
```

## Integration points

- `xrspatial/__init__.py` -- add imports for all three functions.
- `xrspatial/accessor.py` -- add accessor methods for all three functions.
- `docs/source/reference/surface.rst` -- add a "Visibility Analysis" section
  with autosummary entries.
- `README.md` -- add rows for `cumulative_viewshed`, `visibility_frequency`,
  and `line_of_sight` in the feature matrix.
- `examples/user_guide/37_Visibility_Analysis.ipynb` -- new notebook.

## Testing strategy

Tests go in `xrspatial/tests/test_visibility.py`.

**cumulative_viewshed / visibility_frequency:**

- Flat terrain: all cells visible from all observers, count == n_observers.
- Single tall wall: observers on opposite sides, verify cells behind the wall
  are only visible to the observer on their side.
- Single observer: result matches `(viewshed(...) != INVISIBLE).astype(int)`.
- Per-observer parameters: verify that `observer_elev` and `max_distance`
  overrides work.
- Dask backend: verify result matches numpy backend.

**line_of_sight:**

- Flat terrain: all cells visible, LOS height matches linear interpolation.
- Single obstruction: cells behind the peak are not visible.
- Observer/target elevation offsets: verify LOS line shifts up.
- Fresnel zone: known geometry where the zone is partially obstructed.
- Edge case: observer == target (single cell, trivially visible).
- Bresenham correctness: verify the line visits expected cells for known
  endpoints.

## Scope boundaries

**In scope:** the three functions described above, tests, docs, notebook,
README update.

**Out of scope:**

- Refactoring existing `viewshed.py` internals.
- GPU-specific kernels for cumulative viewshed (composition via `viewshed()`
  is sufficient).
- Weighted observer contributions (each observer counts as 1).
- Earth curvature correction for line-of-sight (the transect is typically
  short enough that curvature is negligible; users working at very long
  distances should use geodesic viewshed).
