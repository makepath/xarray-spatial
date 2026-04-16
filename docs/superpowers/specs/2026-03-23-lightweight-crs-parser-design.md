# Lightweight CRS Parser -- Drop-in `pyproj.CRS` Replacement

**Issue:** [#1057](https://github.com/xarray-contrib/xarray-spatial/issues/1057)
**Date:** 2026-03-23

## Problem

The reproject module uses pyproj for two things:

1. **CRS metadata** (~1ms): parsing EPSG codes, extracting projection parameters, comparing CRS objects
2. **Coordinate transforms** (30-700ms): per-pixel projection math

We already replaced #2 with Numba JIT kernels for the most common projection families. But #1 still requires pyproj at import time, making it a hard dependency even when the actual math never touches it.

## Goal

Make `pip install xarray-spatial` sufficient for basic reprojection (UTM, Web Mercator, LCC, Albers, etc.) without pyproj. Users with exotic CRS or datum requirements still benefit from installing pyproj.

## Design

### New file: `xrspatial/reproject/_lite_crs.py`

A `CRS` class that implements the same interface surface as `pyproj.CRS` for EPSG codes that have Numba fast paths.

**Constructor signatures:**

```python
CRS(4326)                    # from EPSG int
CRS("EPSG:4326")             # from authority string
CRS.from_epsg(4326)          # classmethod
CRS.from_wkt(wkt_string)     # extracts EPSG from AUTHORITY["EPSG","XXXX"] via regex
```

Raises `ValueError` if the EPSG code is not in the embedded table.

**Interface (duck-type compatible with pyproj.CRS):**

| Method/Property | Return type | Description |
|---|---|---|
| `.to_dict()` | `dict` | PROJ4-style parameter dict (`k_0` is the canonical scale factor key) |
| `.to_wkt()` | `str` | OGC WKT string |
| `.to_epsg()` | `int` | EPSG code |
| `.to_authority()` | `tuple[str, str]` | `('EPSG', '4326')` |
| `.is_geographic` | `bool` | True for lat/lon CRS |

The `.to_dict()` return value uses PROJ4-style keys: `proj`, `datum`, `ellps`, `lon_0`, `lat_0`, `lat_1`, `lat_2`, `k_0`, `x_0`, `y_0`, `units`, `zone`. The scale factor is always stored as `k_0` (pyproj convention); the `_*_params()` extractors already handle `d.get('k_0', d.get('k', 1.0))` fallback, so this is compatible.

**Equality and hashing:** Two `CRS` objects with the same EPSG code are equal and hash equally. Cross-type comparison with pyproj is one-directional: `lite_crs == pyproj_crs` works (our `__eq__` calls `.to_epsg()` on the other object), but `pyproj_crs == lite_crs` uses pyproj's WKT comparison and may return False. To avoid issues, ensure both CRS objects in any comparison pass through `_resolve_crs()` so they are the same type.

### Embedded EPSG table

Covers codes where Numba fast paths exist:

| Category | EPSG codes |
|---|---|
| Geographic | 4326 (WGS84), 4269 (NAD83), 4267 (NAD27) |
| Web Mercator | 3857 |
| Ellipsoidal Mercator | 3395 |
| UTM North (WGS84) | 32601-32660 |
| UTM South (WGS84) | 32701-32760 |
| UTM (NAD83) | 26901-26923 |
| Lambert Conformal Conic | 2154 |
| Albers Equal Area | 5070 |
| Lambert Azimuthal Equal Area | 3035 |
| Polar Stereographic | 3031, 3413, 3996 |
| Oblique Stereographic | 28992 |
| Cylindrical Equal Area | 6933 |

UTM zones are generated programmatically (not stored individually). Named codes store their full PROJ4 parameter dict and a WKT template string.

**Not in table:**

- **Sinusoidal** -- has a Numba fast path (`_sinu_params`), but MODIS sinusoidal grids use custom WKT or SR-ORG codes, not a standard EPSG code. The fast path dispatches via `.to_dict()['proj'] == 'sinu'`, so it still fires when pyproj is installed and produces the right dict. Without pyproj, sinusoidal falls back to requiring pyproj.
- **Generic Transverse Mercator** (State Plane, national grids) -- `_tmerc_params` dispatches via `.to_dict()['proj'] == 'tmerc'` for hundreds of EPSG codes. Embedding all State Plane codes is out of scope. These fast paths only fire when pyproj provides the `.to_dict()`.
- **Oblique Mercator Hotine** -- `_omerc_params` exists but is disabled in the dispatch pending alignment with PROJ's variant handling.

### Changes to `_crs_utils.py`

`_resolve_crs()` gets a two-tier resolution strategy:

```
_resolve_crs(input):
    1. Try our CRS(input)
       - int -> direct EPSG lookup
       - "EPSG:XXXX" string -> parse and lookup
       - WKT string -> regex for AUTHORITY["EPSG","XXXX"], then lookup
       - existing CRS (ours or pyproj) -> pass through
    2. If step 1 raises ValueError (code not in table):
       -> fall back to pyproj.CRS(input)
       -> if pyproj not installed, raise ImportError
```

New helper `_crs_from_wkt(wkt)` for the chunk functions that reconstruct CRS from WKT strings. Same two-tier logic: try our `CRS.from_wkt()`, fall back to `pyproj.CRS.from_wkt()`.

Note: `_detect_source_crs()` calls `_resolve_crs()` and benefits automatically. The rioxarray fallback path (`raster.rio.crs`) always returns a pyproj CRS, which passes through unchanged.

### Changes to `_grid.py`

`_compute_output_grid()` currently creates a `pyproj.Transformer` to project ~845 boundary and interior sample points. New flow:

1. Build the boundary/interior sample points (same as today).
2. Call a new `_transform_points(src_crs, tgt_crs, xs, ys)` helper that accepts scatter points.
3. That helper extracts the forward/inverse point-level projection functions from `_projections.py` (e.g. `_merc_fwd_point`, UTM kernels, etc.) based on the CRS pair, then applies them in a batch loop over the sample points. This reuses the existing projection math without needing a synthetic grid.
4. If no lite fast path exists, fall back to `pyproj.Transformer` as before.

Note: `_compute_output_grid` also reads `source_crs.is_geographic` (line 44) for coordinate clamping. The lite CRS must return the correct value for this property.

### Changes to chunk functions

`_reproject_chunk_numpy()` and `_reproject_chunk_cupy()` currently call `pyproj.CRS.from_wkt()` to reconstruct CRS objects. Changed to use `_crs_from_wkt()` which tries our `CRS.from_wkt()` first.

`_reproject_chunk_cupy()` also creates a `pyproj.Transformer` unconditionally before checking the CUDA fast path. This must be restructured to defer Transformer creation to after the CUDA fast path check, matching the pattern in `_reproject_chunk_numpy()`.

`_source_footprint_in_target()` (used by `merge()`) also constructs `pyproj.CRS` and Transformer objects. This function needs the same two-tier CRS resolution and Numba-based point transform treatment.

When the Numba/CUDA fast path handles the transform, pyproj is never imported.

### What still requires pyproj

- CRS pairs without Numba fast paths (per-chunk Transformer fallback)
- WKT strings without an AUTHORITY/EPSG tag
- PROJ4 dict input, custom CRS definitions
- Generic Transverse Mercator / State Plane (dispatches via `.to_dict()['proj']`, not EPSG code)
- Sinusoidal (no standard EPSG code)
- Vertical datum transforms (`_vertical.py` and inline geoid code at `__init__.py:725-728`)
- ITRF frame transforms (`_itrf.py`)
- GeoTIFF CRS utilities (`from_user_input`)

All of these paths already have `_require_pyproj()` guards.

### Error messages

When pyproj is not installed and the user hits a code path that needs it:

```
pyproj is required for CRS "EPSG:9999" (not in the built-in table).
Install it with:  pip install pyproj
or:  pip install xarray-spatial[reproject]
```

### Testing

- Unit tests for `CRS` construction, all methods, equality, hashing
- Round-trip: `CRS(epsg).to_wkt()` -> `CRS.from_wkt(wkt)` -> same object
- Parameter correctness: compare `.to_dict()` output against `pyproj.CRS` for every embedded code
- Integration: full reprojection without pyproj on path (mock pyproj as missing)
- Edge cases: unknown EPSG falls back to pyproj, WKT without AUTHORITY tag falls back

### Files touched

| File | Change |
|---|---|
| `xrspatial/reproject/_lite_crs.py` | New -- `CRS` class + EPSG table |
| `xrspatial/reproject/_crs_utils.py` | Two-tier resolution, `_crs_from_wkt()` helper |
| `xrspatial/reproject/__init__.py` | Use `_crs_from_wkt()` in chunk functions; restructure cupy chunk; update `_source_footprint_in_target` |
| `xrspatial/reproject/_grid.py` | Numba-based `_transform_points` for boundary transform |
| `xrspatial/tests/test_lite_crs.py` | New -- unit + integration tests |

### Out of scope

- WKT parser (complex grammar, not worth reimplementing)
- Non-EPSG CRS definitions
- Datum transformations beyond what Helmert already handles
- Changes to the GeoTIFF module's pyproj usage
- Oblique Mercator Hotine (kernel disabled pending PROJ alignment)
- Embedding all State Plane / national grid EPSG codes
