# Hypsometric Integral — Design Spec

## Summary

Add a `hypsometric_integral` function to `xrspatial/zonal.py` that computes
the hypsometric integral (HI) per zone and returns a painted-back raster.

HI is a geomorphic maturity indicator defined as:

```
HI = (mean - min) / (max - min)
```

where mean, min, and max are the elevation values within a zone (basin,
catchment, or arbitrary polygon).

## API

```python
def hypsometric_integral(
    zones,
    values,
    nodata=0,
    column=None,
    rasterize_kw=None,
    name='hypsometric_integral',
) -> xr.DataArray:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `zones` | `DataArray`, `GeoDataFrame`, or list of `(geometry, value)` pairs | 2D integer zone IDs. Vector inputs are rasterized via `_maybe_rasterize_zones` using `values` as the template grid. |
| `values` | `DataArray` | 2D elevation raster (float), same shape as zones. |
| `nodata` | `int`, default `0` | Zone ID that represents "no zone". Cells with this zone ID are excluded from computation and filled with `NaN` in the output. Matches `apply()` convention. Set to `None` to include all zone IDs. |
| `column` | `str` or `None` | Column in a GeoDataFrame containing zone IDs. Required when `zones` is a GeoDataFrame. |
| `rasterize_kw` | `dict` or `None` | Extra keyword arguments passed to `rasterize()` when vector zones are provided. |
| `name` | `str` | Name for the output DataArray. Default `'hypsometric_integral'`. |

### Returns

`xr.DataArray` (dtype `float64`) — same shape, dims, coords, and attrs as
`values`. Each cell contains the HI of its zone. Cells belonging to the
`nodata` zone or with non-finite elevation values get `NaN`. Zones with zero
elevation range (flat) also get `NaN`.

## Placement

Lives in `xrspatial/zonal.py` alongside `stats`, `crosstab`, `apply`, etc.

Rationale: HI requires a zones raster + a values raster (same signature as
other zonal functions) and computes a per-zone aggregate statistic. It is
structurally a zonal operation, not a local neighborhood transform.

## Backends

All four backends via `ArrayTypeFunctionMapping`:

- **numpy**: use `_sort_and_stride` to group values by zone, compute
  min/mean/max per zone, build a zone-to-HI lookup, paint back with
  vectorized indexing.
- **cupy**: same logic using `cupy.unique` and device-side scatter/gather.
- **dask+numpy**: compute per-chunk partial aggregates (min, max, sum, count
  per zone) via `map_blocks`, reduce across chunks to get global per-zone
  stats, then `map_blocks` again to paint HI values back using the global
  lookup. Zones and values chunks must be aligned (use `validate_arrays`).
- **dask+cupy**: same two-pass structure. Follows the existing pattern where
  chunk functions use cupy internally (same as `_stats_dask_cupy`).

## Algorithm

1. Validate inputs (2D, matching shapes via `validate_arrays`).
2. Rasterize vector zones if needed (`_maybe_rasterize_zones`).
3. Identify unique zone IDs, excluding `nodata` zone and NaN.
4. For each zone `z`:
   - Mask: cells where `zones == z` and `values` is finite.
   - Compute `min_z`, `mean_z`, `max_z`.
   - `hi_z = (mean_z - min_z) / (max_z - min_z)` if `max_z != min_z`,
     else `NaN`.
5. Paint `hi_z` back into all cells belonging to zone `z`.
6. Fill remaining cells (nodata zone, non-finite values, flat zones) with
   `NaN`.

### Value nodata handling

Only non-finite values (`NaN`, `inf`) are excluded from per-zone statistics.
Users with sentinel nodata values (e.g., -9999) should mask their DEM before
calling this function. This matches the convention used by `apply()`.

## Accessor

Expose via the existing `.xrs` accessor:

```python
elevation.xrs.zonal_hypsometric_integral(zones)
```

Following the `zonal_` prefix convention used by `zonal_stats`, `zonal_apply`,
and `zonal_crosstab`.

## Tests

- **Hand-crafted case**: zones with known elevation distributions and
  pre-computed HI values.
- **Edge cases**: single-cell zones, flat zones (range=0 returns NaN),
  NaN cells within a zone (ignored in computation), zones with all-NaN values.
- **Cross-backend parity**: standard `general_checks` pattern comparing
  numpy, cupy, dask+numpy, dask+cupy outputs.
- **Vector zones input**: verify GeoDataFrame and list-of-pairs rasterization
  paths work.

## Scope

This is intentionally minimal. Future extensions (not in this iteration):
- Hypsometric curve data (normalized area-altitude distribution)
- Per-zone summary table output
- `zone_ids` parameter to restrict computation to a subset of zones
- Skewness / kurtosis of the hypsometric distribution
- Integration as a stat option in `zonal.stats()`
