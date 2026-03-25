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
    nodata=np.nan,
    name='hypsometric_integral',
) -> xr.DataArray:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `zones` | `DataArray` or `GeoDataFrame` | 2D zone IDs (integer). GeoDataFrame is rasterized via existing `_maybe_rasterize_zones`. |
| `values` | `DataArray` | 2D elevation raster, same shape as zones. |
| `nodata` | `float` | Fill value for cells outside any zone. Default `np.nan`. |
| `name` | `str` | Name for the output DataArray. Default `'hypsometric_integral'`. |

### Returns

`xr.DataArray` — same shape, dims, coords, and attrs as `values`. Each cell
contains the HI of its zone. Cells outside any zone get `nodata`. Zones with
zero elevation range (flat) get `nodata`.

## Placement

Lives in `xrspatial/zonal.py` alongside `stats`, `crosstab`, `apply`, etc.

Rationale: HI requires a zones raster + a values raster (same signature as
other zonal functions) and computes a per-zone aggregate statistic. It is
structurally a zonal operation, not a local neighborhood transform.

## Backends

All four backends via `ArrayTypeFunctionMapping`:

- **numpy**: iterate unique zones, compute min/mean/max per zone, paint back.
  Can reuse existing `_sort_and_stride` infrastructure for grouping values by
  zone.
- **cupy**: same logic on GPU arrays. Use `cupy.unique`, scatter/gather.
- **dask+numpy**: `map_blocks` or blockwise aggregation. Two-pass: first pass
  computes per-zone min/sum/max/count across chunks, second pass reduces and
  paints back.
- **dask+cupy**: same as dask+numpy but with cupy chunk functions.

## Algorithm

1. Validate inputs (2D, matching shapes).
2. Identify unique zones (excluding NaN / 0 if used as nodata).
3. For each zone `z`:
   - Mask: cells where `zones == z` and `values` is finite.
   - Compute `min_z`, `mean_z`, `max_z`.
   - `hi_z = (mean_z - min_z) / (max_z - min_z)` if `max_z != min_z`, else `nodata`.
4. Paint `hi_z` back into all cells belonging to zone `z`.
5. Fill remaining cells with `nodata`.

## Accessor

Expose via `xrspatial.accessor` as:

```python
da.spatial.hypsometric_integral(zones)
```

where `da` is the elevation DataArray.

## Tests

- **Hand-crafted case**: zones with known elevation distributions and
  pre-computed HI values.
- **Edge cases**: single-cell zones, flat zones (range=0 returns nodata),
  NaN cells within a zone (ignored in computation), zones with all-NaN values.
- **Cross-backend parity**: standard `general_checks` pattern comparing
  numpy, cupy, dask+numpy, dask+cupy outputs.
- **GeoDataFrame zones input**: verify rasterization path works.

## Scope

This is intentionally minimal. Future extensions (not in this iteration):
- Hypsometric curve data (normalized area-altitude distribution)
- Per-zone summary table output
- Skewness / kurtosis of the hypsometric distribution
- Integration as a stat option in `zonal.stats()`
