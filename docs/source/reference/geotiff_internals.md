# GeoTIFF backend internals: contract inventory

Phase 1 deliverable for the GeoTIFF refactor epic (issue [#2211]). This page
captures the current shape of the GeoTIFF read and write backends and shows
which contract steps already route through shared helpers and which are still
implemented inline per backend. It is the reference future patches use when
deciding whether a fix belongs in the shared helper or in a single backend.

This is developer-facing internal documentation. Nothing here is part of the
public API. Files referenced live under `xrspatial/geotiff/`.

[#2211]: https://github.com/xarray-contrib/xarray-spatial/issues/2211

## Backend entry points

### Read

| Entry point          | File                              | Returns                |
| -------------------- | --------------------------------- | ---------------------- |
| `open_geotiff`       | `xrspatial/geotiff/__init__.py`   | dispatcher (NumPy / CuPy / Dask / Dask+CuPy / VRT) |
| `read_geotiff_dask`  | `xrspatial/geotiff/_backends/dask.py` | Dask-NumPy DataArray |
| `read_geotiff_gpu`   | `xrspatial/geotiff/_backends/gpu.py`  | CuPy or Dask-CuPy DataArray |
| `read_vrt`           | `xrspatial/geotiff/_backends/vrt.py`  | NumPy / CuPy / Dask DataArray (mosaic) |

### Write

| Entry point          | File                              | Input                  |
| -------------------- | --------------------------------- | ---------------------- |
| `to_geotiff`         | `xrspatial/geotiff/_writers/eager.py` | NumPy / Dask DataArray (auto-dispatches to GPU when input is CuPy-backed) |
| `write_geotiff_gpu`  | `xrspatial/geotiff/_writers/gpu.py`   | CuPy DataArray |
| `write_vrt`          | `xrspatial/geotiff/_writers/vrt.py`   | list of GeoTIFF paths (XML emitter) |

## Contract steps

Every read backend executes the same eight logical steps between the source
kwarg and the returned DataArray. Every writer backend executes the inverse
of steps 1, 3, 4, 6, 7 (no decode, no orientation). The eight read steps:

1. **Source / kwarg validation.** Reject conflicting flags, bad chunks,
   unsupported `overview_level` types, unsafe URLs, file-like buffers paired
   with GPU/dask, etc.
2. **Metadata parse.** Read the IFD chain, geokeys, geotags, and any
   `.tif.ovr` sidecar. Resolve the requested overview level into a single IFD.
3. **Transform / georef classification.** Decide whether the file is fully
   georeferenced, transform-only, CRS-only, no-georef, or rotated. Derive the
   `georef_status` value.
4. **Pixel decode.** Strip or tile decompress, applying the codec, predictor,
   sample format, and planar config.
5. **Orientation / photometric handling.** Apply the TIFF orientation tag and
   the MinIsWhite photometric inversion.
6. **Nodata mask + dtype cast.** Replace the sentinel with NaN (promoting
   integer dtypes to float64 when needed), honour `mask_nodata=False`, apply
   any caller `dtype=` cast.
7. **Attrs finalization.** Stamp the canonical `transform`, `crs`,
   `crs_wkt`, `nodata`, `masked_nodata`, `nodata_pixels_present`,
   `nodata_dtype_cast`, `georef_status`, `rotated_affine`, `vrt_holes`, and
   rich-tag attrs.
8. **DataArray construction.** Build coords (`y`, `x`, optional `band`) and
   wrap the buffer in an `xr.DataArray`.

The write contract collapses to: validate kwargs, validate DataArray
dims / shape / attrs, resolve transform, resolve CRS, resolve nodata, plan
output layout (tile / strip / COG / BigTIFF), encode, and write IFD bytes.

## Shared helpers

Helpers below are the canonical implementations the refactor is moving every
backend onto. Cells in the per-backend tables below cite these names.

### Read helpers

| Helper                              | Location              | Owns |
| ----------------------------------- | --------------------- | ---- |
| `_validate_dispatch_kwargs`         | `_validation.py`      | step 1 |
| `_validate_chunks_arg`              | `_validation.py`      | step 1 (chunks-specific) |
| `_validate_overview_level_arg`      | `_validation.py`      | step 1 (overview-specific) |
| `validate_read_metadata`            | `_validation.py`      | step 1 (rotated / unparseable-CRS / mixed-band) |
| `_read_geo_info`                    | `__init__.py`         | step 2 (metadata-only mmap parse) |
| `_parse_cog_http_meta`              | `_reader.py`          | step 2 (HTTP / fsspec range parse) |
| `extract_geo_info_with_overview_inheritance` | `_geotags.py` | step 2 + step 3 (overview-aware georef) |
| `read_to_array`                     | `_reader.py`          | steps 4 + 5 (CPU decode + orientation + MinIsWhite) |
| `_apply_orientation_gpu` / `_apply_orientation_geo_info` | `_backends/_gpu_helpers.py` | step 5 (GPU side) |
| `_apply_eager_nodata_mask`          | `_attrs.py`           | step 6 (single-sentinel mask) |
| `_validate_dtype_cast`              | `_validation.py`      | step 6 (cast guard) |
| `_set_nodata_attrs`                 | `_attrs.py`           | step 7 (nodata lifecycle attrs) |
| `_populate_attrs_from_geo_info`     | `_attrs.py`           | step 7 (transform / crs / georef_status) |
| `_validate_read_geo_info`           | `_attrs.py`           | step 7 (pre-attrs validation) |
| `_finalize_eager_read`              | `_attrs.py`           | wraps steps 1c + 6 + 7 + 8 for eager backends |
| `_finalize_lazy_read_attrs`         | `_attrs.py`           | wraps steps 1c + 7 for lazy backends |
| `geo_to_coords` / `coords_from_geo_info` / `coords_from_pixel_geometry` | `_coords.py` | step 8 (coord build) |

### Write helpers

| Helper                              | Location              | Owns |
| ----------------------------------- | --------------------- | ---- |
| `_validate_nodata_arg`              | `_validation.py`      | nodata kwarg guard |
| `_validate_tile_size_arg`           | `_validation.py`      | tile-size guard |
| `_validate_3d_writer_dims`          | `_validation.py`      | 3D dim ordering guard |
| `_validate_writer_spatial_shape`    | `_validation.py`      | spatial shape guard |
| `_validate_no_rotated_affine`       | `_validation.py`      | rotated-affine policy |
| `validate_write_metadata`           | `_validation.py`      | conflicting CRS / nodata / non-uniform coords |
| `_resolve_spatial_coords`           | `_runtime.py`         | y/x coord resolution |
| `_has_no_georef_marker`             | `_coords.py`          | no-georef marker detection |
| `_coords_to_transform`              | `_coords.py`          | derive transform from coords |
| `_transform_from_attr`              | `_coords.py`          | parse `attrs['transform']` |
| `_require_transform_for_georeferenced` | `_coords.py`       | guard for missing transform |
| `_validate_crs_arg` / `_validate_crs_fallback` / `_resolve_crs_to_wkt` / `_wkt_to_epsg` | `_crs.py` | CRS resolution |
| `_resolve_nodata_attr`              | `_attrs.py`           | resolve `attrs['nodata']` value |
| `_should_restore_nan_sentinel`      | `_attrs.py`           | NaN-restore-on-write decision |
| `_extract_rich_tags`                | `_attrs.py`           | rich-tag pass-through |

## Per-backend coverage

Cells mark each contract step against each backend as
**shared** (uses the helper named in parentheses), **duplicated** (helper
exists but the backend still inlines the logic, or runs an extra inline check
on top of the helper), or **N/A** (the step does not apply to that backend).
Documented divergences are explicit deviations the refactor is keeping for
now and that the call-site comments justify.

### Read backends

| Step | `open_geotiff` (eager) | `read_geotiff_dask` | `read_geotiff_gpu` (eager) | `read_geotiff_gpu` (chunked) | `read_vrt` (eager) | `read_vrt` (chunked) |
| ---- | ---------------------- | ------------------- | -------------------------- | ---------------------------- | ------------------ | -------------------- |
| 1. source / kwarg validation | shared (`_validate_dispatch_kwargs` then dispatches) | shared (`_validate_dispatch_kwargs`, `_validate_chunks_arg`) | shared (`_validate_dispatch_kwargs`, `_validate_chunks_arg`) | shared (`_validate_dispatch_kwargs`, `_validate_chunks_arg`) | shared (`_validate_dispatch_kwargs`, `_validate_chunks_arg`); duplicated inline overview-level / `missing_sources` / `band_nodata` value rejections | shared (`_validate_dispatch_kwargs`); duplicated inline overview-level / `missing_sources` / `band_nodata` value rejections |
| 2. metadata parse | shared (`read_to_array` -> `_parse_cog_http_meta` for cloud, `parse_header` + `parse_all_ifds` + sidecar otherwise) | shared (`_read_geo_info` for local, `_parse_cog_http_meta` for HTTP/fsspec) | shared (`extract_geo_info_with_overview_inheritance`, `select_overview_ifd`); duplicated inline IFD + sidecar load lifted from `_read_geo_info` | shared (`extract_geo_info_with_overview_inheritance`); duplicated inline IFD + sidecar handling | duplicated (`_parse_vrt` + `_read_vrt_internal` -- VRT-specific, no shared metadata parser) | duplicated (`_parse_vrt` + per-chunk `_vrt_chunk_read`) |
| 3. transform / georef classification | shared (`_populate_attrs_from_geo_info` via `_finalize_eager_read`) | shared (`_populate_attrs_from_geo_info` via `_finalize_lazy_read_attrs`) | shared (`_populate_attrs_from_geo_info` via `_finalize_eager_read`) | shared (`_populate_attrs_from_geo_info` via `_finalize_lazy_read_attrs`) | shared (`_vrt_to_synthetic_geo_info` -> `_finalize_lazy_read_attrs`); documented divergence: rotated VRT promotes to `has_georef=False` + `rotated_affine` instead of the GeoTIFF rotated-drop policy | shared (`_vrt_to_synthetic_geo_info` -> `_finalize_lazy_read_attrs`); same documented divergence |
| 4. pixel decode | shared (`read_to_array`) | shared (per-chunk `read_to_array` / `_fetch_decode_cog_http_tiles`) | duplicated (inline GDS / KvikIO / nvCOMP path with CPU fallback via `read_to_array`) | duplicated (inline GDS + per-chunk delayed; HTTP / fsspec / stripped layouts fall back to `read_geotiff_dask`) | duplicated (`_read_vrt_internal._read_data` per source) | duplicated (per-chunk `_vrt_chunk_read` decodes only sources intersecting the window) |
| 5. orientation / photometric | shared (`read_to_array` applies both) | shared (per chunk via `read_to_array`); rejects non-default orientation on HTTP COG dask path | shared on CPU-fallback (`read_to_array`); duplicated on pure GPU path (`_apply_orientation_gpu`, `_apply_orientation_geo_info`, inline MinIsWhite inversion) | shared on CPU-fallback; duplicated on disk-to-GPU per-chunk path (`_decode_window_gpu_direct`); rejects orientation != 1 in `_gds_chunk_path_available` | duplicated (inline NaN masking in `_vrt._read_data` for float sources; VRT does not carry an orientation tag) | duplicated (per chunk same as eager VRT) |
| 6. nodata mask + dtype cast | shared (`_apply_eager_nodata_mask` + `_validate_dtype_cast` via `_finalize_eager_read`) | duplicated (per-chunk mask inline in `_delayed_read_window`); shared `_validate_dtype_cast` on graph dtype | shared (`_apply_eager_nodata_mask` via `_finalize_eager_read`) on both stripped and tiled paths | duplicated (per-chunk mask inline in `_chunk_task`); shared `_validate_dtype_cast` | duplicated (`_apply_integer_sentinel_mask_with_presence` for per-band integer sentinels, plus inline float-NaN proxy and pre-cast dtype tracking); shared `_validate_dtype_cast` | duplicated (per-chunk integer sentinel mask via `_apply_integer_sentinel_mask_with_presence`); shared `_validate_dtype_cast` |
| 7. attrs finalization | shared (`_finalize_eager_read` -> `_validate_read_geo_info` + `_populate_attrs_from_geo_info` + `_set_nodata_attrs`) | shared (`_finalize_lazy_read_attrs`); documented divergence: `nodata_pixels_present` stays unset on lazy outputs (issue #2135) | shared (`_finalize_eager_read`); GPU MinIsWhite picks `mask_sentinel` from three local stashes (`_mw_mask_nodata`, `_cpu_fallback_geo._mask_nodata`, or raw `nodata`) | shared (`_finalize_lazy_read_attrs`); same `nodata_pixels_present` divergence as the CPU dask path | shared (`_finalize_lazy_read_attrs`); documented divergences: `vrt_holes` injected via `attrs_in` seed; per-band nodata selection runs before the helper; `nodata_pixels_present` stamped post-helper from a VRT-aware scan (`_vrt_mask_with_presence` / `_vrt_scan_for_sentinel`) | shared (`_finalize_lazy_read_attrs`); same VRT divergences as the eager VRT path |
| 8. DataArray construction | shared (`_finalize_eager_read` builds `xr.DataArray` with `coords_from_geo_info`) | duplicated (assembles dask graph + builds `xr.DataArray` inline using `geo_to_coords` / `coords_from_geo_info`) | shared (`_finalize_eager_read`) | duplicated (assembles cupy dask graph + builds `xr.DataArray` inline using `coords_from_geo_info`) | duplicated (builds `xr.DataArray` inline using `coords_from_pixel_geometry`; documented divergence per the VRT branch in `_backends/vrt.py`) | duplicated (per-chunk delayed graph + inline `xr.DataArray` build) |

### Write backends

The TIFF write contract is the inverse of the read contract: validate the
DataArray, resolve transform / CRS / nodata from the attrs, lay out the
output, encode, and emit bytes. Steps 4 and 5 (decode, orientation) have no
write analogue; `to_geotiff` and `write_geotiff_gpu` always emit
Orientation = 1 and rely on the writer assembler (`_writer.write`) for
photometric handling.

| Step | `to_geotiff` (CPU eager / dask) | `write_geotiff_gpu` | `write_vrt` |
| ---- | ------------------------------- | ------------------- | ----------- |
| 1. source / kwarg validation | shared (`_validate_tile_size_arg`, `_validate_3d_writer_dims`, `_validate_writer_spatial_shape`, `_validate_nodata_arg`, `_validate_no_rotated_affine`); duplicated inline compression / `compression_level` / `cog` / `overview_levels` / `bigtiff` / `streaming_buffer_bytes` / `max_z_error` / `photometric` / `allow_internal_only_jpeg` / `allow_experimental_codecs` value rejections | shared (`_validate_tile_size_arg`, `_validate_3d_writer_dims`, `_validate_writer_spatial_shape`, `_validate_nodata_arg`, `_validate_no_rotated_affine`); duplicated inline GPU-specific kwarg rejections (`predictor`, `compression`, `cog`, etc.) | shared (`_validate_nodata_arg`); duplicated inline `path` / `vrt_path` shim, `crs` / `crs_wkt` shim, source path validation |
| 2. metadata parse | N/A (no source to parse; reads attrs off the DataArray) | N/A | duplicated (reads geokeys from the first source file to inherit CRS / nodata; lives in `_vrt.write_vrt`) |
| 3. transform / georef classification | shared (`_transform_from_attr` then `_coords_to_transform` fallback; `_require_transform_for_georeferenced`; `_has_no_georef_marker`; `_resolve_spatial_coords`); shared `validate_write_metadata` (`_check_write_conflicting_crs`, `_check_write_conflicting_nodata`, `_check_write_non_uniform_coords`) | shared (same helpers as CPU eager) | duplicated (VRT XML emitter reads transform from the first source; CRS resolved via `_resolve_crs_to_wkt`) |
| 4. pixel decode | N/A | N/A | N/A |
| 5. orientation / photometric | duplicated (writer assembler `_writer.write` owns photometric tag emission; no shared resolver) | duplicated (same; emits Orientation=1) | N/A |
| 6. nodata mask + dtype cast | shared (`_resolve_nodata_attr`, `_should_restore_nan_sentinel`); duplicated inline NaN-to-sentinel restore step in `_writer.write` | shared (`_resolve_nodata_attr`, `_should_restore_nan_sentinel`); duplicated inline GPU NaN-to-sentinel restore | duplicated (VRT carries the per-band `<NoDataValue>` as a string; XML emitter writes it verbatim) |
| 7. attrs finalization | duplicated (writer pulls `attrs['transform']`, `attrs['crs']`, `attrs['crs_wkt']`, `attrs['nodata']` inline; rich tags via `_extract_rich_tags`) | duplicated (same inline pull as CPU writer) | duplicated (VRT XML emitter writes attrs the source files carry) |
| 8. DataArray construction | N/A (write path); the eager writer optionally returns the path string | N/A | N/A (returns the VRT path) |

## Intended flow

The Phase 2 -- Phase 6 work named in [#2211] moves the **duplicated** cells
above onto the helpers below. Each future PR should land in one row of one
table. New backend code added between now and then should call these helpers
directly rather than re-inlining the logic.

### Read flow

```
source kwarg
  -> _validate_dispatch_kwargs   (step 1)
  -> _validate_chunks_arg / _validate_overview_level_arg as needed
  -> _read_geo_info / _parse_cog_http_meta  (step 2)
  -> extract_geo_info_with_overview_inheritance
  -> validate_read_metadata     (step 3 -- rotated / unparseable / mixed-band)
  -> read_to_array              (steps 4 + 5; per-chunk for dask)
  -> _finalize_eager_read       (steps 6 + 7 + 8 for eager backends)
     OR
     _finalize_lazy_read_attrs  (steps 7 only; caller assembles graph for 4 + 6 + 8)
```

Canonical helper per step:

| Step                        | Canonical helper |
| --------------------------- | ---------------- |
| 1. kwarg validation         | `_validate_dispatch_kwargs` (dispatcher) + `_validate_chunks_arg` / `_validate_overview_level_arg` (per-backend) |
| 2. metadata parse           | `_read_geo_info` (local mmap) / `_parse_cog_http_meta` (cloud) |
| 3. transform classification | `_populate_attrs_from_geo_info` (driven by `geo_info.has_georef` / `rotated_affine` / CRS-only fields) |
| 4. pixel decode             | `read_to_array` (CPU); GPU decoders remain backend-specific but must converge on a single `decode_window` entry point in Phase 5 |
| 5. orientation / photometric | `read_to_array` (CPU); `_apply_orientation_gpu` + `_apply_orientation_geo_info` (GPU) |
| 6. nodata mask + dtype cast | `_apply_eager_nodata_mask` + `_validate_dtype_cast` (eager); for lazy backends the per-chunk mask stays inline but must call the same sentinel-resolution helper |
| 7. attrs finalization       | `_finalize_eager_read` (eager) / `_finalize_lazy_read_attrs` (lazy). Both wrap `_validate_read_geo_info` + `_populate_attrs_from_geo_info` + `_set_nodata_attrs` |
| 8. DataArray construction   | `_finalize_eager_read` (eager). Lazy backends remain inline because the dask graph assembly varies per backend; coords always come from `coords_from_geo_info` / `coords_from_pixel_geometry` |

### Write flow

```
DataArray
  -> _validate_3d_writer_dims, _validate_writer_spatial_shape  (step 1)
  -> _validate_no_rotated_affine, _validate_nodata_arg, _validate_tile_size_arg
  -> _resolve_spatial_coords, _has_no_georef_marker
  -> _transform_from_attr (preferred) or _coords_to_transform (fallback)  (step 3)
  -> _require_transform_for_georeferenced
  -> _validate_crs_arg / _wkt_to_epsg / _resolve_crs_to_wkt
  -> validate_write_metadata    (step 3 -- conflicting CRS / nodata / non-uniform coords)
  -> _resolve_nodata_attr, _should_restore_nan_sentinel  (step 6)
  -> _extract_rich_tags         (step 7 -- attrs surface)
  -> _writer.write / GPU encoder
```

Canonical helper per step:

| Step                        | Canonical helper |
| --------------------------- | ---------------- |
| 1. kwarg / dim validation   | `_validate_3d_writer_dims`, `_validate_writer_spatial_shape`, `_validate_no_rotated_affine`, `_validate_nodata_arg`, `_validate_tile_size_arg` |
| 3. transform / CRS / nodata | `_transform_from_attr` + `_coords_to_transform` fallback; `_require_transform_for_georeferenced`; `_resolve_crs_to_wkt` + `_wkt_to_epsg`; `validate_write_metadata` |
| 6. nodata cast / restore    | `_resolve_nodata_attr`, `_should_restore_nan_sentinel` |
| 7. attrs surface            | `_extract_rich_tags` (and `_populate_attrs_from_geo_info` on the read side defines the canonical attr names the writer should emit) |

## Phase plan reference

| Phase | Target rows from the per-backend tables                           |
| ----- | ----------------------------------------------------------------- |
| 1 (this doc) | none -- inventory only                                     |
| 2     | step 3 (transform / georef) across every read and write backend   |
| 3     | step 6 (nodata mask + dtype cast) across every read backend       |
| 4     | step 7 (attrs finalization) across every read backend             |
| 5     | step 2 (metadata parse) extraction into `_sources.py` / `_decode.py` / `_layout.py` modules |
| 6     | cross-backend parity tests covering every step on every backend   |

See [#2211] for the full epic plan. Subsequent PRs in this series each
target one row of one table and land behind cross-backend parity tests.
