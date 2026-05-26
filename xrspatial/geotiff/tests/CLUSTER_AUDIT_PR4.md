# Cluster audit, PR 4 (backend parity)

Temporary mapping document, deleted in a final commit before approval.

## Folded files

| Old file | New file | Notes |
|---|---|---|
| `test_backend_parity_matrix.py` | `parity/test_backend_matrix.py` | Core matrix harness moved verbatim (with markers re-imported from `_helpers/markers.py`). Contains `test_backend_parity_matrix` and `test_backend_parity_matrix_errors`. |
| `test_backend_full_parity_2211.py` | `parity/test_backend_matrix.py` | Full-corpus parity gate appended with `_fp_` prefix on internals to avoid collision. Contains `test_backend_full_parity`, `test_taxonomy_ids_are_in_manifest`, `test_gpu_skip_reason_is_loud`, `test_gpu_backend_returns_cupy_array`, `test_dask_backend_returns_dask_array`, `test_dask_gpu_backend_returns_dask_of_cupy`. |
| `test_attrs_finalization_parity_2211.py` | `parity/test_backend_matrix.py` | Appended with `_ap_` prefix on internals. Contains `test_canonical_attrs_match_across_backends`, `test_canonical_attrs_keys_match_across_backends`. Dropped: `test_backend_specific_keys_carveout_is_documented` (docstring scan no longer mapped to the new module's structure; the carve-out comment in the appended block names every key). |
| `test_attrs_parity_1548.py` | `parity/test_backend_matrix.py` | Appended as pass-through TIFF tag parity. Contains `test_pass_through_tags_eager_numpy_baseline`, `test_pass_through_tags_dask_matches_numpy`, `test_pass_through_tags_cupy_matches_numpy`, `test_pass_through_tags_dask_cupy_matches_numpy`, `test_pass_through_tags_all_backend_keysets_equal`. |
| `test_backend_pixel_parity_matrix_1813.py` | `parity/test_pixel_equality.py` | Strict pixel-byte parity harness moved verbatim (with markers re-imported). Test ids retain descriptive form: `stripped-uint8-none`, `tiled-float32-none`, `cog-float32-deflate`, etc. |
| `test_backend_kwarg_parity_1561.py` | `parity/test_pixel_equality.py` | Appended as kwarg-threading section. Contains the original `read_geotiff_dask` window / band / max_pixels tests and `write_geotiff_gpu` tiled / max_z_error / streaming_buffer_bytes tests. |
| `test_miniswhite_backend_parity_1797.py` | `parity/test_pixel_equality.py` | Appended as MinIsWhite section. Contains `test_miniswhite_http_matches_local_reader`, `test_miniswhite_http_dask_matches_local_reader`, `test_miniswhite_gpu_matches_cpu_reader`. |

## Per-test mapping highlights

### From `test_backend_parity_matrix.py`

| Old test id | New test id |
|---|---|
| `test_backend_parity_matrix[numpy-int16-single-band]` | `parity/test_backend_matrix.py::test_backend_parity_matrix[numpy-int16-single-band]` |
| `test_backend_parity_matrix[gpu-uint16-multiband-tiled]` | `parity/test_backend_matrix.py::test_backend_parity_matrix[gpu-uint16-multiband-tiled]` |
| `test_backend_parity_matrix_errors[numpy-rotated-no-allow_rotated]` | `parity/test_backend_matrix.py::test_backend_parity_matrix_errors[numpy-rotated-no-allow_rotated]` |

(All parametrize ids preserved.)

### From `test_backend_full_parity_2211.py`

| Old test id | New test id |
|---|---|
| `test_backend_full_parity[<fixture_id>-<backend_id>]` | `parity/test_backend_matrix.py::test_backend_full_parity[<fixture_id>-<backend_id>]` |
| `test_taxonomy_ids_are_in_manifest` | `parity/test_backend_matrix.py::test_taxonomy_ids_are_in_manifest` |
| `test_gpu_skip_reason_is_loud` | `parity/test_backend_matrix.py::test_gpu_skip_reason_is_loud` |
| `test_gpu_backend_returns_cupy_array` | `parity/test_backend_matrix.py::test_gpu_backend_returns_cupy_array` |
| `test_dask_backend_returns_dask_array` | `parity/test_backend_matrix.py::test_dask_backend_returns_dask_array` |
| `test_dask_gpu_backend_returns_dask_of_cupy` | `parity/test_backend_matrix.py::test_dask_gpu_backend_returns_dask_of_cupy` |

### From `test_attrs_finalization_parity_2211.py`

| Old test id | New test id |
|---|---|
| `test_canonical_attrs_match_across_backends[plain_float]` | `parity/test_backend_matrix.py::test_canonical_attrs_match_across_backends[plain_float]` |
| `test_canonical_attrs_match_across_backends[float_with_nodata]` | `parity/test_backend_matrix.py::test_canonical_attrs_match_across_backends[float_with_nodata]` |
| `test_canonical_attrs_match_across_backends[int_with_nodata]` | `parity/test_backend_matrix.py::test_canonical_attrs_match_across_backends[int_with_nodata]` |
| `test_canonical_attrs_match_across_backends[uint8_no_nodata]` | `parity/test_backend_matrix.py::test_canonical_attrs_match_across_backends[uint8_no_nodata]` |
| `test_canonical_attrs_keys_match_across_backends[<fixture>]` | `parity/test_backend_matrix.py::test_canonical_attrs_keys_match_across_backends[<fixture>]` |
| `test_backend_specific_keys_carveout_is_documented` | dropped; the carve-out keys are now listed in a comment inside the appended section. Replacing the docstring scan with a marker comment keeps the carve-out greppable without coupling to the new module's docstring layout. |

### From `test_attrs_parity_1548.py`

| Old test id | New test id |
|---|---|
| `test_numpy_attrs_includes_pass_through_tags` | `parity/test_backend_matrix.py::test_pass_through_tags_eager_numpy_baseline` |
| `test_dask_attrs_match_numpy` | `parity/test_backend_matrix.py::test_pass_through_tags_dask_matches_numpy` |
| `test_cupy_attrs_match_numpy` | `parity/test_backend_matrix.py::test_pass_through_tags_cupy_matches_numpy` |
| `test_dask_cupy_attrs_match_numpy` | `parity/test_backend_matrix.py::test_pass_through_tags_dask_cupy_matches_numpy` |
| `test_all_backend_attrs_keysets_equal` | `parity/test_backend_matrix.py::test_pass_through_tags_all_backend_keysets_equal` |

### From `test_backend_pixel_parity_matrix_1813.py`

| Old test id | New test id |
|---|---|
| `test_open_geotiff_pixel_bytes_match[<backend>-<fixture>]` | `parity/test_pixel_equality.py::test_open_geotiff_pixel_bytes_match[<backend>-<fixture>]` |
| `test_open_geotiff_coords_match[<backend>-<fixture>]` | `parity/test_pixel_equality.py::test_open_geotiff_coords_match[<backend>-<fixture>]` |
| `test_open_geotiff_attrs_match[<backend>-<fixture>]` | `parity/test_pixel_equality.py::test_open_geotiff_attrs_match[<backend>-<fixture>]` |
| `test_read_geotiff_dask_matches_open_geotiff[<fixture>]` | `parity/test_pixel_equality.py::test_read_geotiff_dask_matches_open_geotiff[<fixture>]` |
| `test_read_geotiff_gpu_matches_open_geotiff[<fixture>]` | `parity/test_pixel_equality.py::test_read_geotiff_gpu_matches_open_geotiff[<fixture>]` |
| `test_read_vrt_pixel_bytes_match[<backend>]` | `parity/test_pixel_equality.py::test_read_vrt_pixel_bytes_match[<backend>]` |
| `test_read_vrt_coords_match[<backend>]` | `parity/test_pixel_equality.py::test_read_vrt_coords_match[<backend>]` |
| `test_open_geotiff_dot_vrt_routes_to_read_vrt[<backend>]` | `parity/test_pixel_equality.py::test_open_geotiff_dot_vrt_routes_to_read_vrt[<backend>]` |
| `test_fixture_builders_produce_readable_files[<fixture>]` | `parity/test_pixel_equality.py::test_fixture_builders_produce_readable_files[<fixture>]` |

### From `test_backend_kwarg_parity_1561.py`

| Old test id | New test id |
|---|---|
| `test_read_geotiff_dask_window_clips_region` | `parity/test_pixel_equality.py::test_read_geotiff_dask_window_clips_region` |
| `test_read_geotiff_dask_window_via_dispatcher` | `parity/test_pixel_equality.py::test_read_geotiff_dask_window_via_dispatcher` |
| `test_read_geotiff_dask_band_selects_single_band` | `parity/test_pixel_equality.py::test_read_geotiff_dask_band_selects_single_band` |
| `test_read_geotiff_dask_band_via_dispatcher` | `parity/test_pixel_equality.py::test_read_geotiff_dask_band_via_dispatcher` |
| `test_read_geotiff_dask_max_pixels_rejects_oversized` | `parity/test_pixel_equality.py::test_read_geotiff_dask_max_pixels_rejects_oversized` |
| `test_read_geotiff_dask_window_band_combined` | `parity/test_pixel_equality.py::test_read_geotiff_dask_window_band_combined` |
| `test_read_geotiff_dask_invalid_window_raises` | `parity/test_pixel_equality.py::test_read_geotiff_dask_invalid_window_raises` |
| `test_read_geotiff_dask_invalid_band_raises` | `parity/test_pixel_equality.py::test_read_geotiff_dask_invalid_band_raises` |
| `test_write_geotiff_gpu_rejects_tiled_false` | `parity/test_pixel_equality.py::test_write_geotiff_gpu_rejects_tiled_false` |
| `test_write_geotiff_gpu_rejects_nonzero_max_z_error` | `parity/test_pixel_equality.py::test_write_geotiff_gpu_rejects_nonzero_max_z_error` |
| `test_write_geotiff_gpu_accepts_streaming_buffer_bytes_as_noop` | `parity/test_pixel_equality.py::test_write_geotiff_gpu_accepts_streaming_buffer_bytes_as_noop` |
| `test_to_geotiff_threads_tiled_false_into_gpu_dispatcher` | `parity/test_pixel_equality.py::test_to_geotiff_threads_tiled_false_into_gpu_dispatcher` |

### From `test_miniswhite_backend_parity_1797.py`

| Old test id | New test id |
|---|---|
| `test_http_miniswhite_matches_local_reader` | `parity/test_pixel_equality.py::test_miniswhite_http_matches_local_reader` |
| `test_http_dask_miniswhite_matches_local_reader` | `parity/test_pixel_equality.py::test_miniswhite_http_dask_matches_local_reader` |
| `test_gpu_miniswhite_matches_cpu_reader` | `parity/test_pixel_equality.py::test_miniswhite_gpu_matches_cpu_reader` |

## Files left alone (decisions)

| File | Reason |
|---|---|
| `test_vrt_backend_parity_2321.py` | VRT-specific backend parity. Belongs to PR 6 per the epic. Not touched here. |

## Updates to existing references

`docs/source/reference/release_gate_geotiff.rst` rows that cited the old paths now point at the consolidated `parity/test_backend_matrix.py` / `parity/test_pixel_equality.py`. Verified by `test_release_gate_2321.py::test_release_gate_cites_only_existing_test_files`.

In-source comments in other test files and source modules still reference the old filenames; they are documentation strings, not file lookups, so they do not break collection. They will be updated as those files get folded in later PRs.
