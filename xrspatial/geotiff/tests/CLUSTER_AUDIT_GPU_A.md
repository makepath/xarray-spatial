# CLUSTER_AUDIT_GPU_A: GPU reader test consolidation

Maps every `<old_file>::<test>` to its new location
`xrspatial/geotiff/tests/gpu/test_reader.py::<test_id>`.

Per issue #2438 sub-PR A. This file is deleted on the final commit on this branch before merge.

## Counts

- Pre-consolidation files: 10
- Pre-consolidation tests collected: 66
- Post-consolidation file: 1 (`gpu/test_reader.py`)
- Post-consolidation tests collected: 66

## #1605 -- window / band kwarg forwarding (`test_gpu_window_band_1605.py`)

| Old | New |
| --- | --- |
| `test_read_geotiff_gpu_window_matches_eager` | `test_read_geotiff_gpu_window_matches_eager_1605` |
| `test_open_geotiff_gpu_window_no_longer_silently_dropped` | `test_open_geotiff_gpu_window_no_longer_silently_dropped_1605` |
| `test_read_geotiff_gpu_band_selection` | `test_read_geotiff_gpu_band_selection_1605` |
| `test_open_geotiff_gpu_band_no_longer_silently_dropped` | `test_open_geotiff_gpu_band_no_longer_silently_dropped_1605` |
| `test_read_geotiff_gpu_window_and_band` | `test_read_geotiff_gpu_window_and_band_1605` |
| `test_read_geotiff_gpu_window_bounds_validation` | `test_read_geotiff_gpu_window_bounds_validation_1605` |
| `test_read_geotiff_gpu_band_bounds_validation` | `test_read_geotiff_gpu_band_bounds_validation_1605` |
| `test_read_geotiff_gpu_window_rejected_on_nondefault_orientation` | `test_read_geotiff_gpu_window_rejected_on_nondefault_orientation_1605` |
| `test_read_geotiff_gpu_stripped_chunks_produces_dask` | `test_read_geotiff_gpu_stripped_chunks_produces_dask_1605` |
| `test_read_geotiff_gpu_stripped_chunks_tuple` | `test_read_geotiff_gpu_stripped_chunks_tuple_1605` |

## stripped multiband (`test_gpu_stripped_multiband.py`)

| Old | New |
| --- | --- |
| `test_stripped_3band_uint8` | `test_stripped_3band_uint8_stripped_multiband` |
| `test_stripped_2band_uint16` | `test_stripped_2band_uint16_stripped_multiband` |
| `test_stripped_singleband_still_2d` | `test_stripped_singleband_still_2d_stripped_multiband` |

## #1753 -- stripped no-georef windowed coord parity (`test_gpu_stripped_no_georef_window_1753.py`)

| Old | New |
| --- | --- |
| `TestStrippedGpuWindowedNoGeoref::test_window_at_origin` | `TestStrippedGpuWindowedNoGeoref1753::test_window_at_origin` |
| `TestStrippedGpuWindowedNoGeoref::test_offset_window` | `TestStrippedGpuWindowedNoGeoref1753::test_offset_window` |
| `TestStrippedGpuWindowedNoGeoref::test_dask_cupy_window` | `TestStrippedGpuWindowedNoGeoref1753::test_dask_cupy_window` |
| `TestStrippedGpuWindowedNoGeoref::test_dask_cupy_offset_window` | `TestStrippedGpuWindowedNoGeoref1753::test_dask_cupy_offset_window` |
| `TestStrippedGpuWindowedNoGeoref::test_no_transform_attr` | `TestStrippedGpuWindowedNoGeoref1753::test_no_transform_attr` |
| `TestStrippedGpuWindowedNoGeoref::test_uint16_window` | `TestStrippedGpuWindowedNoGeoref1753::test_uint16_window` |
| `TestStrippedGpuWindowedBackendParity::test_dtype_parity[win0]` (win=(0,0,4,4)) | `TestStrippedGpuWindowedBackendParity1753::test_dtype_parity[win0]` |
| `TestStrippedGpuWindowedBackendParity::test_dtype_parity[win1]` (win=(2,3,6,7)) | `TestStrippedGpuWindowedBackendParity1753::test_dtype_parity[win1]` |
| `TestStrippedGpuWindowedBackendParity::test_dtype_parity[win2]` (win=(1,1,7,7)) | `TestStrippedGpuWindowedBackendParity1753::test_dtype_parity[win2]` |
| `TestStrippedGpuWindowedBackendParity::test_value_parity[win0]` (win=(0,0,4,4)) | `TestStrippedGpuWindowedBackendParity1753::test_value_parity[win0]` |
| `TestStrippedGpuWindowedBackendParity::test_value_parity[win1]` (win=(2,3,6,7)) | `TestStrippedGpuWindowedBackendParity1753::test_value_parity[win1]` |
| `TestStrippedGpuWindowedBackendParity::test_value_parity[win2]` (win=(1,1,7,7)) | `TestStrippedGpuWindowedBackendParity1753::test_value_parity[win2]` |
| `TestStrippedGpuWindowedGeorefStillWorks::test_georef_pixel_is_area_window` | `TestStrippedGpuWindowedGeorefStillWorks1753::test_georef_pixel_is_area_window` |
| `TestStrippedGpuWindowedGeorefStillWorks::test_georef_offset_window` | `TestStrippedGpuWindowedGeorefStillWorks1753::test_georef_offset_window` |

## #1732 -- stripped fallback forwards max_pixels/window/band (`test_gpu_stripped_forwarding_1732.py`)

| Old | New |
| --- | --- |
| `test_stripped_max_pixels_cap_is_enforced` | `test_stripped_max_pixels_cap_is_enforced_1732` |
| `test_stripped_window_returns_only_window` | `test_stripped_window_returns_only_window_1732` |
| `test_stripped_band_selection_returns_2d` | `test_stripped_band_selection_returns_2d_1732` |
| `test_stripped_window_plus_band` | `test_stripped_window_plus_band_1732` |

## #1542 -- nodata propagation (`test_gpu_nodata_1542.py`)

| Old | New |
| --- | --- |
| `test_gpu_uint16_nodata_promoted_and_masked_tiled` | `test_gpu_uint16_nodata_promoted_and_masked_tiled_1542` |
| `test_gpu_uint16_nodata_promoted_and_masked_stripped` | `test_gpu_uint16_nodata_promoted_and_masked_stripped_1542` |
| `test_gpu_float32_sentinel_replaced_with_nan` | `test_gpu_float32_sentinel_replaced_with_nan_1542` |
| `test_gpu_no_nodata_keeps_dtype` | `test_gpu_no_nodata_keeps_dtype_1542` |
| `test_gpu_nan_nodata_passes_through` | `test_gpu_nan_nodata_passes_through_1542` |
| `test_gpu_all_four_backends_agree_on_nodata` | `test_gpu_all_four_backends_agree_on_nodata_1542` |
| `test_gpu_int16_negative_nodata` | `test_gpu_int16_negative_nodata_1542` |

## #2097 -- MinIsWhite band-first guard (`test_gpu_miniswhite_band_first_2097.py`)

| Old | New |
| --- | --- |
| `test_band_first_single_band_miniswhite_rejected` | `test_band_first_single_band_miniswhite_rejected_2097` |
| `test_band_last_single_band_miniswhite_still_rejected` | `test_band_last_single_band_miniswhite_still_rejected_2097` |
| `test_2d_single_band_miniswhite_still_rejected` | `test_2d_single_band_miniswhite_still_rejected_2097` |
| `test_samples_hint_band_first_without_gpu` | `test_samples_hint_band_first_without_gpu_2097` |

## #1876 -- chunks= out-of-core dask pipeline (`test_gpu_chunks_out_of_core_1876.py`)

| Old | New |
| --- | --- |
| `test_read_geotiff_gpu_chunks_yields_dask_cupy_chunks` | `test_read_geotiff_gpu_chunks_yields_dask_cupy_chunks_1876` |
| `test_read_geotiff_gpu_chunks_values_match_eager` | `test_read_geotiff_gpu_chunks_values_match_eager_1876` |
| `test_read_geotiff_gpu_no_chunks_returns_eager_cupy` | `test_read_geotiff_gpu_no_chunks_returns_eager_cupy_1876` |
| `test_open_geotiff_gpu_chunks_propagates_to_dask` | `test_open_geotiff_gpu_chunks_propagates_to_dask_1876` |
| `test_read_geotiff_gpu_chunks_preserves_attrs` | `test_read_geotiff_gpu_chunks_preserves_attrs_1876` |
| `test_read_geotiff_gpu_chunks_uses_gds_path_when_available` | `test_read_geotiff_gpu_chunks_uses_gds_path_when_available_1876` |
| `test_read_geotiff_gpu_chunks_window_subset` | `test_read_geotiff_gpu_chunks_window_subset_1876` |
| `test_read_geotiff_gpu_chunks_multi_band` | `test_read_geotiff_gpu_chunks_multi_band_1876` |
| `test_read_geotiff_gpu_chunks_single_band_selection` | `test_read_geotiff_gpu_chunks_single_band_selection_1876` |
| `test_read_geotiff_gpu_chunks_fallback_when_kvikio_absent` | `test_read_geotiff_gpu_chunks_fallback_when_kvikio_absent_1876` |

## #2324 -- sidecar overview-inheritance parity (`test_gpu_sidecar_georef_parity_2324.py`)

| Old | New |
| --- | --- |
| `test_sidecar_without_geokeys_attrs_match_cpu_vs_dask_2324` | `test_sidecar_without_geokeys_attrs_match_cpu_vs_dask_2324` |
| `test_sidecar_without_geokeys_gpu_matches_cpu_2324` | `test_sidecar_without_geokeys_gpu_matches_cpu_2324` |
| `test_sidecar_with_own_geokeys_gpu_matches_cpu_2324` | `test_sidecar_with_own_geokeys_gpu_matches_cpu_2324` |

## #2161 -- HTTP / fsspec URL on the eager GPU path (`test_read_geotiff_gpu_url_eager_2161.py`)

| Old | New |
| --- | --- |
| `test_local_path_still_returns_cupy` | `test_local_path_still_returns_cupy_2161` |
| `test_http_url_returns_cupy_matching_cpu` | `test_http_url_returns_cupy_matching_cpu_2161` |
| `test_memory_fsspec_uri_returns_cupy_matching_cpu` | `test_memory_fsspec_uri_returns_cupy_matching_cpu_2161` |
| `test_unreachable_http_url_does_not_raise_filenotfound` | `test_unreachable_http_url_does_not_raise_filenotfound_2161` |
| `test_chunked_url_path_still_uses_chunked_helper` | `test_chunked_url_path_still_uses_chunked_helper_2161` |

## #1909 -- GDS chunked declared-dtype cast (`test_chunked_gpu_declared_dtype_1909.py`)

| Old | New |
| --- | --- |
| `test_chunked_gpu_declared_dtype_matches_computed` | `test_chunked_gpu_declared_dtype_matches_computed_1909` |
| `test_chunked_gpu_dtype_matches_cpu_dask` | `test_chunked_gpu_dtype_matches_cpu_dask_1909` |
| `test_chunked_gpu_no_nodata_keeps_source_dtype` | `test_chunked_gpu_no_nodata_keeps_source_dtype_1909` |
| `test_chunked_gpu_explicit_dtype_kwarg_threads_through` | `test_chunked_gpu_explicit_dtype_kwarg_threads_through_1909` |
| `test_chunked_gpu_sentinel_hit_still_promotes` | `test_chunked_gpu_sentinel_hit_still_promotes_1909` |
| `test_chunked_gpu_eager_paths_keep_source_dtype` | `test_chunked_gpu_eager_paths_keep_source_dtype_1909` |

## Cross-cutting touches

- `docs/source/reference/release_gate_geotiff.rst` -- GPU nodata row now cites
  `xrspatial/geotiff/tests/gpu/test_reader.py`.
- `docs/source/reference/geotiff.rst` -- prose pointer updated.
- `xrspatial/geotiff/tests/read/test_nodata.py` -- docstring pointer updated.
