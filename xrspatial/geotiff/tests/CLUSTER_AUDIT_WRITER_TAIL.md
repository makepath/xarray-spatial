# Cluster 9 audit: writer tail (long-tail epic #2424)

Maps every old `file::test` to its new `file::test_id`. Deleted on a
final commit before merge per the epic's hard gate.

## Sub-PR A: folded into `write/test_basic.py`

### `test_lowlevel_write_pushdown_2138.py` -> section "array-level write push-down + byte parity (#2138)"

- `TestCompressionNamePushdown::test_write_rejects_unknown_compression` -> same
- `TestCompressionNamePushdown::test_write_streaming_rejects_unknown_compression` -> same
- `TestJpegOptInPushdown::test_write_rejects_jpeg_without_opt_in` -> same
- `TestJpegOptInPushdown::test_write_accepts_jpeg_with_opt_in` -> same
- `TestJpegOptInPushdown::test_write_streaming_rejects_jpeg_without_opt_in` -> same
- `TestMaxZErrorPushdown::test_write_rejects_negative_max_z_error` -> same
- `TestMaxZErrorPushdown::test_write_rejects_max_z_error_on_non_lerc` -> same
- `TestMaxZErrorPushdown::test_write_streaming_rejects_negative_max_z_error` -> same
- `TestCrsEpsgBoolPushdown::test_write_rejects_bool_crs_epsg` -> same
- `TestCrsEpsgBoolPushdown::test_write_rejects_false_crs_epsg` -> same
- `TestCrsEpsgBoolPushdown::test_write_streaming_rejects_bool_crs_epsg` -> same
- `TestNanToSentinelDefensiveCopy::test_write_does_not_mutate_caller_buffer` -> same
- `TestNanToSentinelDefensiveCopy::test_write_writes_sentinel_in_file` -> same
- `TestDtypePromotionPushdown::test_write_promotes_float16` -> same
- `TestDtypePromotionPushdown::test_write_promotes_bool` -> same
- `test_write_vs_to_geotiff_byte_parity_uint8` -> same
- `test_write_streaming_vs_to_geotiff_byte_parity_uint8` -> same
- `test_write_lerc_lossless_round_trip` -> same
- `test_aliases_match_underscore_names` -> same
- `test_write_not_leaked_into_public_namespace` -> same

Module-level helpers renamed to avoid collision with the host file:
`_codec_available` -> `_codec_available_2138`, `_make_uint8_band` ->
`_make_uint8_band_2138`, `_make_float32_band` -> `_make_float32_band_2138`,
`_bytes` -> `_file_bytes_2138`, `_PARITY_CODECS` -> `_PARITY_CODECS_2138`.

### `test_to_geotiff_3d_dim_validation_1812.py` -> section "3D dim validation (#1812)"

- `test_repro_silent_corruption_now_raises` -> same
- `test_eager_rejects_ambiguous_3d` -> same
- `test_dask_streaming_rejects_ambiguous_3d` -> same
- `test_gpu_writer_rejects_ambiguous_3d` -> same
- `test_happy_3d_round_trip` -> same
- `test_2d_still_works` -> same
- `test_error_message_actionable` -> same
- `test_gpu_writer_happy_path_still_works` -> same

Helpers renamed: `_make_da` -> `_make_da_1812`, `_HAPPY_3D_INPUTS` ->
`_HAPPY_3D_INPUTS_1812`. The per-file `_gpu_available` probe was dropped
in favour of the host file's `_HAS_GPU` / `_gpu_only`.

### `test_temporal_3d_writer_rejection_1972.py` -> section "temporal-trailing 3D writer rejection (#1972)"

- `test_validate_3d_rejects_yx_temporal` -> same
- `test_validate_3d_rejects_yx_temporal_case_insensitive` -> same
- `test_validate_3d_rejects_yx_aliases_with_temporal` -> same
- `test_validate_3d_still_accepts_yx_band` -> same
- `test_validate_3d_still_accepts_recognized_band_alias_trailing_dim` -> same
- `test_validate_3d_still_rejects_time_y_x` -> same
- `test_validate_3d_rejects_temporal_y_x_case_insensitive` -> same
- `test_validate_3d_rejects_temporal_yx_alias_leading` -> same
- `test_validate_3d_still_rejects_other_ambiguous_leading` -> same
- `test_to_geotiff_rejects_yxtime_stack` -> same
- `test_error_message_suggests_isel_and_band_rename` -> same

### `test_to_geotiff_empty_shape_2075.py` -> section "empty spatial dim rejection (#2075)"

- `test_to_geotiff_rejects_empty_numpy` -> same
- `test_write_geotiff_gpu_rejects_empty` -> same
- `test_to_geotiff_rejects_empty_dask` -> same

Helpers renamed: `_EMPTY_SHAPES` -> `_EMPTY_SHAPES_2075`; the per-file
`_cupy_available` / `_HAS_GPU` probe was replaced by the
`@requires_gpu` marker from `_helpers/markers.py`.

### `test_to_geotiff_zero_bands_2095.py` -> section "zero-band axis rejection (#2095)"

- `test_to_geotiff_rejects_zero_bands_numpy` -> same
- `test_to_geotiff_rejects_zero_bands_dask` -> same
- `test_write_band_last_zero_bands_direct` -> same
- `test_write_streaming_zero_bands_direct` -> same
- `test_write_geotiff_gpu_rejects_zero_bands` -> same

Helpers renamed: `_ZERO_BAND_LAYOUTS` -> `_ZERO_BAND_LAYOUTS_2095`; the
per-file `_cupy_available` / `_HAS_GPU` probe was replaced by the
`@requires_gpu` marker.

## Sub-PR B: new `write/test_streaming.py`

### `test_parallel_writer_1800.py` -> section "parallel strip / tile writer (#1800)"

- `test_strip_writer_round_trip_large` -> same
- `test_strip_writer_dtypes` -> same
- `test_strip_writer_small_takes_sequential_path` -> same
- `test_strip_writer_thread_pool_used_when_large` -> same
- `test_strip_writer_uncompressed_stays_sequential` -> same
- `test_tile_writer_large_tile_size_parallelizes` -> same
- `test_tile_writer_small_payload_stays_sequential` -> same
- `test_deflate_compress_zlib_wire_compatible` -> same
- `test_deflate_compress_fallback_when_libdeflate_missing` -> same
- `test_deflate_compress_uses_libdeflate_when_available` -> same
- `test_write_strip_deflate_round_trip_multi_strip` -> same
- `test_write_tiled_deflate_large_tile_round_trip` -> same

Helper renamed: `_make_data` -> `_make_data_1800`.

### `test_streaming_write.py` -> section "streaming write round-trip (#1084 / #1485)"

- `TestStreamingRoundTrip::*` -> same
- `TestStreamingGeoMetadata::*` -> same
- `TestStreamingEdgeCases::*` -> same
- `TestStreamingMultiband::*` -> same
- `TestStreamingBigTiffAndErrors::*` -> same
- `TestCogFallback::*` -> same
- `TestStreamingBufferBudget::*` -> same

Fixtures `sample_raster` / `dask_raster` kept.

### `test_streaming_write_parallel.py` -> section "parallel per-tile streaming compress (P4)"

- `test_streaming_write_round_trip_unchanged` -> same
- `test_streaming_write_parallelism_observed` -> same
- `test_streaming_write_perf_sanity` -> same

Helper renamed: `_make_dataarray` -> `_make_dataarray_parallel`.

### `test_streaming_write_pool_leak_2276.py` -> section "tile-pool shutdown on mid-stream failure (#2276)"

- `test_pool_shutdown_on_compress_failure` -> same
- `test_pool_shutdown_on_file_write_failure` -> same
- `test_pool_shutdown_on_happy_path` -> same

Helper renamed: `_make_dataarray` -> `_make_dataarray_2276`; fixture
`captured_pools` and `_list_writer_pool_worker_threads` kept.
