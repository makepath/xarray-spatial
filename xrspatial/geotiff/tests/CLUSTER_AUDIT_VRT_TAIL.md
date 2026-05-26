# Cluster 13 audit: VRT tail (#2437)

Maps every old `file::test` from the 12 consolidated VRT-tail files to
its new `file::test_id`. Deleted before merge (epic #2424 hard gate).

12 files deleted, 0 added (all fold into existing `vrt/*.py` modules
plus the new `vrt/test_parity.py`, which itself only re-homes three
already-existing files rather than adding a net-new module).

## -> vrt/test_missing_sources.py

### test_vrt_missing_sources_default_raise_1843.py
(internal `_vrt.read_vrt` entry point + STRICT env override preserved)

- `test_read_vrt_default_raises_on_unreadable_source`
  -> `TestInternalEntryPointMissingSources::test_internal_default_raises_on_unreadable_source`
- `test_read_vrt_explicit_warn_preserves_lenient_behaviour`
  -> `TestInternalEntryPointMissingSources::test_internal_explicit_warn_preserves_lenient_behaviour`
- `test_read_vrt_strict_env_still_raises_under_warn`
  -> `TestInternalEntryPointMissingSources::test_internal_strict_env_still_raises_under_warn`

### test_read_vrt_default_missing_sources_1860.py

- `test_public_read_vrt_default_raises_on_unreadable_source`
  -> `TestPublicDefaultMissingSources::test_public_read_vrt_default_raises`
- `test_open_geotiff_vrt_default_raises_on_unreadable_source`
  -> `TestPublicDefaultMissingSources::test_open_geotiff_vrt_default_raises`
- `test_public_read_vrt_explicit_warn_preserves_lenient_behaviour`
  -> `TestPublicDefaultMissingSources::test_public_read_vrt_explicit_warn_preserves_lenient_behaviour`
- `test_open_geotiff_vrt_explicit_warn_preserves_lenient_behaviour`
  -> `TestPublicDefaultMissingSources::test_open_geotiff_vrt_explicit_warn_preserves_lenient_behaviour`

### test_vrt_chunked_missing_sources_1799.py

- `TestChunkedMissingSourcesWarn::test_vrt_holes_populated_at_build`
  -> `TestChunkedMissingSourcesWarn::test_vrt_holes_populated_at_build`
- `TestChunkedMissingSourcesWarn::test_compute_emits_per_task_warning`
  -> `TestChunkedMissingSourcesWarn::test_compute_emits_per_task_warning`
- `TestChunkedMissingSourcesWarn::test_chunks_tuple_form`
  -> `TestChunkedMissingSourcesWarn::test_chunks_tuple_form`
- `TestChunkedMissingSourcesRaise::test_build_raises_immediately`
  -> `TestChunkedMissingSourcesRaiseSmoke::test_build_raises_immediately`
- `TestChunkedMissingSourcesRaise::test_build_raise_message_mentions_policy_kwarg`
  -> `TestChunkedMissingSourcesRaiseSmoke::test_build_raise_message_mentions_policy_kwarg`
- `TestChunkedMissingSourcesRaise::test_window_past_missing_succeeds_under_raise`
  -> `TestChunkedMissingSourcesRaiseSmoke::test_window_past_missing_succeeds_under_raise`
- `TestChunkedMissingSourcesRaise::test_band_selection_skips_other_bands_holes`
  -> `TestChunkedMissingSourcesRaiseSmoke::test_band_selection_single_band_still_raises`
- `TestChunkedMissingSourcesDefault::test_chunked_default_raises_at_build`
  -> `TestChunkedMissingSourcesDefault::test_chunked_default_raises_at_build`
- `TestChunkedMissingSourcesValidation::test_invalid_policy_raises_at_build`
  -> `TestChunkedMissingSourcesValidation::test_invalid_policy_raises_at_build`
- `TestChunkedMissingSourcesValidation::test_invalid_policy_raises_without_chunks_too`
  -> `TestChunkedMissingSourcesValidation::test_invalid_policy_raises_without_chunks_too`

### test_vrt_chunked_missing_raise_at_build_2265.py

- `TestRaiseAtBuild::test_build_raises_immediately`
  -> `TestRaiseAtBuild::test_build_raises_immediately`
- `TestRaiseAtBuild::test_default_raises_at_build`
  -> `TestRaiseAtBuild::test_default_raises_at_build`
- `TestRaiseAtBuild::test_error_message_mentions_opt_in`
  -> `TestRaiseAtBuild::test_error_message_mentions_opt_in`
- `TestWindowScoping::test_window_past_missing_does_not_raise`
  -> `TestRaiseAtBuildWindowScoping::test_window_past_missing_does_not_raise`
- `TestWindowScoping::test_window_intersecting_missing_raises`
  -> `TestRaiseAtBuildWindowScoping::test_window_intersecting_missing_raises`
- `TestBandScoping::test_band_select_skips_other_bands_missing_source`
  -> `TestRaiseAtBuildBandScoping::test_band_select_skips_other_bands_missing_source`
- `TestBandScoping::test_band_select_on_missing_band_raises`
  -> `TestRaiseAtBuildBandScoping::test_band_select_on_missing_band_raises`
- `TestBandScoping::test_no_band_restriction_raises`
  -> `TestRaiseAtBuildBandScoping::test_no_band_restriction_raises`
- `TestWarnPreserved::test_warn_records_holes_at_build`
  -> `TestRaiseAtBuildWarnPreserved::test_warn_records_holes_at_build`
- `TestWarnPreserved::test_warn_compute_emits_per_task_warning`
  -> `TestRaiseAtBuildWarnPreserved::test_warn_compute_emits_per_task_warning`
- `TestMultipleMissingSources::test_two_missing_sources_listed_with_count`
  -> `TestRaiseAtBuildMultipleMissingSources::test_two_missing_sources_listed_with_count`
- `TestMultipleMissingSources::test_many_missing_sources_truncated_with_more_suffix`
  -> `TestRaiseAtBuildMultipleMissingSources::test_many_missing_sources_truncated_with_more_suffix`
- `TestStrictMode::test_strict_overrides_warn_kwarg`
  -> `TestRaiseAtBuildStrictMode::test_strict_overrides_warn_kwarg`
- `TestStrictMode::test_strict_off_warn_still_warns`
  -> `TestRaiseAtBuildStrictMode::test_strict_off_warn_still_warns`

## -> vrt/test_validation.py

### test_geotiff_vrt_srcrect_validation_1784.py

- `test_negative_srcrect_x_size_rejected` -> `TestSrcRectRejection::test_negative_x_size_rejected`
- `test_negative_srcrect_y_size_rejected` -> `TestSrcRectRejection::test_negative_y_size_rejected`
- `test_negative_srcrect_x_off_rejected` -> `TestSrcRectRejection::test_negative_x_off_rejected`
- `test_negative_srcrect_y_off_rejected` -> `TestSrcRectRejection::test_negative_y_off_rejected`
- `test_negative_srcrect_message_names_bad_values` -> `TestSrcRectRejection::test_message_names_bad_values`
- `test_missing_source_still_takes_lenient_warning_path` -> `TestSrcRectRejection::test_missing_source_still_takes_lenient_warning_path`
- `test_valid_srcrect_reads_normally` -> `TestSrcRectRejection::test_valid_srcrect_reads_normally`
- `test_negative_srcrect_raises_under_strict_mode` -> `TestSrcRectRejection::test_negative_srcrect_raises_under_strict_mode`

### test_open_geotiff_vrt_kwarg_drop_1685.py

- `test_open_geotiff_vrt_rejects_overview_level` -> `TestOpenGeotiffVrtKwargRejection::test_rejects_overview_level`
- `test_open_geotiff_vrt_accepts_overview_level_zero` -> `TestOpenGeotiffVrtKwargRejection::test_accepts_overview_level_zero`
- `test_open_geotiff_vrt_rejects_on_gpu_failure_with_gpu_true` -> `TestOpenGeotiffVrtKwargRejection::test_rejects_on_gpu_failure_with_gpu_true`
- `test_open_geotiff_vrt_without_unsupported_kwargs_still_works` -> `TestOpenGeotiffVrtKwargRejection::test_without_unsupported_kwargs_still_works`
- `test_open_geotiff_vrt_with_window_still_works` -> `TestOpenGeotiffVrtKwargRejection::test_with_window_still_works`
- `test_open_geotiff_non_vrt_still_accepts_overview_level` -> `TestOpenGeotiffVrtKwargRejection::test_non_vrt_still_accepts_overview_level`

### test_to_geotiff_vrt_tiled_validation_1862.py

- `test_vrt_rejects_tiled_false_1862` -> `TestVrtTiledValidation::test_rejects_tiled_false`
- `test_vrt_tiled_false_zero_tile_size_raises_value_error_1862` -> `TestVrtTiledValidation::test_tiled_false_zero_tile_size_raises_value_error`
- `test_vrt_zero_tile_size_default_tiled_raises_value_error_1862` -> `TestVrtTiledValidation::test_zero_tile_size_default_tiled_raises_value_error`
- `test_vrt_default_args_still_succeeds_1862` -> `TestVrtTiledValidation::test_default_args_still_succeeds`

## -> vrt/test_window.py

### test_read_vrt_lazy_chunks_1798.py

- `test_read_vrt_chunks_matches_eager_values` -> `TestVrtTailLazyChunks::test_chunks_matches_eager_values`
- `test_read_vrt_chunks_does_not_read_sources_during_construction` -> `TestVrtTailLazyChunks::test_chunks_does_not_read_sources_during_construction`
- `test_read_vrt_chunks_rejects_excessive_task_count` -> `TestVrtTailLazyChunks::test_chunks_rejects_excessive_task_count`

### test_read_geotiff_dask_vrt_kwargs_1795.py

- `test_direct_read_geotiff_dask_vrt_forwards_window_and_band` -> `TestVrtTailDirectDaskKwargs::test_forwards_window_and_band`
- `test_direct_read_geotiff_dask_vrt_forwards_max_pixels` -> `TestVrtTailDirectDaskKwargs::test_forwards_max_pixels`

## -> vrt/test_parity.py (new file; re-homes three existing files)

### test_vrt_backend_parity_2321.py
All tests preserved 1:1 (names unchanged):
`test_vrt_backend_parity`, `test_sidecar_vrt_attrs_match_inline`,
`test_windowed_vrt_shifts_coords_and_transform_consistently`,
`test_sidecar_window_shifts_to_known_coords`,
`test_assert_metadata_parity_flags_transform_drift`.
The `_GOLDEN` fixture path gained one `.parent` because the file moved
from `tests/` into `tests/vrt/`.

### test_vrt_finalization_parity_2162.py
All tests preserved 1:1 (names unchanged): the five
`test_vrt_eager_*_matches_open_geotiff`, the five
`test_vrt_chunked_*`, `test_band_nodata_first_band_attrs`,
`test_band_nodata_chunked_first_band_attrs`,
`test_dtype_cast_no_sentinel_omits_attr_{eager,chunked}`,
`test_missing_sources_{eager,chunked}_surfaces_vrt_holes`,
`test_georef_status_{eager,chunked}_parity`,
`test_vrt_eager_chunked_internal_parity`.

### test_vrt_backend_coverage_2026_05_11.py
All tests preserved 1:1 (names unchanged):
`TestReadVrtGpuBackend` (4 GPU-gated),
`TestReadVrtDtypeKwarg` (2), `TestReadVrtNameKwarg` (2),
`TestOpenGeotiffFileLikeKwargRejection` (3).

## Docs

`docs/source/reference/release_gate_geotiff.rst` rows that cited the
deleted files now cite the consolidated homes
(`vrt/test_parity.py`, `vrt/test_missing_sources.py`,
`vrt/test_validation.py`, `vrt/test_window.py`).
