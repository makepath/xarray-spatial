# Cluster 7, Sub-PR B audit: parity tail -> parity/ siblings

Maps every old `file::test` to its new `file::test_id`. The eight source
files total ~2578 lines; `parity/test_backend_matrix.py` is already 2090
lines, so appending them all would push it past 4600 lines. Per the issue's
2500-line guidance, the tests land in three focused siblings next to
`test_backend_matrix.py` instead, grouped by concern:

- `parity/test_finalization.py` -- the three #2162 read-finalization
  parity files (dispatch validation, eager, lazy).
- `parity/test_signature_contract.py` -- signature / docstring / release-
  contract parity (#1631, #2274, #2389). The issue notes #1631 and #2274
  are signature-flavoured; they live here with the contract gate so all
  three contract-style parity files share a home.
- `parity/test_reference.py` -- degenerate-shape backend parity and the
  rasterio / Zarr external-reference round trip.

Tests are copied verbatim except for moving GPU gating to the shared
`requires_gpu` marker from `_helpers/markers.py` (replacing per-file
`_gpu_available` / `_gpu_only`) and adjusting two imports for the new
`parity/` location (see Notes). No assertion changed. Total: 137 tests,
matching the pre-consolidation total of 137.

## parity/test_finalization.py

### test_dispatch_validation_parity_2162.py -> Section 1
- helpers `_build_local_tif`, `_build_vrt`, `_get_error` -> same
- test_open_geotiff_overview_level_bool (parametrised) -> same id
- test_open_geotiff_overview_level_str -> same id
- test_open_geotiff_overview_level_float -> same id
- test_dask_overview_level_bool (parametrised) -> same id
- test_dask_overview_level_str -> same id
- test_dask_overview_level_float -> same id
- test_gpu_overview_level_bool (parametrised) -> same id
- test_gpu_overview_level_str -> same id
- test_gpu_overview_level_float -> same id
- test_vrt_overview_level_bool (parametrised) -> same id
- test_vrt_overview_level_str -> same id
- test_vrt_overview_level_float -> same id
- test_open_geotiff_dask_rejects_max_cloud_bytes -> same id
- test_open_geotiff_gpu_rejects_max_cloud_bytes -> same id
- test_open_geotiff_vrt_rejects_max_cloud_bytes -> same id
- test_dask_rejects_max_cloud_bytes -> same id
- test_gpu_rejects_max_cloud_bytes -> same id
- test_vrt_rejects_max_cloud_bytes -> same id
- test_explicit_none_max_cloud_bytes_rejected_on_dask_direct -> same id
- test_explicit_none_max_cloud_bytes_rejected_on_gpu_direct -> same id
- test_explicit_none_max_cloud_bytes_rejected_on_vrt_direct -> same id
- test_open_geotiff_rejects_missing_sources_on_tif -> same id
- test_dask_rejects_missing_sources_on_tif -> same id
- test_gpu_rejects_missing_sources_on_tif -> same id
- test_open_geotiff_rejects_band_nodata_on_tif -> same id
- test_dask_rejects_band_nodata_on_tif -> same id
- test_gpu_rejects_band_nodata_on_tif -> same id
- test_open_geotiff_rejects_on_gpu_failure_when_gpu_false -> same id
- test_dask_rejects_on_gpu_failure -> same id
- test_vrt_rejects_on_gpu_failure -> same id
- test_open_geotiff_rejects_file_like_with_chunks -> same id
- test_open_geotiff_rejects_file_like_with_gpu -> same id
- test_dask_rejects_file_like -> same id
- test_gpu_rejects_file_like -> same id
- test_open_geotiff_accepts_path_object -> same id
- test_dask_accepts_path_object -> same id
- test_vrt_accepts_path_object -> same id
- test_gpu_accepts_path_object -> same id (`@requires_gpu`)
- test_gpu_path_object_does_not_raise_file_like_error -> same id
- test_open_geotiff_defaults_round_trip -> same id
- test_dask_defaults_round_trip -> same id
- test_vrt_defaults_round_trip -> same id
- test_max_cloud_bytes_message_parity -> same id
- test_band_nodata_message_parity -> same id
- test_missing_sources_message_parity -> same id
- test_on_gpu_failure_message_parity -> same id
- test_overview_level_message_parity -> same id

### test_eager_finalization_parity_2162.py -> Section 2
- helpers `_write_with_nodata`, `_read_both`,
  `_assert_lifecycle_attrs_match`, constant `_LIFECYCLE_ATTRS` -> same
- test_float_sentinel_match_and_mask -> same id (`@requires_gpu`)
- test_int_in_range_sentinel_promotes_to_float -> same id (`@requires_gpu`)
- test_int_out_of_range_sentinel_is_no_op -> same id (`@requires_gpu`)
- test_mask_nodata_false_keeps_literal_sentinel -> same id (`@requires_gpu`)
- test_no_declared_sentinel_omits_nodata_attrs -> same id (`@requires_gpu`)
- test_dtype_kwarg_records_post_mask_cast -> same id (`@requires_gpu`)
- test_windowed_read_presence_matches_window_contents -> same id (`@requires_gpu`)
- test_miniswhite_post_inversion_sentinel_parity -> same id (`@requires_gpu`)
- test_multiband_stripped_parity -> same id (`@requires_gpu`)
  (module-level `xr` / `to_geotiff` imports reused; in-body imports dropped)

### test_lazy_finalization_parity_2162.py -> Section 3
- helpers `_open_cpu_dask`, `_open_gpu_dask`, fixture builders
  (`_make_full_tiff`, `_make_transform_only_tiff`, `_make_crs_only_tiff`,
  `_make_none_tiff`, `_make_rotated_tiff`, `_make_float_with_nodata_tiff`,
  `_make_int_with_nodata_tiff`), constants `_BACKENDS`, `_GEOREF_FIXTURES`
  -> same (the `_gpu_only` marks on `_BACKENDS` become `requires_gpu`)
- test_georef_status_parity (parametrised) -> same id
- test_attrs_dict_parity (parametrised) -> same id
- test_nodata_pixels_present_absent_on_lazy (parametrised) -> same id
- test_nodata_pixels_present_cross_backend -> same id
- test_dtype_cast_absent_without_caller_dtype (parametrised) -> same id
- test_dtype_cast_records_target (parametrised) -> same id
- test_dtype_cast_parity_cross_backend -> same id
- test_dtype_cast_absent_parity_cross_backend -> same id
- test_dtype_cast_records_integer_target (parametrised) -> same id
- The inline `if _HAS_GPU:` / `if not _HAS_GPU` runtime branches now call
  a local `_gpu_dask_available()` that wraps `_helpers.markers.gpu_available`,
  preserving the original conditional-skip semantics.

## parity/test_signature_contract.py

### test_signature_parity_1631.py -> Section 1
- test_write_vrt_signature_exposes_documented_kwargs -> same id
- test_write_vrt_unknown_kwarg_rejected_at_public_level -> same id
- test_write_vrt_accepts_documented_kwargs -> same id
- test_write_geotiff_gpu_docstring_lists_cubic -> same id
- test_write_geotiff_gpu_data_has_type_hint -> same id
- test_write_geotiff_gpu_cubic_overview_round_trip -> same id (`@requires_gpu`)

### test_read_entry_points_doc_param_parity_2274.py -> Section 2
- constant `READ_ENTRY_POINTS`, helpers `_signature_params`,
  `_documented_params`, regex `_PARAM_HEADING` -> same
- test_read_entry_point_kwargs_have_docstring_entries (parametrised) -> same id
- test_read_entry_point_docstring_does_not_invent_params (parametrised) -> same id
- test_allow_rotated_documented (parametrised) -> same id
- test_allow_unparseable_crs_documented (parametrised) -> same id

### test_release_contract_parity_2389.py -> Section 3
- helper `_contract_rows`, regex `_ROW_RE`, constants `_CONTRACT` -> same
- test_contract_table_parses_into_rows -> same id
- test_contract_keys_are_real_supported_features -> same id
- test_contract_tiers_match_supported_features -> same id
- `_REPO_ROOT` changes from `_HERE.parents[3]` to `_HERE.parents[4]`
  because the file moved one directory deeper (tests/ -> tests/parity/).
  Matches the depth `release_gates/test_stable_features.py` uses.

## parity/test_reference.py

### test_degenerate_shapes_backends_2026_05_11.py -> Section 1
- TestSinglePixelRead::test_* (5 tests, GPU ones `@requires_gpu`) -> same ids
- TestSingleRowRead::test_* (3 tests) -> same ids
- TestSingleColumnRead::test_* (3 tests) -> same ids
- TestGpuWriterDegenerateShapes::test_* (3 tests, class `@requires_gpu`) -> same ids
- TestAllNanRead::test_* (3 tests) -> same ids
- TestInfRead::test_* (3 tests) -> same ids
- TestNanSentinelDaskRead::test_* (3 tests) -> same ids

### test_round_trip_parity_rasterio_zarr_1961.py -> Section 2
- helpers `_as_xrspatial_layout`, `_apply_nodata_to_float`, `_read_via_zarr`,
  `_assert_pixels_equal`, `_assert_transforms_match`, `_assert_crs_match`,
  `_build_rasterio_coords`, `_parity_check_single_band`, constants
  `ATOL_FLOAT32`, `ATOL_COORD`, `RTOL_TRANSFORM` -> same
- TestSingleBandFloat32NodataSentinel::test_round_trip -> same id
- TestMultibandUint16SharedNodata::test_round_trip -> same id
- TestNorthUpVsSouthUp::test_round_trip (parametrised south_up) -> same id
- TestStripeShapes::test_round_trip (parametrised shape) -> same id
- TestTiledCogNoOverviews::test_round_trip -> same id
- TestNoGeorefIntegerCoords::test_round_trip -> same id

## Notes

- The two signature-flavoured files (#1631, #2274) could have landed in
  Sub-PR A per the issue note. They stay in Sub-PR B grouped with the
  release-contract gate so all three contract-style parity files share a
  home; no test is duplicated across the two sub-PRs.
- `parity/test_finalization.py` imports `_write_rotated_tiff` from
  `..read.test_crs` (the same absolute helper the original lazy file used)
  and re-uses module-level `xr` / `to_geotiff` imports.
- HARD GATE per epic #2424: this audit file is deleted in a final
  pre-merge commit on this branch.
