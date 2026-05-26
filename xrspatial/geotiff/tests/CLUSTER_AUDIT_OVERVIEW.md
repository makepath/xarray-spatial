# Cluster 8 audit: overview tests (issue #2432)

Maps every old `file::test` to its new `file::test` location. Deleted on
a final pre-merge commit per epic #2424.

Write-side targets land in `write/test_overview.py` (extended).
Read-side targets land in `read/test_overview.py` (new).

GPU overview files (`test_gpu_writer_overview_inplace_1948.py`,
`test_gpu_writer_overview_mode_and_compression_level_1740.py`) are out of
scope -- they belong to the GPU cluster (#2438).

## Write-side -> write/test_overview.py

### test_overview_block_order_2308.py
- `test_cog_overview_block_order_invariant_2308[None]` -> `test_cog_overview_block_order_invariant_2308[None]`
- `test_cog_overview_block_order_invariant_2308[3]` -> `test_cog_overview_block_order_invariant_2308[3]`
- `test_cog_overview_block_order_three_levels_2308` -> same
- `test_cog_overview_block_order_rio_cogeo_2308[None]` -> same
- `test_cog_overview_block_order_rio_cogeo_2308[3]` -> same
- helpers `_min_block_offset`, `_read_block_order`, `_make_da`, `_rio_cogeo_or_skip` -> module-level, renamed to avoid clashes

### test_overview_levels_decimation_factors_1766.py
- All `test_overview_levels_*` and `test_to_geotiff_*`, `test_validate_*`,
  `test_explicit_factors_match_auto_pyramid_bytewise`,
  `test_overview_pyramid_mean_values_are_correct` -> same names
- helper `_ifd_dimensions`, constant `STEM` -> module-level

### test_overview_nodata_inheritance_1739.py
- `test_overview_inherits_nodata_attr[*]` -> same
- `test_overview_sentinel_pixels_masked_to_nan[*]` -> same
- `test_overview_nanmean_matches_pre_sentinel_value[*]` -> same
- `test_overview_inherits_gdal_metadata[*]` -> same
- `test_overview_inherits_resolution_tags[*]` -> same
- `test_attrs_keysets_consistent_across_overview_levels` -> same
- `test_overview_with_own_nodata_keeps_own_value` -> same
- helpers `_arr_with_partial_nan`, `_arr_with_full_nan_block` already exist
  in the target file (reused); `_make_cog_with_nodata`, `_materialise`,
  `_BACKENDS` -> module-level (reuse existing `_materialise` and
  `_BACKENDS` where the target already defines them)

### test_overview_pixel_is_point_1642.py
- `test_point_overview_first_pixel_center_at_block_centroid[*]` -> same
- `test_point_overview_transform_origin_shifted[*]` -> same
- `test_point_overview_coords_are_uniform` -> same
- `test_helper_pixel_is_point_origin_shift_unit` -> same
- `test_helper_pixel_is_area_no_origin_shift_unit` -> same
- `test_helper_point_overview_with_own_geokeys_not_shifted` -> same
- `test_area_overview_origin_unchanged_regression` -> same
- helpers `_make_pp_cog`, `_make_pa_cog`, `_StubIFD` -> module-level

### test_overview_resampling_min_max_median_2026_05_11.py
- `test_block_reduce_2d_cpu[*]` -> same
- `test_block_reduce_2d_cpu_skips_nan[*]` -> same
- `test_to_geotiff_cog_overview_resampling_cpu[*]` -> same
- `test_to_geotiff_cog_overview_resampling_cpu_nodata[*]` -> same
- `test_block_reduce_2d_gpu[*]` -> same
- `test_block_reduce_2d_gpu_matches_cpu_with_nan[*]` -> same
- `test_write_geotiff_gpu_cog_overview_resampling[*]` -> same
- `test_to_geotiff_gpu_cog_overview_matches_cpu[*]` -> same
- `test_block_reduce_2d_cpu_unknown_method_raises` -> same
- `test_block_reduce_2d_gpu_unknown_method_raises` -> same
- helpers `_arr_4x4_ramp`, `_arr_4x4_with_nan`, expected-array constants
  -> module-level

### test_mode_overview_perf.py
- `test_bit_exact_match_reference[*]` -> same
- `test_tie_break_lowest_wins` -> same
- `test_tie_break_three_way` -> same
- `test_three_of_a_kind_wins` -> same
- `test_all_same_value_block` -> same
- `test_multiple_blocks_independent` -> same
- `test_perf_under_100ms_on_1024sq_uint8` -> same
- helper `_mode_resample_reference` -> module-level

## Read-side -> read/test_overview.py (new)

### test_overview_filter.py
- `TestSelectOverviewIFD::*` -> same class + methods
- `TestOpenGeotiffSkipsMask::*` -> same class + methods
- helpers `_write_tiff_with_mask`, `_write_normal_cog` -> module-level

### test_overview_geo_inheritance_1640.py
- `test_overview_inherits_crs_across_backends[*]` -> same
- `test_overview_transform_scales_by_reduction_factor[*]` -> same
- `test_overview_coords_cover_same_extent[*]` -> same
- `test_overview_with_own_geokeys_is_not_overwritten` -> same
- `test_overview_without_full_res_sibling_falls_back_gracefully` -> same
- `test_overview_level_0_path_unchanged` -> same
- helpers `_make_cog_with_overviews`, `_materialise` -> module-level

### test_overview_level_type_validation_2074.py
- `test_overview_level_bool_raises_typeerror[*]` -> same
- `test_overview_level_str_raises_typeerror` -> same
- `test_overview_level_float_raises_typeerror` -> same
- `test_overview_level_zero_succeeds` -> same
- `test_overview_level_one_succeeds` -> same
- `test_overview_level_none_succeeds` -> same
- `test_overview_level_numpy_int_zero_succeeds[*]` -> same
- `test_overview_level_numpy_int_one_succeeds[*]` -> same
- `test_overview_level_typeerror_names_value` -> same
- helper `_write_cog_with_one_overview`, fixture `cog_with_overview` ->
  renamed to `_write_cog_one_overview_2074` / `cog_with_overview_2074` to
  avoid clash with the 2160 fixture in the same module

### test_overview_level_validation_backends_2160.py
- all `test_dask_overview_level_*` -> same
- all `test_gpu_overview_level_*` -> same
- helper `_write_cog_with_one_overview`, fixture `cog_with_overview` ->
  renamed to `_write_cog_one_overview_2160` / `cog_with_overview_2160`

## Notes on the write-side/read-side split

The issue assigns `test_overview_filter.py`,
`test_overview_nodata_inheritance_1739.py`, and
`test_overview_pixel_is_point_1642.py` to write-side even though their
assertions exercise the read path (`open_geotiff(overview_level=...)`).
The split follows the issue's explicit file lists rather than
re-classifying by call surface; the existing `write/test_overview.py`
already mixes writer-helper and end-to-end read assertions, so the
placement is consistent with the file's current scope.
