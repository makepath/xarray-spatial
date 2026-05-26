# Cluster 6 audit: input validation (#2430 / epic #2424)

Maps every `old_file::test` to its new home in
`xrspatial/geotiff/tests/unit/test_input_validation.py`. This file is
deleted on a final pre-merge commit (epic #2424 hard gate).

Nine source files fold into one, organised by validation axis. The
consolidated file collects 145 tests; the nine originals collected 146.
The single difference is one intentional dedup: the bool file's
`test_read_to_array_band_one_still_works` (`read_to_array(path, band=1)`
asserting `arr[:, :, 1]`) is identical to the type file's
`test_read_to_array_band_int_still_works`, so only one survives.

## Section 1: band type / bool rejection

### test_geotiff_band_bool_rejection_1786.py -> TestBandBoolRejection / TestBandTypeRejection

| old test | new id |
| --- | --- |
| `test_read_to_array_band_true_rejected` | `TestBandBoolRejection::test_read_to_array_band_true_rejected` |
| `test_read_to_array_band_false_rejected` | `TestBandBoolRejection::test_read_to_array_band_false_rejected` |
| `test_read_to_array_band_zero_still_works` | `TestBandTypeRejection::test_read_to_array_band_zero_still_works` |
| `test_read_to_array_band_one_still_works` | dedup -> `TestBandTypeRejection::test_read_to_array_band_int_still_works` |
| `test_open_geotiff_band_true_rejected` | `TestBandBoolRejection::test_open_geotiff_band_true_rejected` |
| `test_open_geotiff_band_false_rejected` | `TestBandBoolRejection::test_open_geotiff_band_false_rejected` |
| `test_read_geotiff_dask_band_true_rejected` | `TestBandBoolRejection::test_read_geotiff_dask_band_true_rejected` |
| `test_read_geotiff_dask_band_false_rejected` | `TestBandBoolRejection::test_read_geotiff_dask_band_false_rejected` |
| `test_read_geotiff_gpu_band_true_rejected` | `TestBandBoolRejection::test_read_geotiff_gpu_band_true_rejected` |
| `test_read_geotiff_gpu_band_false_rejected` | `TestBandBoolRejection::test_read_geotiff_gpu_band_false_rejected` |
| `test_read_vrt_band_true_still_rejected` | `TestBandBoolRejection::test_read_vrt_band_true_still_rejected` |
| `test_read_vrt_band_false_still_rejected` | `TestBandBoolRejection::test_read_vrt_band_false_still_rejected` |
| `test_read_to_array_band_np_bool_rejected` | `TestBandBoolRejection::test_read_to_array_band_np_bool_rejected` |
| `test_open_geotiff_band_np_bool_rejected` | `TestBandBoolRejection::test_open_geotiff_band_np_bool_rejected` |
| `test_read_geotiff_dask_band_np_bool_rejected` | `TestBandBoolRejection::test_read_geotiff_dask_band_np_bool_rejected` |
| `test_read_geotiff_gpu_band_np_bool_rejected` | `TestBandBoolRejection::test_read_geotiff_gpu_band_np_bool_rejected` |
| `test_read_vrt_band_np_bool_still_rejected` | `TestBandBoolRejection::test_read_vrt_band_np_bool_still_rejected` |

### test_geotiff_band_type_rejection_1910.py -> TestBandTypeRejection

| old test | new id |
| --- | --- |
| `test_read_to_array_band_float_rejected` | `TestBandTypeRejection::test_read_to_array_band_float_rejected` |
| `test_read_to_array_band_np_float_rejected` | `TestBandTypeRejection::test_read_to_array_band_np_float_rejected` |
| `test_read_to_array_band_str_rejected` | `TestBandTypeRejection::test_read_to_array_band_str_rejected` |
| `test_read_to_array_band_int_still_works` | `TestBandTypeRejection::test_read_to_array_band_int_still_works` |
| `test_read_to_array_band_np_integer_still_works` | `TestBandTypeRejection::test_read_to_array_band_np_integer_still_works` |
| `test_read_to_array_band_bool_still_rejected` | `TestBandTypeRejection::test_read_to_array_band_bool_still_rejected` |
| `test_open_geotiff_band_float_rejected` | `TestBandTypeRejection::test_open_geotiff_band_float_rejected` |
| `test_open_geotiff_band_str_rejected` | `TestBandTypeRejection::test_open_geotiff_band_str_rejected` |
| `test_read_geotiff_dask_band_float_rejected` | `TestBandTypeRejection::test_read_geotiff_dask_band_float_rejected` |
| `test_read_geotiff_dask_band_str_rejected` | `TestBandTypeRejection::test_read_geotiff_dask_band_str_rejected` |
| `test_read_geotiff_dask_band_int_still_works` | `TestBandTypeRejection::test_read_geotiff_dask_band_int_still_works` |
| `test_read_geotiff_gpu_band_float_rejected` | `TestBandTypeRejection::test_read_geotiff_gpu_band_float_rejected` |
| `test_read_geotiff_gpu_band_str_rejected` | `TestBandTypeRejection::test_read_geotiff_gpu_band_str_rejected` |

## Section 2: size-parameter validation

### test_size_param_validation_1752.py -> TestTileSizePositive / TestReadDaskChunksValidation

| old test | new id |
| --- | --- |
| `test_to_geotiff_tile_size_zero_raises` | `TestTileSizePositive::test_to_geotiff_tile_size_zero_raises` |
| `test_to_geotiff_tile_size_negative_raises` | `TestTileSizePositive::test_to_geotiff_tile_size_negative_raises` |
| `test_to_geotiff_tile_size_non_int_raises` | `TestTileSizePositive::test_to_geotiff_tile_size_non_int_raises` |
| `test_to_geotiff_tile_size_16_writes` | `TestTileSizePositive::test_to_geotiff_tile_size_16_writes` |
| `test_read_geotiff_dask_chunks_zero_raises` | `TestReadDaskChunksValidation::test_chunks_zero_raises` |
| `test_read_geotiff_dask_chunks_negative_raises` | `TestReadDaskChunksValidation::test_chunks_negative_raises` |
| `test_read_geotiff_dask_chunks_tuple_zero_row_raises` | `TestReadDaskChunksValidation::test_chunks_tuple_zero_row_raises` |
| `test_read_geotiff_dask_chunks_tuple_negative_col_raises` | `TestReadDaskChunksValidation::test_chunks_tuple_negative_col_raises` |
| `test_read_geotiff_dask_chunks_tuple_wrong_length_raises` | `TestReadDaskChunksValidation::test_chunks_tuple_wrong_length_raises` |
| `test_read_geotiff_dask_positive_int_chunks_works` | `TestReadDaskChunksValidation::test_positive_int_chunks_works` |
| `test_read_geotiff_dask_positive_tuple_chunks_works` | `TestReadDaskChunksValidation::test_positive_tuple_chunks_works` |
| `test_read_geotiff_dask_numpy_int_scalar_chunks_works` | `TestReadDaskChunksValidation::test_numpy_int_scalar_chunks_works` |
| `test_read_geotiff_dask_numpy_int_tuple_chunks_works` | `TestReadDaskChunksValidation::test_numpy_int_tuple_chunks_works` |

### test_tile_size_multiple_of_16_1767.py -> TestTileSizeMultipleOf16

| old test | new id |
| --- | --- |
| `test_tile_size_17_rejected_1767` | `TestTileSizeMultipleOf16::test_tile_size_17_rejected` |
| `test_tile_size_1_rejected_1767` | `TestTileSizeMultipleOf16::test_tile_size_1_rejected` |
| `test_tile_size_default_256_works_1767` | `TestTileSizeMultipleOf16::test_tile_size_default_256_works` |
| `test_tile_size_512_works_1767` | `TestTileSizeMultipleOf16::test_tile_size_512_works` |
| `test_tile_size_128_works_1767` | `TestTileSizeMultipleOf16::test_tile_size_128_works` |
| `test_tile_size_16_works_1767` | `TestTileSizeMultipleOf16::test_tile_size_16_works` |
| `test_tile_size_17_with_tiled_false_passes_1767` | `TestTileSizeMultipleOf16::test_tile_size_17_with_tiled_false_passes` |
| `test_tile_size_24_message_suggests_16_and_32_1767` | `TestTileSizeMultipleOf16::test_tile_size_24_message_suggests_16_and_32` |
| `test_tile_size_8_message_suggests_16_only_1767` | `TestTileSizeMultipleOf16::test_tile_size_8_message_suggests_16_only` |
| `test_write_geotiff_gpu_tile_size_17_rejected_1767` | `TestTileSizeMultipleOf16::test_write_geotiff_gpu_tile_size_17_rejected` |
| `test_write_geotiff_gpu_tile_size_zero_rejected_1767` | `TestTileSizeMultipleOf16::test_write_geotiff_gpu_tile_size_zero_rejected` |
| `test_write_geotiff_gpu_tile_size_float_rejected_1767` | `TestTileSizeMultipleOf16::test_write_geotiff_gpu_tile_size_float_rejected` |

## Section 3: source-dimension validation

### test_strip_zero_dims_2053.py -> TestCheckSourceDimensions / TestStrippedZeroDimsRejected / TestWindowedEmptyStillAllowed / TestTiledZeroDimsParityPinned / TestHTTPStrippedZeroDimsRejected

| old test | new id |
| --- | --- |
| `TestCheckSourceDimensions::test_zero_width_rejected` | `TestCheckSourceDimensions::test_zero_width_rejected` |
| `TestCheckSourceDimensions::test_zero_height_rejected` | `TestCheckSourceDimensions::test_zero_height_rejected` |
| `TestCheckSourceDimensions::test_zero_samples_rejected` | `TestCheckSourceDimensions::test_zero_samples_rejected` |
| `TestCheckSourceDimensions::test_negative_width_rejected` | `TestCheckSourceDimensions::test_negative_width_rejected` |
| `TestCheckSourceDimensions::test_negative_height_rejected` | `TestCheckSourceDimensions::test_negative_height_rejected` |
| `TestCheckSourceDimensions::test_negative_samples_rejected` | `TestCheckSourceDimensions::test_negative_samples_rejected` |
| `TestCheckSourceDimensions::test_all_positive_passes` | `TestCheckSourceDimensions::test_all_positive_passes` |
| `TestCheckSourceDimensions::test_error_message_contains_each_value` | `TestCheckSourceDimensions::test_error_message_contains_each_value` |
| `TestStrippedZeroDimsRejected::test_zero_image_width_rejected` | `TestStrippedZeroDimsRejected::test_zero_image_width_rejected` |
| `TestStrippedZeroDimsRejected::test_zero_image_length_rejected` | `TestStrippedZeroDimsRejected::test_zero_image_length_rejected` |
| `TestStrippedZeroDimsRejected::test_zero_samples_per_pixel_rejected` | `TestStrippedZeroDimsRejected::test_zero_samples_per_pixel_rejected` |
| `TestStrippedZeroDimsRejected::test_negative_width_via_signed_cast_rejected` | `TestStrippedZeroDimsRejected::test_negative_width_via_signed_cast_rejected` |
| `TestWindowedEmptyStillAllowed::test_windowed_outside_image_returns_empty_not_error` | `TestWindowedEmptyStillAllowed::test_windowed_outside_image_returns_empty_not_error` |
| `TestTiledParityPinned::test_tiled_zero_width_rejected` | `TestTiledZeroDimsParityPinned::test_tiled_zero_width_rejected` |
| `TestTiledParityPinned::test_tiled_zero_height_rejected` | `TestTiledZeroDimsParityPinned::test_tiled_zero_height_rejected` |
| `TestHTTPStrippedZeroDimsRejected::test_zero_image_width_over_http_rejected` | `TestHTTPStrippedZeroDimsRejected::test_zero_image_width_over_http_rejected` |
| `TestHTTPStrippedZeroDimsRejected::test_zero_image_length_over_http_rejected` | `TestHTTPStrippedZeroDimsRejected::test_zero_image_length_over_http_rejected` |

### test_pixel_array_count_cap_1901.py -> TestPixelArrayCountCap

| old test | new id |
| --- | --- |
| `test_tile_offsets_count_exceeds_geometry_rejected` | `TestPixelArrayCountCap::test_tile_offsets_count_exceeds_geometry_rejected` |
| `test_tile_offsets_count_matching_geometry_passes` | `TestPixelArrayCountCap::test_tile_offsets_count_matching_geometry_passes` |
| `test_strip_offsets_count_exceeds_geometry_rejected` | `TestPixelArrayCountCap::test_strip_offsets_count_exceeds_geometry_rejected` |
| `test_strip_byte_counts_planar_multiplies_by_samples` | `TestPixelArrayCountCap::test_strip_byte_counts_planar_multiplies_by_samples` |
| `test_colormap_count_exceeds_bits_per_sample_rejected` | `TestPixelArrayCountCap::test_colormap_count_exceeds_bits_per_sample_rejected` |
| `test_colormap_count_at_expected_passes` | `TestPixelArrayCountCap::test_colormap_count_at_expected_passes` |
| `test_absolute_cap_fires_when_dimensions_missing` | `TestPixelArrayCountCap::test_absolute_cap_fires_when_dimensions_missing` |
| `test_absolute_cap_constant_is_reasonable` | `TestPixelArrayCountCap::test_absolute_cap_constant_is_reasonable` |
| `test_dimensions_listed_after_pixel_array_tag_still_validate` | `TestPixelArrayCountCap::test_dimensions_listed_after_pixel_array_tag_still_validate` |
| `test_strip_byte_counts_chunky_uses_image_length_only` | `TestPixelArrayCountCap::test_strip_byte_counts_chunky_uses_image_length_only` |

## Section 4: 3D writer-dim validation

### test_validate_3d_non_band_trailing_dim_2240.py -> TestValidate3DWriterDims / TestValidate3DWriterEndToEnd

| old test | new id |
| --- | --- |
| `test_validate_3d_rejects_yx_non_band_trailing` | `TestValidate3DWriterDims::test_rejects_yx_non_band_trailing` |
| `test_validate_3d_rejects_yx_aliases_with_non_band_trailing` | `TestValidate3DWriterDims::test_rejects_yx_aliases_with_non_band_trailing` |
| `test_validate_3d_still_accepts_band_alias_trailing` | `TestValidate3DWriterDims::test_still_accepts_band_alias_trailing` |
| `test_validate_3d_still_accepts_band_alias_leading` | `TestValidate3DWriterDims::test_still_accepts_band_alias_leading` |
| `test_validate_3d_still_routes_temporal_to_temporal_message` | `TestValidate3DWriterDims::test_still_routes_temporal_to_temporal_message` |
| `test_validate_3d_still_rejects_other_ambiguous_leading` | `TestValidate3DWriterDims::test_still_rejects_other_ambiguous_leading` |
| `test_validate_3d_2d_dims_unchanged` | `TestValidate3DWriterDims::test_2d_dims_unchanged` |
| `test_to_geotiff_rejects_yxz_dataarray` | `TestValidate3DWriterEndToEnd::test_to_geotiff_rejects_yxz_dataarray` |
| `test_to_geotiff_rejects_lat_lon_scenario_dataarray` | `TestValidate3DWriterEndToEnd::test_to_geotiff_rejects_lat_lon_scenario_dataarray` |
| `test_error_message_is_actionable` | `TestValidate3DWriterEndToEnd::test_error_message_is_actionable` |
| `test_to_geotiff_still_accepts_yx_band_dataarray` | `TestValidate3DWriterEndToEnd::test_to_geotiff_still_accepts_yx_band_dataarray` |
| `test_to_geotiff_still_accepts_band_yx_dataarray` | `TestValidate3DWriterEndToEnd::test_to_geotiff_still_accepts_band_yx_dataarray` |
| `test_raw_ndarray_band_last_still_writes` | `TestValidate3DWriterEndToEnd::test_raw_ndarray_band_last_still_writes` |
| `test_raw_ndarray_unusual_third_axis_still_writes` | `TestValidate3DWriterEndToEnd::test_raw_ndarray_unusual_third_axis_still_writes` |

## Section 5: window-bounds validation

### test_window_out_of_bounds_1634.py -> TestWindowOutOfBoundsEager / TestWindowInBoundsEager / TestWindowBackendParity

| old test | new id |
| --- | --- |
| `test_eager_negative_start_raises_value_error` | `TestWindowOutOfBoundsEager::test_negative_start_raises_value_error` |
| `test_eager_past_right_edge_raises_value_error` | `TestWindowOutOfBoundsEager::test_past_right_edge_raises_value_error` |
| `test_eager_past_bottom_edge_raises_value_error` | `TestWindowOutOfBoundsEager::test_past_bottom_edge_raises_value_error` |
| `test_eager_past_both_edges_raises_value_error` | `TestWindowOutOfBoundsEager::test_past_both_edges_raises_value_error` |
| `test_eager_zero_size_window_raises_value_error` | `TestWindowOutOfBoundsEager::test_zero_size_window_raises_value_error` |
| `test_eager_inverted_window_raises_value_error` | `TestWindowOutOfBoundsEager::test_inverted_window_raises_value_error` |
| `test_eager_full_extent_window_returns_full_array` | `TestWindowInBoundsEager::test_full_extent_window_returns_full_array` |
| `test_eager_interior_window_returns_correct_subset` | `TestWindowInBoundsEager::test_interior_window_returns_correct_subset` |
| `test_eager_edge_aligned_window_returns_correct_subset` | `TestWindowInBoundsEager::test_edge_aligned_window_returns_correct_subset` |
| `test_eager_and_dask_paths_share_window_validation` | `TestWindowBackendParity::test_eager_and_dask_paths_share_window_validation` |
| `test_eager_and_dask_paths_share_window_message_format` | `TestWindowBackendParity::test_eager_and_dask_paths_share_window_message_format` |
| `test_issue_1634_reproducer_raises_clean_error` | `TestWindowBackendParity::test_reproducer_raises_clean_error` |

## Section 6: degenerate pixel-size fail-closed

### test_degenerate_pixel_size_2214.py -> TestDegenerateWritesFailClosed / TestDegenerateWritesWithExplicitTransform / TestDegenerateWritesWithOptIn / TestMultiRowMultiColumnUnchanged / TestCoordsToTransformHelperContract / TestDegenerateFailClosedAcrossBackends

| old test | new id |
| --- | --- |
| `TestDegenerateWritesFailClosed::test_1xN_without_transform_or_optin_raises` | `TestDegenerateWritesFailClosed::test_1xN_without_transform_or_optin_raises` |
| `TestDegenerateWritesFailClosed::test_Nx1_without_transform_or_optin_raises` | `TestDegenerateWritesFailClosed::test_Nx1_without_transform_or_optin_raises` |
| `TestDegenerateWritesWithExplicitTransform::test_1xN_with_attrs_transform_round_trips_true_pixel_size` | same |
| `TestDegenerateWritesWithExplicitTransform::test_Nx1_with_attrs_transform_round_trips_true_pixel_size` | same |
| `TestDegenerateWritesWithOptIn::test_1xN_optin_borrows_from_x_axis` | same |
| `TestDegenerateWritesWithOptIn::test_Nx1_optin_borrows_from_y_axis` | same |
| `TestDegenerateWritesWithOptIn::test_optin_must_be_boolean_True_not_truthy_string` | same |
| `TestMultiRowMultiColumnUnchanged::test_2x2_writes_without_optin` | same |
| `TestMultiRowMultiColumnUnchanged::test_3x5_writes_without_optin` | same |
| `TestCoordsToTransformHelperContract::test_degenerate_without_optin_returns_None` | same |
| `TestCoordsToTransformHelperContract::test_degenerate_with_optin_returns_borrowed_transform` | same |
| `TestCoordsToTransformHelperContract::test_multi_axis_ignores_optin_flag` | same |
| `TestDegenerateFailClosedAcrossBackends::test_dask_numpy_1xN_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_dask_numpy_Nx1_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_vrt_1xN_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_vrt_Nx1_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_gpu_1xN_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_gpu_Nx1_raises` | same |
| `TestDegenerateFailClosedAcrossBackends::test_dask_cupy_1xN_raises` | same |
