# Cluster audit -- coords

Issue: #2428 (cluster 4 of long-tail epic #2424).

This file is deleted on the pre-merge commit. Do not let it land on main.

## Scope

Extend `read/test_coords.py` (already exists from #2390) with the seven
coord-edge-case files still at the top level, plus the alias-validation
file flagged as a boundary case. Tests-only -- no source changes.

## Sections in the extended file

1. `coords_from_pixel_geometry` / `transform_tuple_from_pixel_geometry`
   / `coords_from_geo_info` -- the shared GeoTransform-to-(y, x) helpers.
2. `_extract_transform` multi-tiepoint consistency (single, multi-
   consistent, GCP-warp rejection, tolerance scaling, helper edge cases).
3. Zero-denominator `RATIONAL` / `SRATIONAL` rejection on the reader.
4. Descending / ascending coord round trip + orientation tag selection.
5. `_coords_to_transform` writer-side validation: regularity (1D),
   3D `(y, x, band)` / `(band, y, x)`, and alias-aware
   `NonUniformCoordsError` across `y/x`, `lat/lon`, `latitude/longitude`,
   `row/col`.
6. Integer-coord round trip + `_NO_GEOREF_KEY` marker. Integration
   sub-cluster gated on `XRSPATIAL_RUN_INTEGRATION=1`.

## Boundary

`test_non_uniform_coords_alias_2215.py` was flagged as "fold if it slots
cleanly". It does -- the alias resolution layers cleanly onto the
existing `_coords_to_transform` writer-side checks, so the file is
folded as Section 5b (`TestNonUniformCoordsAliasResolution` and
`TestNonUniformCoordsAlias`) rather than left as a follow-up.

## File mapping

### `test_coords_1813.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `TestCoordsFromPixelGeometry::test_basic_area_north_up` | `TestCoordsFromPixelGeometry::test_basic_area_north_up` |
| `TestCoordsFromPixelGeometry::test_windowed_area` | `TestCoordsFromPixelGeometry::test_windowed_area` |
| `TestCoordsFromPixelGeometry::test_pixel_is_point_skips_half_pixel_shift` | `TestCoordsFromPixelGeometry::test_pixel_is_point_skips_half_pixel_shift` |
| `TestCoordsFromPixelGeometry::test_negative_y_resolution_north_up` | `TestCoordsFromPixelGeometry::test_negative_y_resolution_north_up` |
| `TestCoordsFromPixelGeometry::test_no_georef_returns_integer_pixel_coords` | `TestCoordsFromPixelGeometry::test_no_georef_returns_integer_pixel_coords` |
| `TestCoordsFromPixelGeometry::test_no_georef_windowed_returns_integer_window_indices` | `TestCoordsFromPixelGeometry::test_no_georef_windowed_returns_integer_window_indices` |
| `TestTransformTupleFromPixelGeometry::test_basic_tuple_ordering` | `TestTransformTupleFromPixelGeometry::test_basic_tuple_ordering` |
| `TestTransformTupleFromPixelGeometry::test_windowed_origin_shifts` | `TestTransformTupleFromPixelGeometry::test_windowed_origin_shifts` |
| `TestCoordsFromGeoInfo::test_area_full_extent` | `TestCoordsFromGeoInfo::test_area_full_extent` |
| `TestCoordsFromGeoInfo::test_windowed` | `TestCoordsFromGeoInfo::test_windowed` |
| `TestCoordsFromGeoInfo::test_pixel_is_point` | `TestCoordsFromGeoInfo::test_pixel_is_point` |
| `TestCoordsFromGeoInfo::test_no_georef_returns_integer_coords` | `TestCoordsFromGeoInfo::test_no_georef_returns_integer_coords` |
| `TestCoordsFromGeoInfo::test_none_transform_treated_as_no_georef` | `TestCoordsFromGeoInfo::test_none_transform_treated_as_no_georef` |

### `test_multi_tiepoint_validation_2117.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `test_single_tiepoint_unchanged` | `TestMultiTiepointValidation::test_single_tiepoint_unchanged` |
| `test_multiple_consistent_tiepoints_pass` | `TestMultiTiepointValidation::test_multiple_consistent_tiepoints_pass` |
| `test_inconsistent_tiepoints_raise` | `TestMultiTiepointValidation::test_inconsistent_tiepoints_raise` |
| `test_tolerance_scales_with_pixel_size` | `TestMultiTiepointValidation::test_tolerance_scales_with_pixel_size` |
| `test_validate_helper_no_op_for_single_tuple` | `TestMultiTiepointValidation::test_validate_helper_no_op_for_single_tuple` |
| `test_validate_helper_rejects_disagreement` | `TestMultiTiepointValidation::test_validate_helper_rejects_disagreement` |
| `test_validate_helper_y_axis_sign` | `TestMultiTiepointValidation::test_validate_helper_y_axis_sign` |
| `test_tiepoint_without_scale_also_validates` | `TestMultiTiepointValidation::test_tiepoint_without_scale_also_validates` |
| `test_validate_helper_honours_custom_rel_tol` | `TestMultiTiepointValidation::test_validate_helper_honours_custom_rel_tol` |
| `test_short_tiepoint_is_treated_as_single_tuple` | `TestMultiTiepointValidation::test_short_tiepoint_is_treated_as_single_tuple` |

### `test_rational_zero_denominator_2313.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `TestRationalZeroDenominator::test_rational_zero_denominator_surfaces_from_parse_all_ifds` | `TestRationalZeroDenominator::test_rational_zero_denominator_surfaces_from_parse_all_ifds` |
| `TestRationalZeroDenominator::test_rational_zero_denominator_message_includes_denominator` | `TestRationalZeroDenominator::test_rational_zero_denominator_message_includes_denominator` |
| `TestRationalZeroDenominator::test_srational_zero_denominator_surfaces_from_parse_all_ifds` | `TestRationalZeroDenominator::test_srational_zero_denominator_surfaces_from_parse_all_ifds` |
| `TestRationalZeroDenominator::test_rational_zero_denominator_fails_open_geotiff` | `TestRationalZeroDenominator::test_rational_zero_denominator_fails_open_geotiff` |
| `TestRationalZeroDenominator::test_yresolution_zero_denominator_named_in_error` | `TestRationalZeroDenominator::test_yresolution_zero_denominator_named_in_error` |
| `TestRationalZeroDenominator::test_tag_constants_present` | `TestRationalZeroDenominator::test_tag_constants_present` |

### `test_coord_regularity_1720.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `test_uniform_coords_ok` | `TestCoordsToTransformRegularity::test_uniform_coords_ok` |
| `test_uniform_coords_roundtrip_to_geotiff_1720` | `TestCoordsToTransformRegularity::test_uniform_coords_roundtrip_to_geotiff` |
| `test_non_uniform_x_raises_1720` | `TestCoordsToTransformRegularity::test_non_uniform_x_raises` |
| `test_non_uniform_y_raises_1720` | `TestCoordsToTransformRegularity::test_non_uniform_y_raises` |
| `test_jitter_within_tolerance_ok_1720` | `TestCoordsToTransformRegularity::test_jitter_within_tolerance_ok` |
| `test_jitter_just_above_tolerance_raises_1720` | `TestCoordsToTransformRegularity::test_jitter_just_above_tolerance_raises` |
| `test_two_sample_coords_ok_1720` | `TestCoordsToTransformRegularity::test_two_sample_coords_ok` |
| `test_constant_coords_raises_1720` | `TestCoordsToTransformRegularity::test_constant_coords_raises` |

### `test_coords_to_transform_3d_1643.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `test_coords_to_transform_yxband_returns_yx_spacing` | `TestCoordsToTransform3D::test_yxband_returns_yx_spacing` |
| `test_coords_to_transform_bandyx_returns_yx_spacing` | `TestCoordsToTransform3D::test_bandyx_returns_yx_spacing` |
| `test_coords_to_transform_3d_band_name_variants[band]` | `TestCoordsToTransform3D::test_3d_band_name_variants[band]` |
| `test_coords_to_transform_3d_band_name_variants[bands]` | `TestCoordsToTransform3D::test_3d_band_name_variants[bands]` |
| `test_coords_to_transform_3d_band_name_variants[channel]` | `TestCoordsToTransform3D::test_3d_band_name_variants[channel]` |
| `test_coords_to_transform_2d_unchanged` | `TestCoordsToTransform3D::test_2d_unchanged` |
| `test_to_geotiff_roundtrip_3d_yxband_no_transform_attr` | `TestCoordsToTransform3D::test_to_geotiff_roundtrip_3d_yxband` |
| `test_to_geotiff_roundtrip_3d_bandyx_no_transform_attr` | `TestCoordsToTransform3D::test_to_geotiff_roundtrip_3d_bandyx` |
| `test_to_geotiff_3d_without_transform_attr_does_not_invent_unit_pixels` | `TestCoordsToTransform3D::test_to_geotiff_3d_does_not_invent_unit_pixels` |
| `test_write_geotiff_gpu_roundtrip_3d_no_transform_attr` | `TestCoordsToTransform3D::test_write_geotiff_gpu_roundtrip_3d` |

### `test_non_uniform_coords_alias_2215.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `test_resolve_spatial_coords_finds_alias[y-x]` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_finds_alias[y-x]` |
| `test_resolve_spatial_coords_finds_alias[lat-lon]` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_finds_alias[lat-lon]` |
| `test_resolve_spatial_coords_finds_alias[latitude-longitude]` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_finds_alias[latitude-longitude]` |
| `test_resolve_spatial_coords_finds_alias[row-col]` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_finds_alias[row-col]` |
| `test_resolve_spatial_coords_picks_canonical_first` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_picks_canonical_first` |
| `test_resolve_spatial_coords_missing_returns_none` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_missing_returns_none` |
| `test_resolve_spatial_coords_handles_none_input` | `TestNonUniformCoordsAliasResolution::test_resolve_spatial_coords_handles_none_input` |
| `test_non_uniform_y_alias_raises_non_uniform_coords_error[*]` | `TestNonUniformCoordsAlias::test_non_uniform_y_alias_raises_typed[*]` |
| `test_non_uniform_x_alias_raises_non_uniform_coords_error[*]` | `TestNonUniformCoordsAlias::test_non_uniform_x_alias_raises_typed[*]` |
| `test_constant_y_alias_raises_non_uniform_coords_error[*]` | `TestNonUniformCoordsAlias::test_constant_y_alias_raises_typed[*]` |
| `test_uniform_alias_coords_write_successfully[*]` | `TestNonUniformCoordsAlias::test_uniform_alias_coords_write_successfully[*]` |
| `test_alias_pairs_cover_every_documented_name` | `TestNonUniformCoordsAlias::test_alias_pairs_cover_every_documented_name` |
| `test_legacy_except_value_error_still_catches` | `TestNonUniformCoordsAlias::test_legacy_except_value_error_still_catches` |

### `test_int_coord_sentinel_2087.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `test_marker_predicate_identity_check[*]` | `TestNoGeorefMarkerPredicate::test_marker_predicate_identity_check[*]` |
| `test_arange_int64_shape_helper_accepts[*]` | `TestNoGeorefMarkerPredicate::test_arange_int64_shape_helper_accepts[*]` |
| `test_arange_int64_shape_helper_rejects[*]` | `TestNoGeorefMarkerPredicate::test_arange_int64_shape_helper_rejects[*]` |
| `test_user_authored_int_grid_writes_real_transform` | `TestIntCoordRoundTrip::test_user_authored_int_grid_writes_real_transform` |
| `test_both_axes_ascending_int64_step1_round_trips_with_georef` | `TestIntCoordRoundTrip::test_both_axes_ascending_int64_step1_writes_real_transform` |
| `test_user_authored_int_grid_with_explicit_transform` | `TestIntCoordRoundTrip::test_user_authored_int_grid_with_explicit_transform` |
| `test_non_uniform_int_coords_raise` | `TestIntCoordRoundTrip::test_non_uniform_int_coords_raise` |
| `test_int_x_float_y_writes_transform` | `TestIntCoordRoundTrip::test_int_x_float_y_writes_transform` |
| `test_no_georef_roundtrip_preserved` | `TestIntCoordRoundTrip::test_no_georef_roundtrip_preserved` |
| `test_windowed_no_georef_roundtrip_with_marker` | `TestIntCoordRoundTrip::test_windowed_no_georef_roundtrip_with_marker` |

### `test_int_coords_round_trip_hotfix_1962.py` -> `read/test_coords.py`

| Old test | New test / param id |
|---|---|
| `TestIntCoordRoundTripHotfix1962::test_int_coords_2d_round_trip` | `TestIntCoordRoundTripIntegration::test_int_coords_2d_round_trip` |
| `TestIntCoordRoundTripHotfix1962::test_int_coords_3d_band_y_x_round_trip` | `TestIntCoordRoundTripIntegration::test_int_coords_3d_band_y_x_round_trip` |

## Verification

- `pytest xrspatial/geotiff/tests/read/test_coords.py -v`: 101 passed,
  2 skipped (integration-gated).
- `pytest xrspatial/geotiff/tests/ -x -q`: 5817 passed, 68 skipped,
  2 xfailed.
- `find xrspatial/geotiff/tests -name 'test_*.py' | wc -l`: 242
  (was 249; -7 deleted, +0 added because `read/test_coords.py` already
  existed).
