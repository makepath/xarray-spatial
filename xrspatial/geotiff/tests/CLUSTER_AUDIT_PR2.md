# CLUSTER_AUDIT_PR2.md — VRT validation cluster

Temporary audit table tracking every old `file::test` and where it lands
in the consolidated `vrt/test_validation.py`. Deleted in a follow-up
commit on the same branch before merge per the epic #2390 contract.

## File mapping summary

| Old file | New file | Status |
|---|---|---|
| `test_vrt_validation_2321.py` | `vrt/test_validation.py` | folded |
| `test_vrt_capability_validator_2371.py` | `vrt/test_validation.py` | folded |
| `test_vrt_unsupported_2370.py` | `vrt/test_validation.py` | folded |
| `test_vrt_narrow_except_1670.py` | `vrt/test_validation.py` | folded |
| `test_vrt_path_containment_1671.py` | `vrt/test_validation.py` | folded |

## Test mapping (old → new)

### From `test_vrt_validation_2321.py`

| Old test | New location | Notes |
|---|---|---|
| `test_vrt_unsupported_error_is_geotiff_metadata_error` | `test_vrt_unsupported_error_subclass_contract` | identical assertions |
| `test_zero_bands_raises_vrt_unsupported` | `TestValidatorRules::test_zero_bands_rejected` | identical assertion |
| `test_zero_bands_parity_across_entry_points` | `test_zero_bands_parity_across_entry_points` | helper now `_expect_same_error` |
| `test_complex_dtype_band_rejected_by_validator` | `TestValidatorRules::test_complex_dtype_band_rejected` | identical assertions |
| `test_rotated_transform_rejected_without_opt_in` | `TestValidatorRules::test_rotated_transform_rejected_without_opt_in` | both opt-out and opt-in paths preserved |
| `test_negative_src_rect_size_rejected` | `TestValidatorRules::test_geometry_rules_rejected[reject[negative-src-size]]` | parametrised |
| `test_negative_src_rect_offset_rejected` | `TestValidatorRules::test_geometry_rules_rejected[reject[negative-src-offset]]` | parametrised |
| `test_negative_dst_rect_size_rejected` | `TestValidatorRules::test_geometry_rules_rejected[reject[negative-dst-size]]` | parametrised |
| `test_dst_rect_outside_vrt_extent_rejected` | `TestValidatorRules::test_geometry_rules_rejected[reject[dst-outside-extent]]` | parametrised |
| `test_zero_pixel_size_rejected` | `TestValidatorRules::test_geometry_rules_rejected[reject[zero-pixel-size]]` | parametrised |
| `test_unsupported_resample_alg_rejected_at_validate` | `TestValidatorRules::test_unsupported_resample_alg_rejected` | identical assertion |
| `test_mixed_band_nodata_rejected_without_opt_in` | `test_mixed_band_nodata_rejected_without_opt_in` | identical assertions |
| `test_unparseable_crs_rejected_without_opt_in` | `TestValidatorRules::test_unparseable_crs_rejected_without_opt_in` | both opt-out and opt-in paths preserved |
| `test_resample_parity_across_entry_points` | `test_resample_parity_across_entry_points` | uses `_expect_same_error` |
| `test_rotated_parity_across_entry_points` | `test_rotated_parity_across_entry_points` | uses `_expect_same_error` |
| `test_unsupported_resample_chunked_raises_at_build` | `test_unsupported_resample_chunked_raises_at_build` | identical assertion |
| `test_well_formed_vrt_validates_silently` | `TestValidatorRules::test_well_formed_vrt_validates_silently` | identical assertions |

### From `test_vrt_capability_validator_2371.py`

| Old test | New location | Notes |
|---|---|---|
| `test_validate_vrt_capability_alias_resolves_to_validate_parsed_vrt` | `test_validate_vrt_capability_is_validate_parsed_vrt` | identical assertion |
| `test_nested_vrt_rejected_at_validator` | `test_nested_vrt_message_names_outer_and_inner` | identical assertions (message, outer path, inner basename, keyword) |
| `test_nested_vrt_uppercase_extension_rejected` | `test_nested_vrt_uppercase_extension_rejected` | identical assertion |
| `test_nested_vrt_rejected_via_public_read_vrt` | `test_nested_vrt_rejected_via_entry_points[entry[package-read_vrt]]` | parametrised entry-point matrix |
| `test_nested_vrt_rejected_via_open_geotiff` | `test_nested_vrt_rejected_via_entry_points[entry[open_geotiff]]` | parametrised |
| `test_nested_vrt_rejected_via_internal_read_vrt` | `test_nested_vrt_rejected_via_entry_points[entry[internal-read_vrt]]` | parametrised |
| `test_warp_options_dataset_level_rejected_at_parse` | `test_warp_options_rejected_at_parse[warp[dataset-level]]` | parametrised over dataset / band scope |
| `test_warp_options_dataset_level_rejected_via_public_read_vrt` | `test_warp_options_dataset_rejected_via_entry_points[entry[package-read_vrt]]` | parametrised |
| `test_warp_options_dataset_level_rejected_via_internal_read_vrt` | `test_warp_options_dataset_rejected_via_entry_points[entry[internal-read_vrt]]` | parametrised |
| `test_warp_options_band_level_rejected` | `test_warp_options_rejected_at_parse[warp[band-level]]` | parametrised |
| `test_use_mask_band_true_rejected_at_validator` | `test_use_mask_band_message_names_source` | identical assertions |
| `test_use_mask_band_truthy_spellings_rejected[true/True/TRUE/1]` | `test_use_mask_band_truthy_spellings_rejected[truthy[true]/[True]/[TRUE]/[1]]` | descriptive IDs |
| `test_use_mask_band_false_is_accepted` | `test_use_mask_band_false_is_accepted` | identical |
| `test_use_mask_band_non_canonical_truthy_accepted[yes/on/Y]` | `test_use_mask_band_non_canonical_truthy_accepted[non-canonical[yes]/[on]/[Y]]` | descriptive IDs |
| `test_use_mask_band_rejected_via_public_read_vrt` | `test_use_mask_band_rejected_via_entry_points[entry[package-read_vrt]]` | parametrised |
| `test_use_mask_band_rejected_via_internal_read_vrt` | `test_use_mask_band_rejected_via_entry_points[entry[internal-read_vrt]]` | parametrised |
| `test_per_source_mask_band_rejected_at_validator` | `test_per_source_mask_band_message_names_source` | identical assertions |
| `test_resample_alg_now_rejected_at_internal_read_vrt` | `test_resample_alg_rejected_at_internal_read_vrt` | identical assertion |
| `test_nested_vrt_error_is_value_error` | `test_nested_vrt_error_remains_value_error_subclass` | identical assertions |

### From `test_vrt_unsupported_2370.py`

The `_assert_raises_or_xfail` helper from the original file is gone; PR
1's validator landed, so most cases assert directly. Two cases that
were already `xfail` in the original (mixed-CRS, mixed-dtype widening)
stay under `pytest.mark.xfail(strict=False)` until the validator delivers
the rejection contract.

| Old test | New location | Notes |
|---|---|---|
| `test_warped_vrt_subclass_raises` | `test_warped_subclass_band_rejected_via_open_geotiff` | direct assertion (no xfail wrapper) |
| `test_warped_vrt_gdalwarpoptions_raises` | `test_warp_options_rejected_at_parse[warp[dataset-level]]` (and `..._via_entry_points`) | already covered by 2371 fold |
| `test_warped_vrt_open_geotiff_raises` | `test_warped_subclass_band_rejected_via_open_geotiff` | open_geotiff path preserved |
| `test_nested_vrt_source_raises` | `test_nested_vrt_rejected_via_entry_points[entry[package-read_vrt]]` | parametrised matrix |
| `test_nested_vrt_open_geotiff_raises` | `test_nested_vrt_rejected_via_entry_points[entry[open_geotiff]]` | parametrised matrix |
| `test_mixed_source_crs_raises` | `test_mixed_source_crs_rejected` | preserved as `xfail(strict=False)`; same assertion shape |
| `test_mixed_source_dtype_unsupported_complex_raises` | `test_mixed_source_dtype_complex_rejected` | direct assertion |
| `test_mixed_source_dtype_ambiguous_widening_raises` | `test_mixed_source_dtype_ambiguous_widening_rejected` | preserved as `xfail(strict=False)` |
| `test_mixed_source_band_count_raises` | `test_mixed_source_band_count_rejected` | direct assertion |
| `test_complex_mask_source_raises` | `test_dataset_level_mask_band_rejected` | direct assertion |
| `test_unsupported_resample_alg_raises[Bilinear/Cubic/Lanczos/Average/Mode]` | `test_unsupported_resample_alg_rejected_end_to_end[entry[package-read_vrt]-resample[<alg>]]` | merged with open_geotiff parametrise |
| `test_unsupported_resample_alg_open_geotiff` | `test_unsupported_resample_alg_rejected_end_to_end[entry[open_geotiff]-resample[cubic]]` | covered by full alg × entry matrix |
| `test_supported_simple_vrt_round_trips_via_open_geotiff` | `test_supported_simple_vrt_round_trips_via_open_geotiff` | identical assertion |

### From `test_vrt_narrow_except_1670.py`

The matrix is parametrised over exception class × mode rather than one
test per exception. The fixtures `clear_strict_env` and
`set_strict_env` are reused unchanged.

| Old test | New location | Notes |
|---|---|---|
| `test_runtime_error_propagates_default_mode` | `test_narrow_except_bug_classes_propagate_in_default_mode[bug[runtime-error]]` | parametrised |
| `test_runtime_error_propagates_strict_mode` | `test_narrow_except_runtime_error_propagates_in_strict_mode` | dedicated case |
| `test_file_not_found_warns_and_continues` | `test_narrow_except_io_or_parse_warns_in_default_mode[io[file-not-found]]` | parametrised |
| `test_file_not_found_strict_reraises` | `test_narrow_except_io_or_parse_reraises_in_strict_mode[io[file-not-found]]` | parametrised |
| `test_value_error_warns_and_continues` | `test_narrow_except_io_or_parse_warns_in_default_mode[parse[value-error]]` | parametrised |
| `test_value_error_strict_reraises` | `test_narrow_except_io_or_parse_reraises_in_strict_mode[parse[value-error]]` | parametrised |
| `test_struct_error_warns_and_continues` | `test_narrow_except_io_or_parse_warns_in_default_mode[parse[struct-error]]` | parametrised |
| `test_permission_error_warns_and_continues` | `test_narrow_except_io_or_parse_warns_in_default_mode[io[permission-error]]` | parametrised |
| `test_memory_error_propagates_default_mode` | `test_narrow_except_bug_classes_propagate_in_default_mode[bug[memory-error]]` | parametrised |
| `test_zlib_error_warns_and_continues` | `test_narrow_except_io_or_parse_warns_in_default_mode[codec[zlib-error]]` | parametrised |
| `test_zlib_error_strict_reraises` | `test_narrow_except_io_or_parse_reraises_in_strict_mode[codec[zlib-error]]` | parametrised |
| `test_zstd_error_warns_and_continues_if_available` | `test_narrow_except_zstd_error_warns_in_default_mode` | kept as standalone; `pytest.importorskip` replaced with module-level `skipif(not _has_zstandard())` |
| (new) | `test_narrow_except_zstd_error_reraises_in_strict_mode` | added strict-mode case for parity with zlib (closes the matrix; previously only the warn path was covered for zstd) |

### From `test_vrt_path_containment_1671.py`

Folded into two classes (`TestPathContainment`, `TestPathContainmentAllowlist`).
The `_clear_allowlist_env` autouse fixture is replaced by an explicit
`clear_allowlist_env` fixture on the non-allowlist tests so the
allowlist class can set the env var via `monkeypatch.setenv` without a
race against the autouse delenv.

| Old test | New location | Notes |
|---|---|---|
| `test_relative_source_with_dotdot_traversal_rejected` | `TestPathContainment::test_relative_dotdot_traversal_rejected` | identical assertion |
| `test_relative_source_symlink_traversal_rejected` | `TestPathContainment::test_relative_symlink_traversal_rejected` | identical assertion |
| `test_absolute_source_outside_vrt_dir_rejected` | `TestPathContainment::test_absolute_outside_vrt_dir_rejected` | identical assertion |
| `test_absolute_source_inside_vrt_dir_ok` | `TestPathContainment::test_absolute_inside_vrt_dir_ok` | identical assertion |
| `test_absolute_source_allowlisted_root_passes` | `TestPathContainmentAllowlist::test_single_root_allows_outside_absolute` | identical assertion |
| `test_allowlist_supports_multiple_roots` | `TestPathContainmentAllowlist::test_multiple_roots_pathsep_separated` | identical assertion |
| `test_allowlist_does_not_cover_traversal_via_relative_source` | `TestPathContainmentAllowlist::test_relative_source_escape_still_rejected` | identical assertion |
| `test_allowlist_empty_entries_ignored` | `TestPathContainmentAllowlist::test_empty_entries_ignored` | identical assertion |
| `test_normal_relative_source_under_vrt_dir` | `TestPathContainment::test_normal_relative_source_under_vrt_dir_ok` | identical assertion |
| `test_error_message_names_rejected_path` | `TestPathContainment::test_error_message_names_rejected_path` | identical assertion |
