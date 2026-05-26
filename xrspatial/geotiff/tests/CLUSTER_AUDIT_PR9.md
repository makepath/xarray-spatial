# CLUSTER_AUDIT_PR9.md

PR 9 of the GeoTIFF test consolidation epic (#2390): fold the
integration / HTTP / dask-pipeline cluster into three files under
`xrspatial/geotiff/tests/integration/`.

This file is deleted on the final commit on this branch before the PR
is approved (epic convention).

Each old file lands as one named section inside the consolidated module.
Helper functions, fixtures, and classes are suffixed with the section id
(`_<section>`) so cross-section names cannot collide. Top-level
`autouse=True` fixtures from each source file lose their autouse flag and
apply via an explicit `@pytest.mark.usefixtures(...)` marker on the tests
and classes of that section, so a fixture that monkey-patches a global
like `XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1` no longer leaks to tests
that need the production default (the `scheme_case` SSRF rejection tests).

Issue-number suffixes on test names (`_2266`, `_2026_05_15`, `_issue_A`)
are stripped per epic convention. Issue numbers are preserved in git log
and PR descriptions.

For parametrised tests, the "New `file::test_id`" column lists the first
collected parametrize variant. A single row in this table can therefore
cover several parametrize variants of the same test function -- the
original test moved as one unit and pytest expands the matrix from the
preserved `@pytest.mark.parametrize` decorators.

## HTTP sources -> `integration/test_http_sources.py`

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_http_band_validation_1695.py::test_http_negative_band_rejected` | `integration/test_http_sources.py::test_http_negative_band_rejected` |  |
| `test_http_band_validation_1695.py::test_http_negative_band_rejected_via_low_level` | `integration/test_http_sources.py::test_http_negative_band_rejected_via_low_level` |  |
| `test_http_band_validation_1695.py::test_http_band_equal_to_samples_rejected` | `integration/test_http_sources.py::test_http_band_equal_to_samples_rejected` |  |
| `test_http_band_validation_1695.py::test_http_band_far_above_samples_rejected` | `integration/test_http_sources.py::test_http_band_far_above_samples_rejected` |  |
| `test_http_band_validation_1695.py::test_http_nonzero_band_on_single_band_rejected` | `integration/test_http_sources.py::test_http_nonzero_band_on_single_band_rejected` |  |
| `test_http_band_validation_1695.py::test_http_band_zero_on_single_band_still_works` | `integration/test_http_sources.py::test_http_band_zero_on_single_band_still_works` |  |
| `test_http_band_validation_1695.py::test_http_band_none_returns_all_bands` | `integration/test_http_sources.py::test_http_band_none_returns_all_bands` |  |
| `test_http_band_validation_1695.py::test_local_and_http_negative_band_parity` | `integration/test_http_sources.py::test_local_and_http_negative_band_parity` |  |
| `test_http_band_validation_1695.py::test_local_and_http_band_equal_to_samples_parity` | `integration/test_http_sources.py::test_local_and_http_band_equal_to_samples_parity` |  |
| `test_http_band_validation_1695.py::test_local_and_http_single_band_nonzero_parity` | `integration/test_http_sources.py::test_local_and_http_single_band_nonzero_parity` |  |
| `test_http_band_validation_1695.py::test_open_geotiff_http_negative_band_rejected` | `integration/test_http_sources.py::test_open_geotiff_http_negative_band_rejected` |  |
| `test_http_cog_coalesce.py::test_coalesce_empty_input` | `integration/test_http_sources.py::test_coalesce_empty_input` |  |
| `test_http_cog_coalesce.py::test_coalesce_single_range` | `integration/test_http_sources.py::test_coalesce_single_range` |  |
| `test_http_cog_coalesce.py::test_coalesce_merges_adjacent_ranges` | `integration/test_http_sources.py::test_coalesce_merges_adjacent_ranges` |  |
| `test_http_cog_coalesce.py::test_coalesce_does_not_merge_when_gap_exceeds_threshold` | `integration/test_http_sources.py::test_coalesce_does_not_merge_when_gap_exceeds_threshold` |  |
| `test_http_cog_coalesce.py::test_coalesce_with_unsorted_input` | `integration/test_http_sources.py::test_coalesce_with_unsorted_input` |  |
| `test_http_cog_coalesce.py::test_coalesce_negative_threshold_disables_merging` | `integration/test_http_sources.py::test_coalesce_negative_threshold_disables_merging` |  |
| `test_http_cog_coalesce.py::test_coalesce_split_recovers_per_tile_bytes` | `integration/test_http_sources.py::test_coalesce_split_recovers_per_tile_bytes` |  |
| `test_http_cog_coalesce.py::test_coalesce_caps_merged_range_size_2266` | `integration/test_http_sources.py::test_coalesce_caps_merged_range_size` |  |
| `test_http_cog_coalesce.py::test_coalesce_cap_round_trips_bytes_2266` | `integration/test_http_sources.py::test_coalesce_cap_round_trips_bytes` |  |
| `test_http_cog_coalesce.py::test_coalesce_default_cap_bounds_adversarial_input_2266` | `integration/test_http_sources.py::test_coalesce_default_cap_bounds_adversarial_input` |  |
| `test_http_cog_coalesce.py::test_coalesce_cap_zero_disables_size_check_2266` | `integration/test_http_sources.py::test_coalesce_cap_zero_disables_size_check` |  |
| `test_http_cog_coalesce.py::test_coalesce_cap_does_not_split_legitimate_back_to_back_2266` | `integration/test_http_sources.py::test_coalesce_cap_does_not_split_legitimate_back_to_back` |  |
| `test_http_cog_coalesce.py::test_coalesce_cap_respects_env_override_2266` | `integration/test_http_sources.py::test_coalesce_cap_respects_env_override` |  |
| `test_http_cog_coalesce.py::test_coalesce_cap_preserves_oversized_single_input_2266` | `integration/test_http_sources.py::test_coalesce_cap_preserves_oversized_single_input` |  |
| `test_http_cog_coalesce.py::test_http_source_read_ranges_coalesced_respects_cap_2266` | `integration/test_http_sources.py::test_http_source_read_ranges_coalesced_respects_cap` |  |
| `test_http_cog_coalesce.py::test_read_cog_http_uses_coalesced_fetches` | `integration/test_http_sources.py::test_read_cog_http_uses_coalesced_fetches` |  |
| `test_http_cog_coalesce.py::test_read_cog_http_perf_with_mock_rtt` | `integration/test_http_sources.py::test_read_cog_http_perf_with_mock_rtt` |  |
| `test_http_cog_coalesce.py::test_dask_local_correctness` | `integration/test_http_sources.py::test_dask_local_correctness` |  |
| `test_http_cog_coalesce.py::test_dask_http_parses_ifds_once` | `integration/test_http_sources.py::test_dask_http_parses_ifds_once` |  |
| `test_http_cog_range_contract_2286.py::test_windowed_tile_read_bounded_bytes_and_range_count` | `integration/test_http_sources.py::test_windowed_tile_read_bounded_bytes_and_range_count` |  |
| `test_http_cog_range_contract_2286.py::test_windowed_multi_tile_read_range_count_bounded` | `integration/test_http_sources.py::test_windowed_multi_tile_read_range_count_bounded` |  |
| `test_http_cog_range_contract_2286.py::test_overview_read_does_not_fetch_full_resolution_pixels` | `integration/test_http_sources.py::test_overview_read_does_not_fetch_full_resolution_pixels` |  |
| `test_http_cog_range_contract_2286.py::test_band_selection_multiband_chunky_bounded_reads` | `integration/test_http_sources.py::test_band_selection_multiband_chunky_bounded_reads` |  |
| `test_http_cog_range_contract_2286.py::test_band_selection_with_window_bounded_range_count` | `integration/test_http_sources.py::test_band_selection_with_window_bounded_range_count` |  |
| `test_http_cog_range_contract_2286.py::test_dask_read_parses_ifds_once_across_chunks` | `integration/test_http_sources.py::test_dask_read_parses_ifds_once_across_chunks` |  |
| `test_http_cog_range_contract_2286.py::test_dask_header_gets_independent_of_chunk_count` | `integration/test_http_sources.py::test_dask_header_gets_independent_of_chunk_count` |  |
| `test_http_cog_range_contract_2286.py::test_truncated_cog_closes_http_source` | `integration/test_http_sources.py::test_truncated_cog_closes_http_source` |  |
| `test_http_cog_range_contract_2286.py::test_malformed_ifd_chain_closes_http_source` | `integration/test_http_sources.py::test_malformed_ifd_chain_closes_http_source` |  |
| `test_http_cog_range_contract_2286.py::test_short_body_during_pixel_fetch_closes_source` | `integration/test_http_sources.py::test_short_body_during_pixel_fetch_closes_source` |  |
| `test_http_cog_range_contract_2286.py::test_coalesce_does_not_silently_exceed_explicit_cap` | `integration/test_http_sources.py::test_coalesce_does_not_silently_exceed_explicit_cap` |  |
| `test_http_cog_range_contract_2286.py::test_coalesce_default_cap_bounds_adversarial_input` | `integration/test_http_sources.py::test_coalesce_default_cap_bounds_adversarial_input` |  |
| `test_http_cog_range_contract_2286.py::test_coalesced_get_size_capped_on_real_http_source` | `integration/test_http_sources.py::test_coalesced_get_size_capped_on_real_http_source` |  |
| `test_http_cog_range_contract_2286.py::test_split_coalesced_bytes_round_trips_under_cap` | `integration/test_http_sources.py::test_split_coalesced_bytes_round_trips_under_cap` |  |
| `test_http_cog_range_contract_2286.py::test_loopback_end_to_end_windowed_byte_budget` | `integration/test_http_sources.py::test_loopback_end_to_end_windowed_byte_budget` |  |
| `test_http_dask_allow_rotated_2130.py::test_http_dask_rotated_default_raises` | `integration/test_http_sources.py::test_http_dask_rotated_default_raises` |  |
| `test_http_dask_allow_rotated_2130.py::test_http_dask_rotated_allow_rotated_reads` | `integration/test_http_sources.py::test_http_dask_rotated_allow_rotated_reads` |  |
| `test_http_dask_orientation_1794.py::test_http_dask_read_rejects_non_default_orientation` | `integration/test_http_sources.py::test_http_dask_read_rejects_non_default_orientation` |  |
| `test_http_meta_buffer_1718.py::test_small_cog_uses_single_initial_read` | `integration/test_http_sources.py::test_small_cog_uses_single_initial_read` |  |
| `test_http_meta_buffer_1718.py::test_ifd_chain_past_64kib_resolves` | `integration/test_http_sources.py::test_ifd_chain_past_64kib_resolves` |  |
| `test_http_meta_buffer_1718.py::test_end_to_end_http_read_with_big_metadata` | `integration/test_http_sources.py::test_end_to_end_http_read_with_big_metadata` |  |
| `test_http_meta_buffer_1718.py::test_cap_raises_clear_error_on_excessive_chain` | `integration/test_http_sources.py::test_cap_raises_clear_error_on_excessive_chain` |  |
| `test_http_no_stdlib_fallback_2050.py::test_urllib3_is_importable` | `integration/test_http_sources.py::test_urllib3_is_importable` |  |
| `test_http_no_stdlib_fallback_2050.py::test_reader_imports_urllib3_at_module_level` | `integration/test_http_sources.py::test_reader_imports_urllib3_at_module_level` |  |
| `test_http_no_stdlib_fallback_2050.py::test_get_http_pool_returns_a_pool_manager` | `integration/test_http_sources.py::test_get_http_pool_returns_a_pool_manager` |  |
| `test_http_no_stdlib_fallback_2050.py::test_stdlib_opener_helper_is_removed` | `integration/test_http_sources.py::test_stdlib_opener_helper_is_removed` |  |
| `test_http_no_stdlib_fallback_2050.py::test_validating_redirect_handler_is_removed` | `integration/test_http_sources.py::test_validating_redirect_handler_is_removed` |  |
| `test_http_no_stdlib_fallback_2050.py::test_reader_does_not_import_urllib_request` | `integration/test_http_sources.py::test_reader_does_not_import_urllib_request` |  |
| `test_http_no_stdlib_fallback_2050.py::test_read_range_source_has_no_stdlib_branch` | `integration/test_http_sources.py::test_read_range_source_has_no_stdlib_branch` |  |
| `test_http_no_stdlib_fallback_2050.py::test_read_all_source_has_no_stdlib_branch` | `integration/test_http_sources.py::test_read_all_source_has_no_stdlib_branch` |  |
| `test_http_no_stdlib_fallback_2050.py::test_read_range_uses_urllib3_pool` | `integration/test_http_sources.py::test_read_range_uses_urllib3_pool` |  |
| `test_http_no_stdlib_fallback_2050.py::test_read_all_uses_urllib3_pool` | `integration/test_http_sources.py::test_read_all_uses_urllib3_pool` |  |
| `test_http_no_stdlib_fallback_2050.py::test_read_range_short_circuits_zero_length` | `integration/test_http_sources.py::test_read_range_short_circuits_zero_length` |  |
| `test_http_no_stdlib_fallback_2050.py::test_install_requires_lists_urllib3` | `integration/test_http_sources.py::test_install_requires_lists_urllib3` |  |
| `test_http_orientation_1717.py::test_http_full_read_matches_local_for_orientation` | `integration/test_http_sources.py::test_http_full_read_matches_local_for_orientation[2]` |  |
| `test_http_orientation_1717.py::test_http_windowed_read_rejects_non_default_orientation` | `integration/test_http_sources.py::test_http_windowed_read_rejects_non_default_orientation[5]` |  |
| `test_http_orientation_1717.py::test_http_default_orientation_still_works` | `integration/test_http_sources.py::test_http_default_orientation_still_works` |  |
| `test_http_range_validation_1735.py::test_range_request_ignored_for_nonzero_start_raises` | `integration/test_http_sources.py::test_range_request_ignored_for_nonzero_start_raises` |  |
| `test_http_range_validation_1735.py::test_range_request_wrong_content_range_raises` | `integration/test_http_sources.py::test_range_request_wrong_content_range_raises` |  |
| `test_http_range_validation_1735.py::test_range_request_short_body_raises` | `integration/test_http_sources.py::test_range_request_short_body_raises` |  |
| `test_http_range_validation_1735.py::test_range_request_well_formed_succeeds` | `integration/test_http_sources.py::test_range_request_well_formed_succeeds` |  |
| `test_http_range_validation_1735.py::test_read_range_zero_length_returns_empty_without_request` | `integration/test_http_sources.py::test_read_range_zero_length_returns_empty_without_request` |  |
| `test_http_range_validation_1735.py::test_range_ignored_200_oversize_rejected_via_content_length` | `integration/test_http_sources.py::test_range_ignored_200_oversize_rejected_via_content_length` |  |
| `test_http_range_validation_1735.py::test_range_ignored_200_full_object_sliced_within_cap` | `integration/test_http_sources.py::test_range_ignored_200_full_object_sliced_within_cap` |  |
| `test_http_range_validation_1735.py::test_range_ignored_200_short_body_returned_as_is` | `integration/test_http_sources.py::test_range_ignored_200_short_body_returned_as_is` |  |
| `test_http_range_validation_1735.py::test_range_ignored_200_no_content_length_is_streamed_and_capped` | `integration/test_http_sources.py::test_range_ignored_200_no_content_length_is_streamed_and_capped` |  |
| `test_http_range_validation_1735.py::test_range_request_uses_streaming_response` | `integration/test_http_sources.py::test_range_request_uses_streaming_response` |  |
| `test_http_read_all_bounded_2051.py::test_budget_uses_max_strip_end_plus_slack` | `integration/test_http_sources.py::test_budget_uses_max_strip_end_plus_slack` |  |
| `test_http_read_all_bounded_2051.py::test_budget_empty_strip_table_falls_back_to_per_strip_cap` | `integration/test_http_sources.py::test_budget_empty_strip_table_falls_back_to_per_strip_cap` |  |
| `test_http_read_all_bounded_2051.py::test_budget_all_sparse_falls_back_to_per_strip_cap` | `integration/test_http_sources.py::test_budget_all_sparse_falls_back_to_per_strip_cap` |  |
| `test_http_read_all_bounded_2051.py::test_read_all_no_budget_returns_full_body` | `integration/test_http_sources.py::test_read_all_no_budget_returns_full_body` |  |
| `test_http_read_all_bounded_2051.py::test_read_all_rejects_oversized_content_length` | `integration/test_http_sources.py::test_read_all_rejects_oversized_content_length` |  |
| `test_http_read_all_bounded_2051.py::test_read_all_truncates_when_server_lies_about_content_length_small` | `integration/test_http_sources.py::test_read_all_truncates_when_server_lies_about_content_length_small` |  |
| `test_http_read_all_bounded_2051.py::test_read_all_catches_missing_content_length` | `integration/test_http_sources.py::test_read_all_catches_missing_content_length` |  |
| `test_http_read_all_bounded_2051.py::test_read_all_passes_when_body_fits_budget` | `integration/test_http_sources.py::test_read_all_passes_when_body_fits_budget` |  |
| `test_http_read_all_bounded_2051.py::test_full_image_http_read_still_works_for_legitimate_cog` | `integration/test_http_sources.py::test_full_image_http_read_still_works_for_legitimate_cog` |  |
| `test_http_read_all_bounded_2051.py::test_full_image_http_read_rejects_padded_body` | `integration/test_http_sources.py::test_full_image_http_read_rejects_padded_body` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_http_schemes_match` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_http_schemes_match[HTTPS://example.com/x.tif]` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_non_http_schemes_do_not_match` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_non_http_schemes_do_not_match[C:\\windows\\file.tif]` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_non_string_does_not_match` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_non_string_does_not_match[42]` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_empty_string_does_not_match` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_empty_string_does_not_match` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_scheme_only_prefix_does_not_match` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_scheme_only_prefix_does_not_match` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_scheme_colon_no_slashes_classifies_as_http` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_scheme_colon_no_slashes_classifies_as_http` |  |
| `test_http_scheme_case_2321.py::TestIsHttpSourceHelper::test_open_source_http_colon_no_hostname_raises` | `integration/test_http_sources.py::TestIsHttpSourceHelper_http_scheme_case::test_open_source_http_colon_no_hostname_raises` |  |
| `test_http_scheme_case_2321.py::TestOpenSourceRoutesUppercase::test_uppercase_http_routes_to_http_source` | `integration/test_http_sources.py::TestOpenSourceRoutesUppercase_http_scheme_case::test_uppercase_http_routes_to_http_source` |  |
| `test_http_scheme_case_2321.py::TestOpenSourceRoutesUppercase::test_uppercase_https_routes_to_http_source` | `integration/test_http_sources.py::TestOpenSourceRoutesUppercase_http_scheme_case::test_uppercase_https_routes_to_http_source` |  |
| `test_http_scheme_case_2321.py::TestOpenSourceRoutesUppercase::test_mixed_case_routes_to_http_source` | `integration/test_http_sources.py::TestOpenSourceRoutesUppercase_http_scheme_case::test_mixed_case_routes_to_http_source` |  |
| `test_http_scheme_case_2321.py::TestDispatchBooleansAreCaseInsensitive::test_helper_recognizes_uppercase` | `integration/test_http_sources.py::TestDispatchBooleansAreCaseInsensitive_http_scheme_case::test_helper_recognizes_uppercase[HTTPS://example.com/x.tif]` |  |
| `test_http_scheme_case_2321.py::TestDispatchBooleansAreCaseInsensitive::test_is_fsspec_uri_excludes_uppercase_http` | `integration/test_http_sources.py::TestDispatchBooleansAreCaseInsensitive_http_scheme_case::test_is_fsspec_uri_excludes_uppercase_http` |  |
| `test_http_scheme_case_2321.py::TestDispatchBooleansAreCaseInsensitive::test_writer_is_fsspec_uri_excludes_uppercase_http` | `integration/test_http_sources.py::TestDispatchBooleansAreCaseInsensitive_http_scheme_case::test_writer_is_fsspec_uri_excludes_uppercase_http` |  |
| `test_http_scheme_case_2321.py::TestDispatchBooleansAreCaseInsensitive::test_sidecar_helper_is_case_insensitive` | `integration/test_http_sources.py::TestDispatchBooleansAreCaseInsensitive_http_scheme_case::test_sidecar_helper_is_case_insensitive` |  |
| `test_http_scheme_case_2321.py::TestUppercaseSchemeStillRejectsPrivateHosts::test_private_host_rejected_regardless_of_scheme_case` | `integration/test_http_sources.py::TestUppercaseSchemeStillRejectsPrivateHosts_http_scheme_case::test_private_host_rejected_regardless_of_scheme_case[127.0.0.1-HTTPS]` |  |
| `test_http_scheme_case_2321.py::TestUppercaseSchemeStillRejectsPrivateHosts::test_localhost_rejected_regardless_of_scheme_case` | `integration/test_http_sources.py::TestUppercaseSchemeStillRejectsPrivateHosts_http_scheme_case::test_localhost_rejected_regardless_of_scheme_case[HTTPS]` |  |
| `test_http_scheme_case_2321.py::TestUppercaseSchemeStillRejectsPrivateHosts::test_uppercase_scheme_to_127_literal_rejected` | `integration/test_http_sources.py::TestUppercaseSchemeStillRejectsPrivateHosts_http_scheme_case::test_uppercase_scheme_to_127_literal_rejected[HTTP]` |  |
| `test_http_scheme_case_2321.py::TestUppercaseSchemeStillRejectsPrivateHosts::test_open_source_uppercase_private_host_raises` | `integration/test_http_sources.py::TestUppercaseSchemeStillRejectsPrivateHosts_http_scheme_case::test_open_source_uppercase_private_host_raises` |  |
| `test_http_scheme_case_2321.py::TestWriterRejectsHttpTargets::test_write_bytes_rejects_http` | `integration/test_http_sources.py::TestWriterRejectsHttpTargets_http_scheme_case::test_write_bytes_rejects_http[HTTP://example.com/x.tif]` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_stripped_http_fetches_only_intersecting_strips` | `integration/test_http_sources.py::test_windowed_stripped_http_fetches_only_intersecting_strips` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_max_pixels_honoured_for_stripped_http_read` | `integration/test_http_sources.py::test_windowed_max_pixels_honoured_for_stripped_http_read` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_max_pixels_too_small_raises` | `integration/test_http_sources.py::test_windowed_max_pixels_too_small_raises` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_full_stripped_http_read_honours_caller_max_pixels` | `integration/test_http_sources.py::test_full_stripped_http_read_honours_caller_max_pixels` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_stripped_http_matches_full_read` | `integration/test_http_sources.py::test_windowed_stripped_http_matches_full_read[window2]` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_strip_byte_cap_skips_unrelated_oversized_strip` | `integration/test_http_sources.py::test_windowed_strip_byte_cap_skips_unrelated_oversized_strip` |  |
| `test_http_stripped_window_max_pixels_issue_A_1842.py::test_windowed_strip_decoded_dim_guard_rejects_oversized_strip` | `integration/test_http_sources.py::test_windowed_strip_decoded_dim_guard_rejects_oversized_strip` |  |
| `test_http_window_band_planar_1669.py::test_http_window_parity_single_band` | `integration/test_http_sources.py::test_http_window_parity_single_band` |  |
| `test_http_window_band_planar_1669.py::test_http_window_parity_full_tile_aligned` | `integration/test_http_sources.py::test_http_window_parity_full_tile_aligned` |  |
| `test_http_window_band_planar_1669.py::test_http_window_via_read_to_array_low_level` | `integration/test_http_sources.py::test_http_window_via_read_to_array_low_level` |  |
| `test_http_window_band_planar_1669.py::test_http_window_via_low_level_read_cog_http` | `integration/test_http_sources.py::test_http_window_via_low_level_read_cog_http` |  |
| `test_http_window_band_planar_1669.py::test_http_window_out_of_bounds_rejected` | `integration/test_http_sources.py::test_http_window_out_of_bounds_rejected` |  |
| `test_http_window_band_planar_1669.py::test_http_band_parity_multi_band` | `integration/test_http_sources.py::test_http_band_parity_multi_band` |  |
| `test_http_window_band_planar_1669.py::test_http_band_parity_via_read_to_array` | `integration/test_http_sources.py::test_http_band_parity_via_read_to_array` |  |
| `test_http_window_band_planar_1669.py::test_http_window_and_band_combined` | `integration/test_http_sources.py::test_http_window_and_band_combined` |  |
| `test_http_window_band_planar_1669.py::test_http_planar2_full_read` | `integration/test_http_sources.py::test_http_planar2_full_read` |  |
| `test_http_window_band_planar_1669.py::test_http_planar2_windowed` | `integration/test_http_sources.py::test_http_planar2_windowed` |  |
| `test_http_window_band_planar_1669.py::test_http_planar2_band_selection` | `integration/test_http_sources.py::test_http_planar2_band_selection` |  |
| `test_http_window_band_planar_1669.py::test_http_window_on_oriented_tiff_rejected` | `integration/test_http_sources.py::test_http_window_on_oriented_tiff_rejected` |  |
| `test_cog_http_close_on_error_1816.py::test_http_source_closed_on_success` | `integration/test_http_sources.py::test_http_source_closed_on_success` |  |
| `test_cog_http_close_on_error_1816.py::test_http_source_closed_when_tile_fetch_raises` | `integration/test_http_sources.py::test_http_source_closed_when_tile_fetch_raises` |  |
| `test_cog_http_close_on_error_1816.py::test_http_source_closed_when_post_processing_raises` | `integration/test_http_sources.py::test_http_source_closed_when_post_processing_raises` |  |
| `test_cog_http_concurrent.py::test_read_ranges_returns_results_in_input_order` | `integration/test_http_sources.py::test_read_ranges_returns_results_in_input_order` |  |
| `test_cog_http_concurrent.py::test_read_ranges_empty_list` | `integration/test_http_sources.py::test_read_ranges_empty_list` |  |
| `test_cog_http_concurrent.py::test_read_ranges_single_request_skips_pool` | `integration/test_http_sources.py::test_read_ranges_single_request_skips_pool` |  |
| `test_cog_http_concurrent.py::test_read_ranges_dispatches_concurrently` | `integration/test_http_sources.py::test_read_ranges_dispatches_concurrently` |  |
| `test_cog_http_concurrent.py::test_cog_http_round_trip_matches_local_read` | `integration/test_http_sources.py::test_cog_http_round_trip_matches_local_read` |  |
| `test_cog_http_concurrent.py::test_read_to_array_dispatches_to_http` | `integration/test_http_sources.py::test_read_to_array_dispatches_to_http` |  |
| `test_cog_http_parallel_decode_2026_05_15.py::test_parallel_decode_matches_reference` | `integration/test_http_sources.py::test_parallel_decode_matches_reference` |  |
| `test_cog_http_parallel_decode_2026_05_15.py::test_serial_decode_matches_reference` | `integration/test_http_sources.py::test_serial_decode_matches_reference` |  |
| `test_cog_http_parallel_decode_2026_05_15.py::test_parallel_pool_used_above_threshold` | `integration/test_http_sources.py::test_parallel_pool_used_above_threshold` |  |
| `test_cog_http_parallel_decode_2026_05_15.py::test_serial_path_below_threshold` | `integration/test_http_sources.py::test_serial_path_below_threshold` |  |
| `test_cog_http_parallel_decode_2026_05_15.py::test_each_tile_decoded_once` | `integration/test_http_sources.py::test_each_tile_decoded_once` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_sentinel_returns_default` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_sentinel_returns_default` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_none_disables_check` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_none_disables_check` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_int_kwarg_wins` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_int_kwarg_wins` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_env_override` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_env_override` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_kwarg_overrides_env` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_kwarg_overrides_env` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_invalid_env_falls_back_to_default` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_invalid_env_falls_back_to_default` |  |
| `test_cloud_read_byte_limit_1928.py::TestResolveMaxCloudBytes::test_zero_or_negative_env_falls_back` | `integration/test_http_sources.py::TestResolveMaxCloudBytes_cloud_read_byte_limit::test_zero_or_negative_env_falls_back` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_small_cloud_object_under_budget_reads` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_small_cloud_object_under_budget_reads` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_oversized_cloud_object_rejected_before_read` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_oversized_cloud_object_rejected_before_read` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_none_disables_limit` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_none_disables_limit` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_env_var_threshold_applied` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_env_var_threshold_applied` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_open_geotiff_plumbs_max_cloud_bytes` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_open_geotiff_plumbs_max_cloud_bytes` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_local_file_unaffected` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_local_file_unaffected` |  |
| `test_cloud_read_byte_limit_1928.py::TestCloudByteLimit::test_http_path_unaffected` | `integration/test_http_sources.py::TestCloudByteLimit_cloud_read_byte_limit::test_http_path_unaffected` |  |

## Dask pipeline + accessor -> `integration/test_dask_pipeline.py`

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_dask_chunk_tile_misalignment.py::test_chunk_smaller_than_tile` | `integration/test_dask_pipeline.py::test_chunk_smaller_than_tile` |  |
| `test_dask_chunk_tile_misalignment.py::test_chunk_larger_than_tile_nonmultiple` | `integration/test_dask_pipeline.py::test_chunk_larger_than_tile_nonmultiple` |  |
| `test_dask_chunk_tile_misalignment.py::test_chunk_tuple_doubly_unaligned` | `integration/test_dask_pipeline.py::test_chunk_tuple_doubly_unaligned` |  |
| `test_dask_int_nodata_chunks_1597.py::test_eager_promotes_to_float64_and_masks` | `integration/test_dask_pipeline.py::test_eager_promotes_to_float64_and_masks` |  |
| `test_dask_int_nodata_chunks_1597.py::test_dask_chunks_4_matches_eager` | `integration/test_dask_pipeline.py::test_dask_chunks_4_matches_eager` |  |
| `test_dask_int_nodata_chunks_1597.py::test_dask_chunks_2_per_chunk_dtype_uniform` | `integration/test_dask_pipeline.py::test_dask_chunks_2_per_chunk_dtype_uniform` |  |
| `test_dask_int_nodata_chunks_1597.py::test_dask_keeps_dtype_for_out_of_range_sentinel` | `integration/test_dask_pipeline.py::test_dask_keeps_dtype_for_out_of_range_sentinel` |  |
| `test_dask_int_nodata_chunks_1597.py::test_dask_float_input_with_sentinel_in_one_chunk` | `integration/test_dask_pipeline.py::test_dask_float_input_with_sentinel_in_one_chunk` |  |
| `test_dask_max_pixels_default_guard_1838.py::test_default_max_pixels_guard_fires_for_full_region` | `integration/test_dask_pipeline.py::test_default_max_pixels_guard_fires_for_full_region` |  |
| `test_dask_max_pixels_default_guard_1838.py::test_explicit_max_pixels_still_enforced` | `integration/test_dask_pipeline.py::test_explicit_max_pixels_still_enforced` |  |
| `test_dask_max_pixels_default_guard_1838.py::test_small_region_unaffected` | `integration/test_dask_pipeline.py::test_small_region_unaffected` |  |
| `test_dask_no_op_astype_1624.py::test_uint16_mask_path_still_promotes` | `integration/test_dask_pipeline.py::test_uint16_mask_path_still_promotes` |  |
| `test_dask_no_op_astype_1624.py::test_astype_skipped_when_dtypes_match` | `integration/test_dask_pipeline.py::test_astype_skipped_when_dtypes_match` |  |
| `test_dask_no_op_astype_1624.py::test_caller_supplied_dtype_still_casts` | `integration/test_dask_pipeline.py::test_caller_supplied_dtype_still_casts` |  |
| `test_dask_overview_level.py::test_dask_overview_level_zero_matches_full_res` | `integration/test_dask_pipeline.py::test_dask_overview_level_zero_matches_full_res` |  |
| `test_dask_overview_level.py::test_dask_overview_level_one_returns_half_res` | `integration/test_dask_pipeline.py::test_dask_overview_level_one_returns_half_res` |  |
| `test_dask_overview_level.py::test_dask_overview_level_two_returns_quarter_res` | `integration/test_dask_pipeline.py::test_dask_overview_level_two_returns_quarter_res` |  |
| `test_dask_overview_level.py::test_dask_overview_level_none_returns_full_res` | `integration/test_dask_pipeline.py::test_dask_overview_level_none_returns_full_res` |  |
| `test_dask_planar_multiband.py::test_dask_planar_multiband_matches_numpy` | `integration/test_dask_pipeline.py::test_dask_planar_multiband_matches_numpy[uint8-4-False-separate]` |  |
| `test_dask_planar_multiband.py::test_dask_planar_separate_chunks_tuple` | `integration/test_dask_pipeline.py::test_dask_planar_separate_chunks_tuple` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1x1::test_1x1_chunk_matches_shape` | `integration/test_dask_pipeline.py::TestStreamingWrite1x1_dask_streaming_write_degenerate::test_1x1_chunk_matches_shape` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1x1::test_1x1_with_nodata_attr` | `integration/test_dask_pipeline.py::TestStreamingWrite1x1_dask_streaming_write_degenerate::test_1x1_with_nodata_attr` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1x1::test_1x1_uint16` | `integration/test_dask_pipeline.py::TestStreamingWrite1x1_dask_streaming_write_degenerate::test_1x1_uint16` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1xN::test_1xN_single_chunk` | `integration/test_dask_pipeline.py::TestStreamingWrite1xN_dask_streaming_write_degenerate::test_1xN_single_chunk` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1xN::test_1xN_chunks_split_columns` | `integration/test_dask_pipeline.py::TestStreamingWrite1xN_dask_streaming_write_degenerate::test_1xN_chunks_split_columns` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWrite1xN::test_1xN_wide_segmented_by_buffer` | `integration/test_dask_pipeline.py::TestStreamingWrite1xN_dask_streaming_write_degenerate::test_1xN_wide_segmented_by_buffer` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteNx1::test_Nx1_single_chunk` | `integration/test_dask_pipeline.py::TestStreamingWriteNx1_dask_streaming_write_degenerate::test_Nx1_single_chunk` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteNx1::test_Nx1_chunks_split_rows` | `integration/test_dask_pipeline.py::TestStreamingWriteNx1_dask_streaming_write_degenerate::test_Nx1_chunks_split_rows` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteAllNan::test_all_nan_with_sentinel` | `integration/test_dask_pipeline.py::TestStreamingWriteAllNan_dask_streaming_write_degenerate::test_all_nan_with_sentinel` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteAllNan::test_all_nan_default_nodata` | `integration/test_dask_pipeline.py::TestStreamingWriteAllNan_dask_streaming_write_degenerate::test_all_nan_default_nodata` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteMixedNanInf::test_mixed_nan_plus_minus_inf` | `integration/test_dask_pipeline.py::TestStreamingWriteMixedNanInf_dask_streaming_write_degenerate::test_mixed_nan_plus_minus_inf` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteAllInf::test_all_plus_inf` | `integration/test_dask_pipeline.py::TestStreamingWriteAllInf_dask_streaming_write_degenerate::test_all_plus_inf` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteAllInf::test_all_minus_inf` | `integration/test_dask_pipeline.py::TestStreamingWriteAllInf_dask_streaming_write_degenerate::test_all_minus_inf` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteFloatPredictor::test_predictor3_float32_round_trip` | `integration/test_dask_pipeline.py::TestStreamingWriteFloatPredictor_dask_streaming_write_degenerate::test_predictor3_float32_round_trip` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteFloatPredictor::test_predictor3_float64_round_trip` | `integration/test_dask_pipeline.py::TestStreamingWriteFloatPredictor_dask_streaming_write_degenerate::test_predictor3_float64_round_trip` |  |
| `test_dask_streaming_write_degenerate_2026_05_15.py::TestStreamingWriteFloatPredictor::test_predictor3_int_input_rejected` | `integration/test_dask_pipeline.py::TestStreamingWriteFloatPredictor_dask_streaming_write_degenerate::test_predictor3_int_input_rejected` |  |
| `test_accessor_io.py::TestDataArrayToGeotiff::test_round_trip` | `integration/test_dask_pipeline.py::TestDataArrayToGeotiff_accessor_io::test_round_trip` |  |
| `test_accessor_io.py::TestDataArrayToGeotiff::test_with_kwargs` | `integration/test_dask_pipeline.py::TestDataArrayToGeotiff_accessor_io::test_with_kwargs` |  |
| `test_accessor_io.py::TestDataArrayToGeotiff::test_preserves_crs` | `integration/test_dask_pipeline.py::TestDataArrayToGeotiff_accessor_io::test_preserves_crs` |  |
| `test_accessor_io.py::TestDatasetToGeotiff::test_round_trip` | `integration/test_dask_pipeline.py::TestDatasetToGeotiff_accessor_io::test_round_trip` |  |
| `test_accessor_io.py::TestDatasetToGeotiff::test_explicit_var` | `integration/test_dask_pipeline.py::TestDatasetToGeotiff_accessor_io::test_explicit_var` |  |
| `test_accessor_io.py::TestDatasetToGeotiff::test_no_yx_raises` | `integration/test_dask_pipeline.py::TestDatasetToGeotiff_accessor_io::test_no_yx_raises` |  |
| `test_accessor_io.py::TestDatasetOpenGeotiff::test_windowed_read` | `integration/test_dask_pipeline.py::TestDatasetOpenGeotiff_accessor_io::test_windowed_read` |  |
| `test_accessor_io.py::TestDatasetOpenGeotiff::test_full_extent_returns_all` | `integration/test_dask_pipeline.py::TestDatasetOpenGeotiff_accessor_io::test_full_extent_returns_all` |  |
| `test_accessor_io.py::TestDatasetOpenGeotiff::test_no_coords_raises` | `integration/test_dask_pipeline.py::TestDatasetOpenGeotiff_accessor_io::test_no_coords_raises` |  |
| `test_accessor_io.py::TestDatasetOpenGeotiff::test_kwargs_forwarded` | `integration/test_dask_pipeline.py::TestDatasetOpenGeotiff_accessor_io::test_kwargs_forwarded` |  |

## GPU pipeline -> `integration/test_gpu_pipeline.py`

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_dask_cupy_combined.py::test_open_geotiff_gpu_chunks_int_round_trip` | `integration/test_gpu_pipeline.py::test_open_geotiff_gpu_chunks_int_round_trip` |  |
| `test_dask_cupy_combined.py::test_read_geotiff_gpu_chunks_tuple_round_trip` | `integration/test_gpu_pipeline.py::test_read_geotiff_gpu_chunks_tuple_round_trip` |  |
| `test_dask_cupy_combined.py::test_open_geotiff_gpu_chunks_multiband` | `integration/test_gpu_pipeline.py::test_open_geotiff_gpu_chunks_multiband` |  |
| `test_dask_cupy_combined.py::test_open_geotiff_gpu_chunks_partial_last_chunk` | `integration/test_gpu_pipeline.py::test_open_geotiff_gpu_chunks_partial_last_chunk` |  |
| `test_dask_cupy_combined.py::test_open_geotiff_gpu_chunks_preserves_geo_attrs` | `integration/test_gpu_pipeline.py::test_open_geotiff_gpu_chunks_preserves_geo_attrs` |  |
