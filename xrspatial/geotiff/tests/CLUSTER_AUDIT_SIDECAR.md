# Cluster 12 audit: sidecar and remote hardening

Maps every old `file::test` to its new `file::test`. Cluster 12 of the
long-tail GeoTIFF test consolidation epic (#2424), issue #2436. Eleven
top-level files fold into two integration targets:

- Sub-PR A (this branch): the six sidecar files -> new
  `integration/test_sidecar.py`.
- Sub-PR B (separate branch): the five remote-hardening files extend
  `integration/test_http_sources.py`.

Tests are copied verbatim except for:

- Per-file module helpers that collided across sources are namespaced by
  source issue (e.g. `_make_dataarray` -> `_make_dataarray_2315`,
  `_start_http_server` -> `_start_http_server_2112` /
  `_start_http_server_2121`, `_start_range_http_server` ->
  `_start_range_http_server_2239` / `_start_range_http_server_2314`,
  `_make_base` -> `_make_base_2416`, the `_http_with_sidecar` /
  `_fsspec_memory_with_sidecar` fixtures gain a `_2239` suffix).
- GPU gating moves to the shared `requires_gpu` marker from
  `_helpers/markers.py`, replacing the per-file `_gpu_available` probe
  and `_gpu_or_skip` fixture.
- The bundled `golden_corpus` fixture path resolves against
  `Path(__file__).resolve().parents[1]` (the `tests` directory) because
  the new file sits one level deeper in `integration/`.
- `_fixture_or_skip` is shared once at module scope rather than redefined
  per source.

No assertion changed.

`test_gpu_sidecar_georef_parity_2324.py` imported `_write_pair` from
`test_sidecar_own_geokeys_2315.py`; its import now points at
`integration/test_sidecar.py`, where `_write_pair` keeps its name.

## Sub-PR A: integration/test_sidecar.py (64 test functions, 83 collected)

### test_sidecar_ovr_2112.py -> Section: sidecar_ovr
- test_find_sidecar_returns_path_for_local_file_with_sidecar -> same
- test_find_sidecar_returns_none_when_sidecar_missing -> same
- test_find_sidecar_returns_none_for_file_like_object -> same
- test_find_sidecar_returns_none_for_remote_uri (parametrised) -> same
- test_load_sidecar_returns_two_ifds -> same
- test_open_geotiff_base_level_unchanged -> same
- test_open_geotiff_sidecar_level_1 -> same
- test_open_geotiff_sidecar_level_2 -> same
- test_open_geotiff_out_of_range_after_sidecar_appended -> same
- test_sidecar_level_1_matches_rasterio -> same
- test_sidecar_level_2_matches_rasterio -> same
- test_read_to_array_sidecar_level_1 -> same
- test_metadata_only_includes_sidecar_levels -> same
- test_missing_sidecar_raises_overview_out_of_range -> same
- test_gpu_eager_reads_sidecar_level_1 -> same (now `@requires_gpu`)
- test_gpu_eager_reads_sidecar_level_2 -> same (now `@requires_gpu`)
- test_gpu_eager_base_level_unchanged -> same (now `@requires_gpu`)
- test_find_sidecar_http_probe_returns_url_when_present -> same
- test_find_sidecar_http_probe_returns_none_when_missing -> same
- test_find_sidecar_http_probe_rejects_loopback_without_env_override -> same
- test_load_sidecar_http_returns_ifds -> same
- test_find_sidecar_fsspec_probe_returns_uri_when_present -> same
- test_file_like_source_reads_base_without_sidecar -> same

### test_sidecar_own_geokeys_2315.py -> Section: sidecar_own_geokeys
- test_ifd_has_georef_payload_true_for_pixel_scale -> same
- test_ifd_has_georef_payload_false_after_strip -> same
- test_sidecar_with_own_geokeys_wins_eager -> same
- test_sidecar_with_own_geokeys_wins_metadata_only -> same
- test_sidecar_without_geokeys_inherits_from_base_eager -> same
- test_sidecar_without_geokeys_inherits_from_base_metadata_only -> same

### test_sidecar_max_cloud_bytes_2121.py -> Section: sidecar_max_cloud_bytes
- test_local_sidecar_ignores_max_cloud_bytes -> same
- test_fsspec_sidecar_rejects_when_exceeds_max_cloud_bytes -> same
- test_fsspec_sidecar_succeeds_when_under_max_cloud_bytes -> same
- test_fsspec_sidecar_max_cloud_bytes_none_is_unbounded -> same
- test_http_sidecar_rejects_when_exceeds_max_cloud_bytes -> same
- test_http_sidecar_succeeds_when_under_max_cloud_bytes -> same
- test_http_sidecar_max_cloud_bytes_none_is_unbounded -> same
- test_read_to_array_propagates_max_cloud_bytes_to_sidecar -> same
- test_env_var_propagates_to_sidecar -> same

### test_sidecar_bad_does_not_break_base_2416.py -> Section: sidecar_bad_does_not_break_base
- test_open_geotiff_base_read_survives_unreadable_sidecar -> same
- test_open_geotiff_overview_level_zero_survives_unreadable_sidecar -> same
- test_read_to_array_base_read_survives_unreadable_sidecar -> same
- test_open_geotiff_base_read_survives_various_sidecar_payloads (parametrised) -> same
- test_open_geotiff_requesting_sidecar_level_still_raises -> same
- test_cloud_size_limit_error_from_sidecar_is_not_silenced -> same
- test_read_geo_info_base_survives_unreadable_sidecar -> same
- test_read_geo_info_cloud_size_limit_error_is_not_silenced -> same
- test_open_geotiff_gpu_base_read_survives_unreadable_sidecar -> same (now `@requires_gpu`)

### test_remote_sidecar_chunked_2239.py -> Section: remote_sidecar_chunked
- test_fsspec_chunked_open_resolves_sidecar_overview (parametrised) -> same
- test_http_chunked_open_resolves_sidecar_overview (parametrised) -> same
- test_http_eager_reads_sidecar_overview -> same
- test_http_eager_vs_local_parity -> same
- test_read_geo_info_fsspec_reports_sidecar_dimensions (parametrised) -> same
- test_fsspec_chunked_open_rejects_overview_past_sidecar -> same
- test_http_chunked_open_rejects_overview_past_sidecar -> same
- test_discover_remote_sidecar_falls_back_when_load_fails -> same
- test_discover_remote_sidecar_propagates_cloud_size_limit -> same
- test_parse_cog_http_meta_requires_source_path_when_return_sidecar -> same
- test_file_like_chunked_open_unaffected_by_sidecar_discovery -> same

### test_remote_sidecar_byte_order_2314.py -> Section: remote_sidecar_byte_order
- test_local_eager_mixed_endian_sidecar (parametrised) -> same
- test_http_eager_mixed_endian_sidecar (parametrised) -> same
- test_http_chunked_mixed_endian_sidecar (parametrised) -> same
- test_fsspec_chunked_mixed_endian_sidecar (parametrised) -> same
- test_http_eager_mixed_endian_sidecar_tiled (parametrised) -> same
- test_parse_cog_http_meta_returns_sidecar_header (parametrised) -> same

## Sub-PR B: integration/test_http_sources.py (extended)

### test_ssrf_hardening_1664.py -> Section: ssrf_hardening
- TestSchemeAllowList.* -> same
- TestPrivateHostBlocking.* -> same
- TestHTTPTimeouts.* -> same
- test_redirect_cap_is_set -> same
- TestRedirectRevalidation.* -> same
- TestHTTPSourceConstructor.* -> same
- test_read_to_array_rejects_file_url -> same
- test_open_source_rejects_loopback_http -> same
- module helpers `_MockPool` / `_MockPoolResponse` -> `_MockPool_ssrf_1664` /
  `_MockPoolResponse_ssrf_1664` (referenced by the #1846 section)

### test_dns_rebinding_pin_issue_1846.py -> Section: dns_rebinding
- TestValidatorReturnsPinnedIP.* -> same
- TestPinnedConnectionTarget.* -> same
- TestRedirectRevalidates.* -> same
- cross-file import of `_MockPool` / `_MockPoolResponse` from the 1664
  file -> in-file `_MockPool_ssrf_1664` / `_MockPoolResponse_ssrf_1664`

### test_uppercase_scheme_ssrf_2323.py -> Section: uppercase_scheme_ssrf
- TestIsHttpUrlCaseInsensitive.* -> same
- TestIsFsspecUriExcludesHttpCaseInsensitive.* -> same
- TestOpenSourceUppercaseDispatch.* -> same
- TestReadToArrayUppercaseDispatch.* -> same
- TestDaskBackendUppercaseDispatch.* -> same

### test_max_cloud_bytes_dispatcher_silent_drop_2026_05_15.py -> Section: max_cloud_bytes_dispatcher
- TestEagerLocalPathAcceptsMaxCloudBytes.* -> same
- TestEagerFileLikeAcceptsMaxCloudBytes.* -> same
- test_dispatcher_gpu_path_rejects_max_cloud_bytes -> same
- test_dispatcher_dask_path_rejects_max_cloud_bytes -> same
- test_dispatcher_vrt_path_rejects_max_cloud_bytes -> same
- test_dispatcher_dask_gpu_path_rejects_max_cloud_bytes -> same
- test_default_kwarg_does_not_trigger_guard_on_gpu_path -> same
- test_default_kwarg_does_not_trigger_guard_on_dask_path -> same
- test_default_kwarg_does_not_trigger_guard_on_vrt_path -> same
- test_explicit_none_max_cloud_bytes_rejected_on_gpu_path -> same
- test_explicit_none_max_cloud_bytes_rejected_on_dask_path -> same
- test_explicit_none_max_cloud_bytes_rejected_on_vrt_path -> same
- helpers `_build_local_tif` / `_build_vrt` -> `_build_local_tif_2026_05_15`
  / `_build_vrt_2026_05_15`; `_skip_if_no_cupy_cuda` -> `@requires_gpu`

### test_open_geotiff_max_cloud_bytes_annot_2106.py -> Section: max_cloud_bytes_annot
- test_open_geotiff_max_cloud_bytes_has_type_annotation -> same
- test_public_entry_point_kwargs_have_type_annotations (parametrised) -> same

## Docs

`docs/source/reference/geotiff.rst` and
`docs/source/reference/release_gate_geotiff.rst` cited the eleven source
filenames; the checklist-parity gate
(`release_gates/test_stable_features.py::test_release_gate_cites_only_existing_test_files`)
enforces that every cited file exists on disk. Both docs now cite the two
consolidated targets.
