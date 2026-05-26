# CLUSTER_AUDIT_PR10 — Release-gate registry

This audit table maps every test currently living in a
`test_release_gate_*.py` file under `xrspatial/geotiff/tests/` to its new
home inside the single consolidated registry,
`release_gates/test_stable_features.py`. Deleted before merge per the
epic protocol (see `xarray-contrib/xarray-spatial#2390`).

## Inputs

13 source files, 159 tests collected, 134 of those previously carried
`@pytest.mark.release_gate`. The remaining 25 lived in
`test_release_gate_*.py` files but did not carry the marker; the epic
specifies all such tests fold in and pick up the marker.

## Mapping

| Old file:test | New `release_gates/test_stable_features.py::test_id` | Notes |
|---|---|---|
| `test_release_gate_local_read.py::test_release_gate_local_read_pixels` | `test_release_gate_local_read_pixels` | unchanged |
| `test_release_gate_local_read.py::test_release_gate_local_read_crs` | `test_release_gate_local_read_crs` | unchanged |
| `test_release_gate_local_read.py::test_release_gate_local_read_transform` | `test_release_gate_local_read_transform` | unchanged |
| `test_release_gate_local_read.py::test_release_gate_local_read_nodata` | `test_release_gate_local_read_nodata` | unchanged |
| `test_release_gate_local_write.py::test_release_gate_local_write_round_trips_pixels` | `test_release_gate_local_write_round_trips_pixels` | unchanged |
| `test_release_gate_local_write.py::test_release_gate_local_write_preserves_crs` | `test_release_gate_local_write_preserves_crs` | unchanged |
| `test_release_gate_local_write.py::test_release_gate_local_write_preserves_transform` | `test_release_gate_local_write_preserves_transform` | unchanged |
| `test_release_gate_local_write.py::test_release_gate_local_write_preserves_nodata` | `test_release_gate_local_write_preserves_nodata` | unchanged |
| `test_release_gate_codecs.py::test_release_gate_codec_round_trip_uint16[codec]` (5) | `test_release_gate_codec_round_trip_uint16[codec]` | parametrized over `STABLE_LOSSLESS_CODECS` |
| `test_release_gate_codecs.py::test_release_gate_codec_round_trip_float32[codec]` (5) | `test_release_gate_codec_round_trip_float32[codec]` | parametrized over `STABLE_LOSSLESS_CODECS` |
| `test_release_gate_codecs.py::test_release_gate_codec_stable_set_matches_supported_features` | `test_release_gate_codec_stable_set_matches_supported_features` | unchanged |
| `test_release_gate_cog.py::test_release_gate_cog_round_trips_pixels[codec]` (5) | `test_release_gate_cog_round_trips_pixels[codec]` | shared `STABLE_LOSSLESS_CODECS` constant, no cross-file import |
| `test_release_gate_cog.py::test_release_gate_cog_preserves_crs_transform[codec]` (5) | `test_release_gate_cog_preserves_crs_transform[codec]` | unchanged |
| `test_release_gate_cog.py::test_release_gate_cog_preserves_nodata[codec]` (5) | `test_release_gate_cog_preserves_nodata[codec]` | unchanged |
| `test_release_gate_windowed_read.py::test_release_gate_windowed_read_returns_subset` | `test_release_gate_windowed_read_returns_subset` | unchanged |
| `test_release_gate_windowed_read.py::test_release_gate_windowed_read_preserves_crs` | `test_release_gate_windowed_read_preserves_crs` | unchanged |
| `test_release_gate_windowed_read.py::test_release_gate_windowed_read_shifts_transform_origin` | `test_release_gate_windowed_read_shifts_transform_origin` | unchanged |
| `test_release_gate_windowed_read.py::test_release_gate_windowed_read_full_extent_matches_unwindowed` | `test_release_gate_windowed_read_full_extent_matches_unwindowed` | unchanged |
| `test_release_gate_dask_parity.py::test_release_gate_dask_read_matches_eager_pixels` | `test_release_gate_dask_read_matches_eager_pixels` | unchanged |
| `test_release_gate_dask_parity.py::test_release_gate_dask_read_matches_eager_attrs` | `test_release_gate_dask_read_matches_eager_attrs` | unchanged |
| `test_release_gate_dask_parity.py::test_release_gate_dask_read_is_lazy` | `test_release_gate_dask_read_is_lazy` | unchanged |
| `test_release_gate_eager_dask_parity_2341.py::test_release_gate_eager_dask_full_parity[fixture]` (4) | `test_release_gate_eager_dask_full_parity[fixture]` | corpus list preserved |
| `test_release_gate_eager_dask_parity_2341.py::test_release_gate_corpus_is_non_empty` | `test_release_gate_corpus_is_non_empty` | now carries `@pytest.mark.release_gate` (previously unmarked despite living in a `test_release_gate_*.py` file) |
| `test_release_gate_attrs_contract.py::test_release_gate_attrs_canonical_keys_present` | `test_release_gate_attrs_canonical_keys_present` | unchanged |
| `test_release_gate_attrs_contract.py::test_release_gate_attrs_georef_status_full` | `test_release_gate_attrs_georef_status_full` | unchanged |
| `test_release_gate_attrs_contract.py::test_release_gate_attrs_contract_version_is_int` | `test_release_gate_attrs_contract_version_is_int` | unchanged |
| `test_release_gate_attrs_contract.py::test_release_gate_attrs_round_trip_preserves_crs_transform_nodata` | `test_release_gate_attrs_round_trip_preserves_crs_transform_nodata` | unchanged |
| `test_release_gate_codec_round_trip_2341.py::test_release_gate_codec_round_trip[codec-dtype]` (20) | `test_release_gate_codec_round_trip[codec-dtype]` | unchanged |
| `test_release_gate_codec_round_trip_2341.py::test_release_gate_codec_round_trip_stable_set_matches_supported_features` | `test_release_gate_codec_round_trip_stable_set_matches_supported_features` | unchanged |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_cog_internal_overview_metadata_survives[reader]` (2) | `test_release_gate_cog_internal_overview_metadata_survives[reader]` | renamed for the `release_gate_` test-name prefix; now carries `@pytest.mark.release_gate` |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_cog_internal_overview_transform_scales[reader]` (2) | `test_release_gate_cog_internal_overview_transform_scales[reader]` | renamed; marker added |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_cog_internal_overview_shape_matches_factors` | `test_release_gate_cog_internal_overview_shape_matches_factors` | renamed; marker added |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_sidecar_overview_metadata_survives[reader]` (2) | `test_release_gate_sidecar_overview_metadata_survives[reader]` | renamed; marker added |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_sidecar_overview_transform_scales[reader]` (2) | `test_release_gate_sidecar_overview_transform_scales[reader]` | renamed; marker added |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_sidecar_overview_shape_matches_factors` | `test_release_gate_sidecar_overview_shape_matches_factors` | renamed; marker added |
| `test_release_gate_overview_sidecar_metadata_2341.py::test_internal_vs_sidecar_metadata_agree[reader]` (2) | `test_release_gate_internal_vs_sidecar_metadata_agree[reader]` | renamed; marker added |
| `test_release_gate_windowed_reads_2341.py::test_release_gate_windowed_read_shape[r-corpus-window]` (16) | `test_release_gate_windowed_read_shape[r-corpus-window]` | corpus fixture renamed to `_wsp_corpus_file`; same IDs |
| `test_release_gate_windowed_reads_2341.py::test_release_gate_windowed_read_coords_slice[r-corpus-window]` (16) | `test_release_gate_windowed_read_coords_slice[r-corpus-window]` | same |
| `test_release_gate_windowed_reads_2341.py::test_release_gate_windowed_read_transform_shifted[r-corpus-window]` (16) | `test_release_gate_windowed_read_transform_shifted[r-corpus-window]` | same |
| `test_release_gate_windowed_reads_2341.py::test_release_gate_windowed_read_canonical_attrs_unchanged[r-corpus-window]` (16) | `test_release_gate_windowed_read_canonical_attrs_unchanged[r-corpus-window]` | same |
| `test_release_gate_negative_2341.py::test_release_gate_negative_conflicting_aux_xml_crs` | `test_release_gate_negative_conflicting_aux_xml_crs` | unchanged; remains `xfail strict=False` |
| `test_release_gate_negative_2341.py::test_release_gate_negative_integer_nodata_float_promoted` | `test_release_gate_negative_integer_nodata_float_promoted` | unchanged |
| `test_release_gate_negative_2341.py::test_release_gate_negative_rotated_eager` | `test_release_gate_negative_rotated_eager` | now carries `@pytest.mark.release_gate` (previously unmarked) |
| `test_release_gate_negative_2341.py::test_release_gate_negative_rotated_dask` | `test_release_gate_negative_rotated_dask` | marker added |
| `test_release_gate_negative_2341.py::test_release_gate_negative_rotated_windowed` | `test_release_gate_negative_rotated_windowed` | marker added |
| `test_release_gate_negative_2341.py::test_release_gate_negative_rotated_gpu` | `test_release_gate_negative_rotated_gpu` | marker added; `requires_gpu` imported from `_helpers.markers` instead of the slim conftest re-export |
| `test_release_gate_negative_2341.py::test_release_gate_negative_mixed_tier_vrt_children` | `test_release_gate_negative_mixed_tier_vrt_children` | unchanged |
| `test_release_gate_2321.py::test_release_gate_cites_only_existing_test_files` | `test_release_gate_cites_only_existing_test_files` | now carries `@pytest.mark.release_gate`; self-reference path updated to `release_gates/test_stable_features.py` |
| `test_release_gate_2321.py::test_release_gate_lists_every_promised_supported_feature` | `test_release_gate_lists_every_promised_supported_feature` | marker added |
| `test_release_gate_2321.py::test_release_gate_http_ssrf_rejects_loopback` | `test_release_gate_http_ssrf_rejects_loopback` | marker added |
| `test_release_gate_2321.py::test_release_gate_http_ssrf_rejects_loopback_uppercase_scheme` | `test_release_gate_http_ssrf_rejects_loopback_uppercase_scheme` | marker added; xfail kept |
| `test_release_gate_2321.py::test_release_gate_vrt_rows_point_at_real_test_functions` | `test_release_gate_vrt_rows_point_at_real_test_functions` | marker added |

## Helper-function collisions

Two source files defined `_write_known_good` and two defined
`_make_data_array`. Helpers carry section prefixes in the consolidated
file (`_local_read_write_known_good`, `_local_write_make_data_array`,
`_dask_parity_write_known_good`, `_attrs_write_known_good`,
`_cog_make_data_array`, etc.) so the consolidation does not introduce
cross-section coupling.

## Drops / dismissals

None. Every test from every folded file moved. The `release_gate`
marker now covers all 159 tests rather than the previous 134.

## Verification

```
pytest xrspatial/geotiff/tests/release_gates/ -v -m release_gate
# 155 passed, 3 xfailed, 1 xpassed
pytest xrspatial/geotiff/tests/ -m release_gate -v
# same 159 tests selected -- no other file carries the marker now
```

`-m release_gate` from the wider tests root resolves to the single
registry file. Deletion: this file is removed in the final commit
before merge.
