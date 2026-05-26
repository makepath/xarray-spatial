# Cluster 7, Sub-PR A audit: kwarg / signature -> unit/test_signatures.py

Maps every old `file::test` to its new `file::test_id`. Tests are copied
verbatim except for moving GPU gating to the shared `requires_gpu` marker
from `_helpers/markers.py` (replacing per-file `_gpu_available` /
`_gpu_only` / `from .conftest import gpu_available`) and lifting shared
helpers (`_annotated_smoke_da`) to module scope. No assertion changed.

New file: `xrspatial/geotiff/tests/unit/test_signatures.py` (167 tests,
matches the pre-consolidation total of 167).

## Section 1 -- annotations (#1654, #1705)

### test_signature_annotations_1654.py
- test_open_geotiff_window_annotated -> same id
- test_read_vrt_window_annotated -> same id
- test_read_geotiff_dask_window_annotated -> same id
- test_read_geotiff_gpu_window_annotated -> same id
- test_to_geotiff_path_annotated -> same id
- test_write_geotiff_gpu_path_annotated -> same id
- test_write_vrt_path_annotated -> same id
- test_write_vrt_vrt_path_annotated -> same id
- test_open_geotiff_source_annotated -> same id
- test_read_geotiff_dask_source_str_only -> same id
- test_read_geotiff_gpu_source_str_only -> same id
- test_read_vrt_source_str_only -> same id
- test_open_geotiff_dtype_annotated -> same id
- test_read_geotiff_dask_dtype_annotated -> same id
- test_read_geotiff_gpu_dtype_annotated -> same id
- test_read_vrt_dtype_annotated -> same id
- test_open_geotiff_on_gpu_failure_annotated -> same id
- test_read_geotiff_gpu_on_gpu_failure_annotated -> same id
- test_read_geotiff_gpu_deprecated_gpu_alias_annotated -> same id
- test_open_geotiff_window_kwarg_runtime -> same id (uses module `_annotated_smoke_da`)
- test_open_geotiff_bytesio_source_runtime -> same id (uses module `_annotated_smoke_da`)
- test_open_geotiff_dtype_kwarg_runtime -> same id (uses module `_annotated_smoke_da`)

### test_signature_annotations_1705.py
- test_to_geotiff_nodata_annotated -> same id
- test_write_geotiff_gpu_nodata_annotated -> same id
- test_write_vrt_nodata_annotated -> same id
- test_to_geotiff_streaming_buffer_bytes_annotated -> same id
- test_write_geotiff_gpu_streaming_buffer_bytes_annotated -> same id
- test_to_geotiff_nodata_int_runtime -> same id
- test_write_geotiff_gpu_streaming_buffer_bytes_runtime_noop -> same id
  (GPU gate now `@requires_gpu` instead of `from .conftest import gpu_available`)

## Section 2 -- canonical reader kwarg order (#1935)

### test_reader_kwarg_order_1935.py
- module constant `_CANONICAL_ORDER` -> same constant
- _kwonly_params / _assert_canonical -> same helpers
- test_open_geotiff_defines_canonical_order -> same id
- test_read_geotiff_gpu_matches_canonical_order -> same id
- test_read_geotiff_dask_matches_canonical_order -> same id
- test_read_vrt_matches_canonical_order -> same id
- test_no_pairwise_order_inversions -> same id

## Section 3 -- experimental / internal-only opt-in (#2352)

### test_experimental_internal_optin_2352.py
- helpers `_make_float32_da`, `_write_test_tif` -> same helpers
- test_read_signature_has_codec_optin (parametrised fn) -> same id
- test_validate_read_codec_optin_accepts_stable_codecs -> same id
- test_validate_read_codec_optin_rejects_experimental (parametrised codec_name) -> same id
- test_validate_read_codec_optin_rejects_jpeg -> same id
- test_validate_read_codec_optin_accepts_jpeg_with_flag -> same id
- test_validate_read_codec_optin_accepts_experimental_with_flag (parametrised) -> same id
- test_validate_read_codec_optin_message_names_feature_and_tier -> same id
- test_validate_write_rich_tag_optin_accepts_empty_attrs -> same id
- test_validate_write_rich_tag_optin_rejects_gdal_metadata_xml -> same id
- test_validate_write_rich_tag_optin_rejects_extra_tags -> same id
- test_validate_write_rich_tag_optin_accepts_with_flag -> same id
- test_validate_write_rich_tag_optin_exempts_round_trip -> same id
- test_open_geotiff_rejects_experimental_codec (parametrised codec) -> same id
- test_open_geotiff_accepts_experimental_codec_with_flag (parametrised) -> same id
- test_open_geotiff_rejects_jpeg2000 -> same id
- test_open_geotiff_rejects_jpeg_internal_only -> same id
- test_open_geotiff_accepts_jpeg_internal_only_with_flag -> same id
- test_read_geotiff_dask_rejects_experimental_codec -> same id
- test_read_geotiff_dask_accepts_experimental_codec_with_flag -> same id
- test_to_geotiff_rejects_gdal_metadata_xml_without_flag -> same id
- test_to_geotiff_rejects_extra_tags_without_flag -> same id
- test_to_geotiff_accepts_rich_tags_with_flag -> same id
- test_write_geotiff_gpu_rejects_rich_tags_without_flag -> same id
- test_allow_rotated_default_raises_already_gated -> same id
  (dropped the unused `tmp_path` arg -- the body is a signature pin only)
- test_allow_unparseable_crs_default_raises_already_gated -> same id
- test_gpu_read_requires_explicit_optin -> same id
- test_gpu_write_requires_explicit_optin -> same id

## Section 4 -- photometric kwarg + extra_tags override (#1769)

### test_photometric_kwarg_1769.py
- helpers `_read_primary_ifd`, `_to_da` -> same helpers
- test_four_band_default_is_minisblack_with_unspecified_extras -> same id
- test_four_band_photometric_rgba_writes_rgb_plus_alpha -> same id
- test_four_band_photometric_rgb_writes_unspecified_extras -> same id
- test_three_band_default_is_minisblack_regression_1769 -> same id
- test_single_band_default_unchanged_1769 -> same id
- test_user_extra_tags_override_extra_samples_1769 -> same id
- test_user_extra_tags_override_photometric_1769 -> same id
- test_explicit_integer_photometric_1769 -> same id
- test_invalid_photometric_name_raises_1769 -> same id
- test_rgba_requires_four_bands_1769 -> same id
- test_rgb_requires_three_bands_1769 -> same id
- test_explicit_int_rgb_requires_three_bands_1769 -> same id
- test_dask_streaming_default_is_minisblack_1769 -> same id
- test_cog_overviews_carry_same_photometric_1769 -> same id

## Section 5 -- gil_friendly deflate kwarg (#1830)

### test_gil_friendly_kwarg_1830.py
- helper `_payload`, class `_DeflateCallSpy` -> same
- test_deflate_compress_gil_friendly_true_bypasses_libdeflate -> same id
- test_deflate_compress_gil_friendly_false_uses_libdeflate -> same id
- test_deflate_compress_gil_friendly_round_trip_both_directions -> same id
- test_deflate_compress_fallback_warning_fires_when_libdeflate_missing -> same id
- test_deflate_compress_fallback_warning_is_one_shot -> same id
- test_deflate_compress_fallback_no_warning_when_latch_set -> same id
- test_compress_forwards_gil_friendly_to_deflate -> same id
- test_compress_gil_friendly_ignored_for_non_deflate_codecs -> same id
- test_compress_default_gil_friendly_is_false -> same id
- test_write_stripped_parallel_path_uses_gil_friendly -> same id
- test_write_stripped_sequential_path_uses_default -> same id
- test_write_tiled_parallel_path_uses_gil_friendly -> same id
- test_write_tiled_sequential_path_uses_default -> same id
- test_prepare_strip_forwards_gil_friendly -> same id
- test_prepare_tile_forwards_gil_friendly -> same id
- test_write_tiled_parallel_passes_gil_friendly_positionally -> same id
  (module-level `import inspect` reused; in-body `import inspect` dropped)
- test_compress_block_forwards_gil_friendly_true -> same id
- test_compress_block_default_gil_friendly_is_false -> same id
- test_write_streaming_parallel_segment_uses_gil_friendly -> same id
- test_write_deflate_round_trip_across_parallelism_modes (parametrised) -> same id

## Section 6 -- reader / writer kwarg behaviour (2026-05-12 sweep)

### test_kwarg_coverage_2026_05_11_r4.py (6b: name / max_pixels)
- fixture `small_tiff_path` -> same fixture
- test_read_geotiff_dask_name_kwarg_sets_name -> same id
- test_read_geotiff_dask_default_name_from_path -> same id
- test_read_geotiff_gpu_name_kwarg_sets_name -> same id (`@requires_gpu`)
- test_read_geotiff_gpu_default_name_from_path -> same id (`@requires_gpu`)
- test_read_geotiff_gpu_chunks_name_kwarg_sets_name -> same id (`@requires_gpu`)
- test_read_geotiff_gpu_max_pixels_accepts_within_budget -> same id (`@requires_gpu`)
- test_read_geotiff_gpu_max_pixels_rejects_oversized -> same id (`@requires_gpu`)
- test_read_geotiff_gpu_chunks_max_pixels_rejects_oversized -> same id (`@requires_gpu`)
- test_open_geotiff_chunks_name_flows_through -> same id
- test_open_geotiff_gpu_name_flows_through -> same id (`@requires_gpu`)
- test_open_geotiff_gpu_chunks_name_flows_through -> same id (`@requires_gpu`)
- test_open_geotiff_gpu_max_pixels_rejects -> same id (`@requires_gpu`)

### test_kwarg_behaviour_2026_05_12.py (6a write_vrt + 6b dtype/bigtiff)
- fixtures `source_tif`, `float64_tif`, `uint16_tif` -> same fixtures
- TestWriteVrtRelativeBehaviour::test_relative_true_writes_relative_path -> same id
- TestWriteVrtRelativeBehaviour::test_relative_false_writes_absolute_path -> same id
- TestWriteVrtRelativeBehaviour::test_relative_true_parses_back_to_same_source -> same id
- TestWriteVrtRelativeBehaviour::test_relative_false_parses_back_to_same_source -> same id
- TestWriteVrtCrsWktBehaviour::test_crs_wkt_override_wins -> same id
- TestWriteVrtCrsWktBehaviour::test_crs_wkt_none_falls_back_to_first_source -> same id
- TestWriteVrtCrsWktBehaviour::test_crs_wkt_override_distinct_from_default -> same id
- TestWriteVrtNodataBehaviour::test_nodata_override_wins -> same id
- TestWriteVrtNodataBehaviour::test_nodata_none_takes_first_source -> same id
- TestWriteVrtNodataBehaviour::test_nodata_override_writes_xml_element -> same id
- TestWriteVrtEmptySourceFiles::test_empty_list_raises -> same id
- TestWriteVrtEmptySourceFiles::test_empty_list_does_not_create_file -> same id
- TestReadGeotiffGpuDtype::test_* (7 tests) -> same ids (`@requires_gpu`)
- TestOpenGeotiffGpuDispatchDtype::test_* (2 tests) -> same ids (`@requires_gpu`)
- TestReadGeotiffGpuChunksDtype::test_chunks_float64_to_float32 -> same id (`@requires_gpu`)
- TestWriteGeotiffGpuBigtiff::test_* (4 tests) -> same ids (`@requires_gpu`)
  (in-body `parse_header` now from module import as `parse_header`)

### test_kwarg_behaviour_2026_05_12_v2.py (6c predictor + read_vrt window)
- helpers `_read_predictor_tag`, `_da_with_float_coords`,
  `_write_tile_to_vrt`, `_make_single_tile_vrt`, `_make_2x1_mosaic_vrt`
  -> same helpers (`_write_tile_to_vrt` uses the module-level `write`
  import rather than an in-body import)
- TestWriteGeotiffGpuPredictor2Uint8::test_* (4 tests) -> same ids (`@requires_gpu`)
- TestWriteGeotiffGpuPredictor2Uint16::test_predictor_2_uint16_round_trip -> same id
- TestWriteGeotiffGpuPredictor2Int32::test_predictor_2_int32_round_trip -> same id
- TestWriteGeotiffGpuPredictor3Float::test_* (3 tests) -> same ids
- TestWriteGeotiffGpuPredictorCpuParity::test_* (2 tests) -> same ids
- TestReadVrtWindowEager::test_* (9 tests) -> same ids
- TestReadVrtWindowWithBand::test_window_plus_band_selection -> same id
- TestReadVrtWindowDask::test_window_chunks_returns_dask -> same id
- TestReadVrtWindowGpu::test_* (2 tests) -> same ids (`@requires_gpu`)

## Notes

- `test_experimental_internal_optin_2352.py` overlaps conceptually with
  the `allow_internal_only_jpeg` signature pin already in
  `unit/test_photometric.py` Section 2 (from PR #2451), but the two do
  not duplicate: photometric.py pins only the one writer signature, while
  this file's Section 3 covers the read-side codec gate, the writer
  rich-tag gate, validator unit tests, and the full opt-in inventory. No
  test was dropped or merged across the two files.
- HARD GATE per epic #2424: this audit file is deleted in a final
  pre-merge commit on this branch.
