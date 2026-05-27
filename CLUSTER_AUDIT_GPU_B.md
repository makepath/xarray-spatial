# Cluster 14 audit: GPU writer consolidation (sub-PR B of #2438)

Cluster 14 of long-tail epic #2424 folds eight GPU-only writer test
files into a single parametrised home at
`xrspatial/geotiff/tests/gpu/test_writer.py`. This audit maps every
`old_file::test_name` to its `new_file::test_name`. The file is
deleted in the final pre-merge commit on this branch per the epic's
hard gate.

All 87 tests collected before consolidation remain collected after.
Per-file `_gpu_only` / `_gpu_available` helpers were replaced with the
shared `requires_gpu` marker imported as `_gpu_only` for brevity.

## `test_gpu_writer_attrs_1563.py` (10 tests)
- `test_crs_wkt_only_attr_round_trips` -> `gpu/test_writer.py::test_crs_wkt_only_attr_round_trips`
- `test_image_description_round_trips_via_gpu_writer` -> ditto
- `test_extra_samples_single_band_writer_compat` -> ditto
- `test_extra_samples_round_trips_multiband_via_gpu_writer` -> ditto
- `test_colormap_round_trips_via_gpu_writer` -> ditto
- `test_extra_tags_custom_tag_round_trips_via_gpu_writer` -> ditto
- `test_resolution_tags_round_trip_via_gpu_writer` -> ditto
- `test_gdal_metadata_round_trips_via_gpu_writer` -> ditto
- `test_transform_attr_round_trip_bit_stable` -> ditto
- `test_no_data_attr_still_round_trips_after_fix` -> ditto

## `test_gpu_writer_band_first_1580.py` (6 tests, 3 parametrise IDs)
- `test_band_first_layout_written_correctly_via_write_geotiff_gpu[band|bands|channel]` -> 3 IDs preserved
- `test_band_first_layout_via_to_geotiff_auto_dispatch` -> preserved
- `test_yxbands_layout_unchanged` -> preserved
- `test_gpu_band_first_matches_cpu_byte_for_byte_on_pixel_values` -> preserved

## `test_gpu_writer_compression_modes_2026_05_11.py` (12 tests)
- `test_write_geotiff_gpu_zstd_roundtrip` -> preserved
- `test_write_geotiff_gpu_zstd_default_matches_explicit` -> preserved
- `test_write_geotiff_gpu_jpeg_rgb_roundtrip` -> preserved
- `test_write_geotiff_gpu_jpeg_uint8_single_band_roundtrip` -> preserved
- `test_write_geotiff_gpu_jpeg_uses_nvjpeg_when_available` -> preserved (gated on `_nvjpeg_only` now defined locally)
- `test_write_geotiff_gpu_compression_tag[none|deflate|zstd]` -> 3 IDs preserved
- `test_write_geotiff_gpu_jpeg_compression_tag` -> preserved
- `test_write_geotiff_gpu_deflate_roundtrip` -> preserved
- `test_write_geotiff_gpu_none_roundtrip` -> preserved
- `test_write_geotiff_gpu_lossless_codecs_agree` -> preserved

Helpers `_make_int_da`, `_make_rgb_uint8_da`, `_make_mono_uint8_da`,
`_CallSpy`, `_read_compression_tag`, and the `_COMPRESSION_TAGS` table
are reused by Section 4. `_COMPRESSION_TAGS` is now a superset of the
two original tables.

## `test_gpu_writer_cpu_fallback_codecs_2026_05_12.py` (17 tests, 4 parametrise IDs)
- `test_write_geotiff_gpu_lzw_roundtrip` -> preserved
- `test_write_geotiff_gpu_lzw_compression_tag` -> preserved
- `test_write_geotiff_gpu_packbits_roundtrip` -> preserved
- `test_write_geotiff_gpu_packbits_compression_tag` -> preserved
- `test_write_geotiff_gpu_lz4_roundtrip` -> preserved
- `test_write_geotiff_gpu_lz4_compression_tag` -> preserved
- `test_write_geotiff_gpu_lerc_float_lossless_roundtrip` -> preserved
- `test_write_geotiff_gpu_lerc_int_roundtrip` -> preserved
- `test_write_geotiff_gpu_lerc_compression_tag` -> preserved
- `test_write_geotiff_gpu_jpeg2000_uint8_lossless_roundtrip` -> preserved
- `test_write_geotiff_gpu_jpeg2000_rgb_roundtrip` -> preserved
- `test_write_geotiff_gpu_j2k_alias_matches_jpeg2000` -> preserved
- `test_write_geotiff_gpu_jpeg2000_compression_tag` -> preserved
- `test_write_geotiff_gpu_cpu_parity_lossless[lzw|packbits]` -> 2 IDs preserved
- `test_to_geotiff_gpu_true_dispatches_through_fallback_codec[lzw|packbits]` -> 2 IDs preserved

Local helpers `_make_int_da_small`, `_make_float_da_small`,
`_make_uint8_rgb_da_small` are named with the `_small` suffix to
disambiguate from the same-named helpers from Section 3 (which used
larger 64x64 defaults).

## `test_gpu_writer_nan_sentinel_1599.py` (7 tests)
- `test_gpu_writer_substitutes_nan_with_sentinel` -> preserved
- `test_gpu_and_cpu_writers_byte_equivalent_on_nan_input` -> preserved
- `test_gpu_writer_preserves_caller_cupy_buffer` -> preserved
- `test_gpu_writer_no_rewrite_when_no_nans` -> preserved
- `test_gpu_writer_nan_nodata_skips_substitution` -> preserved
- `test_gpu_writer_external_reader_sees_correct_nodata_mask` -> preserved
- `test_gpu_writer_multiband_nan_substitution` -> preserved

## `test_gpu_writer_overview_inplace_1948.py` (3 tests)
- `test_gpu_writer_overview_loop_uses_putmask_1948` -> preserved (no `_gpu_only` gate; source-level read only)
- `test_gpu_writer_cog_overview_sentinel_roundtrip_1948` -> preserved; `_make_float_raster_with_nodata` -> `_make_float_raster_with_nodata_1948`
- `test_gpu_writer_overview_uses_make_overview_gpu_fresh_buffer_1948` -> preserved

The source-locate path inside `test_gpu_writer_overview_loop_uses_putmask_1948`
walks `__file__.parent.parent.parent / "_writers" / "gpu.py"` from the
new `gpu/` directory; the original walked `.parent.parent / "_writers"`.

## `test_gpu_writer_overview_mode_and_compression_level_1740.py` (16 tests, 4 + 2x4 parametrise IDs)
- `test_block_reduce_2d_gpu_mode_matches_cpu_4x4` -> preserved
- `test_block_reduce_2d_gpu_mode_matches_cpu_random_8x8` -> preserved
- `test_block_reduce_2d_gpu_mode_dtype_preserved[uint8|uint16|int16|int32]` -> 4 IDs preserved
- `test_write_geotiff_gpu_cog_overview_resampling_mode` -> preserved
- `test_to_geotiff_gpu_cog_overview_resampling_mode` -> preserved
- `test_gpu_vs_cpu_mode_overview_pixel_parity` -> preserved
- `test_write_geotiff_gpu_compression_level_in_range_accepted[zstd-1|zstd-22|deflate-1|deflate-9]` -> 4 IDs preserved
- `test_write_geotiff_gpu_compression_level_out_of_range_accepted[zstd-999|zstd--5|deflate-50|deflate-0]` -> 4 IDs preserved
- `test_to_geotiff_gpu_compression_level_out_of_range_accepted` -> preserved
- `test_to_geotiff_cpu_compression_level_out_of_range_raises` -> preserved

## `test_to_geotiff_gpu_fallback_1674.py` (12 tests, 5 parametrise IDs)
- `test_runtime_error_without_gpu_signal_propagates` -> preserved
- `test_value_error_propagates` -> preserved
- `test_import_error_falls_back_with_warning` -> preserved
- `test_import_error_strict_mode_reraises` -> preserved
- `test_runtime_error_with_gpu_signal_falls_back[CUDA not available|no device found|nvCOMP library not loadable|cuInit failed: no driver|no GPU on this host]` -> 5 IDs preserved
- `test_runtime_error_with_gpu_signal_strict_reraises` -> preserved
- `test_auto_detected_gpu_fallback_warns` -> preserved
- `test_auto_detected_gpu_runtime_error_falls_back_with_warning` -> preserved
- `test_explicit_gpu_false_then_true_uses_explicit_template` -> preserved

The fallback section's tests intentionally have no `@_gpu_only`
marker. They monkeypatch `write_geotiff_gpu` and exercise the
CPU dispatcher's exception classification, so they run on every host
(matching the original module).

## Verification

- `pytest xrspatial/geotiff/tests/gpu/test_writer.py -v`: 87 passed locally
  (with cupy + CUDA available).
- `pytest --collect-only xrspatial/geotiff/tests/gpu/test_writer.py`: 87
  tests collected, matching the pre-consolidation baseline across the
  eight source files.
- `pytest xrspatial/geotiff/tests/release_gates/test_stable_features.py`:
  158 passed, 1 xfailed. The `writer.gpu` row now cites
  `gpu/test_writer.py`, and the `experimental` GPU codec note in
  `docs/source/reference/geotiff.rst` was updated to match.
