# Cluster audit -- tile / strip decode

Issue: #2429 (cluster 5 of long-tail epic #2424).

This file is deleted on the pre-merge commit. Do not let it land on main.

## Scope

Extend `xrspatial/geotiff/tests/read/test_tiling.py` with 7 top-level
tile / strip decoder test files. Parametrise by layout (tiled / stripped),
planar configuration, bits-per-sample, and sample format. Tests-only --
no source changes.

## Files folded

- `test_geotiff_planar_strip_truncation_1782.py`
- `test_mixed_bps.py`
- `test_mixed_sample_format.py`
- `test_parallel_decode_default_tile_1551.py`
- `test_parallel_strip_decode_2100.py`
- `test_planar_multiband.py`
- `test_unpack_bits_vectorised_1713.py`

## New sections in `read/test_tiling.py`

The existing file already covers tile / strip byte caps (CPU + GPU). The
new sections sit after the existing ones, in failure-mode order:

3. Mixed BitsPerSample dispatch (`resolve_bits_per_sample` unit cases
   + end-to-end open_geotiff rejection).
4. Mixed SampleFormat dispatch (`resolve_sample_format` unit cases +
   end-to-end open_geotiff rejection).
5. Sub-byte BPS unpack (`unpack_bits` reference parity, boundary cases).
6. Planar=2 strip table truncation (issue #1782).
7. Planar configuration multiband round-trips (CPU + GPU; planar x
   layout x bands x dtype).
8. Parallel tile decode gate (issue #1551 boundary).
9. Parallel strip decode -- local + HTTP COG, planar=1 and planar=2
   (issue #2100).

## File mapping

### `test_mixed_bps.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `TestResolveBitsPerSample::test_scalar` | `test_resolve_bps_scalar` |
| `TestResolveBitsPerSample::test_one_element_tuple` | `test_resolve_bps_one_element_tuple` |
| `TestResolveBitsPerSample::test_uniform_tuple` | `test_resolve_bps_uniform[tuple]` |
| `TestResolveBitsPerSample::test_uniform_list` | `test_resolve_bps_uniform[list]` |
| `TestResolveBitsPerSample::test_mixed_tuple_raises` | `test_resolve_bps_mixed_tuple_raises` |
| `TestResolveBitsPerSample::test_error_message_contains_values` | `test_resolve_bps_error_message_contains_values` |
| `TestResolveBitsPerSample::test_error_message_ot_matches_widest_bps` | `test_resolve_bps_error_message_ot_matches_widest[uint32]` + `[uint16]` |
| `TestResolveBitsPerSample::test_error_message_ot_uses_sample_format_hint` | `test_resolve_bps_error_message_ot_uses_sample_format_hint[float32]` + `[int16]` |
| `TestResolveBitsPerSample::test_empty_tuple_raises` | `test_resolve_bps_empty_tuple_raises` |
| `TestMixedBitsPerSampleTiff::test_uniform_bps_reads_fine` | `test_mixed_bps_uniform_reads_fine` |
| `TestMixedBitsPerSampleTiff::test_mixed_bps_rgb_plus_8bit_alpha_rejected` | `test_mixed_bps_rgb_plus_8bit_alpha_rejected` |

### `test_mixed_sample_format.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `TestResolveSampleFormat::test_scalar` | `test_resolve_sf_scalar` |
| `TestResolveSampleFormat::test_one_element_tuple` | `test_resolve_sf_one_element_tuple` |
| `TestResolveSampleFormat::test_uniform_tuple` | `test_resolve_sf_uniform[tuple]` |
| `TestResolveSampleFormat::test_uniform_list` | `test_resolve_sf_uniform[list]` |
| `TestResolveSampleFormat::test_mixed_tuple_raises` | `test_resolve_sf_mixed_tuple_raises` |
| `TestResolveSampleFormat::test_error_message_contains_values` | `test_resolve_sf_error_message_contains_values` |
| `TestResolveSampleFormat::test_mixed_signed_unsigned_raises` | `test_resolve_sf_mixed_signed_unsigned_raises` |
| `TestResolveSampleFormat::test_empty_tuple_falls_back_to_default` | `test_resolve_sf_empty_tuple_falls_back_to_default` |
| `TestMixedSampleFormatTiff::test_uniform_sample_format_reads_fine` | `test_mixed_sample_format_uniform_reads_fine` |
| `TestMixedSampleFormatTiff::test_mixed_float_uint_rejected` | `test_mixed_sample_format_float_uint_rejected` |
| `TestMixedSampleFormatTiff::test_mixed_signed_unsigned_rejected` | `test_mixed_sample_format_signed_unsigned_rejected` |

### `test_unpack_bits_vectorised_1713.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `test_vectorised_matches_reference` | `test_unpack_bits_matches_reference[bps=2,4,12 x pixel_count x data_factor]` (unchanged parametrise) |
| `test_bps1_unchanged` | `test_unpack_bits_bps1_unchanged` |
| `test_bps12_three_byte_buffer_decodes_one_pair` | `test_unpack_bits_bps12_three_byte_buffer_decodes_one_pair` |
| `test_bps12_two_byte_buffer_no_pair_decoded` | `test_unpack_bits_bps12_two_byte_buffer_no_pair_decoded` |
| `test_unsupported_bps_raises` | `test_unpack_bits_unsupported_bps_raises` |

### `test_geotiff_planar_strip_truncation_1782.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `test_planar_strip_table_truncated_raises_typed_error` | `test_planar_strip_table_truncated_raises_typed_error` |
| `test_planar_strip_table_complete_reads_correct_pixels` | `test_planar_strip_table_complete_reads_correct_pixels` |
| `test_chunky_single_band_truncated_strip_table_still_raises` | `test_chunky_single_band_truncated_strip_table_still_raises` |

### `test_planar_multiband.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `test_cpu_planar_multiband[planar x tiled x bands x dtype]` | `test_planar_multiband_cpu[<planar>-<layout>-bands<N>-<dtype>]` |
| `test_gpu_matches_cpu_planar_multiband[...]` | `test_planar_multiband_gpu_matches_cpu[<planar>-<layout>-bands<N>-<dtype>]` |
| `test_cpu_singleband_sanity[tiled]` | `test_planar_singleband_cpu[<layout>]` |
| `test_gpu_singleband_sanity[tiled]` | `test_planar_singleband_gpu[<layout>]` |
| `test_a3_repro_stripped_planar_separate` | `test_planar_stripped_separate_axis_order` |

### `test_parallel_decode_default_tile_1551.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `test_parallel_engages_at_default_tile_size_256` | `test_parallel_tile_decode_engages_at_default_tile_size_256` |
| `test_sequential_for_small_tiles` | `test_parallel_tile_decode_sequential_for_small_tiles` |
| `test_sequential_when_only_one_tile` | `test_parallel_tile_decode_sequential_when_only_one_tile` |

### `test_parallel_strip_decode_2100.py` -> `read/test_tiling.py`

| Old test | New test / param id |
|---|---|
| `TestReadStripsParallelGate::test_parallel_decode_matches_serial` | `test_parallel_strip_decode_matches_serial_local` |
| `TestReadStripsParallelGate::test_parallel_pool_engages_on_multi_strip` | `test_parallel_strip_decode_pool_engages_on_multi_strip` |
| `TestReadStripsParallelGate::test_serial_path_used_for_small_strip` | `test_parallel_strip_decode_serial_for_single_strip` |
| `TestReadStripsParallelGate::test_windowed_strip_read_parallel` | `test_parallel_strip_decode_windowed_matches_serial` |
| `TestHttpStripParallelDecode::test_parallel_decode_matches_serial` | `test_parallel_strip_decode_http_matches_serial` |
| `TestHttpStripParallelDecode::test_serial_gate_engages_on_single_strip` | `test_parallel_strip_decode_http_serial_on_single_strip` |
| `TestPlanar2MultibandStripParallel::test_parallel_matches_serial_planar2` | `test_parallel_strip_decode_planar2_matches_serial` |
| `TestPlanar2MultibandStripParallel::test_windowed_planar2_parallel` | `test_parallel_strip_decode_planar2_windowed_matches_serial` |
| `TestPlanar2MultibandStripParallel::test_http_windowed_planar2_parallel` | `test_parallel_strip_decode_http_planar2_windowed_matches_serial` |

## Verification

- `pytest xrspatial/geotiff/tests/read/test_tiling.py -v`
- `pytest xrspatial/geotiff/tests/ -x -q`
- `find xrspatial/geotiff/tests -name 'test_*.py' | wc -l` drops by 7
  (7 deleted, 0 added).
