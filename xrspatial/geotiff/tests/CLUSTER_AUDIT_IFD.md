# Cluster audit -- IFD parser

Issue: #2426 (cluster 2 of long-tail epic #2424).

This file is deleted on the pre-merge commit. Do not let it land on main.

## Scope

Consolidate seven IFD-parser test files into `unit/test_ifd.py`,
parametrised by failure mode (cycle / cap / out-of-bounds / sparse /
malformed). Tests-only -- no source changes.

## Sections in the new file

1. Entry-table bounds: `num_entries`, entry-table, next-IFD reads
   (classic + BigTIFF), negative offset guards.
2. Entry value bounds: `MAX_IFD_ENTRY_COUNT`, `MAX_IFD_ENTRY_BYTES`,
   value-range vs EOF, pixel-array exemptions.
3. Chain length cap (`MAX_IFDS`): classic + big-endian, boundary,
   real-COG sanity.
4. Chain cycle detection: A->B->A, self-cycle, sibling malformed
   branches (EOF / MAX_IFDS) still raise.
5. Chain malformed offsets (`next_ifd_offset` / `first_ifd_offset`
   past EOF).
6. Sparse blocks (tile + strip).
7. Sparse strip parallel-decode (local + HTTP COG).

## File mapping

### `test_ifd_chain_cap.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `TestIFDChainCap::test_ifd_chain_at_limit_rejected` | `test_chain_cap_rejects_oversized[chain_cap[over-limit-le]]` |
| `TestIFDChainCap::test_chain_at_boundary_passes` | `test_chain_cap_boundary_passes_at_max_and_fails_at_max_plus_one` |
| `TestIFDChainCap::test_error_message_mentions_dos_and_limit` | `test_chain_cap_error_message_mentions_dos_and_limit` |
| `TestIFDChainCap::test_short_chain_passes` | `test_chain_cap_short_chain_passes` |
| `TestIFDChainCap::test_legitimate_cog_with_overviews_passes` | `test_chain_cap_legitimate_cog_with_overviews_passes` |
| `TestIFDChainCapBigEndian::test_big_endian_chain_rejected` | `test_chain_cap_rejects_oversized[chain_cap[over-limit-be]]` |

### `test_ifd_cycle_1913.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `TestIFDChainCycle::test_two_ifd_cycle_rejected` | `test_chain_cycle_rejected[chain_cycle[a-b-a-le]]` |
| `TestIFDChainCycle::test_self_cycle_rejected` | `test_chain_cycle_rejected[chain_cycle[self-le]]` |
| `TestIFDChainCycle::test_cycle_error_message_mentions_offset_and_malformed` | `test_chain_cycle_error_message_mentions_offset_and_malformed` |
| `TestIFDChainCycle::test_big_endian_cycle_rejected` | `test_chain_cycle_rejected[chain_cycle[a-b-a-be]]` |
| `TestMalformedChainSiblingsStillRaise::test_offset_past_eof_still_raises` | `test_first_ifd_offset_past_eof_rejected` (de-duplicated -- same coverage as malformed-chain section) |
| `TestMalformedChainSiblingsStillRaise::test_max_ifds_still_raises` | `test_chain_cap_rejects_oversized[chain_cap[over-limit-le]]` (de-duplicated) |
| `TestNormalChainStillParses::test_short_acyclic_chain_parses` | `test_chain_cap_short_chain_passes` (de-duplicated) |
| `TestNormalChainStillParses::test_single_ifd_chain_parses` | `test_chain_single_ifd_parses` |

### `test_ifd_entry_table_bounds_1672.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `test_classic_num_entries_truncated_raises_valueerror` | `test_entry_table_bounds_rejected[entry_table[classic-num_entries-truncated]]` |
| `test_classic_num_entries_zero_buffer_raises_valueerror` | `test_entry_table_bounds_rejected[entry_table[classic-num_entries-zero-buffer]]` |
| `test_classic_entry_table_truncated_raises_valueerror` | `test_entry_table_bounds_rejected[entry_table[classic-entry-table-truncated]]` |
| `test_classic_next_ifd_truncated_raises_valueerror` (zero-byte slice) | `test_entry_table_bounds_rejected[entry_table[classic-next_ifd-truncated]]` |
| `test_classic_next_ifd_truncated_raises_valueerror` (3-byte slice) | `test_entry_table_bounds_rejected[entry_table[classic-next_ifd-one-short]]` |
| `test_bigtiff_num_entries_truncated_raises_valueerror` | `test_entry_table_bounds_rejected[entry_table[bigtiff-num_entries-truncated]]` |
| `test_bigtiff_entry_table_truncated_raises_valueerror` | `test_entry_table_bounds_rejected[entry_table[bigtiff-entry-table-truncated]]` |
| `test_bigtiff_next_ifd_truncated_raises_valueerror` (zero-byte slice) | `test_entry_table_bounds_rejected[entry_table[bigtiff-next_ifd-truncated]]` |
| `test_bigtiff_next_ifd_truncated_raises_valueerror` (7-byte slice) | `test_entry_table_bounds_rejected[entry_table[bigtiff-next_ifd-one-short]]` |
| `test_classic_complete_buffer_parses_ok` | `test_entry_table_complete_buffer_parses[entry_table[classic-complete]]` |
| `test_bigtiff_complete_buffer_parses_ok` | `test_entry_table_complete_buffer_parses[entry_table[bigtiff-complete]]` |
| `test_offset_past_eof_raises_valueerror` | `test_entry_table_offset_past_eof_rejected` |
| `test_classic_negative_offset_rejected_at_num_entries` | `test_entry_table_negative_offset_rejected[entry_table[classic-neg1]]` |
| `test_bigtiff_negative_offset_rejected_at_num_entries` | `test_entry_table_negative_offset_rejected[entry_table[bigtiff-neg1]]` |
| `test_classic_large_negative_offset_rejected` | `test_entry_table_negative_offset_rejected[entry_table[classic-neg-10000]]` |
| `test_entry_table_negative_offset_guard` | `test_entry_table_overrun_via_huge_num_entries_rejected` |
| `test_next_ifd_negative_offset_guard` | `test_entry_table_bounds_rejected[entry_table[classic-next_ifd-truncated]]` (de-duplicated with the next-IFD case above) |

### `test_parallel_strip_decode_sparse_2100.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `TestReadStripsSparseParallel::test_full_image_parallel_matches_serial` | `test_sparse_strips_full_image_parallel_matches_serial` |
| `TestReadStripsSparseParallel::test_parallel_pool_engages_on_sparse_multi_strip` | `test_sparse_strips_parallel_pool_engages_on_multi_strip` |
| `TestReadStripsSparseParallel::test_windowed_across_sparse_boundary` | `test_sparse_strips_windowed_across_boundary` |
| `TestReadStripsSparseParallel::test_all_sparse_image_returns_fill` | `test_sparse_strips_all_sparse_image_returns_fill` |
| `TestReadStripsSparsePlanar2::test_planar2_sparse_parallel_matches_serial` | `test_sparse_strips_planar2_parallel_matches_serial` |
| `TestHttpStripsSparseParallel::test_http_windowed_strict_subset_parallel` | `test_sparse_strips_http_windowed_strict_subset_parallel` |
| `TestHttpStripsSparseParallel::test_http_windowed_across_sparse_boundary` | `test_sparse_strips_http_windowed_across_boundary` |

### `test_parse_all_ifds_malformed_chain_1863.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `TestParseAllIFDsMalformedChain1863::test_next_ifd_offset_past_eof_raises` | `test_chain_next_ifd_offset_past_eof_rejected[chain_offset[next-past-eof-le]]` |
| `TestParseAllIFDsMalformedChain1863::test_next_ifd_offset_equals_file_length_raises` | `test_chain_next_ifd_offset_at_file_length_rejected` |
| `TestParseAllIFDsMalformedChain1863::test_first_ifd_offset_past_eof_raises` | `test_first_ifd_offset_past_eof_rejected` |
| `TestParseAllIFDsMalformedChain1863::test_first_ifd_offset_past_eof_raises_synthetic_header` | `test_first_ifd_offset_past_eof_rejected_synthetic_header` |
| `TestParseAllIFDsMalformedChain1863::test_big_endian_next_offset_past_eof_raises` | `test_chain_next_ifd_offset_past_eof_rejected[chain_offset[next-past-eof-be]]` |
| `TestParseAllIFDsMalformedChain1863::test_valid_chain_terminator_still_parses` | `test_chain_valid_terminator_still_parses` |

### `test_parse_ifd_bounds.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `test_count_overflow_rejected` | `test_entry_value_count_cap_rejected_at_default` |
| `test_byte_size_overflow_rejected_with_legal_count` | `test_entry_value_byte_cap_rejected_under_lowered_caps` |
| `test_byte_cap_fires_at_default_values` | `test_entry_value_byte_cap_rejected_at_default` |
| `test_value_offset_past_eof_rejected` | `test_entry_value_range_past_eof_rejected` |
| `test_pixel_array_tag_exemption` | `test_entry_value_pixel_array_tag_exempt_from_caps` |
| `test_count_cap_fires_for_non_pixel_tag_under_lowered_caps` | `test_entry_value_count_cap_rejected_under_lowered_caps` |
| `test_normal_tag_with_legal_count_passes` | `test_entry_value_normal_tag_with_legal_count_passes` |
| `test_short_count_at_cap_still_passes_for_pixel_tag` | `test_entry_value_short_count_pixel_tag_passes` |

### `test_sparse_cog.py` -> `unit/test_ifd.py`

| Old test | New test / param id |
|---|---|
| `TestSparseTiles::test_sparse_tile_with_nodata_round_trips` | `test_sparse_tile_with_nodata_round_trips` |
| `TestSparseTiles::test_sparse_tile_without_nodata_fills_zero` | `test_sparse_tile_without_nodata_fills_zero` |
| `TestSparseTiles::test_sparse_tile_raw_read_uses_nodata_sentinel` | `test_sparse_tile_raw_read_uses_nodata_sentinel` |
| `TestSparseStrips::test_sparse_strip_with_nodata` | `test_sparse_strip_with_nodata` |
| `TestSparseTilesGPU::test_sparse_tile_gpu_round_trip` | `test_sparse_tile_gpu_round_trip` |

## File delta

- Deleted: 7 top-level test files (listed above).
- Added: `xrspatial/geotiff/tests/unit/test_ifd.py`.
- Net: -6 files in `xrspatial/geotiff/tests/test_*.py`.

## Verification

```bash
pytest xrspatial/geotiff/tests/unit/test_ifd.py -v
pytest xrspatial/geotiff/tests/ -x -q
```
