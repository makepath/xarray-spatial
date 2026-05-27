# Cluster 14 (Sub-PR C) audit: GPU codec test consolidation

Folds 11 GPU codec test files into `xrspatial/geotiff/tests/gpu/test_codec.py`.
Baseline collection: 71 tests across the 11 source files. Consolidated
collection: 71 tests in the new file. Run-time outcome on this checkout
(no nvCOMP / nvJPEG / lerc beyond what the host ships): 68 passed, 3
skipped.

Source -> destination mapping (every old test landed under the same
issue-suffixed name unless noted):

## test_nvcomp_batch_compress_batched_1712.py

| old `file::test`                                    | new `test_codec.py::test_id`                                  |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `test_no_per_tile_cupy_empty_in_compressed_pool`    | `test_no_per_tile_cupy_empty_in_compressed_pool_1712`         |
| `test_no_per_tile_get_in_result_loop`               | `test_no_per_tile_get_in_result_loop_1712`                    |
| `test_gpu_write_roundtrip_after_batched_compress[deflate]` | `test_gpu_write_roundtrip_after_batched_compress_1712[deflate]` |
| `test_gpu_write_roundtrip_after_batched_compress[zstd]`    | `test_gpu_write_roundtrip_after_batched_compress_1712[zstd]`    |
| `test_gpu_write_zero_tile_edge_case`                | `test_gpu_write_zero_tile_edge_case_1712`                     |

## test_nvcomp_batch_upload_p3.py

| old `file::test`                                       | new `test_codec.py::test_id`                            |
| ------------------------------------------------------ | ------------------------------------------------------- |
| `test_nvcomp_batch_upload_correctness[256-tile0]`      | `test_nvcomp_batch_upload_correctness_p3[256-tile0]`    |
| `test_nvcomp_batch_upload_correctness[1024-tile1]`     | `test_nvcomp_batch_upload_correctness_p3[1024-tile1]`   |
| `test_nvcomp_batch_upload_correctness[2048-tile2]`     | `test_nvcomp_batch_upload_correctness_p3[2048-tile2]`   |
| `test_nvcomp_kvikio_fallback_skips_zstd`               | `test_nvcomp_kvikio_fallback_skips_zstd_p3`             |
| `test_nvcomp_batch_upload_perf_regression_guard`       | `test_nvcomp_batch_upload_perf_regression_guard_p3`     |

## test_nvcomp_decompress_cumsum_offsets_1950.py

| old `file::test`                                          | new `test_codec.py::test_id`                            |
| --------------------------------------------------------- | ------------------------------------------------------- |
| `test_nvcomp_decompress_uses_cumsum_for_offsets_1950`     | `test_nvcomp_decompress_uses_cumsum_for_offsets_1950`   |
| `test_cumsum_matches_loop_prefix_sum_1950`                | `test_cumsum_matches_loop_prefix_sum_1950`              |
| `test_nvcomp_batch_decompress_roundtrip_1950`             | `test_nvcomp_batch_decompress_roundtrip_1950`           |

## test_nvcomp_from_device_bufs_single_alloc_1659.py

| old `file::test`                                                | new `test_codec.py::test_id`                                       |
| --------------------------------------------------------------- | ------------------------------------------------------------------ |
| `test_unsupported_codec_short_circuits_before_allocation`       | `test_unsupported_codec_short_circuits_before_allocation_1659`     |
| `test_no_nvcomp_lib_returns_none`                               | `test_no_nvcomp_lib_returns_none_1659`                             |
| `test_memory_guard_runs_with_full_decomp_size`                  | `test_memory_guard_runs_with_full_decomp_size_1659`                |
| `test_zstd_decompress_roundtrip_returns_single_contiguous_buffer` | `test_zstd_decompress_roundtrip_returns_single_contiguous_buffer_1659` |
| `test_no_orphan_decomp_buffers_after_call`                      | `test_no_orphan_decomp_buffers_after_call_1659`                    |

## test_nvjpeg_encode_stream_sync_2212.py

| old `file::test`                                                          | new `test_codec.py::test_id`                                                  |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `TestNvjpegEncodeStreamSync::test_no_device_synchronize_inside_encode_loop` | `TestNvjpegEncodeStreamSync_2212::test_no_device_synchronize_inside_encode_loop` |
| `TestNvjpegEncodeStreamSync::test_stream_null_synchronize_present`        | `TestNvjpegEncodeStreamSync_2212::test_stream_null_synchronize_present`       |
| `TestNvjpeg2kEncodeStreamSync::test_no_device_synchronize_inside_encode_loop` | `TestNvjpeg2kEncodeStreamSync_2212::test_no_device_synchronize_inside_encode_loop` |
| `TestNvjpeg2kEncodeStreamSync::test_stream_null_synchronize_present`      | `TestNvjpeg2kEncodeStreamSync_2212::test_stream_null_synchronize_present`     |
| `TestDecodeReferencePattern::test_decoder_uses_stream_null_sync_in_loop`  | `TestDecodeReferencePattern_2212::test_decoder_uses_stream_null_sync_in_loop` |

## test_nvjpeg2k_single_alloc_2107.py

| old `file::test`                                                            | new `test_codec.py::test_id`                                                 |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `TestNvjpeg2kSingleAllocStructural::test_no_cupy_empty_inside_decode_loop`  | `TestNvjpeg2kSingleAllocStructural_2107::test_no_cupy_empty_inside_decode_loop` |
| `TestNvjpeg2kSingleAllocStructural::test_no_device_synchronize_inside_decode_loop` | `TestNvjpeg2kSingleAllocStructural_2107::test_no_device_synchronize_inside_decode_loop` |
| `TestNvjpeg2kSingleAllocStructural::test_pool_allocation_present`           | `TestNvjpeg2kSingleAllocStructural_2107::test_pool_allocation_present`        |
| `TestNvjpeg2kSingleAllocStructural::test_check_gpu_memory_guard_present`    | `TestNvjpeg2kSingleAllocStructural_2107::test_check_gpu_memory_guard_present` |
| `TestNvjpeg2kLibAbsentShortCircuit::test_returns_none_when_lib_missing`     | `TestNvjpeg2kLibAbsentShortCircuit_2107::test_returns_none_when_lib_missing`  |
| `TestNvjpeg2kLibAbsentShortCircuit::test_returns_none_for_unsupported_dtype` | `TestNvjpeg2kLibAbsentShortCircuit_2107::test_returns_none_for_unsupported_dtype` |
| `TestNvjpeg2kPoolWithCupy::test_pool_slabs_are_non_overlapping`             | `TestNvjpeg2kPoolWithCupy_2107::test_pool_slabs_are_non_overlapping`          |

Note: the source had `@pytest.mark.gpu` on `TestNvjpeg2kPoolWithCupy`,
which raised an `UnknownMark` warning because the project does not
register a `gpu` mark. The new section uses `@requires_gpu` from
`_helpers/markers.py` -- same skip behaviour, no warning.

## test_jpeg_gpu_1549.py

| old `file::test`                                | new `test_codec.py::test_id`                          |
| ----------------------------------------------- | ----------------------------------------------------- |
| `test_rgb_jpeg_gpu_no_crash`                    | `test_rgb_jpeg_gpu_no_crash_1549`                     |
| `test_rgb_jpeg_gpu_matches_cpu`                 | `test_rgb_jpeg_gpu_matches_cpu_1549`                  |
| `test_grayscale_jpeg_gpu_matches_cpu`           | `test_grayscale_jpeg_gpu_matches_cpu_1549`            |
| `test_cuda_context_survives_after_jpeg_gpu_read` | `test_cuda_context_survives_after_jpeg_gpu_read_1549` |

## test_lerc_valid_mask_gpu.py

| old `file::test`                                       | new `test_codec.py::test_id`                            |
| ------------------------------------------------------ | ------------------------------------------------------- |
| `TestGpuLercValidMask::test_float32_nan_nodata`        | `TestGpuLercValidMask::test_float32_nan_nodata`         |
| `TestGpuLercValidMask::test_float32_sentinel_nodata`   | `TestGpuLercValidMask::test_float32_sentinel_nodata`    |
| `TestGpuLercValidMask::test_uint16_sentinel_nodata`    | `TestGpuLercValidMask::test_uint16_sentinel_nodata`     |
| `TestGpuLercValidMask::test_no_mask_roundtrip_bitexact` | `TestGpuLercValidMask::test_no_mask_roundtrip_bitexact` |

## test_predictor2_big_endian_gpu_1517.py

| old `file::test`                                            | new `test_codec.py::test_id`                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| `test_gpu_predictor2_big_endian_int32_tiled_reproducer`     | `test_gpu_predictor2_big_endian_int32_tiled_reproducer_1517` |
| `test_gpu_predictor2_big_endian_dtypes_tiled[uint16]`       | `test_gpu_predictor2_big_endian_dtypes_tiled_1517[uint16]`   |
| `test_gpu_predictor2_big_endian_dtypes_tiled[int16]`        | `test_gpu_predictor2_big_endian_dtypes_tiled_1517[int16]`    |
| `test_gpu_predictor2_big_endian_dtypes_tiled[uint32]`       | `test_gpu_predictor2_big_endian_dtypes_tiled_1517[uint32]`   |
| `test_gpu_predictor2_big_endian_dtypes_tiled[int32]`        | `test_gpu_predictor2_big_endian_dtypes_tiled_1517[int32]`    |
| `test_gpu_predictor2_big_endian_stripped_uint16`            | `test_gpu_predictor2_big_endian_stripped_uint16_1517`        |
| `test_gpu_predictor2_little_endian_still_works`             | `test_gpu_predictor2_little_endian_still_works_1517`         |
| `test_gpu_predictor3_big_endian_still_works`                | `test_gpu_predictor3_big_endian_still_works_1517`            |
| `test_swap_byte_lanes_numpy_bps2`                           | `test_swap_byte_lanes_numpy_bps2_1517`                       |
| `test_swap_byte_lanes_numpy_bps4`                           | `test_swap_byte_lanes_numpy_bps4_1517`                       |
| `test_swap_byte_lanes_numpy_bps8`                           | `test_swap_byte_lanes_numpy_bps8_1517`                       |
| `test_swap_byte_lanes_uint8_noop`                           | `test_swap_byte_lanes_uint8_noop_1517`                       |
| `test_swap_byte_lanes_rejects_unsupported_bps`              | `test_swap_byte_lanes_rejects_unsupported_bps_1517`          |
| `test_swap_byte_lanes_rejects_misaligned_size`              | `test_swap_byte_lanes_rejects_misaligned_size_1517`          |
| `test_swap_byte_lanes_numpy_is_zero_temp`                   | `test_swap_byte_lanes_numpy_is_zero_temp_1517`               |
| `test_swap_byte_lanes_cupy_kernel[2-uint16]`                | `test_swap_byte_lanes_cupy_kernel_1517[2-uint16]`            |
| `test_swap_byte_lanes_cupy_kernel[4-uint32]`                | `test_swap_byte_lanes_cupy_kernel_1517[4-uint32]`            |
| `test_swap_byte_lanes_cupy_kernel[8-uint64]`                | `test_swap_byte_lanes_cupy_kernel_1517[8-uint64]`            |
| `test_swap_byte_lanes_cupy_uint8_noop`                      | `test_swap_byte_lanes_cupy_uint8_noop_1517`                  |

## test_predictor3_int_dtype_gpu_1933.py

| old `file::test`                                                              | new `test_codec.py::test_id`                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `TestGPUEagerRejectsMalformedFile::test_gpu_eager_stripped_raises`            | `TestGPUEagerRejectsMalformedFile_1933::test_gpu_eager_stripped_raises`        |
| `TestGPUEagerRejectsMalformedFile::test_gpu_eager_tiled_raises`               | `TestGPUEagerRejectsMalformedFile_1933::test_gpu_eager_tiled_raises`           |
| `TestGPUEagerRejectsMalformedFile::test_gpu_dispatcher_eager_raises`          | `TestGPUEagerRejectsMalformedFile_1933::test_gpu_dispatcher_eager_raises`      |
| `TestGPUChunkedRejectsMalformedFile::test_read_geotiff_gpu_chunked_stripped_raises` | `TestGPUChunkedRejectsMalformedFile_1933::test_read_geotiff_gpu_chunked_stripped_raises` |
| `TestGPUChunkedRejectsMalformedFile::test_read_geotiff_gpu_chunked_tiled_raises` | `TestGPUChunkedRejectsMalformedFile_1933::test_read_geotiff_gpu_chunked_tiled_raises` |
| `TestGPUChunkedRejectsMalformedFile::test_open_geotiff_chunks_gpu_dispatcher_raises` | `TestGPUChunkedRejectsMalformedFile_1933::test_open_geotiff_chunks_gpu_dispatcher_raises` |
| `TestValidPredictor3StillWorksOnGPU::test_predictor3_float32_gpu_round_trip`  | `TestValidPredictor3StillWorksOnGPU_1933::test_predictor3_float32_gpu_round_trip` |
| `TestValidPredictor3StillWorksOnGPU::test_predictor3_float32_dask_gpu_round_trip` | `TestValidPredictor3StillWorksOnGPU_1933::test_predictor3_float32_dask_gpu_round_trip` |
| `TestErrorMessageStable::test_gpu_error_message_matches_eager`                | `TestErrorMessageStable_1933::test_gpu_error_message_matches_eager`            |

## test_gpu_jpeg_interop_reject_issue_D_1845.py

| old `file::test`                                                  | new `test_codec.py::test_id`                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `test_write_geotiff_gpu_rejects_jpeg_without_opt_in`              | `test_write_geotiff_gpu_rejects_jpeg_without_opt_in_1845`          |
| `test_write_geotiff_gpu_rejects_jpeg_message_mentions_alternatives` | `test_write_geotiff_gpu_rejects_jpeg_message_mentions_alternatives_1845` |
| `test_write_geotiff_gpu_rejects_jpeg_case_insensitive`            | `test_write_geotiff_gpu_rejects_jpeg_case_insensitive_1845`        |
| `test_write_geotiff_gpu_jpeg_opt_in_emits_warning`                | `test_write_geotiff_gpu_jpeg_opt_in_emits_warning_1845`            |
| `test_write_geotiff_gpu_non_jpeg_unaffected_by_flag`              | `test_write_geotiff_gpu_non_jpeg_unaffected_by_flag_1845`          |

## Cross-references updated

* `docs/source/reference/release_gate_geotiff.rst` -- codec ``jpeg``
  row now cites `gpu/test_codec.py` instead of the deleted
  `test_gpu_jpeg_interop_reject_issue_D_1845.py`.
* `xrspatial/geotiff/tests/unit/test_predictor.py` -- the GPU
  predictor file pointers in the module docstring now point at
  `gpu/test_codec.py`.

This audit file is deleted in a final pre-merge commit on this branch
(epic #2424 hard gate).
