# Cluster Audit: Predictor (issue #2427)

Long-tail epic #2424, cluster 3 of 16. Maps every old `file::test` (with
parametrise expansion) to its new `unit/test_predictor.py::test_id` so a
reviewer can confirm zero coverage loss.

Deleted on the pre-merge commit so it does not land on `main`.

## Old -> new map

### `test_predictor2_big_endian.py`

| Old test_id                                              | New test_id                                                                     |
|----------------------------------------------------------|---------------------------------------------------------------------------------|
| `test_big_endian_predictor2_round_trip[uint16]`          | `test_predictor2_round_trip_stripped[uint16->]`                                 |
| `test_big_endian_predictor2_round_trip[int16]`           | `test_predictor2_round_trip_stripped[int16->]`                                  |
| `test_big_endian_predictor2_round_trip[uint32]`          | `test_predictor2_round_trip_stripped[uint32->]`                                 |
| `test_big_endian_predictor2_round_trip[int32]`           | `test_predictor2_round_trip_stripped[int32->]`                                  |
| `test_little_endian_predictor2_still_round_trips`        | `test_predictor2_round_trip_stripped[uint16-<]` (covers same path with uint16)  |
| `test_big_endian_predictor2_uint8_unaffected`            | `test_predictor2_round_trip_stripped[uint8->]`                                  |

The new parametrise matrix also adds `uint8-<`, `int8-<`, `int8->`,
`uint16-<`, `int16-<`, `uint32-<`, `int32-<` -- not coverage loss,
coverage gain. The pre-fix bug only fires on multi-byte BE, but the LE
sanity case from the old file generalises to all dtypes cheaply.

### `test_predictor2_int8.py`

| Old test_id                                              | New test_id                                          |
|----------------------------------------------------------|------------------------------------------------------|
| `test_cpu_predictor2_int8_round_trip[<]`                 | `test_predictor2_round_trip_stripped[int8-<]`        |
| `test_cpu_predictor2_int8_round_trip[>]`                 | `test_predictor2_round_trip_stripped[int8->]`        |
| `test_cpu_predictor2_int8_tiled`                         | `test_predictor2_round_trip_tiled_int8`              |
| `test_gpu_predictor2_int8_tiled_matches_cpu`             | `test_gpu_predictor2_int8_matches_cpu[tiled]`        |
| `test_gpu_predictor2_int8_stripped_matches_cpu`          | `test_gpu_predictor2_int8_matches_cpu[stripped]`     |

The new int8 grid generator is the same `_signed_int8_grid` helper.

### `test_predictor3_big_endian.py`

| Old test_id                                              | New test_id                                            |
|----------------------------------------------------------|--------------------------------------------------------|
| `test_big_endian_predictor3_round_trip[float32]`         | `test_predictor3_round_trip_stripped[float32->]`       |
| `test_big_endian_predictor3_round_trip[float64]`         | `test_predictor3_round_trip_stripped[float64->]`       |
| `test_little_endian_predictor3_still_round_trips`        | `test_predictor3_round_trip_stripped[float32-<]`       |
| `test_big_endian_predictor3_tiled`                       | `test_predictor3_round_trip_tiled_big_endian`          |
| `test_big_endian_predictor3_gpu`                         | `test_gpu_predictor3_big_endian_matches_cpu`           |

`float64-<` is added by the parametrise expansion.

### `test_predictor3_int_dtype_1933.py`

| Old test_id                                                                                        | New test_id                                                                                            |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf1_uint`                    | `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf1_uint`                        |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf2_int`                     | `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf2_int`                         |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf4_undefined`               | `TestPredictor3IntegerSampleFormatRejected::test_helper_rejects_pred3_sf4_undefined`                   |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred3_sf3_float`                   | `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred3_sf3_float`                       |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred1_with_any_sf`                 | `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred1_with_any_sf`                     |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred2_with_any_sf`                 | `TestPredictor3IntegerSampleFormatRejected::test_helper_accepts_pred2_with_any_sf`                     |
| `TestPredictor3IntegerSampleFormatRejected::test_helper_normalizes_tuple_predictor`                | `TestPredictor3IntegerSampleFormatRejected::test_helper_normalizes_tuple_predictor`                    |
| `TestEagerReadRejectsMalformedFile::test_open_geotiff_eager_raises`                                | `TestEagerReadRejectsMalformedFile::test_open_geotiff_eager_raises`                                    |
| `TestEagerReadRejectsMalformedFile::test_open_geotiff_dask_raises`                                 | `TestEagerReadRejectsMalformedFile::test_open_geotiff_dask_raises`                                     |
| `TestValidPredictor3StillWorks::test_predictor3_float32_round_trip`                                | covered by `test_predictor3_round_trip_stripped[float32-<]` + `test_predictor3_writer_round_trip[*]`   |

Class names preserved; helper / validator test bodies preserved
verbatim.

### `test_predictor_fp_write_1313.py`

| Old test_id                                                  | New test_id                                                       |
|--------------------------------------------------------------|-------------------------------------------------------------------|
| `test_predictor3_round_trip[True-deflate-float32]`           | `test_predictor3_writer_round_trip[tiled-deflate-float32]`        |
| `test_predictor3_round_trip[True-deflate-float64]`           | `test_predictor3_writer_round_trip[tiled-deflate-float64]`        |
| `test_predictor3_round_trip[True-zstd-float32]`              | `test_predictor3_writer_round_trip[tiled-zstd-float32]`           |
| `test_predictor3_round_trip[True-zstd-float64]`              | `test_predictor3_writer_round_trip[tiled-zstd-float64]`           |
| `test_predictor3_round_trip[False-deflate-float32]`          | `test_predictor3_writer_round_trip[stripped-deflate-float32]`     |
| `test_predictor3_round_trip[False-deflate-float64]`          | `test_predictor3_writer_round_trip[stripped-deflate-float64]`     |
| `test_predictor3_round_trip[False-zstd-float32]`             | `test_predictor3_writer_round_trip[stripped-zstd-float32]`        |
| `test_predictor3_round_trip[False-zstd-float64]`             | `test_predictor3_writer_round_trip[stripped-zstd-float64]`        |
| `test_predictor3_better_than_predictor2_on_smooth_floats`    | `test_predictor3_better_than_predictor2_on_smooth_floats`         |
| `test_predictor3_rejects_integer_dtype`                      | `test_predictor3_rejects_integer_dtype`                           |
| `test_predictor_legacy_bool_unchanged`                       | `test_predictor_legacy_bool_unchanged`                            |
| `test_predictor_false_emits_no_tag`                          | `test_predictor_false_emits_no_tag`                               |
| `test_predictor3_with_compression_none_is_silent`            | `test_predictor3_with_compression_none_is_silent`                 |
| `test_normalize_predictor_table`                             | `test_normalize_predictor_table`                                  |
| `test_predictor3_streaming_dask`                             | `test_predictor3_streaming_dask`                                  |
| `test_predictor3_multiband_round_trip`                       | `test_predictor3_multiband_round_trip`                            |
| `test_predictor3_large_round_trip_value_exact`               | `test_predictor3_large_round_trip_value_exact`                    |
| `test_predictor3_encode_within_2x_of_predictor2`             | `test_predictor3_encode_within_2x_of_predictor2`                  |

### `test_predictor_multisample.py`

| Old test_id                                                                | New test_id                                                                  |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `test_gpu_predictor2_multisample_matches_cpu[3-uint8]`                     | `test_gpu_predictor2_multisample_matches_cpu[s3-uint8]`                      |
| `test_gpu_predictor2_multisample_matches_cpu[4-uint8]`                     | `test_gpu_predictor2_multisample_matches_cpu[s4-uint8]`                      |
| `test_gpu_predictor2_multisample_matches_cpu[3-uint16]`                    | `test_gpu_predictor2_multisample_matches_cpu[s3-uint16]`                     |
| `test_gpu_predictor2_multisample_uneven_tiles`                             | `test_gpu_predictor2_multisample_uneven_tiles`                               |
| `test_cpu_predictor3_multisample_reads_correctly_1247[3-float32]`          | `test_cpu_predictor3_multisample_reads_correctly[s3-float32]`                |
| `test_cpu_predictor3_multisample_reads_correctly_1247[4-float32]`          | `test_cpu_predictor3_multisample_reads_correctly[s4-float32]`                |
| `test_cpu_predictor3_multisample_reads_correctly_1247[3-float64]`          | `test_cpu_predictor3_multisample_reads_correctly[s3-float64]`                |
| `test_cpu_predictor3_multisample_reads_correctly_1247[2-float32]`          | `test_cpu_predictor3_multisample_reads_correctly[s2-float32]`                |
| `test_cpu_predictor3_single_sample_still_works_1247`                       | `test_cpu_predictor3_single_sample_still_works`                              |
| `test_apply_predictor3_matches_tn3_reference_1247`                         | `test_apply_predictor3_matches_tn3_reference`                                |
| `test_predictor2_reads_libtiff_multibyte_correctly[uint16]`                | `test_predictor2_reads_libtiff_multibyte_correctly[pred2-libtiff[uint16]]`   |
| `test_predictor2_reads_libtiff_multibyte_correctly[int16]`                 | `test_predictor2_reads_libtiff_multibyte_correctly[pred2-libtiff[int16]]`    |
| `test_predictor2_reads_libtiff_multibyte_correctly[uint32]`                | `test_predictor2_reads_libtiff_multibyte_correctly[pred2-libtiff[uint32]]`   |
| `test_predictor2_reads_libtiff_multibyte_correctly[int32]`                 | `test_predictor2_reads_libtiff_multibyte_correctly[pred2-libtiff[int32]]`    |
| `test_predictor2_reads_libtiff_multiband_uint16`                           | `test_predictor2_reads_libtiff_multiband_uint16`                             |
| `test_predictor2_writer_interops_with_libtiff[uint16]`                     | `test_predictor2_writer_interops_with_libtiff[pred2-writer[uint16]]`         |
| `test_predictor2_writer_interops_with_libtiff[int16]`                      | `test_predictor2_writer_interops_with_libtiff[pred2-writer[int16]]`          |
| `test_predictor2_writer_interops_with_libtiff[uint32]`                     | `test_predictor2_writer_interops_with_libtiff[pred2-writer[uint32]]`         |
| `test_predictor2_writer_interops_with_libtiff[int32]`                      | `test_predictor2_writer_interops_with_libtiff[pred2-writer[int32]]`          |
| `test_gpu_predictor2_multibyte_matches_cpu[uint16]`                        | `test_gpu_predictor2_multibyte_matches_cpu[gpu-pred2[uint16]]`               |
| `test_gpu_predictor2_multibyte_matches_cpu[int16]`                         | `test_gpu_predictor2_multibyte_matches_cpu[gpu-pred2[int16]]`                |
| `test_gpu_predictor2_multibyte_matches_cpu[uint32]`                        | `test_gpu_predictor2_multibyte_matches_cpu[gpu-pred2[uint32]]`               |
| `test_gpu_predictor2_multibyte_writer_round_trip[uint16]`                  | `test_gpu_predictor2_multibyte_writer_round_trip[gpu-pred2-writer[uint16]]`  |
| `test_gpu_predictor2_multibyte_writer_round_trip[int16]`                   | `test_gpu_predictor2_multibyte_writer_round_trip[gpu-pred2-writer[int16]]`   |
| `test_gpu_predictor2_multibyte_writer_round_trip[uint32]`                  | `test_gpu_predictor2_multibyte_writer_round_trip[gpu-pred2-writer[uint32]]`  |
| `test_gpu_predictor2_multiband_uint16_matches_cpu`                         | `test_gpu_predictor2_multiband_uint16_matches_cpu`                           |
| `test_gpu_predictor2_writer_round_trip[uint16]`                            | `test_gpu_predictor2_writer_round_trip[gpu-pred2-encoder[uint16]]`           |
| `test_gpu_predictor2_writer_round_trip[int16]`                             | `test_gpu_predictor2_writer_round_trip[gpu-pred2-encoder[int16]]`            |
| `test_gpu_predictor2_writer_round_trip[uint32]`                            | `test_gpu_predictor2_writer_round_trip[gpu-pred2-encoder[uint32]]`           |
| `test_gpu_predictor3_multisample_matches_cpu_1479[3-float32]`              | `test_gpu_predictor3_multisample_matches_cpu[s3-float32]`                    |
| `test_gpu_predictor3_multisample_matches_cpu_1479[4-float32]`              | `test_gpu_predictor3_multisample_matches_cpu[s4-float32]`                    |
| `test_gpu_predictor3_multisample_matches_cpu_1479[3-float64]`              | `test_gpu_predictor3_multisample_matches_cpu[s3-float64]`                    |
| `test_gpu_predictor3_multisample_matches_cpu_1479[4-float64]`              | `test_gpu_predictor3_multisample_matches_cpu[s4-float64]`                    |

## Boundary

GPU predictor files stay in the GPU cluster #2438:
- `test_predictor2_big_endian_gpu_1517.py`
- `test_predictor3_int_dtype_gpu_1933.py`

GPU tests that lived inside CPU-named files move with this cluster
(int8 tiled+stripped, BE GPU, multi-sample GPU parity) so CPU and GPU
coverage for the same behaviour stays co-located, mirroring the
photometric cluster (PR #2451) which kept its GPU regressions.

## Counts

- Old test count (collected): 77 (76 passed + 1 skipped perf gate).
- New test count (collected): 81 (80 passed + 1 skipped perf gate).
- Net delta: +4 cases from filling out the BE x LE x dtype matrix for
  predictor=2 (uint8-<, int8-<, plus the four LE multi-byte cases that
  were only LE-uint16 before).

## File delta

- Deleted: 6 files (1332 lines).
- Added: 1 file (`unit/test_predictor.py`) plus this audit file
  (deleted pre-merge).
- Net: -5 test files.
