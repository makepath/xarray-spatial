# Cluster audit -- photometric MinIsWhite + internal-only-JPEG tier

Issue: #2445 (PR B of cluster #2425, long-tail epic #2424).

This file is deleted on the pre-merge commit. Do not let it land on main.

## Section 1: MinIsWhite read/write semantics

### `test_miniswhite_nodata_1809.py` -> `unit/test_photometric.py`

| Old test | New test / param id |
|---|---|
| `test_eager_numpy_uint8_nodata_zero` | `test_eager_numpy_miniswhite_nodata[miniswhite_read[uint8-nodata-0]]` |
| `test_dask_uint8_nodata_zero` | `test_dask_miniswhite_nodata[miniswhite_read[uint8-dask]]` |
| `test_gpu_eager_uint8_nodata_zero` | `test_gpu_eager_miniswhite_uint8_nodata_zero` |
| `test_eager_numpy_uint16_nodata_max` | `test_eager_numpy_miniswhite_nodata[miniswhite_read[uint16-nodata-max]]` |
| `test_eager_numpy_float32_nodata` | `test_eager_numpy_miniswhite_nodata[miniswhite_read[float32-nodata-neg9999]]` |
| `test_dask_float32_nodata` | `test_dask_miniswhite_nodata[miniswhite_read[float32-dask]]` |
| `test_eager_numpy_uint8_nodata_no_collision` | `test_eager_miniswhite_uint8_no_collision` |
| `test_eager_numpy_no_nodata_stays_integer` | `test_eager_miniswhite_no_nodata_stays_integer` |
| `test_gpu_sparse_tile_miniswhite_nodata_zero` | `test_gpu_sparse_tile_miniswhite_nodata_zero` |
| `test_backend_parity_uint8_nodata_zero` | `test_miniswhite_backend_parity_uint8_nodata_zero` |

### `test_miniswhite_writer_roundtrip_1836.py` -> `unit/test_photometric.py`

| Old test | New test / param id |
|---|---|
| `test_uint8_miniswhite_round_trip` | `test_miniswhite_round_trip[miniswhite_roundtrip[uint8]]` |
| `test_uint16_miniswhite_round_trip` | `test_miniswhite_round_trip[miniswhite_roundtrip[uint16]]` |
| `test_float32_miniswhite_round_trip` | `test_miniswhite_round_trip[miniswhite_roundtrip[float32]]` |
| `test_miniswhite_with_nodata_round_trip` | `test_miniswhite_float_with_nodata_round_trips_nan` |
| `test_miniswhite_int_rejected_at_write_2278` | `test_write_signed_miniswhite_int16_error_message_shape` |
| `test_uint16_miniswhite_with_in_range_nodata_round_trip` | `test_miniswhite_uint16_in_range_nodata_round_trips_nan` |
| `test_miniswhite_rejected_with_cog_no_overviews` | `test_miniswhite_rejected_with_cog_no_overviews` |
| `test_miniswhite_rejected_with_explicit_overviews_no_cog` | `test_miniswhite_rejected_with_explicit_overviews_no_cog` |
| `test_miniswhite_rejected_on_dask_path` | `test_miniswhite_rejected_on_dask_path` |
| `test_miniswhite_rejected_with_extra_tags_photometric_override` | `test_miniswhite_rejected_with_extra_tags_photometric_override` |

### `test_signed_miniswhite_rejected_2278.py` -> `unit/test_photometric.py`

| Old test | New test / param id |
|---|---|
| `test_write_signed_miniswhite_rejected_2278[int8-8]` | `test_write_signed_miniswhite_rejected[signed_miniswhite_rejected[int8]]` |
| `test_write_signed_miniswhite_rejected_2278[int16-16]` | `test_write_signed_miniswhite_rejected[signed_miniswhite_rejected[int16]]` |
| `test_write_signed_miniswhite_rejected_2278[int32-32]` | `test_write_signed_miniswhite_rejected[signed_miniswhite_rejected[int32]]` |
| `test_write_signed_miniswhite_rejected_2278[int64-64]` | `test_write_signed_miniswhite_rejected[signed_miniswhite_rejected[int64]]` |
| `test_write_signed_miniswhite_does_not_partial_write_2278` | `test_write_signed_miniswhite_no_partial_file` |
| `test_read_signed_miniswhite_rejected_2278` | `test_read_signed_miniswhite_rejected` |
| `test_unsigned_miniswhite_still_round_trips_2278` | `test_unsigned_and_float_miniswhite_still_round_trip[signed_miniswhite_nonregression[uint8]]` |
| `test_float_miniswhite_still_round_trips_2278` | `test_unsigned_and_float_miniswhite_still_round_trip[signed_miniswhite_nonregression[float32]]` |
| `test_read_signed_miniswhite_rejected_on_gpu_path_2278` | `test_read_signed_miniswhite_rejected_on_gpu_path` |

## Section 2: `allow_internal_only_jpeg` API tier

### `test_to_geotiff_allow_internal_only_jpeg_parity.py` -> `unit/test_photometric.py`

| Old test | New test / param id |
|---|---|
| `test_to_geotiff_signature_has_allow_internal_only_jpeg` | `test_to_geotiff_signature_has_allow_internal_only_jpeg` |
| `test_to_geotiff_rejects_jpeg_without_opt_in` | `test_to_geotiff_rejects_jpeg_without_opt_in[allow_internal_only_jpeg[default-rejects]]` |
| `test_to_geotiff_jpeg_rejection_message_mentions_opt_in` | `test_to_geotiff_rejects_jpeg_without_opt_in[allow_internal_only_jpeg[error-mentions-flag]]` |
| `test_to_geotiff_jpeg_opt_in_emits_warning_and_writes` | `test_to_geotiff_jpeg_opt_in_emits_warning_and_writes` |
| `test_to_geotiff_non_jpeg_unaffected_by_flag` | `test_to_geotiff_non_jpeg_unaffected_by_flag` |
| `test_to_geotiff_and_gpu_writer_share_kwarg_default` | `test_to_geotiff_and_gpu_writer_share_kwarg_default` |
| `test_to_geotiff_gpu_dispatch_forwards_allow_internal_only_jpeg` | `test_to_geotiff_gpu_dispatch_forwards_allow_internal_only_jpeg` |
| `test_to_geotiff_gpu_dispatch_emits_single_jpeg_opt_in_warning` | `test_to_geotiff_gpu_dispatch_emits_single_jpeg_opt_in_warning` |

## File delta

- Deleted: 4 top-level test files (listed above).
- Added: `xrspatial/geotiff/tests/unit/test_photometric.py`.
- Net: -3 files in `xrspatial/geotiff/tests/test_*.py`.
