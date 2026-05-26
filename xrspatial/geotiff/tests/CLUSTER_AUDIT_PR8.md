# CLUSTER_AUDIT_PR8.md — Reader-path tests

Temporary audit table mapping every old `file::test` to its new home in
the `read/` cluster. Deleted in a follow-up commit on the same branch
before merge, per the epic #2390 contract.

## Cluster split

PR 8 owns the reader-side cluster. The following eight files land under
`xrspatial/geotiff/tests/read/`:

- `read/test_basic.py` — minimal read paths, band validation.
- `read/test_dtypes.py` — reader dtype handling (eager / dask / GPU).
- `read/test_compression.py` — decompression-codec round-trips and
  bomb caps (DEFLATE / LZW / ZSTD / PACKBITS / LZ4 / LERC / JPEG2000 /
  JPEG).
- `read/test_tiling.py` — tile / strip byte-count cap on CPU and GPU.
- `read/test_endianness.py` — big-endian multi-byte read paths.
- `read/test_nodata.py` — nodata propagation on read (GPU helper).
- `read/test_coords.py` — descending / ascending coord round-trip.
- `read/test_streaming.py` — streaming BigTIFF threshold (folds in
  `xrspatial/tests/test_geotiff_streaming_bigtiff_threshold_1785.py`).

PR 3's `read/test_crs.py` (rotated / dropped / missing CRS) is the
parallel sibling and is left for that PR.

## Folded files

### `test_band_validation_1673.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_band_validation_1673.py::test_read_to_array_negative_band_rejected` | `read/test_basic.py::TestBandValidationLocal::test_negative_band_rejected` | Renamed, class-grouped. Same assertion. |
| `test_band_validation_1673.py::test_read_to_array_band_equal_to_samples_rejected` | `read/test_basic.py::TestBandValidationLocal::test_band_equal_to_samples_rejected` | Same. |
| `test_band_validation_1673.py::test_read_to_array_band_far_above_samples_rejected` | `read/test_basic.py::TestBandValidationLocal::test_band_far_above_samples_rejected` | Same. |
| `test_band_validation_1673.py::test_read_to_array_valid_band_still_works` | `read/test_basic.py::TestBandValidationLocal::test_valid_band_still_works` | Same. |
| `test_band_validation_1673.py::test_read_to_array_band_none_still_returns_all_bands` | `read/test_basic.py::TestBandValidationLocal::test_band_none_returns_all_bands` | Same. |
| `test_band_validation_1673.py::test_backend_parity_negative_band` | `read/test_basic.py::TestBandValidationBackendParity::test_negative_band` | Class-grouped. |
| `test_band_validation_1673.py::test_backend_parity_band_equal_to_samples` | `read/test_basic.py::TestBandValidationBackendParity::test_band_equal_to_samples` | Class-grouped. |
| (fixture) `multiband_tiff_path` | same fixture in `read/test_basic.py` | Filename in tmp_path renamed `mb_1673.tif` -> `mb_band_validation.tif`. |

### `test_dtype_read.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_dtype_read.py::TestDtypeEager::*` | `read/test_dtypes.py::TestDtypeEager::*` | Verbatim. Fixture filenames renamed `test_1083_*.tif` -> `dtype_*.tif`. |
| `test_dtype_read.py::TestDtypeDask::*` | `read/test_dtypes.py::TestDtypeDask::*` | Same. |

### `test_float16_read_1941.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestDtypeMap::*` | `read/test_dtypes.py::TestFloat16DtypeMap::*` | Renamed to disambiguate from generic dtype map tests. Body unchanged. |
| `TestEagerFloat16Read::*` | `read/test_dtypes.py::TestEagerFloat16Read::*` | Verbatim. |
| `TestPredictor3Float16::*` | `read/test_dtypes.py::TestPredictor3Float16::*` | Verbatim. |
| `TestRegressionGuards::*` | `read/test_dtypes.py::TestFloat16RegressionGuards::*` | Class renamed (no name collisions with other regression-guard classes). |

### `test_float16_read_gpu_1941.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestEagerGPUReadFloat16::*` | `read/test_dtypes.py::TestEagerGPUReadFloat16::*` | Body unchanged. Module-level `pytestmark` skip replaced with per-method `@_gpu_only` since the consolidated file mixes GPU and non-GPU tests. |
| `TestGPUWindowedFloat16::*` | `read/test_dtypes.py::TestGPUWindowedFloat16::*` | Same. |
| `TestDaskGPUFloat16::*` | `read/test_dtypes.py::TestDaskGPUFloat16::*` | Same. |
| `TestGDSPathGatedOffForFloat16::*` | `read/test_dtypes.py::TestGDSPathGatedOffForFloat16::*` | Same. |
| `TestBackendParityFloat16::*` | `read/test_dtypes.py::TestBackendParityFloat16::*` | Same. |
| `TestPredictor3Float16GPU::*` | `read/test_dtypes.py::TestPredictor3Float16GPU::*` | Same. |

### `test_compression.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestDeflate::*` | `read/test_compression.py::TestDeflate::*` | Verbatim. |
| `TestLZW::*` | `read/test_compression.py::TestLZW::*` | Verbatim. |
| `TestPredictor::*` | `read/test_compression.py::TestPredictor::*` | Verbatim. |
| `TestDispatch::*` | `read/test_compression.py::TestDispatch::*` | Verbatim. |

### `test_decompression_caps.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestCodecDirect::*` | `read/test_compression.py::TestCodecDirect::*` | Verbatim. |
| `TestZstdDirect::*` | `read/test_compression.py::TestZstdDirect::*` | Verbatim. |
| `TestLz4Direct::*` | `read/test_compression.py::TestLz4Direct::*` | Verbatim. |
| `test_deflate_bomb_rejected` | `read/test_compression.py::test_deflate_bomb_rejected` | Verbatim. |
| `test_zstd_bomb_rejected` | `read/test_compression.py::test_zstd_bomb_rejected` | Verbatim. |
| `test_lz4_bomb_rejected` | `read/test_compression.py::test_lz4_bomb_rejected` | Verbatim. |
| `test_packbits_bomb_rejected` | `read/test_compression.py::test_packbits_bomb_rejected` | Verbatim. |
| `test_legitimate_high_compression_passes` | `read/test_compression.py::test_legitimate_high_compression_passes` | Verbatim. |
| `test_cap_includes_metadata_margin` | `read/test_compression.py::test_cap_includes_metadata_margin` | Verbatim. |
| `TestLercDirect::*` | `read/test_compression.py::TestLercDirect::*` | Verbatim. |
| `TestJpeg2000Direct::*` | `read/test_compression.py::TestJpeg2000Direct::*` | Verbatim. |
| `TestJpegDirect::*` | `read/test_compression.py::TestJpegDirect::*` | Verbatim. |

### `test_local_tile_byte_cap_1664.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestLocalTileByteCap::*` | `read/test_tiling.py::TestLocalTileByteCap::*` | Verbatim. Fixture filenames renamed `forged_local_*_1664.tif` -> `forged_*.tif`. |
| `TestLocalStripByteCap::*` | `read/test_tiling.py::TestLocalStripByteCap::*` | Same. |
| `test_max_tile_bytes_env_negative_falls_back` | `read/test_tiling.py::test_max_tile_bytes_env_negative_falls_back` | Verbatim. |
| `test_max_tile_bytes_env_zero_falls_back` | `read/test_tiling.py::test_max_tile_bytes_env_zero_falls_back` | Verbatim. |
| `test_max_tile_bytes_env_garbage_falls_back` | `read/test_tiling.py::test_max_tile_bytes_env_garbage_falls_back` | Verbatim. |
| Import: `from ._helpers.tiff_surgery import ...` | `from .._helpers.tiff_surgery import ...` | One-level deeper under `read/`. |

### `test_gpu_tile_byte_cap_2026_05_18.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestGpuTileByteCap::*` | `read/test_tiling.py::TestGpuTileByteCap::*` | Verbatim. Shares `_build_forged_tiled_cog` helper with the CPU class via a `basename` parameter so the two CPU-vs-GPU forged-tile groups do not collide on `tmp_path`. |
| `TestGpuChunkedTileByteCap::*` | `read/test_tiling.py::TestGpuChunkedTileByteCap::*` | Verbatim. |

### `test_gpu_byteswap_1508.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_read_geotiff_gpu_big_endian_multibyte[*]` | `read/test_endianness.py::test_read_geotiff_gpu_big_endian_multibyte[*]` | Verbatim. |
| `test_read_geotiff_gpu_big_endian_uncompressed` | `read/test_endianness.py::test_read_geotiff_gpu_big_endian_uncompressed` | Verbatim. |
| `test_xp_byteswap_preserves_dtype` | `read/test_endianness.py::test_xp_byteswap_preserves_dtype` | Verbatim. |
| `test_xp_byteswap_uint8_passthrough` | `read/test_endianness.py::test_xp_byteswap_uint8_passthrough` | Verbatim. |

### `test_apply_nodata_mask_gpu_inplace_1934.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_apply_nodata_mask_gpu_float_masks_sentinel_to_nan_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_float_masks_sentinel_to_nan` | Issue number dropped from name. Body unchanged. |
| `test_apply_nodata_mask_gpu_float_in_place_no_copy_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_float_in_place_no_copy` | Same. |
| `test_apply_nodata_mask_gpu_float_alloc_count_unchanged_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_float_alloc_count_unchanged` | Same. |
| `test_apply_nodata_mask_gpu_int_promotes_and_masks_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_int_promotes_and_masks` | Same. |
| `test_apply_nodata_mask_gpu_int_no_extra_buffer_after_astype_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_int_no_extra_buffer_after_astype` | Same. |
| `test_apply_nodata_mask_gpu_float_nan_sentinel_noop_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_float_nan_sentinel_noop` | Same. |
| `test_apply_nodata_mask_gpu_none_nodata_passthrough_1934` | `read/test_nodata.py::test_apply_nodata_mask_gpu_none_nodata_passthrough` | Same. |

### `test_apply_nodata_mask_gpu_with_presence_removed_2208.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_apply_nodata_mask_gpu_with_presence_not_importable_2208` | `read/test_nodata.py::test_apply_nodata_mask_gpu_with_presence_not_importable` | Issue number dropped. Same `ImportError` assertion. |
| `test_apply_nodata_mask_gpu_still_present_2208` | `read/test_nodata.py::test_apply_nodata_mask_gpu_still_present` | Same. |

### `test_descending_coords_1716.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_descending_x_roundtrip` | `read/test_coords.py::TestDescendingCoordsRoundTrip::test_descending_x_roundtrip` | Class-grouped. tmp_path filenames renamed (`tmp_1716_desc_x.tif` -> `desc_x.tif`). |
| `test_ascending_y_roundtrip` | `read/test_coords.py::TestDescendingCoordsRoundTrip::test_ascending_y_roundtrip` | Same. |
| `test_descending_x_and_ascending_y_roundtrip` | `read/test_coords.py::TestDescendingCoordsRoundTrip::test_descending_x_and_ascending_y_roundtrip` | Same. |
| `test_north_up_still_uses_pixel_scale_and_tiepoint` | `read/test_coords.py::TestOrientationTagSelection::test_north_up_uses_pixel_scale_and_tiepoint` | Class-grouped, name slimmed. |
| `test_descending_x_uses_transformation_tag` | `read/test_coords.py::TestOrientationTagSelection::test_descending_x_uses_transformation_tag` | Same. |
| `test_ascending_y_uses_transformation_tag` | `read/test_coords.py::TestOrientationTagSelection::test_ascending_y_uses_transformation_tag` | Same. |

### `xrspatial/tests/test_geotiff_streaming_bigtiff_threshold_1785.py` (deleted — cross-directory move)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `TestShouldUseBigTIFFStreaming::*` | `read/test_streaming.py::TestShouldUseBigTIFFStreaming::*` | Verbatim. |
| `TestStreamingBigTIFFUserOverride::*` | `read/test_streaming.py::TestStreamingBigTIFFUserOverride::*` | Verbatim. Fixture filenames renamed `*_1785.tif` -> issue-number-free. |

## Files NOT folded in (justified)

Several files in the prompt's "key examples" list turned out to be
writer-side or unit-level on inspection and would conflict with another
PR's surface. They are left in place for their natural cluster:

| File | Reason left in place |
|---|---|
| `test_accuracy_1081.py` | Mixed read/write numerical accuracy with parity surface area; folding into `read/test_basic.py` would expand PR scope beyond the reader-only contract. Defer to PR 11 unit-cleanup. |
| `test_ambiguous_metadata_hooks_1987.py` | Metadata contract / parity surface — overlaps with PR 4 (parity) and PR 5 (attrs contract). |
| `test_assemble_layout_no_bytes_copy_1756.py` | Tests `_assemble_standard_layout`, `_assemble_cog_layout`, `_assemble_tiff` — writer internals. Belongs to PR 7. |
| `test_bytesio_source.py` | Mixed BytesIO read/write; round-trip surface area is large and the file already groups its own concerns coherently. Defer to PR 11. |
| `test_chunked_gpu_declared_dtype_1909.py` | Mixed dtype/dask coverage that overlaps with the parity matrix (PR 4). |
| `test_compression_docstring_1644.py` | Tests `write_geotiff_gpu` docstring + GPU writer codec acceptance — writer-side. Belongs to PR 7. |
| `test_compression_level.py` | Tests `to_geotiff(compression_level=...)` — writer-side. Belongs to PR 7. |
| `test_conflicting_crs_write_1987.py` | Writer-side (CRS conflict on write). Belongs to PR 7. |
| `test_coord_regularity_1720.py` | Tests `_coords_to_transform` validation on the writer path. Belongs to PR 7. |
| `test_coords_1813.py` | Unit tests of `xrspatial.geotiff._coords` helpers — fits `unit/` (PR 11). |
| `test_coords_to_transform_3d_1643.py` | Writer-side coord-to-transform. Belongs to PR 7. |
| `test_predictor2_big_endian.py` / `test_predictor2_big_endian_gpu_1517.py` / `test_predictor3_big_endian.py` / `test_predictor3_int_dtype*` / `test_predictor_fp_write_*` | Predictor coverage overlaps with the writer codec matrix (PR 7) and the parity matrix (PR 4). Defer to a future endianness/predictor sub-cluster rather than risk colliding mid-PR. |

## Verification

- 134 tests collected in `xrspatial/geotiff/tests/read/` after PR 8 (8
  modules, including PR 3's `test_crs.py` once that PR lands).
- Total `test_*.py` files removed across the PR: 13 (12 inside
  `geotiff/tests/`, plus the one cross-directory move from
  `xrspatial/tests/test_geotiff_streaming_bigtiff_threshold_1785.py`).
- New `test_*.py` files added under `read/`: 8 (plus the empty
  `__init__.py`).
- Net delta inside `geotiff/tests/`: -12 + 8 = -4 `test_*.py` files
  (`find xrspatial/geotiff/tests -name 'test_*.py' | wc -l` goes from
  352 to 348).
- Net delta inside `xrspatial/tests/`: -1 `test_*.py` file.
- Total PR-wide `test_*.py` delta: -5.
