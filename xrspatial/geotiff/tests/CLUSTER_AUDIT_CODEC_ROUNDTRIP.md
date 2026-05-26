# Cluster Audit: Codec Round-Trip Consolidation (#2446 / cluster #2425 PR A)

Mapping every old `file::test` to its new `file::test_id` in
`xrspatial/geotiff/tests/unit/test_codec_roundtrip.py`.

This file is temporary. **Deleted on a final commit on the same branch
before the PR is approved.** Do not let it land on `main`.

| Old `file::test` | New `file::test_id` | Notes |
| --- | --- | --- |
| `test_jpeg.py::TestJpegCodec::test_grayscale_round_trip` | `TestJpegCodec::test_grayscale_round_trip` | unchanged |
| `test_jpeg.py::TestJpegCodec::test_rgb_round_trip` | `TestJpegCodec::test_rgb_round_trip` | unchanged |
| `test_jpeg.py::TestJpegCodec::test_quality_affects_size` | `TestJpegCodec::test_quality_affects_size` | unchanged |
| `test_jpeg.py::TestJpegCodec::test_invalid_samples_raises` | `TestJpegCodec::test_invalid_samples_raises` | unchanged |
| `test_jpeg.py::TestCompressionTagJpeg::test_jpeg_tag_value` | `TestJpegCompressionTag::test_jpeg_tag_value` | class renamed |
| `test_jpeg.py::TestCompressionTagJpeg::test_tag_value_is_7` | `TestJpegCompressionTag::test_tag_value_is_7` | class renamed |
| `test_jpeg.py::TestJpegWriteRoundTrip::test_grayscale_tiled` | `TestJpegWriteRoundTrip::test_jpeg_grayscale_roundtrip[jpeg[uint8-gray-tiled]]` | parametrised over tiled |
| `test_jpeg.py::TestJpegWriteRoundTrip::test_grayscale_stripped` | `TestJpegWriteRoundTrip::test_jpeg_grayscale_roundtrip[jpeg[uint8-gray-stripped]]` | parametrised |
| `test_jpeg.py::TestJpegWriteRoundTrip::test_rgb_tiled` | `TestJpegWriteRoundTrip::test_jpeg_rgb_tiled` | unchanged |
| `test_jpeg.py::TestJpegValidation::test_float_data_rejected` | `TestJpegValidation::test_jpeg_rejects_invalid_input[jpeg[reject-float32]]` | parametrised |
| `test_jpeg.py::TestJpegValidation::test_uint16_data_rejected` | `TestJpegValidation::test_jpeg_rejects_invalid_input[jpeg[reject-uint16]]` | parametrised |
| `test_jpeg.py::TestJpegValidation::test_4band_rejected` | `TestJpegValidation::test_jpeg_rejects_invalid_input[jpeg[reject-4-band]]` | parametrised |
| `test_jpeg.py::TestWriteGeotiffJpeg::test_to_geotiff_jpeg_rejected` | `TestPublicToGeotiffJpegRejected::test_to_geotiff_jpeg_rejected` | class renamed |
| `test_jpeg.py::TestJpegTablesSplice::test_splice_reconstructs_complete_jpeg` | `TestJpegTablesSplice::test_splice_reconstructs_complete_jpeg` | shared helper |
| `test_jpeg.py::TestJpegTablesSplice::test_splice_passthrough_on_empty_tables` | `TestJpegTablesSplice::test_splice_passthrough_on_empty_tables` | unchanged |
| `test_jpeg.py::TestJpegTablesSplice::test_splice_passthrough_on_invalid_input` | `TestJpegTablesSplice::test_splice_passthrough_on_invalid_input` | unchanged |
| `test_jpeg.py::TestJpegTablesSplice::test_jpeg_decompress_accepts_jpeg_tables_kwarg` | `TestJpegTablesSplice::test_jpeg_decompress_accepts_jpeg_tables_kwarg` | uses shared helper |
| `test_jpeg.py::TestGdalTiledJpegRead::test_tiled_ycbcr_jpeg` | `TestGdalTiledJpegRead::test_tiled_ycbcr_jpeg` | unchanged |
| `test_jpeg.py::TestGdalTiledJpegRead::test_tiled_grayscale_jpeg` | `TestGdalTiledJpegRead::test_tiled_grayscale_jpeg` | unchanged |
| `test_jpeg2000.py::TestJPEG2000Codec::test_roundtrip_uint8` | `TestJpeg2000Codec::test_jpeg2000_codec_roundtrip[jpeg2000[uint8-1band]]` | parametrised dtype×samples |
| `test_jpeg2000.py::TestJPEG2000Codec::test_roundtrip_uint16` | `TestJpeg2000Codec::test_jpeg2000_codec_roundtrip[jpeg2000[uint16-1band]]` | parametrised |
| `test_jpeg2000.py::TestJPEG2000Codec::test_roundtrip_multiband` | `TestJpeg2000Codec::test_jpeg2000_codec_roundtrip[jpeg2000[uint8-3band]]` | parametrised |
| `test_jpeg2000.py::TestJPEG2000Codec::test_single_pixel` | `TestJpeg2000Codec::test_jpeg2000_single_pixel` | unchanged |
| `test_jpeg2000.py::TestJPEG2000Codec::test_lossy_produces_smaller_output` | `TestJpeg2000Codec::test_jpeg2000_lossy_at_most_as_large_as_lossless` | renamed |
| `test_jpeg2000.py::TestJPEG2000Codec::test_dispatch_decompress` | `TestJpeg2000Codec::test_jpeg2000_dispatcher_decompress` | renamed |
| `test_jpeg2000.py::TestJPEG2000WriteRoundTrip::test_tiled_uint8` | `TestJpeg2000WriteRoundTrip::test_jpeg2000_writer_roundtrip[jpeg2000[uint8-tiled]]` | parametrised |
| `test_jpeg2000.py::TestJPEG2000WriteRoundTrip::test_tiled_uint16` | `TestJpeg2000WriteRoundTrip::test_jpeg2000_writer_roundtrip[jpeg2000[uint16-tiled]]` | parametrised |
| `test_jpeg2000.py::TestJPEG2000WriteRoundTrip::test_stripped_uint8` | `TestJpeg2000WriteRoundTrip::test_jpeg2000_writer_roundtrip[jpeg2000[uint8-stripped]]` | parametrised |
| `test_jpeg2000.py::TestJPEG2000WriteRoundTrip::test_with_geo_info` | `TestJpeg2000WriteRoundTrip::test_jpeg2000_with_geo_info` | unchanged |
| `test_jpeg2000.py::TestJPEG2000WriteRoundTrip::test_public_api_roundtrip` | `TestJpeg2000WriteRoundTrip::test_jpeg2000_public_api_roundtrip` | renamed |
| `test_jpeg2000.py::TestJPEG2000Availability::test_compression_constant` | `TestJpeg2000Availability::test_compression_constant` | unchanged |
| `test_jpeg2000.py::TestJPEG2000Availability::test_compression_tag_mapping` | `TestJpeg2000Availability::test_compression_tag_mapping` | unchanged |
| `test_jpeg2000.py::TestJPEG2000Availability::test_unavailable_raises_import_error` | `TestJpeg2000Availability::test_unavailable_raises_import_error` | unchanged |
| `test_lerc.py::TestLERCCodec::test_roundtrip_float32_lossless` | `TestLercCodec::test_lerc_codec_roundtrip_lossless[lerc[float32-lossless]]` | parametrised dtype |
| `test_lerc.py::TestLERCCodec::test_roundtrip_uint8_lossless` | `TestLercCodec::test_lerc_codec_roundtrip_lossless[lerc[uint8-lossless]]` | parametrised |
| `test_lerc.py::TestLERCCodec::test_roundtrip_uint16_lossless` | `TestLercCodec::test_lerc_codec_roundtrip_lossless[lerc[uint16-lossless]]` | parametrised |
| `test_lerc.py::TestLERCCodec::test_lossy_within_tolerance` | `TestLercCodec::test_lerc_lossy_within_tolerance` | renamed |
| `test_lerc.py::TestLERCCodec::test_lossy_smaller_than_lossless` | `TestLercCodec::test_lerc_lossy_smaller_than_lossless` | renamed |
| `test_lerc.py::TestLERCCodec::test_dispatch_decompress` | `TestLercCodec::test_lerc_dispatcher_decompress` | renamed |
| `test_lerc.py::TestLERCWriteRoundTrip::test_tiled_float32` | `TestLercWriteRoundTrip::test_lerc_writer_roundtrip[lerc[float32-tiled]]` | parametrised |
| `test_lerc.py::TestLERCWriteRoundTrip::test_tiled_uint8` | `TestLercWriteRoundTrip::test_lerc_writer_roundtrip[lerc[uint8-tiled]]` | parametrised |
| `test_lerc.py::TestLERCWriteRoundTrip::test_stripped_float32` | `TestLercWriteRoundTrip::test_lerc_writer_roundtrip[lerc[float32-stripped]]` | parametrised |
| `test_lerc.py::TestLERCWriteRoundTrip::test_public_api_roundtrip` | `TestLercWriteRoundTrip::test_lerc_public_api_roundtrip` | renamed |
| `test_lerc.py::TestLERCAvailability::test_compression_constant` | `TestLercAvailability::test_compression_constant` | unchanged |
| `test_lerc.py::TestLERCAvailability::test_compression_tag_mapping` | `TestLercAvailability::test_compression_tag_mapping` | unchanged |
| `test_lerc.py::TestLERCAvailability::test_unavailable_raises_import_error` | `TestLercAvailability::test_unavailable_raises_import_error` | unchanged |
| `test_lerc_max_z_error.py::TestLerclessLossless::test_lossless_roundtrip_bit_exact` | `TestLercMaxZError::test_lerc_max_z_error_lossless_bit_exact` | merged into one class |
| `test_lerc_max_z_error.py::TestLossyShrinksAndStaysWithinTolerance::test_lossy_smaller_and_bounded` | `TestLercMaxZError::test_lerc_max_z_error_lossy_smaller_and_bounded` | tolerance pinned verbatim |
| `test_lerc_max_z_error.py::TestStreamingDaskPath::test_dask_lerc_with_max_z_error` | `TestLercMaxZError::test_lerc_max_z_error_dask_streaming` | tolerance pinned verbatim |
| `test_lerc_max_z_error.py::TestValidation::test_max_z_error_with_non_lerc_codec_raises` | `TestLercMaxZError::test_lerc_max_z_error_with_non_lerc_codec_raises` | renamed |
| `test_lerc_max_z_error.py::TestValidation::test_negative_max_z_error_raises` | `TestLercMaxZError::test_lerc_negative_max_z_error_raises` | renamed |
| `test_lerc_max_z_error.py::TestValidation::test_max_z_error_zero_with_other_codec_is_allowed` | `TestLercMaxZError::test_lerc_max_z_error_zero_allowed_with_other_codec` | renamed |
| `test_lerc_valid_mask.py::TestLercDecompressWithMask::test_no_mask_returns_none` | `TestLercDecompressWithMask::test_lerc_no_mask_returns_none` | renamed |
| `test_lerc_valid_mask.py::TestLercDecompressWithMask::test_all_valid_mask_collapses_to_none` | `TestLercDecompressWithMask::test_lerc_all_valid_mask_collapses_to_none` | renamed |
| `test_lerc_valid_mask.py::TestLercDecompressWithMask::test_partial_mask_returns_array` | `TestLercDecompressWithMask::test_lerc_partial_mask_returns_array` | renamed |
| `test_lerc_valid_mask.py::TestLercDecompressWithMask::test_legacy_decompress_drops_mask` | `TestLercDecompressWithMask::test_lerc_legacy_decompress_drops_mask` | renamed |
| `test_lerc_valid_mask.py::TestLercTiffRoundTripWithMask::test_float32_nan_nodata` | `TestLercTiffRoundTripWithMask::test_lerc_valid_mask_roundtrip[lerc-mask[float32-nan]]` | parametrised dtype × nodata |
| `test_lerc_valid_mask.py::TestLercTiffRoundTripWithMask::test_float32_sentinel_nodata` | `TestLercTiffRoundTripWithMask::test_lerc_valid_mask_roundtrip[lerc-mask[float32-sentinel]]` | parametrised |
| `test_lerc_valid_mask.py::TestLercTiffRoundTripWithMask::test_uint16_sentinel_nodata` | `TestLercTiffRoundTripWithMask::test_lerc_valid_mask_roundtrip[lerc-mask[uint16-sentinel]]` | parametrised |
| `test_lerc_valid_mask.py::TestLercTiffRoundTripWithMask::test_no_mask_roundtrip_bitexact` | `TestLercTiffRoundTripWithMask::test_lerc_no_mask_roundtrip_bit_exact` | renamed |
| `test_lz4.py::TestLZ4Codec::test_roundtrip_simple` | `TestLz4Codec::test_lz4_codec_roundtrip[lz4[repetitive-text]]` | parametrised payload factory |
| `test_lz4.py::TestLZ4Codec::test_roundtrip_binary` | `TestLz4Codec::test_lz4_codec_roundtrip[lz4[binary-256]]` | parametrised |
| `test_lz4.py::TestLZ4Codec::test_roundtrip_empty` | `TestLz4Codec::test_lz4_codec_roundtrip[lz4[empty]]` | parametrised |
| `test_lz4.py::TestLZ4Codec::test_roundtrip_large` | `TestLz4Codec::test_lz4_codec_roundtrip[lz4[random-50k]]` | parametrised |
| `test_lz4.py::TestLZ4Codec::test_dispatch_roundtrip` | `TestLz4Codec::test_lz4_dispatcher_roundtrip` | renamed |
| `test_lz4.py::TestLZ4WriteRoundTrip::test_tiled_uint8` | `TestLz4WriteRoundTrip::test_lz4_writer_roundtrip[lz4[uint8-tiled]]` | parametrised |
| `test_lz4.py::TestLZ4WriteRoundTrip::test_tiled_float32` | `TestLz4WriteRoundTrip::test_lz4_writer_roundtrip[lz4[float32-tiled]]` | parametrised |
| `test_lz4.py::TestLZ4WriteRoundTrip::test_stripped_uint8` | `TestLz4WriteRoundTrip::test_lz4_writer_roundtrip[lz4[uint8-stripped]]` | parametrised |
| `test_lz4.py::TestLZ4WriteRoundTrip::test_with_predictor` | `TestLz4WriteRoundTrip::test_lz4_writer_roundtrip[lz4[float32-tiled-predictor]]` | parametrised |
| `test_lz4.py::TestLZ4WriteRoundTrip::test_public_api_roundtrip` | `TestLz4WriteRoundTrip::test_lz4_public_api_roundtrip` | renamed |
| `test_lz4.py::TestLZ4Availability::test_compression_constant` | `TestLz4Availability::test_compression_constant` | unchanged |
| `test_lz4.py::TestLZ4Availability::test_compression_tag_mapping` | `TestLz4Availability::test_compression_tag_mapping` | unchanged |
| `test_lz4.py::TestLZ4Availability::test_unavailable_raises_import_error` | `TestLz4Availability::test_unavailable_raises_import_error` | unchanged |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelRoundTrip::test_lz4_level_round_trip[0/1/9/16]` | `TestLz4CompressionLevel::test_lz4_level_round_trip[lz4-level[0/1/9/16]]` | id renamed |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelRoundTrip::test_lz4_default_level_round_trip` | `TestLz4CompressionLevel::test_lz4_default_level_round_trip` | unchanged |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelSizeEffect::test_lz4_higher_level_not_larger` | `TestLz4CompressionLevel::test_lz4_higher_level_not_larger` | flattened into one class |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelOutOfRange::test_lz4_out_of_range_level_raises_eager[*]` | `TestLz4CompressionLevel::test_lz4_out_of_range_level_raises_eager[lz4-level[-1/-10/17/100]]` | id renamed |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelOutOfRange::test_lz4_out_of_range_message_includes_range` | `TestLz4CompressionLevel::test_lz4_out_of_range_message_includes_range` | unchanged |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelDaskStreaming::test_lz4_dask_streaming_level_round_trip[*]` | `TestLz4CompressionLevelDask::test_lz4_dask_streaming_level_round_trip[lz4-dask[*]]` | id renamed |
| `test_lz4_compression_level_2026_05_11.py::TestLZ4LevelDaskStreaming::test_lz4_dask_streaming_out_of_range_raises[*]` | `TestLz4CompressionLevelDask::test_lz4_dask_streaming_out_of_range_raises[lz4-dask[*]]` | id renamed |
| `test_compression_level.py::TestRoundTrip::test_zstd_level_1` | `TestCompressionLevelRoundTrip::test_level_round_trip[zstd-level[1]]` | parametrised codec × level |
| `test_compression_level.py::TestRoundTrip::test_zstd_level_22` | `TestCompressionLevelRoundTrip::test_level_round_trip[zstd-level[22]]` | parametrised |
| `test_compression_level.py::TestRoundTrip::test_deflate_level_1` | `TestCompressionLevelRoundTrip::test_level_round_trip[deflate-level[1]]` | parametrised |
| `test_compression_level.py::TestRoundTrip::test_deflate_level_9` | `TestCompressionLevelRoundTrip::test_level_round_trip[deflate-level[9]]` | parametrised |
| `test_compression_level.py::TestLevelEffect::test_zstd_higher_level_not_larger` | `TestCompressionLevelEffect::test_higher_level_not_larger[zstd-level[1-vs-22]]` | parametrised |
| `test_compression_level.py::TestLevelEffect::test_deflate_higher_level_not_larger` | `TestCompressionLevelEffect::test_higher_level_not_larger[deflate-level[1-vs-9]]` | parametrised |
| `test_compression_level.py::TestDefaultLevel::test_none_uses_default_zstd` | `TestCompressionLevelDefault::test_zstd_none_uses_default` | renamed |
| `test_compression_level.py::TestDefaultLevel::test_omitted_uses_default_deflate` | `TestCompressionLevelDefault::test_deflate_omitted_uses_default` | renamed |
| `test_compression_level.py::TestLZWIgnoresLevel::test_lzw_with_level_does_not_raise` | `TestLzwIgnoresLevel::test_lzw_with_level_does_not_raise` | unchanged |
| `test_compression_level.py::TestInvalidLevels::test_zstd_level_0_raises` | `TestCompressionLevelOutOfRange::test_out_of_range_raises[zstd-level[0-reject]]` | parametrised |
| `test_compression_level.py::TestInvalidLevels::test_zstd_level_23_raises` | `TestCompressionLevelOutOfRange::test_out_of_range_raises[zstd-level[23-reject]]` | parametrised |
| `test_compression_level.py::TestInvalidLevels::test_deflate_level_0_raises` | `TestCompressionLevelOutOfRange::test_out_of_range_raises[deflate-level[0-reject]]` | parametrised |
| `test_compression_level.py::TestInvalidLevels::test_deflate_level_10_raises` | `TestCompressionLevelOutOfRange::test_out_of_range_raises[deflate-level[10-reject]]` | parametrised |
| `test_compression_level.py::TestInvalidLevels::test_negative_level_raises` | `TestCompressionLevelOutOfRange::test_out_of_range_raises[zstd-level[-1-reject]]` | parametrised |
| `test_compression_docstring_1644.py::test_write_geotiff_gpu_docstring_lists_full_codec_set` | `test_write_geotiff_gpu_docstring_lists_full_codec_set` | unchanged |
| `test_compression_docstring_1644.py::test_write_geotiff_gpu_accepts_cpu_fallback_codecs[*]` | `test_write_geotiff_gpu_accepts_cpu_fallback_codecs[*]` | unchanged |
