# Cluster 14 sub-PR D audit -- GPU kernels + kwargs

Source-file -> new-home mapping for the 12 files folded into
`xrspatial/geotiff/tests/gpu/test_kernels_and_kwargs.py`.

Every old `file::test` maps to a `file::test_id` in the consolidated
module. Test bodies are unchanged; helper names and fixture names are
suffixed with the source issue number where they collided across
sources.

This file is deleted in a final pre-merge commit per the epic #2424
hard gate.

---

## `test_gpu_cuda_preflight_1903.py`

| Old                                                       | New                                                              |
| ---                                                       | ---                                                              |
| `test_preflight_raises_on_runtime_error`                  | `test_preflight_raises_on_runtime_error_1903`                    |
| `test_preflight_raises_on_zero_devices`                   | `test_preflight_raises_on_zero_devices_1903`                     |
| `test_preflight_returns_silently_when_device_present`     | `test_preflight_returns_silently_when_device_present_1903`       |
| `test_read_geotiff_gpu_preflight_surface`                 | `test_read_geotiff_gpu_preflight_surface_1903`                   |
| `test_preflight_when_real_cupy_present`                   | `test_preflight_when_real_cupy_present_1903`                     |

## `test_gpu_kwarg_rename_1560.py`

| Old                                                  | New                                                       |
| ---                                                  | ---                                                       |
| `test_on_gpu_failure_invalid_value_raises_value_error` | `test_on_gpu_failure_invalid_value_raises_value_error_1560` |
| `test_gpu_alias_emits_deprecation_warning`           | `test_gpu_alias_emits_deprecation_warning_1560`           |
| `test_gpu_alias_accepts_old_values_without_validation_error` | `test_gpu_alias_accepts_old_values_without_validation_error_1560` |
| `test_passing_both_raises_type_error`                | `test_passing_both_raises_type_error_1560`                |
| `test_passing_both_raises_regardless_of_values`      | `test_passing_both_raises_regardless_of_values_1560`      |
| `test_gpu_alias_bool_no_longer_misleading_value_error` | `test_gpu_alias_bool_no_longer_misleading_value_error_1560` |

## `test_gpu_strict_fallback_1516.py`

| Old                                                | New                                                  |
| ---                                                | ---                                                  |
| `test_default_mode_warns_on_gpu_failure`           | `test_default_mode_warns_on_gpu_failure_1516`        |
| `test_strict_mode_reraises`                        | `test_strict_mode_reraises_1516`                     |
| `test_strict_mode_reraises_second_stage`           | `test_strict_mode_reraises_second_stage_1516`        |
| `test_default_mode_warns_on_second_stage_failure`  | `test_default_mode_warns_on_second_stage_failure_1516` |
| `test_invalid_gpu_kwarg_rejected`                  | `test_invalid_gpu_kwarg_rejected_1516`               |

## `test_gpu_fallback_forwards_kwargs_2238.py`

| Old                                                       | New                                                          |
| ---                                                       | ---                                                          |
| `test_stripped_fallback_forwards_allow_rotated`           | `test_stripped_fallback_forwards_allow_rotated_2238`         |
| `test_sparse_tile_fallback_forwards_all_kwargs`           | `test_sparse_tile_fallback_forwards_all_kwargs_2238`         |
| `test_gpu_decode_failure_fallback_forwards_all_kwargs`    | `test_gpu_decode_failure_fallback_forwards_all_kwargs_2238`  |
| `test_planar2_fallback_forwards_all_kwargs`               | `test_planar2_fallback_forwards_all_kwargs_2238`             |
| `test_decode_failure_fallback_applies_window_band`        | `test_decode_failure_fallback_applies_window_band_2238`      |

## `test_kvikio_batched_pread_1688.py`

| Old                                                  | New                                                       |
| ---                                                  | ---                                                       |
| `test_empty_tile_list_returns_empty_list`            | `test_empty_tile_list_returns_empty_list_1688`            |
| `test_kvikio_missing_returns_none`                   | `test_kvikio_missing_returns_none_1688`                   |
| `test_single_buffer_allocation`                      | `test_single_buffer_allocation_1688`                      |
| `test_all_preads_submitted_before_any_get`           | `test_all_preads_submitted_before_any_get_1688`           |
| `test_memory_guard_runs_with_total_byte_count`       | `test_memory_guard_runs_with_total_byte_count_1688`       |
| `test_partial_read_returns_none`                     | `test_partial_read_returns_none_1688`                     |
| `test_round_trip_data_preserved`                     | `test_round_trip_data_preserved_1688`                     |
| `test_zero_size_tile_returns_zero_length_view`       | `test_zero_size_tile_returns_zero_length_view_1688`       |
| `test_all_zero_size_tiles_returns_zero_length_views` | `test_all_zero_size_tiles_returns_zero_length_views_1688` |

## `test_gds_chunked_gpu_parity_1896.py`

| Old                                              | New                                                |
| ---                                              | ---                                                |
| `test_gds_chunked_band_true_rejected`            | `test_gds_chunked_band_true_rejected_1896`         |
| `test_gds_chunked_band_false_rejected`           | `test_gds_chunked_band_false_rejected_1896`        |
| `test_gds_chunked_band_np_bool_rejected`         | `test_gds_chunked_band_np_bool_rejected_1896`      |
| `test_gds_chunked_band_int_still_works`          | `test_gds_chunked_band_int_still_works_1896`       |
| `test_gds_chunked_lerc_mask_matches_eager`       | `test_gds_chunked_lerc_mask_matches_eager_1896`    |
| `test_gds_chunked_lerc_mask_sentinel_nodata`     | `test_gds_chunked_lerc_mask_sentinel_nodata_1896`  |

## `test_gds_fallback_batched_d2h_1552.py`

| Old                                                  | New                                                       |
| ---                                                  | ---                                                       |
| `test_batched_d2h_empty_list`                        | `test_batched_d2h_empty_list_1552`                        |
| `test_batched_d2h_matches_per_tile_get`              | `test_batched_d2h_matches_per_tile_get_1552`              |
| `test_batched_d2h_single_tile`                       | `test_batched_d2h_single_tile_1552`                       |
| `test_batched_d2h_zero_size_tile_in_list`            | `test_batched_d2h_zero_size_tile_in_list_1552`            |
| `test_batched_d2h_checks_gpu_memory_before_concat`   | `test_batched_d2h_checks_gpu_memory_before_concat_1552`   |
| `test_batched_d2h_many_small_tiles`                  | `test_batched_d2h_many_small_tiles_1552`                  |

## `test_orientation_gpu.py`

| Old                                                  | New                                                |
| ---                                                  | ---                                                |
| `test_gpu_tiled_matches_cpu`                         | `test_gpu_tiled_matches_cpu_orientation`           |
| `test_gpu_stripped_matches_cpu`                      | `test_gpu_stripped_matches_cpu_orientation`        |
| `test_gpu_3band_tiled_matches_cpu`                   | `test_gpu_3band_tiled_matches_cpu_orientation`     |
| `test_gpu_orient_2_3_4_coords_track_pixel_flip`      | `test_gpu_orient_2_3_4_coords_track_pixel_flip`    |
| `test_gpu_default_orientation_unchanged`             | `test_gpu_default_orientation_unchanged`           |
| `test_gpu_orientation_5_to_8_raise_on_georef`        | `test_gpu_orientation_5_to_8_raise_on_georef`      |
| `test_gpu_orientation_5_to_8_transform_only_raises`  | `test_gpu_orientation_5_to_8_transform_only_raises` |
| `test_gpu_orientation_5_to_8_no_georef_still_swaps`  | `test_gpu_orientation_5_to_8_no_georef_still_swaps` |

(The orientation tests already had unique-enough names; no rename needed.)

## `test_size_param_validation_gpu_vrt_1776.py`

| Old class                                             | New class                                              |
| ---                                                   | ---                                                    |
| `TestWriteGeotiffGpuTileSize`                         | `TestWriteGeotiffGpuTileSize_1776`                     |
| `TestReadGeotiffGpuChunks`                            | `TestReadGeotiffGpuChunks_1776`                        |
| `TestReadVrtChunks`                                   | `TestReadVrtChunks_1776`                               |
| `TestOpenGeotiffGpuChunksDispatch`                    | `TestOpenGeotiffGpuChunksDispatch_1776`                |
| `TestToGeotiffGpuTileSizeAlreadyChecked`              | `TestToGeotiffGpuTileSizeAlreadyChecked_1776`          |
| `TestNoDoubleValidationSideEffects`                   | `TestNoDoubleValidationSideEffects_1776`               |
| `TestChunksNoneAcrossEntryPoints`                     | `TestChunksNoneAcrossEntryPoints_1776`                 |

Test method names inside each class are unchanged.

## `test_mask_nodata_gpu_vrt_2052.py`

Every top-level function in the source file gets a `_2052` suffix in
the consolidated module. Examples:

| Old                                                          | New                                                               |
| ---                                                          | ---                                                               |
| `test_read_geotiff_gpu_mask_nodata_false_preserves_uint16`   | `test_read_geotiff_gpu_mask_nodata_false_preserves_uint16_2052`   |
| `test_cross_backend_parity_eager_gpu`                        | `test_cross_backend_parity_eager_gpu_2052`                        |
| `test_read_geotiff_dask_direct_mask_nodata_false`            | `test_read_geotiff_dask_direct_mask_nodata_false_2052`            |

(20 functions total, all suffixed the same way.)

## `test_open_geotiff_on_gpu_failure_1615.py`

| Old                                                  | New                                                       |
| ---                                                  | ---                                                       |
| `test_open_geotiff_signature_includes_on_gpu_failure` | `test_open_geotiff_signature_includes_on_gpu_failure_1615` |
| `test_on_gpu_failure_with_gpu_false_raises_value_error` | `test_on_gpu_failure_with_gpu_false_raises_value_error_1615` |
| `test_on_gpu_failure_with_explicit_gpu_false_raises` | `test_on_gpu_failure_with_explicit_gpu_false_raises_1615` |
| `test_on_gpu_failure_with_chunks_only_raises`        | `test_on_gpu_failure_with_chunks_only_raises_1615`        |
| `test_default_dispatch_unchanged_cpu`                | `test_default_dispatch_unchanged_cpu_1615`                |
| `test_default_dispatch_unchanged_dask`               | `test_default_dispatch_unchanged_dask_1615`               |
| `test_open_geotiff_gpu_forwards_on_gpu_failure_auto` | `test_open_geotiff_gpu_forwards_on_gpu_failure_auto_1615` |
| `test_open_geotiff_gpu_forwards_on_gpu_failure_strict` | `test_open_geotiff_gpu_forwards_on_gpu_failure_strict_1615` |
| `test_open_geotiff_gpu_rejects_invalid_on_gpu_failure` | `test_open_geotiff_gpu_rejects_invalid_on_gpu_failure_1615` |
| `test_invalid_on_gpu_failure_reaches_gpu_validator_on_cpu` | `test_invalid_on_gpu_failure_reaches_gpu_validator_on_cpu_1615` |

## `test_crs_fail_closed_gpu_1929.py`

| Old class                                  | New class                                       |
| ---                                        | ---                                             |
| `TestWriteGeotiffGpuFailClosed`            | `TestWriteGeotiffGpuFailClosed_1929`            |
| `TestToGeotiffGpuDispatcherFailClosed`     | `TestToGeotiffGpuDispatcherFailClosed_1929`     |
| `TestErrorMessageParity`                   | `TestErrorMessageParity_1929`                   |
| `TestKwargDefaultParity`                   | `TestKwargDefaultParity_1929`                   |

Test method names inside each class are unchanged.

---

## Test counts

- Old: 163 tests collected across the 12 source files.
- New: 163 tests collected in `gpu/test_kernels_and_kwargs.py`.

Verified via `pytest --collect-only -q` against both layouts.

## Cross-cutting follow-ups

- `docs/source/reference/geotiff.rst`: pointer to
  `test_gpu_strict_fallback_1516.py` updated to the new module path.
- `docs/source/reference/release_gate_geotiff.rst`: three pointers
  (`test_mask_nodata_gpu_vrt_2052.py`, `test_crs_fail_closed_gpu_1929.py`,
  `test_gpu_strict_fallback_1516.py` + `test_gpu_fallback_forwards_kwargs_2238.py`)
  redirected to the new module.
- `pytest.importorskip("lerc")` and `pytest.importorskip("tifffile")`
  moved from module scope to per-test gating so unrelated sections
  still collect when one optional extra is missing.
- The `TestKwargDefaultParity_1929` signature-introspection class
  no longer carries the section-level `@_gpu_only` (the original
  module-level `pytestmark` swept it up). The test only inspects
  function signatures and runs cleanly on CPU.
