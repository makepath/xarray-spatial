# CLUSTER_AUDIT_PR6.md

PR 6 of epic #2390 (issue #2399). Maps every test in the 28 folded
issue-numbered VRT files to its new home in `vrt/test_metadata.py`,
`vrt/test_window.py`, or `vrt/test_dtype_conversion.py`. Helpers were
prefixed per source file (e.g. `_holes_attr_*`) so cross-file folds do
not collide; test names dropped trailing `_NNNN` suffixes where the
originating file already namespaced them. No tests were dropped.

This file is removed on a final commit on the same branch before the
PR is approved -- it must not land on `main`.

## vrt/test_metadata.py (was 12 files)

| Old file:test | New file:test_id |
|---|---|
| `test_vrt_holes_attr_1734.py::test_skipped_source_records_vrt_holes_attr` | `vrt/test_metadata.py::test_skipped_source_records_vrt_holes_attr` |
| `test_vrt_holes_attr_1734.py::test_no_holes_attr_when_all_sources_read` | `vrt/test_metadata.py::test_no_holes_attr_when_all_sources_read` |
| `test_vrt_holes_attr_1734.py::test_strict_mode_still_raises` | `vrt/test_metadata.py::test_strict_mode_still_raises` |
| `test_vrt_holes_attr_1734.py::test_warning_mentions_how_to_detect_holes` | `vrt/test_metadata.py::test_warning_mentions_how_to_detect_holes` |
| `test_vrt_masked_nodata_attr_2159.py::*` (9 tests) | `vrt/test_metadata.py::*` (same 9 names) |
| `test_vrt_band_nodata_1598.py::*` (7 tests) | `vrt/test_metadata.py::*` (same 7 names) |
| `test_vrt_source_nodata_zero_1655.py::TestVRTSourceNodataZero::*` (5 tests) | `vrt/test_metadata.py::TestVRTSourceNodataZero::*` (same 5 names) |
| `test_vrt_int_nodata_1564.py::*` (7 tests) | `vrt/test_metadata.py::*` (same 7 names) |
| `test_vrt_mask_nodata_float_source_2158.py::*` (10 tests) | `vrt/test_metadata.py::*` (same 10 names) |
| `test_vrt_tiled_metadata_1606.py::TestVrtTiledMetadataParity::*` etc. (12 tests across 3 classes) | `vrt/test_metadata.py::TestVrtTiledMetadataParity::*` etc. |
| `test_vrt_single_parse_1825.py::*` (6 tests) | `vrt/test_metadata.py::*` (same 6 names) |
| `test_vrt_xml_escape_1607.py::*` (4 tests) | `vrt/test_metadata.py::*` (same 4 names) |
| `test_vrt_xml_size_cap_1815.py::*` (4 tests + parametrize) | `vrt/test_metadata.py::*` (same names) |
| `test_vrt_xml_size_cap_chunked_1831.py::*` (3 tests) | `vrt/test_metadata.py::*` (same 3 names) |
| `test_vrt_metadata_parity_2321.py::*` (~30 tests, parametrised over backends) | `vrt/test_metadata.py::*` (same names) |

Section count: 96 tests + 1 xfail collected.

## vrt/test_window.py (was 9 files)

| Old file:test | New file:test_id |
|---|---|
| `test_vrt_window_validation_1697.py::*` (16 tests, uses `vrt_4x4` fixture) | `vrt/test_window.py::*` (fixture renamed to `window_validation_vrt_4x4`) |
| `test_vrt_resample_window_inverse_1704.py::*` (8 tests incl. parametrised) | `vrt/test_window.py::*` |
| `test_vrt_dstrect_resample_cap_1737.py::*` (7 tests) | `vrt/test_window.py::*` |
| `test_vrt_scaled_rects_1694.py::*` (8 tests incl. parametrised) | `vrt/test_window.py::*` |
| `test_vrt_source_tile_check_1823.py::TestPerTileCheckDoesNotUseCallerBudget`, `TestOutputWindowCheckStillEnforced`, `TestPerTileCheckStillRejectsCraftedHeader` | `vrt/test_window.py::*` (same class names) |
| `test_vrt_source_max_pixels_1796.py::*` (2 tests) | `vrt/test_window.py::*` |
| `test_vrt_lazy_chunks_1814.py::*` (11 tests, uses fixtures) | `vrt/test_window.py::*` (fixtures prefixed `lazy_chunks_*`) |
| `test_vrt_chunked_shared_dataset_1923.py::*` (3 tests) | `vrt/test_window.py::*` |
| `test_vrt_tiled_scheduler_1714.py::*` (3 tests) | `vrt/test_window.py::*` |

Section count: 69 tests collected.

## vrt/test_dtype_conversion.py (was 7 files)

| Old file:test | New file:test_id |
|---|---|
| `test_vrt_dtype_1783.py::*` (~25 tests incl. parametrised) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_dtype_12bit_1914.py::*` (~6 tests) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_int_source_float_dtype_1616.py::*` (7 tests) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_multiband_dtype_1696.py::*` (10 tests) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_multiband_int_nodata_1611.py::*` (7 tests) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_resample_alg_1751.py::*` (6 tests, parametrised) | `vrt/test_dtype_conversion.py::*` |
| `test_vrt_simple_mosaic_2369.py::*` (7 tests, uses `mosaic_*` fixtures) | `vrt/test_dtype_conversion.py::*` (fixtures prefixed `simple_mosaic_*`) |

Section count: 95 tests collected.

## Out of scope (left untouched)

- PR 2 validation: `test_vrt_validation_2321.py`, `test_vrt_capability_validator_2371.py`, `test_vrt_unsupported_2370.py`, `test_vrt_narrow_except_1670.py`, `test_vrt_path_containment_1671.py`.
- PR 4 parity: `test_vrt_backend_parity_2321.py`, `test_vrt_backend_coverage_2026_05_11.py`, `test_vrt_finalization_parity_2162.py`.
- PR 7 writer: `test_vrt_write.py`, `test_vrt_writer_int64_1833.py`, `test_vrt_writer_photometric_1861.py`, `test_vrt_writer_source_compat_1733.py`, all `test_write_vrt_*.py`.
- PR 11 missing-sources flavour: `test_vrt_missing_sources_default_raise_1843.py`, `test_vrt_chunked_missing_raise_at_build_2265.py`, `test_vrt_chunked_missing_sources_1799.py`.

## Doc cross-references updated

The release-gate checklist (`docs/source/reference/release_gate_geotiff.rst`)
and the GeoTIFF reference (`docs/source/reference/geotiff.rst`) had rows
citing the old issue-numbered file names. Each citation was repointed at
the matching consolidated file (`vrt/test_metadata.py`,
`vrt/test_window.py`, or `vrt/test_dtype_conversion.py`). The presence
gate `test_release_gate_2321.py::test_release_gate_cites_only_existing_test_files`
is green on the new paths.

## Verification

- `pytest xrspatial/geotiff/tests/vrt/ -v`: 281 passed, 1 xfailed.
- `pytest xrspatial/geotiff/tests/ -q`: 5706 passed, 68 skipped, 6 xfailed, 1 xpassed.
