# Cluster 10 audit: nodata semantics consolidation (#2434)

Cluster 10 of long-tail epic #2424 folds 13 nodata test files into two
consolidated homes: `read/test_nodata.py` (extends the existing GPU
helper coverage) and `write/test_nodata.py` (new). One file (#2434
chose the split based on read/write surface).

This audit file maps every `old_file::test_name` to its
`new_file::test_name`. The file is deleted in the final pre-merge commit
on this branch per the epic's hard gate.

## Sub-PR A: read-side -> `read/test_nodata.py`

### `test_masked_nodata_attr_2092.py` (12 tests)
- `test_eager_mask_nodata_false_reports_false` -> `read/test_nodata.py::test_eager_mask_nodata_false_reports_false`
- `test_eager_mask_nodata_true_reports_true` -> `read/test_nodata.py::test_eager_mask_nodata_true_reports_true`
- `test_eager_int_file_mask_nodata_true_no_match_reports_false` -> ditto
- `test_eager_explicit_float_dtype_mask_off_reports_false` -> ditto
- `test_dask_mask_nodata_false_reports_false` -> ditto
- `test_dask_mask_nodata_true_reports_true` -> ditto
- `test_dask_explicit_float_dtype_mask_off_reports_false` -> ditto
- `test_vrt_int_source_mask_nodata_false_reports_false` -> ditto
- `test_vrt_int_source_mask_nodata_true_reports_true` -> ditto
- `test_vrt_int_source_mask_off_with_float_cast_reports_false` -> ditto
- `test_gpu_mask_nodata_false_reports_false` -> ditto
- `test_gpu_mask_nodata_true_reports_true` -> ditto

### `test_nodata_attr_aliases_1582.py` (11 tests)
- All 8 free-function tests preserved (helper `_da_float` -> `_da_float_1582`).
- GPU parametrised `test_gpu_writer_resolves_alias[...]` 3 cases preserved.

### `test_nodata_lifecycle_attrs_2135.py` (19 tests)
- All eager / dask / VRT / GPU tests preserved.
- Helper functions suffixed `_2135` to avoid collisions with sibling sections.

### `test_nodata_lifecycle_parity_2211.py` (57 tests)
- `TestRawSentinelExposure` (3 methods) preserved.
- `TestEffectiveSentinelUnderMinIsWhite` (6) preserved.
- `TestSentinelFitsBuffer` (6) preserved.
- `TestMaskingOccurredDecision` (5) preserved.
- `TestDtypeCastDecision` (2) preserved.
- `TestWriterRestoreSentinelDecision` (9) preserved.
- `TestInvertForMinIsWhiteLockstep` (1) preserved.
- `TestPixelsPresentSlot` (2) preserved.
- `TestIntegerSentinelParity` (3) preserved.
- `TestFloatSentinelParity` (3) preserved.
- `TestNaNSentinelParity` (2) preserved.
- `TestOutOfRangeSentinelParity` (2) preserved.
- `TestMinIsWhiteSentinelInversion` (3) preserved.
- `TestMaskNodataFalseParity` (3) preserved.
- `TestExplicitDtypeRequestParity` (2) preserved.
- `TestVRTEagerParity` (2) preserved.
- `TestWriterRestoreParity` (3) preserved.
- Module-level helper functions `_make_int_raster` / `_make_float_raster`
  suffixed `_2211` to avoid collisions with the 2135 helpers.

### `test_nodata_nan_int_1774.py` (21 tests)
- Eager parametrised `nan` / `Inf` cases preserved.
- Dask + GPU helper parity preserved.
- Fractional sentinel cases preserved.
- Helper `_build_uint16_tiff` -> `_build_uint16_tiff_1774`; the sibling
  module `test_invalid_int_nodata_rejection_2441.py` (not in this
  cluster) is updated to import the renamed helper from the new location
  via an aliased import.

### `test_nodata_no_extra_copy_1553.py` (8 tests)
- All preserved. Helpers `_make_float_with_sentinel` /
  `_make_uint16_with_sentinel` suffixed `_1553`.

### `test_nodata_semantics_split_1988.py` (39 tests)
- All class-based tests (`TestEagerNumpy`, `TestDaskNumpy`, `TestVRTEager`,
  `TestVRTChunked`, `TestGPU`, `TestSetNodataAttrsHelper`,
  `TestShouldRestoreNanSentinelHelper`, `TestWriterRoundTripEager`,
  `TestWriteStreamingRestoreSentinelKwarg`,
  `TestWriteCOGOverviewGateInteraction`, `TestWriterGPU`) preserved.
- Free function `test_int_source_with_out_of_range_sentinel` preserved.
- Module-level helpers (`_write_float_tiff`, `_write_int_tiff`,
  `_build_uint16_with_out_of_range_nodata`, `_write_uint16_vrt_source`,
  `_build_vrt`) suffixed `_1988`.
- The original module-level `rasterio = pytest.importorskip("rasterio")`
  was replaced with per-helper / per-test `pytest.importorskip` so the
  rest of `read/test_nodata.py` collects even when rasterio is absent.

### `test_helper_band_nodata_2210.py` (9 tests)
- All preserved. `_FakeTransform` / `_FakeGeoInfo` -> `_FakeTransform2210`
  / `_FakeGeoInfo2210`.

### Pre-existing in `read/test_nodata.py` (9 tests)
Unchanged: GPU helper in-place mask + removal-pin coverage.

**Read-side count**: 9 (existing) + 12 + 11 + 19 + 57 + 21 + 8 + 39 + 9
= 185 tests collected from `read/test_nodata.py` (matches baseline sum
exactly: each consolidated file contributes the same count it had
before).

## Sub-PR B: write-side -> `write/test_nodata.py` (new file)

### `test_nodata_validation_1973.py` (27 tests)
- All free functions + `pytest.importorskip("rasterio")` removed
  (rasterio not imported in this file; the bool / non-numeric checks
  are pure validator tests).

### `test_nodata_bool_rejection_1911.py` (17 tests)
- All preserved.

### `test_nodata_int64_precision_1847.py` (25 tests)
- `TestParseNodataStr` (14), `TestOpenGeotiffEager` (6),
  `TestReadGeotiffDask` (2), `TestVrtRoundTrip` (2),
  `TestGpuPathParity` (1) preserved.

### `test_nodata_out_of_range_1581.py` (8 tests)
- All preserved.

### `test_mask_nodata_kwarg_2052.py` (10 tests)
- All preserved.

**Write-side count**: 27 + 17 + 25 + 8 + 10 = 87 tests
collected from `write/test_nodata.py` (matches baseline exactly).

## Cross-cutting updates

- `docs/source/reference/release_gate_geotiff.rst`: the nodata
  round-trip release-gate row now cites the consolidated files. The
  release-gate checklist-parity test
  (`release_gates/test_stable_features.py::test_release_gate_cites_only_existing_test_files`)
  passes.
- `docs/source/reference/geotiff.rst`: prose pointers to deleted files
  updated to the consolidated homes.
- `xrspatial/geotiff/tests/test_invalid_int_nodata_rejection_2441.py`:
  import updated to use `_build_uint16_tiff_1774` from
  `.read.test_nodata`.
- `xrspatial/geotiff/tests/test_roundtrip_properties.py`: stale
  docstring pointer updated.

## Verification

- `pytest xrspatial/geotiff/tests/read/test_nodata.py xrspatial/geotiff/tests/write/test_nodata.py xrspatial/geotiff/tests/test_invalid_int_nodata_rejection_2441.py xrspatial/geotiff/tests/test_roundtrip_properties.py`:
  298 passed (185 + 87 + 21 + 5).
- `pytest xrspatial/geotiff/tests/read/ xrspatial/geotiff/tests/write/`:
  1586 passed, 5 skipped.
- File-count drop: -12 (13 deleted, 1 new).

This audit file is deleted in a final pre-merge commit on this branch.
