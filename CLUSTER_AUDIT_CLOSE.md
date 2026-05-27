# CLUSTER 16 (CLOSE) AUDIT — Epic #2424 / Issue #2440

Pre-merge gate for the closing cluster of the long-tail GeoTIFF test
consolidation epic. This file is removed in a final commit before merge.

## Scope landed in this PR

### Sub-PR A: features / tiers (4) → `release_gates/test_features.py`

- `test_features.py` → relocated (rename + folded in)
- `test_supported_features_shape_2348.py` → folded as section
- `test_supported_features_tiers_2137.py` → folded as section
- `test_unsupported_features_2349.py` → folded as section
- `test_vrt_stable_only_2443.py` → folded as section (release-contract gate)

`Path(__file__).resolve().parents[1]` rewired to `parents[2]` for the
source-parsing dedup test now that the file sits one level deeper.

### Sub-PR B: golden-corpus relocation (12) → `golden_corpus/`

Moved with rename (drop `_1930` suffix):

- `test_golden_corpus_compression_1930.py` → `golden_corpus/test_compression.py`
- `test_golden_corpus_dask_gpu_1930.py` → `golden_corpus/test_dask_gpu.py`
- `test_golden_corpus_dask_numpy_1930.py` → `golden_corpus/test_dask_numpy.py`
- `test_golden_corpus_dtype_variants_1930.py` → `golden_corpus/test_dtype_variants.py`
- `test_golden_corpus_eager_numpy_1930.py` → `golden_corpus/test_eager_numpy.py`
- `test_golden_corpus_fsspec_1930.py` → `golden_corpus/test_fsspec.py`
- `test_golden_corpus_gpu_1930.py` → `golden_corpus/test_gpu.py`
- `test_golden_corpus_http_1930.py` → `golden_corpus/test_http.py`
- `test_golden_corpus_layout_endian_1930.py` → `golden_corpus/test_layout_endian.py`
- `test_golden_corpus_manifest_1930.py` → `golden_corpus/test_manifest.py`
- `test_golden_corpus_metadata_tags_1930.py` → `golden_corpus/test_metadata_tags.py`
- `test_golden_corpus_overview_cog_1930.py` → `golden_corpus/test_overview_cog.py`
- `test_golden_corpus_vrt_1930.py` → `golden_corpus/test_vrt.py`

`from ._helpers.markers import` in `test_http.py` rewired to
`from .._helpers.markers import` after the relocation.

### Sub-PR C: uncategorised tail (11) → individual existing homes

- `test_accuracy_1081.py` → `read/test_basic.py`
- `test_assemble_layout_no_bytes_copy_1756.py` → `read/test_tiling.py`
- `test_bytesio_source.py` → `read/test_basic.py`
- `test_eager_source_close_on_error_2322.py` → `read/test_basic.py`
- `test_finalization_helpers_2162.py` → `parity/test_backend_matrix.py`
- `test_orientation.py` → `read/test_basic.py`
- `test_remaining_fail_closed_1987.py` → `unit/test_metadata.py`
- `test_runtime_sentinels_identity_1880.py` → `unit/test_metadata.py`
- `test_streaming_codecs_2026_05_11.py` → `write/test_streaming.py`
- `test_streaming_photometric_override_2073.py` → `write/test_basic.py`
- `test_strict_mode_1662.py` → `unit/test_signatures.py`

`from .conftest import make_minimal_tiff` rewired to `from ..conftest import ...`
across `release_gates/test_features.py` (TestBigEndian section). The
self-referential import in `read/test_nodata.py` (former cross-file
`_build_uint16_tiff_1774` reference) replaced with a local alias.

### Out-of-scope cleanup (residue from prior clusters)

Five numbered files were still at the top level after cluster 15 landed.
Folded each into the matching existing home so the ≤5 cap can hold:

- `test_inconsistent_geokeys_2417.py` → `unit/test_geotags.py`
- `test_overview_kernels_2413.py` → `read/test_overview.py`
- `test_invalid_int_nodata_rejection_2441.py` → `read/test_nodata.py`
- `test_compound_crs_reject_2418.py` → `write/test_crs.py`
- `test_vrt_stable_only_2443.py` → `release_gates/test_features.py`

Three remaining numbered top-level files renamed to drop the suffix per
the audit rule (no filename matches `test_*_[0-9]{4,}.py`):

- `test_fuzz_hypothesis_1661.py` → `test_fuzz_hypothesis.py`
- `test_namespace_no_leak_1708.py` → `unit/test_signatures.py` (folded)
- `test_polish_1488.py` → `test_polish.py`

`test_round_trip_invariants.py` + `test_roundtrip_properties.py` consolidated
into `test_round_trip.py` (one round-trip module instead of two near-duplicates).

### Release-gate citation updates

Updated paths in:

- `docs/source/reference/release_gate_geotiff.rst`
- `docs/source/reference/geotiff.rst`
- `docs/source/contributing/golden_corpus_baselines.rst`
- `xrspatial/geotiff/__init__.py` (docstring citation)
- `xrspatial/geotiff/_header.py` (code-comment citation)
- `xrspatial/geotiff/tests/integration/test_sidecar.py` (cross-test reference)
- `xrspatial/geotiff/tests/golden_corpus/test_eager_numpy.py` (cross-test reference)
- `xrspatial/geotiff/tests/golden_corpus/test_dask_numpy.py` (cross-test reference)
- `xrspatial/geotiff/tests/test_round_trip.py` (cross-test reference)

The release-gate meta-gate
(`test_release_gate_cites_only_existing_test_files`) now passes.

## Definition-of-done audit (epic #2424 targets)

- `find xrspatial/geotiff/tests -name 'test_*.py' | wc -l` → **72** (target: 60–80) ✓
- `find xrspatial/geotiff/tests -maxdepth 1 -name 'test_*.py' | wc -l` → **5** (target: ≤5) ✓
- Filenames matching `test_*_[0-9]{4,}.py` → **0** ✓
- `pytest xrspatial/geotiff/tests/ -q` → green (5597 + 339 passed,
  56 + 25 skipped, 2 xfailed across the two batches).

## Top-level survivors (5 cross-cutting modules)

- `test_edge_cases.py` — invalid / corrupt / boundary inputs.
- `test_fuzz_hypothesis.py` — Hypothesis property/fuzz suite (#1661).
- `test_polish.py` — multi-section polish bundle (#1488).
- `test_round_trip.py` — invariants + property-based round-trip (#1986, #2134).
- `test_security.py` — unbounded-alloc / path-traversal guards (#1184, #1185).

## Closes (residual epic admin)

This PR carries `Closes #2425`, `Closes #2434`, `Closes #2438`,
`Closes #2440`, `Closes #2424` so the squash merge closes every cluster
issue that the closing PR ties off.
