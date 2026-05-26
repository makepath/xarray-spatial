# CLUSTER_AUDIT_PR7.md — Writer / COG / BigTIFF cluster (epic #2390 PR 7)

Temporary audit table tracking the 26 deleted writer-cluster files and
where their tests land in the consolidated layout. **This file is
deleted on a follow-up commit on the same branch before the PR is
approved per the epic contract.**

Result: 497 collected tests in `write/` mirror 497 collected tests
across the 26 source files (counts captured before and after the move
with `pytest --collect-only -q`). All pass on the run captured below
the table.

## Counts

| Bucket            | Source files | New file                          | Tests collected |
|-------------------|--------------|-----------------------------------|-----------------|
| Generic writer    | 14           | `write/test_basic.py`             | (see below)     |
| COG writer        | 6            | `write/test_cog.py`               | (see below)     |
| BigTIFF + COG     | 1            | `write/test_bigtiff.py`           | (see below)     |
| Overview policy   | 5            | `write/test_overview.py`          | (see below)     |
| **Total**         | **26**       | **4**                             | **497**         |

`pytest xrspatial/geotiff/tests/write/ --collect-only -q | tail -1`
reports `497 tests collected`. The same count was captured against the
26 source files immediately before the move (see PR description).

## File map

### `write/test_basic.py` — generic writer paths

| Old file                                                  | New module section                                  | Notes |
|-----------------------------------------------------------|------------------------------------------------------|-------|
| `test_writer.py`                                          | `# Folded from: test_writer.py`                      | verbatim |
| `test_writer_matrix.py`                                   | `# Folded from: test_writer_matrix.py`               | verbatim |
| `test_writer_kwarg_order_1922.py`                         | `# Folded from: test_writer_kwarg_order_1922.py`     | verbatim |
| `test_writer_return_path_1938.py`                         | `# Folded from: test_writer_return_path_1938.py`     | verbatim |
| `test_writer_uncompressed_tiled_no_dead_alloc_1736.py`    | section banner                                       | verbatim |
| `test_write_layout_monkeypatch_contract_2248.py`          | section banner                                       | verbatim |
| `test_write_vrt_path_kwarg_1946.py`                       | section banner                                       | verbatim |
| `test_write_vrt_crs_1715.py`                              | section banner                                       | verbatim |
| `test_write_vrt_bool_nodata_1921.py`                      | section banner                                       | verbatim |
| `test_write_vrt_int_nodata_1684.py`                       | section banner                                       | verbatim |
| `test_vrt_writer_int64_1833.py`                           | section banner                                       | verbatim |
| `test_vrt_writer_photometric_1861.py`                     | section banner                                       | verbatim |
| `test_vrt_writer_source_compat_1733.py`                   | section banner                                       | `write_vrt` import aliased to `_priv_write_vrt` to avoid shadowing the public re-export used by the kwarg-order tests above |
| `test_vrt_write.py`                                       | section banner                                       | verbatim |

Module-level dedup: the identical `_build_source_tif(tmp_path, name='src.tif')` helper that the `test_write_vrt_path_kwarg_1946.py` and `test_write_vrt_crs_1715.py` sections both defined was reduced to one copy (the first), since the two source bodies were byte-for-byte identical.

### `write/test_cog.py` — COG writer compliance + overview/nodata

| Old file                                  | New module section            | Notes |
|-------------------------------------------|-------------------------------|-------|
| `test_cog.py`                             | section banner                | verbatim; `from .conftest import gpu_available` rewritten to `from ..conftest import gpu_available` since `write/` is one level deeper |
| `test_cog_writer_compliance.py`           | section banner                | verbatim |
| `test_cog_invalid_input_errors_2286.py`   | section banner                | the local `_float_da` helper (shape default `(8, 8)`) was renamed to `_float_da_small` for this section so the later sections' `_float_da` (`(64, 64)`) does not shadow it; all in-section call sites updated |
| `test_cog_parity_2286.py`                 | section banner                | verbatim |
| `test_cog_requires_tiled_2312.py`         | section banner                | verbatim |
| `test_cog_tile_size_hang_2311.py`         | section banner                | verbatim |

Module-level dedup: the docstring-only-differing duplicate `_float_da(shape=(64, 64))` from the third section was removed (the second section's definition is reused).

### `write/test_bigtiff.py` — BigTIFF threshold + COG compliance

| Old file                                  | New module section            | Notes |
|-------------------------------------------|-------------------------------|-------|
| `test_bigtiff_cog_compliance_2286.py`     | section banner                | verbatim |

No other current files match the `bigtiff` writer-side criterion. The eager-overhead test (`test_eager_bigtiff_overhead_exact_1905.py`) stays in the read/streaming cluster (PR 8) per the epic, since it exercises the threshold from the *read* path.

### `write/test_overview.py` — overview-level + nodata-aware overviews

| Old file                                       | New module section   | Notes |
|------------------------------------------------|----------------------|-------|
| `test_cog_overview_ceil_2105.py`               | section banner       | verbatim |
| `test_cog_overview_nodata_1613.py`             | section banner       | verbatim |
| `test_cog_cubic_overview_nodata_1623.py`       | section banner       | verbatim |
| `test_cog_cubic_int_overview_nodata_1975.py`   | section banner       | verbatim |
| `test_cog_int_overview_nodata_2026_05_12.py`   | section banner       | verbatim |

Module-level dedup: the four sections each defined an identical-body `_gpu_available() -> bool` helper and a `_HAS_GPU = _gpu_available()` module assignment. The first definition is kept and the three later identical copies (and their reassignments) were dropped.

## Cross-suite reference updates

Several files outside the writer cluster cited the old filenames; the
release-gate test (`test_release_gate_2321.py`) fails if a cited file
is gone, so the following docs were updated:

| File                                                     | Change |
|----------------------------------------------------------|--------|
| `docs/source/reference/release_gate_geotiff.rst`         | `writer.local_file`, `writer.cog`, `writer.overviews`, `writer.bigtiff_cog`, `writer.cog (tile-layout / tile-size pre-flight)`, and `reader.local_cog` rows now point at `write/test_cog.py`, `write/test_overview.py`, and `write/test_bigtiff.py` instead of the deleted files. |
| `docs/source/reference/geotiff.rst`                      | Prose references and the "Known unsupported combinations" table updated to the new paths. |

Comment-only references in `xrspatial/geotiff/_writer.py`,
`xrspatial/geotiff/_attrs.py`,
`xrspatial/geotiff/tests/test_overview_block_order_2308.py`, and
`xrspatial/geotiff/tests/test_release_gate_cog.py` were left in place:
they read as historical notes ("see test_cog_writer_compliance for the
skip semantics") and the test gate only checks the rst checklist, not
Python comments. PR 11's lint pass can refresh these in bulk.

## Out of scope (per PR 7's "MUST NOT touch" list)

- HTTP-side COG tests: `test_cog_http_close_on_error_1816.py`,
  `test_cog_http_concurrent.py`,
  `test_cog_http_parallel_decode_2026_05_15.py`. Integration cluster
  for PR 9.
- Reader-side files: PR 8.
- VRT validation / metadata / window / dtype / missing-sources /
  parity files: PRs 1, 2, 4, 6.

## Verification

```
pytest xrspatial/geotiff/tests/write/ -v        # 497 passed
pytest xrspatial/geotiff/tests/ -x -q           # 5706 passed, 68 skipped, 6 xfailed, 1 xpassed
```

Captured locally on Python 3.14, branch `issue-2400`.
