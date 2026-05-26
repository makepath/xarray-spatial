# PR 11 cluster audit — unit-level consolidation + final conftest slim

Closes epic #2390.

## Unit-level moves

Source files folded into `xrspatial/geotiff/tests/unit/`. Test
function names are unchanged unless noted; the move only changes
the file path and the relative import for `make_minimal_tiff`.

| Old file                                              | New file                       | Notes                                  |
|-------------------------------------------------------|--------------------------------|----------------------------------------|
| `test_header.py`                                      | `unit/test_header.py`          | verbatim move (import path adjusted)   |
| `test_geotags.py`                                     | `unit/test_geotags.py`         | verbatim move (import path adjusted)   |
| `test_gdal_metadata_xml_escape_1614.py`               | `unit/test_safe_xml.py`        | renamed (no content change)            |
| `test_packbits_jit_2048.py`                           | `unit/test_compression.py`     | merged with `test_packbits_jit_2049.py` under sectioned headings |
| `test_packbits_jit_2049.py`                           | `unit/test_compression.py`     | merged with `test_packbits_jit_2048.py`; classes renamed `TestPackBitsJIT*` → `TestPackBitsEncode*` |

The release-gate checklist row for `writer.gdal_metadata_xml` was
updated from the old filename to `unit/test_safe_xml.py`.

## Left alone

The following files were considered for folding into `unit/` but
kept where they are because they bundle a pure-unit slice with an
integration-flavoured slice. Splitting them is a future-epic
problem.

| File                              | Reason                                                                                       |
|-----------------------------------|----------------------------------------------------------------------------------------------|
| `test_extra_tags_safe_filter_1657.py` | end-to-end overview / SubIFDs leakage test against `open_geotiff` / `to_geotiff`           |
| `test_compression_docstring_1644.py`  | docstring check + GPU codec acceptance smoke test                                          |
| `test_compression_level.py`           | write/read round-trip through `to_geotiff` / `open_geotiff`                                |
| `test_lz4.py`, `test_lerc.py`         | codec helper round-trip plus full TIFF writer round-trip                                    |
| `test_jpeg.py`, `test_jpeg2000.py`    | codec round-trip plus writer / public-API tests                                            |
| `test_mixed_bps.py`, `test_mixed_sample_format.py` | helper-resolution unit tests bundled with end-to-end `open_geotiff` tests       |
| `test_lerc_max_z_error.py`            | end-to-end through `to_geotiff` / `open_geotiff`                                            |
| `test_packbits_jit_*`                 | now folded — see Unit-level moves                                                          |

## conftest.py slim

* Dropped the `pytest_collection_modifyitems` socketserver hook.
* Kept the marker / helper re-exports under `__all__` so test files
  that import via `from .conftest import make_minimal_tiff` or
  `from .conftest import requires_loopback` keep working.

## `@requires_loopback` additions

Every HTTP/loopback test outside of `integration/` now carries the
marker explicitly (the hook used to skip them by source inspection).
Files touched:

* `parity/test_backend_matrix.py` — added `@requires_loopback` to
  `test_backend_parity_matrix` and `test_backend_parity_matrix_errors`
  (both depend on the `_matrix_http_server` fixture).
* `parity/test_pixel_equality.py` — added to
  `test_miniswhite_http_matches_local_reader` and
  `test_miniswhite_http_dask_matches_local_reader`.
* `test_golden_corpus_http_1930.py` — module-level `pytestmark`.
* `test_read_geotiff_gpu_url_eager_2161.py` — added to
  `test_http_url_returns_cupy_matching_cpu`,
  `test_unreachable_http_url_does_not_raise_filenotfound`,
  `test_chunked_url_path_still_uses_chunked_helper`.
* `test_remote_sidecar_byte_order_2314.py` — added to the four
  `test_http_*` and `test_parse_cog_http_meta_*` cases.
* `test_remote_sidecar_chunked_2239.py` — added to every
  `test_http_*` case.
* `test_sidecar_max_cloud_bytes_2121.py` — added to the three
  `test_http_sidecar_*` cases.
* `test_sidecar_ovr_2112.py` — added to the four
  `test_find_sidecar_http_*` and `test_load_sidecar_http_*` cases.
* `write/test_cog.py` — added to `test_row5_golden_cog_xrspatial_http`
  and `test_row6_golden_cog_xrspatial_dask_http`.
* `test_parallel_strip_decode_2100.py` — added to
  `TestHttpStripParallelDecode` class and the
  `test_http_windowed_planar2_parallel` outside it.
* `test_parallel_strip_decode_sparse_2100.py` — added to
  `TestHttpStripsSparseParallel` class.

## Cleanup

* Deleted `CLUSTER_AUDIT_PR5.md`, `CLUSTER_AUDIT_PR8.md`,
  `CLUSTER_AUDIT_PR9.md` (stale audit notes that leaked from earlier
  PRs and should have been removed at merge time).
* This audit file (`CLUSTER_AUDIT_PR11.md`) is deleted in a final
  commit on the same branch before approval.

## File-count delta

* Top-level `xrspatial/geotiff/tests/*.py` test files: 229 → 224
  (five removed: `test_header.py`, `test_geotags.py`,
  `test_gdal_metadata_xml_escape_1614.py`, `test_packbits_jit_2048.py`,
  `test_packbits_jit_2049.py`).
* `unit/` adds four files: `test_header.py`, `test_geotags.py`,
  `test_safe_xml.py`, `test_compression.py`.
* Net change: one fewer file overall, with the slice now grouped
  under a named directory instead of scattered across the long tail.

## Verification

```
pytest xrspatial/geotiff/tests/unit/             # 135 passed
pytest xrspatial/geotiff/tests/                  # 5719 passed, 68 skipped, 6 xfailed
grep -n "pytest_collection_modifyitems\|socketserver.TCPServer\|_serve(" \
    xrspatial/geotiff/tests/conftest.py          # empty
grep -rL "@requires_loopback" \
    $(grep -rl "socketserver\|_serve(" xrspatial/geotiff/tests/)
                                                  # empty
```
