# Cluster audit -- epic #2390 PR 3 (rotated / dropped CRS)

Maps every old `file::test` from the three folded files to its new
`read/test_crs.py::test_id`. **This file is deleted on the final commit
before the PR is approved; it must not land on `main`.**

The HTTP rotated read (`test_http_dask_allow_rotated_2130.py`) is left
in place; it belongs to PR 9 (integration).

## `test_allow_rotated_geotiff_2115.py`

| Old file::test | New file::test_id | Notes |
|---|---|---|
| `test_allow_rotated_geotiff_2115.py::test_extract_transform_rejects_rotated_by_default` | `read/test_crs.py::test_extract_transform_rotated_default_raises` | renamed for clarity |
| `test_allow_rotated_geotiff_2115.py::test_extract_transform_allow_rotated_returns_no_georef` | `read/test_crs.py::test_extract_transform_rotated_optin_returns_no_georef` | renamed |
| `test_allow_rotated_geotiff_2115.py::test_extract_transform_allow_rotated_passes_through_axis_aligned` | `read/test_crs.py::test_extract_transform_axis_aligned_optin_passes_through` | renamed |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_rotated_default_raises` | `read/test_crs.py::test_open_geotiff_rotated_no_crs_default_raises[eager]` | parametrized |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_rotated_allow_rotated_reads_pixels` | `read/test_crs.py::test_open_geotiff_rotated_no_crs_optin_reads_pixels[eager]` | parametrized |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_rotated_default_raises_with_dask` | `read/test_crs.py::test_open_geotiff_rotated_no_crs_default_raises[dask]` | parametrized |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_rotated_allow_rotated_with_dask` | `read/test_crs.py::test_open_geotiff_rotated_no_crs_optin_reads_pixels[dask]` | parametrized |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_http_rotated_default_raises` | -- | DROPPED. HTTP rotated coverage already exists in `test_http_dask_allow_rotated_2130.py` (PR 9 integration cluster). Local-file rotated raise is pinned by `test_open_geotiff_rotated_no_crs_default_raises[eager]`. |
| `test_allow_rotated_geotiff_2115.py::test_open_geotiff_http_rotated_allow_rotated_reads_pixels` | -- | DROPPED. Same rationale as above; PR 9 integration cluster owns end-to-end HTTP rotated reads. |

## `test_allow_rotated_crs_drop_2126.py`

| Old file::test | New file::test_id | Notes |
|---|---|---|
| `test_allow_rotated_crs_drop_2126.py::test_rotated_optin_drops_crs_epsg` | `read/test_crs.py::test_populate_attrs_rotated_optin_drops_attr[crs]` | parametrized over `crs` / `crs_wkt` / `transform` |
| `test_allow_rotated_crs_drop_2126.py::test_rotated_optin_drops_crs_wkt` | `read/test_crs.py::test_populate_attrs_rotated_optin_drops_attr[crs_wkt]` | parametrized |
| `test_allow_rotated_crs_drop_2126.py::test_plain_no_georef_keeps_crs` | `read/test_crs.py::test_populate_attrs_plain_no_georef_keeps_crs` | unchanged |
| `test_allow_rotated_crs_drop_2126.py::test_axis_aligned_georef_keeps_crs_and_transform` | `read/test_crs.py::test_populate_attrs_axis_aligned_keeps_crs_and_transform` | renamed |
| `test_allow_rotated_crs_drop_2126.py::test_open_geotiff_rotated_with_crs_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_geokey_only_drops_crs[eager]` | parametrized; tifffile-written variant with Geographic-only GeoKey |
| `test_allow_rotated_crs_drop_2126.py::test_open_geotiff_rotated_with_crs_dask_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_geokey_only_drops_crs[dask]` | parametrized |

## `test_allow_rotated_no_crs_2122.py`

| Old file::test | New file::test_id | Notes |
|---|---|---|
| `test_allow_rotated_no_crs_2122.py::test_eager_rotated_read_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_drops_crs[eager]` | parametrized; uses the hand-rolled writer with full `_GEO_KEYS_4326` block |
| `test_allow_rotated_no_crs_2122.py::test_dask_rotated_read_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_drops_crs[dask]` | parametrized |
| `test_allow_rotated_no_crs_2122.py::test_cupy_rotated_read_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_drops_crs_gpu[eager]` | parametrized; `@requires_gpu` |
| `test_allow_rotated_no_crs_2122.py::test_dask_cupy_rotated_read_drops_crs` | `read/test_crs.py::test_open_geotiff_rotated_with_crs_drops_crs_gpu[dask]` | parametrized; `@requires_gpu` |
| `test_allow_rotated_no_crs_2122.py::test_axis_aligned_read_still_emits_crs` | `read/test_crs.py::test_open_geotiff_axis_aligned_with_crs_keeps_crs` | renamed |
| `test_allow_rotated_no_crs_2122.py::test_vrt_eager_rotated_read_drops_crs` | `read/test_crs.py::test_vrt_rotated_with_crs_drops_crs[eager]` | parametrized |
| `test_allow_rotated_no_crs_2122.py::test_vrt_chunked_rotated_read_drops_crs` | `read/test_crs.py::test_vrt_rotated_with_crs_drops_crs[dask]` | parametrized |
| `test_allow_rotated_no_crs_2122.py::test_vrt_axis_aligned_still_emits_crs` | `read/test_crs.py::test_vrt_axis_aligned_with_crs_keeps_crs` | renamed |

## Coverage delta

* No coverage was dropped except the two HTTP cases from 2115, which are
  superseded by the more thorough HTTP+dask coverage in
  `test_http_dask_allow_rotated_2130.py` (PR 9). Local-file rotated
  raise and pixel-grid read are still pinned by the eager+dask
  parametrizations.
* The Geographic-only GeoKey path (tifffile writer, 2126's
  ``_write_rotated_tiff_with_geokeys``) and the full
  `_GEO_KEYS_4326`-block path (hand-rolled writer, 2122's
  ``_write_rotated_tiff_with_crs``) are both kept as distinct
  scenarios; each exercises a different GeoKey-parser branch.

## Pre-merge action

Delete this file (`CLUSTER_AUDIT_PR3.md`) on the final commit before the
PR is approved. The audit is a review artifact, not a documentation
deliverable; the git history and the PR description retain the trail.
