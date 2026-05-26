# Cluster 11 audit: georef and rotation (#2435)

Twelve top-level georef / no-georef / rotation test files folded into
`xrspatial/geotiff/tests/read/test_georef.py`. Read-side and write-side
cases share the rotated-affine and no-georef-marker fixtures, so the
write-side tests stay in the same module under a `# write side:`
heading rather than splitting into `write/test_georef.py`.

`read/test_crs.py` (rotated-CRS read surface, epic #2390 PR 3) is not
touched. The status tests reuse its `_write_rotated_tiff` helper.

## Source files removed (12)

| Old file | Concern | Section in `read/test_georef.py` |
| --- | --- | --- |
| `test_georef_edges.py` | non-georef reads, south-up writes (#1482) | `TestNonGeoreferencedRead`, `TestDescendingYWrite` |
| `test_georef_resolver_parity_2211.py` | `resolve_georef` parity (#2225) | `test_resolve_georef_*`, `test_writer_and_reader_*`, `test_bare_input_*` |
| `test_georef_status_2136.py` | canonical `georef_status` attr (#2136) | `test_contract_version_*`, `test_compute_status_*`, `test_*_status*`, `test_vrt_*_status*`, `test_roundtrip_preserves_status` |
| `test_no_georef_attr_migration_2133.py` | marker migration to attrs (#2133) | `test_non_uniform_*`, `test_uniform_int_coords_still_write`, `TestMarkerOnRead`, `test_georef_file_has_no_marker`, `test_marker_survives_*`, `test_3d_no_georef_round_trip` |
| `test_no_georef_marker_2120.py` | int64 step-1 grids keep georef (#2120) | `test_int64_step1_*`, `test_no_georef_file_carries_marker_*`, `test_marker_on_user_grid_*` |
| `test_no_georef_windowed_coords_1710.py` | windowed coord parity (#1710) | `TestEagerWindowedCoords`, `TestDaskWindowedCoords`, `TestGpuWindowedCoords`, `TestBackendParity`, `TestGeorefStillWorks` |
| `test_rotated_transform_attr_1764.py` | `_transform_from_attr` rejection (#1764) | `TestTransformFromAttrRejection`, `TestToGeotiffRejectsRotated` |
| `test_rotated_affine_attr_2129.py` | `attrs['rotated_affine']` (#2129) | `test_rotated_optin_*`, `test_*_omits_rotated_affine`, `test_rotated_affine_is_tuple_*`, `test_attrs_to_metadata_drops_rotated_affine`, `test_open_geotiff_*rotated*` |
| `test_rotated_typed_error_2267.py` | typed rotated-read error (#2267) | `test_extract_transform_*`, `test_rotated_error_*`, `test_open_geotiff_rotated_*reads*` |
| `test_degenerate_georef_1945.py` | 1xN/Nx1/1x1 writes (#1945) | `TestEagerWriterDegenerateGeoref`, `TestDaskNumpyWriterDegenerateGeoref`, `TestGpuWriterDegenerateGeoref`, `TestDaskCupyWriterDegenerateGeoref`, `TestVrtTiledWriterDegenerateGeoref`, `TestEagerWriterPointRasterDegenerate`, `TestCoordsToTransformBorrowSignPinning` |
| `test_no_georef_writer_round_trip_1949.py` | no-georef int-coord round trip (#1949) | `test_coords_to_transform_*`, `test_round_trip_preserves_int_coords_*`, `test_round_trip_dask_streaming_*`, `test_double_round_trip_stable`, `test_explicit_transform_attr_*`, `test_gpu_writer_preserves_int_coords` |
| `test_to_geotiff_drop_rotation_2216.py` | fail-closed rotated write (#2216) | `test_to_geotiff_rejects_*`, `test_to_geotiff_drop_rotation_*`, `test_to_geotiff_*_unchanged`, `test_round_trip_*rotated*requires_opt_in`, `test_*write_vrt_tiled*` |

## Helper renames (collision avoidance)

The three rotated-TIFF writers and two VRT writers in the source files
shared names (`_write_rotated_tiff`, `_write_rotated_vrt`,
`_make_da`, `_rotated_dataarray`, `_ROTATED_TUPLE`). Each kept its
distinct byte layout, so they were suffixed per source issue in the
merged file: `_write_rotated_tiff_2129`, `_write_rotated_tiff_2267`,
`_write_rotated_tiff_2216`, `_write_rotated_vrt_2129`,
`_write_vrt_georef_status`, `_make_da_1764`, `_rotated_dataarray_2216`,
`_rotated_geo_info_2129`, `_ROTATED_TUPLE_2129`, `_ROTATED_TUPLE_2216`,
`_ROTATED_M_2267`, `_make_rotated_ifd_2267`, `_make_no_georef_tiff_1949`.
The status suite's `_write_rotated_tiff` import now points at
`read/test_crs.py` (same file, relative import) as
`_write_rotated_tiff_crs`. The per-issue local `_gpu_available`
helpers were dropped in favour of the shared `requires_gpu` marker and
`gpu_available()` probe from `_helpers/markers.py`.

## Verification

- `pytest xrspatial/geotiff/tests/read/test_georef.py -q`: 171 passed.
  Same total as the 12 source files (171 passed before the move).
- `pytest xrspatial/geotiff/tests/ --co -q`: 5890 collected, no errors.
- File count: 12 deleted, 1 added (net -11).

This file is deleted in a pre-merge commit per epic #2424.
