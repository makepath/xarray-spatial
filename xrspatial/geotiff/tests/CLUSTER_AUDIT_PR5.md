# CLUSTER_AUDIT_PR5.md — Attrs contract consolidation

Temporary audit table tracking every old `file::test` and where it
lands in `attrs/test_contract.py`. Deleted on the final commit on the
branch before approval (epic #2390 contract).

## Source files folded

- `test_attrs_contract_canonical_1984.py`
- `test_attrs_contract_aliases_1984.py`
- `test_attrs_contract_passthrough_1984.py`
- `test_attrs_contract_version_1984.py`

All four deleted in the same commit that added `attrs/test_contract.py`.
File count delta: `-4 +1 = -3`.

## Canonical tier mapping

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_attrs_contract_canonical_1984.py::test_every_canonical_key_present` | `attrs/test_contract.py::test_canonical_every_key_present` | Renamed for the new prefix scheme; logic identical. |
| `test_attrs_contract_canonical_1984.py::test_crs_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[crs]]` | Folded into the per-key parametrize; assertion unchanged. |
| `test_attrs_contract_canonical_1984.py::test_crs_wkt_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[crs_wkt]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_transform_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[transform]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_nodata_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[GDAL_NODATA]]` | Folded; id renamed to surface the underlying TIFF tag. |
| `test_attrs_contract_canonical_1984.py::test_extra_tags_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[extra_tags]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_gdal_metadata_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[gdal_metadata]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_gdal_metadata_xml_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[gdal_metadata_xml]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_resolution_group_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[resolution_group]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_contract_version_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[contract_version]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_georef_status_roundtrip` | `attrs/test_contract.py::test_canonical_value_roundtrip[canonical[georef_status]]` | Folded. |
| `test_attrs_contract_canonical_1984.py::test_canonical_keys_present_per_backend[eager-numpy]` | `attrs/test_contract.py::test_canonical_keys_present_per_backend[canonical[eager-numpy]]` | Param IDs renamed to the dimensional scheme; marker now `requires_gpu` from `_helpers/markers.py`. |
| `test_attrs_contract_canonical_1984.py::test_canonical_keys_present_per_backend[dask-numpy]` | `attrs/test_contract.py::test_canonical_keys_present_per_backend[canonical[dask-numpy]]` | Same. |
| `test_attrs_contract_canonical_1984.py::test_canonical_keys_present_per_backend[gpu]` | `attrs/test_contract.py::test_canonical_keys_present_per_backend[canonical[gpu]]` | Same; marker swapped to `requires_gpu`. |
| `test_attrs_contract_canonical_1984.py::test_canonical_keys_present_per_backend[dask-gpu]` | `attrs/test_contract.py::test_canonical_keys_present_per_backend[canonical[dask-gpu]]` | Same. |
| `test_attrs_contract_canonical_1984.py::test_raster_type_area_omitted_on_roundtrip` | `attrs/test_contract.py::test_canonical_raster_type_area_omitted_on_roundtrip` | Renamed for prefix consistency. |
| `test_attrs_contract_canonical_1984.py::test_raster_type_point_roundtrip` | `attrs/test_contract.py::test_canonical_raster_type_point_roundtrip` | Renamed. |

## Aliases tier mapping

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_attrs_contract_aliases_1984.py::test_read_uses_nodatavals_when_nodata_absent` | `attrs/test_contract.py::test_alias_read_used_when_nodata_absent[alias[nodatavals->nodata]]` | Folded into the two-alias parametrize. |
| `test_attrs_contract_aliases_1984.py::test_read_uses_fill_value_when_nodata_absent` | `attrs/test_contract.py::test_alias_read_used_when_nodata_absent[alias[_FillValue->nodata]]` | Folded. |
| `test_attrs_contract_aliases_1984.py::test_canonical_nodata_wins_over_aliases_at_resolver` | `attrs/test_contract.py::test_alias_canonical_nodata_wins_at_resolver` | Renamed for prefix. |
| `test_attrs_contract_aliases_1984.py::test_canonical_nodata_wins_over_aliases_at_write` | `attrs/test_contract.py::test_alias_canonical_nodata_wins_at_write` | Renamed; `ConflictingNodataError` import moved to module top. |
| `test_attrs_contract_aliases_1984.py::test_write_does_not_emit_aliases_when_canonical_present` | `attrs/test_contract.py::test_alias_write_does_not_emit_aliases_when_canonical_present` | Renamed. |
| `test_attrs_contract_aliases_1984.py::test_nan_in_nodatavals_is_skipped` | `attrs/test_contract.py::test_alias_nan_in_nodatavals_is_skipped` | Renamed. |

## Passthrough tier mapping

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_attrs_contract_passthrough_1984.py::test_passthrough_cases_cover_all_keys` | `attrs/test_contract.py::test_passthrough_cases_cover_all_keys` | Verbatim. |
| `test_attrs_contract_passthrough_1984.py::test_passthrough_key_roundtrip[image_description]` | `attrs/test_contract.py::test_passthrough_key_roundtrip[passthrough[image_description]]` | Param ID prefixed with the tier. |
| `test_attrs_contract_passthrough_1984.py::test_passthrough_key_roundtrip[extra_samples]` | `attrs/test_contract.py::test_passthrough_key_roundtrip[passthrough[extra_samples]]` | Same. |
| `test_attrs_contract_passthrough_1984.py::test_passthrough_key_roundtrip[colormap]` | `attrs/test_contract.py::test_passthrough_key_roundtrip[passthrough[colormap]]` | Same. |
| `test_attrs_contract_passthrough_1984.py::test_passthrough_dropped_when_no_crs` | `attrs/test_contract.py::test_passthrough_dropped_when_no_crs` | Verbatim. |
| `test_attrs_contract_passthrough_1984.py::test_passthrough_does_not_promote_to_canonical` | `attrs/test_contract.py::test_passthrough_does_not_promote_to_canonical` | Verbatim. |
| `test_attrs_contract_passthrough_1984.py::test_removed_attrs_not_emitted` | `attrs/test_contract.py::test_passthrough_removed_attrs_not_emitted` | Renamed for prefix. |
| `test_attrs_contract_passthrough_1984.py::test_removed_attrs_absent_after_roundtrip` | `attrs/test_contract.py::test_passthrough_removed_attrs_absent_after_roundtrip` | Renamed. |
| `test_attrs_contract_passthrough_1984.py::test_contract_version_is_current` | `attrs/test_contract.py::test_passthrough_contract_version_is_current` | Renamed; assertion unchanged. |

## Version tier mapping

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_attrs_contract_version_1984.py::test_attrs_contract_version_constant_is_current` | `attrs/test_contract.py::test_version_constant_is_current` | Renamed for prefix. |
| `test_attrs_contract_version_1984.py::test_attrs_module_docstring_version_matches_constant` | `attrs/test_contract.py::test_version_module_docstring_matches_constant` | Renamed. |
| `test_attrs_contract_version_1984.py::test_eager_numpy_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_tiff_backend[version[eager-numpy]]` | Folded into TIFF-backend parametrize. |
| `test_attrs_contract_version_1984.py::test_dask_numpy_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_tiff_backend[version[dask-numpy]]` | Folded. |
| `test_attrs_contract_version_1984.py::test_gpu_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_tiff_backend[version[gpu]]` | Folded; marker swapped to `requires_gpu`. |
| `test_attrs_contract_version_1984.py::test_dask_gpu_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_tiff_backend[version[dask-gpu]]` | Folded. |
| `test_attrs_contract_version_1984.py::test_vrt_eager_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_vrt_backend[version[vrt-eager]]` | Folded into VRT-backend parametrize. |
| `test_attrs_contract_version_1984.py::test_vrt_chunked_stamps_contract_version` | `attrs/test_contract.py::test_version_stamp_present_per_vrt_backend[version[vrt-chunked]]` | Folded. |

## Coverage parity check

Old surface (deduplicated functions/params):
- Canonical: 1 presence test + 10 per-key value tests + 4 per-backend
  params + 2 raster_type tests = 17 cases.
- Aliases: 6 tests (2 read-side, 2 precedence, 1 write-side, 1 NaN).
- Passthrough: 1 covers-all + 3 param cases + 5 misc = 9 cases.
- Version: 2 constant/docstring + 4 TIFF backends + 2 VRT backends = 8.
- Total old: 40 cases.

New surface:
- Canonical: 1 + 10 (parametrize) + 4 (parametrize) + 2 = 17 cases.
- Aliases: 2 (parametrize) + 4 = 6 cases.
- Passthrough: 1 + 3 (parametrize) + 5 = 9 cases.
- Version: 2 + 4 (parametrize) + 2 (parametrize) = 8 cases.
- Total new: 40 cases. **No coverage drop.**

## Out of scope (intentionally untouched)

- `test_attrs_finalization_parity_2211.py` — parity-flavoured (PR 4).
- `test_attrs_parity_1548.py` — parity (PR 4).
- `test_attrs_kwarg_parity_1561.py` — parity (PR 4).
- `test_release_gate_attrs_contract.py` — release-gate registry (PR 10).
- `test_nodata_attr_aliases_1582.py` — finer-grained alias regression
  tests; the original aliases file noted these as a sibling, not as
  contract scope. Stays for now; can fold into a later cluster if it
  fits.

## Roundtrip slice deferral

The epic mentions a future `attrs/test_roundtrip.py`. The four source
files do not contain a clean roundtrip slice that would naturally fall
out during this consolidation: every roundtrip case here is bound to a
specific tier (canonical key presence, alias promotion, passthrough
reconstruction) and is exercised inside that tier's section. Adding a
standalone `test_roundtrip.py` now would duplicate the canonical
fixture without adding coverage. Leaving for a follow-up so it can be
designed against a real coverage gap.
