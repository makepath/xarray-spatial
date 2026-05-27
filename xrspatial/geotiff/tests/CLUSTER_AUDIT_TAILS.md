# Cluster 15 audit: metadata / CRS-write / COG-BigTIFF / reader tails (#2439)

This audit maps every old `file::test` from the four sub-clusters to its
new `file::test_id` in the consolidated suite. The file is deleted in
the final pre-merge commit per epic #2424's hard gate.

## Sub-cluster 1: metadata (5 files -> new `unit/test_metadata.py` + extending `unit/test_safe_xml.py`)

### test_ambiguous_metadata_hooks_1987.py -> unit/test_metadata.py
- all tests preserved under section "Ambiguous metadata hooks (#1987)"
- fixture `_reset_metadata_check_registries` localised to this section

### test_geotiff_metadata_2139.py -> unit/test_metadata.py
- all tests preserved under section "GeoTIFFMetadata dataclass round-trip (#2139)"

### test_metadata_round_trip_1484.py -> unit/test_metadata.py
- all tests preserved under section "Transform / CRS / tag metadata round-trip (#1484)"
- helpers `_make_palette_uint8_tiff_1484` / `_write_simple_tiff_with_image_description_1484`

### test_mixed_band_metadata_fail_closed_1987.py -> unit/test_metadata.py
- all tests preserved under section "Mixed-band metadata fail-closed (#1987)"
- helpers `_write_mixed_band_vrt_1987`, `_write_shared_sentinel_vrt_1987`, `_write_one_band_no_sentinel_vrt_1987`, `_wrap_2d_1987`

### test_extra_tags_safe_filter_1657.py -> unit/test_safe_xml.py (extension)
- all tests preserved under section "Safe extra_tags filter (#1657)"
- `_gpu_only` collapses onto the shared `requires_gpu` marker
- helper `_make_cog_1657`, `_read_subfile_type_1657`

## Sub-cluster 2: CRS write (6 files -> new `write/test_crs.py`)

### test_conflicting_crs_write_1987.py -> write/test_crs.py
- all tests preserved under section "Conflicting CRS write check (#1987)"
- helper `_da_1987`

### test_crs_arg_validation_1971.py -> write/test_crs.py
- all tests preserved under section "CRS argument validation (#1971)"
- helper `_square_1971`

### test_crs_fail_closed_1929.py -> write/test_crs.py
- all tests preserved under section "CRS fail-closed citation guard (#1929)"
- helper `_make_da_1929`

### test_numpy_int_crs_2082.py -> write/test_crs.py
- all tests preserved under section "Numpy integer CRS round-trip (#2082)"
- helper `_square_2082`

### test_user_defined_crs_wkt_1632.py -> write/test_crs.py
- all tests preserved under section "User-defined CRS WKT promotion (#1632)"
- helper `_write_user_defined_crs_tif_1632`, `_gpu_only` collapses onto shared marker

### test_wkt_only_crs_warning_1768.py -> write/test_crs.py
- all tests preserved under section "WKT-only CRS warning (#1768)"
- helper `_wkt_only_da_1768`

## Sub-cluster 3: COG / BigTIFF tail (2 files -> extending `write/test_bigtiff.py`)

### test_eager_bigtiff_overhead_exact_1905.py -> write/test_bigtiff.py
- all tests preserved under section "Eager writer BigTIFF auto-detection (#1905)"
- helper `_make_4x4_float32_1905`

### test_to_geotiff_bigtiff_doc_1683.py -> write/test_bigtiff.py
- all tests preserved under section "BigTIFF docstring parity (#1683)"
- helper `_documented_params_1683`

## Sub-cluster 4: reader tail (2 files -> extending `read/test_basic.py`)

### test_open_geotiff_missing_sources_1810.py -> read/test_basic.py
- all tests preserved under section "open_geotiff missing_sources (#1810)"
- helper `_write_missing_source_vrt_1810`

### test_reader.py -> read/test_basic.py
- all tests preserved under sections "Reader strips/tiles/array (low-level)" and "Partial tile validation (#1486)"
- helper `_zero_out_last_tile_1486`

## Cross-references updated

- `docs/source/reference/release_gate_geotiff.rst`: updated rows citing
  the moved files.
- `xrspatial/geotiff/_attrs.py:846`: docstring pointer updated.
- `xrspatial/geotiff/tests/write/test_bigtiff.py:316`: prose pointer
  updated.

## Verification

```
pytest xrspatial/geotiff/tests/unit/ xrspatial/geotiff/tests/read/ \
       xrspatial/geotiff/tests/write/ -v
pytest xrspatial/geotiff/tests/ -x -q
pytest xrspatial/geotiff/tests/release_gates/test_stable_features.py \
       ::test_release_gate_cites_only_existing_test_files
```
