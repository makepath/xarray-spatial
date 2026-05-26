# CLUSTER_AUDIT.md — PR 1 (Foundation + VRT missing-sources)

Temporary audit table tracking every old `file::test` and where it lands
in the consolidated layout. Deleted in a follow-up commit on the same
branch before merge per the epic #2390 contract.

## Foundation moves

| Old location | New location | Notes |
|---|---|---|
| `conftest.py::make_minimal_tiff` (function) | `_helpers/tiff_builders.py::make_minimal_tiff` | Re-exported from `conftest.py` so existing `from .conftest import make_minimal_tiff` keeps working. |
| `conftest.py::gpu_available` | `_helpers/markers.py::gpu_available` | Re-exported from `conftest.py`. |
| `conftest.py::loopback_available` | `_helpers/markers.py::loopback_available` | Re-exported from `conftest.py`. |
| `conftest.py::requires_gpu` | `_helpers/markers.py::requires_gpu` | Marker name unchanged; re-exported. |
| `conftest.py::requires_loopback` | `_helpers/markers.py::requires_loopback` | Marker name unchanged; re-exported. |
| `conftest.py::requires_integration` | `_helpers/markers.py::requires_integration` | Marker name unchanged; re-exported. |
| `conftest.py::pytest_collection_modifyitems` | `conftest.py::pytest_collection_modifyitems` | Left in place; PR 11 removes it per the epic. |
| `_tiff_surgery.py` (whole module) | `_helpers/tiff_surgery.py` | Verbatim relocation. Direct importers updated. |

### Updated import sites

| File | Change |
|---|---|
| `test_local_tile_byte_cap_1664.py` | `from ._tiff_surgery import ...` -> `from ._helpers.tiff_surgery import ...` |
| `test_gpu_tile_byte_cap_2026_05_18.py` | `from ._tiff_surgery import ...` -> `from ._helpers.tiff_surgery import ...` |

All other test files import `make_minimal_tiff`, `gpu_available`, and
the `requires_*` markers from `conftest.py` (or
`xrspatial.geotiff.tests.conftest`), which now re-exports them. No
further changes needed.

## VRT missing-sources cluster

### `test_vrt_missing_sources_policy_1799.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_vrt_missing_sources_policy_1799.py::test_read_vrt_missing_sources_warns_and_records_hole` | `vrt/test_missing_sources.py::TestWarnPolicyEmitsWarningAndFillsNodata::test_eager_byte_warn_records_hole` | Byte-band variant carried over verbatim. Asserts on the `"could not be read"` message and `vrt_holes`. |
| `test_vrt_missing_sources_policy_1799.py::test_read_vrt_missing_sources_raise_fails_fast` | `vrt/test_missing_sources.py::TestExplicitRaisePolicy::test_eager_byte_explicit_raise` | Byte-band variant. Renamed; assertion unchanged (raises `OSError` or `ValueError`). |
| `test_vrt_missing_sources_policy_1799.py::test_read_vrt_missing_sources_validates_policy` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_eager_byte_invalid_policy` | Byte-band invalid-policy smoke check. Parametrised matrix below covers more bad values across both readers; this stays as a literal port to keep byte-band coverage. |

### `test_vrt_missing_sources_policy_2367.py` (deleted)

| Old `file::test` | New `file::test_id` | Notes |
|---|---|---|
| `test_vrt_missing_sources_policy_2367.py::TestDefaultPolicyRaises::test_default_raises_filenotfound_naming_source[eager_read_vrt]` | `vrt/test_missing_sources.py::TestDefaultPolicyRaises::test_default_raises_filenotfound_naming_source[eager]` | Renamed reader id from `eager_read_vrt` to `eager`. Filename in the VRT helper renamed `missing_2367.tif` -> `missing_source.tif` (no issue numbers in fixtures). |
| `test_vrt_missing_sources_policy_2367.py::TestDefaultPolicyRaises::test_default_raises_filenotfound_naming_source[dask_open_geotiff_chunks]` | `vrt/test_missing_sources.py::TestDefaultPolicyRaises::test_default_raises_filenotfound_naming_source[dask]` | Renamed reader id; same coverage. |
| `test_vrt_missing_sources_policy_2367.py::TestExplicitRaisePolicy::test_explicit_raise_matches_default[eager_read_vrt]` | `vrt/test_missing_sources.py::TestExplicitRaisePolicy::test_explicit_raise_matches_default[eager]` | Renamed reader id. |
| `test_vrt_missing_sources_policy_2367.py::TestExplicitRaisePolicy::test_explicit_raise_matches_default[dask_open_geotiff_chunks]` | `vrt/test_missing_sources.py::TestExplicitRaisePolicy::test_explicit_raise_matches_default[dask]` | Renamed reader id. |
| `test_vrt_missing_sources_policy_2367.py::TestWarnPolicyEmitsWarningAndFillsNodata::test_eager_warn_emits_and_fills` | `vrt/test_missing_sources.py::TestWarnPolicyEmitsWarningAndFillsNodata::test_eager_warn_emits_and_fills` | Body unchanged except missing-source filename rename. |
| `test_vrt_missing_sources_policy_2367.py::TestWarnPolicyEmitsWarningAndFillsNodata::test_dask_warn_emits_at_compute_and_fills` | `vrt/test_missing_sources.py::TestWarnPolicyEmitsWarningAndFillsNodata::test_dask_warn_emits_at_compute_and_fills` | Body unchanged except missing-source filename rename. |
| `test_vrt_missing_sources_policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-ignore]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[ignore-eager]` | Reader id renamed; parametrize order unchanged. |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-RAISE]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[RAISE-eager]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-raises]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[raises-eager]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[-eager]` | Empty-string bad value. |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-warn ]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[warn -eager]` | Trailing-space bad value. |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[eager_read_vrt-1]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[1-eager]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-ignore]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[ignore-dask]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-RAISE]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[RAISE-dask]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-raises]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[raises-dask]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[-dask]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-warn ]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[warn -dask]` | |
| `..._policy_2367.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[dask_open_geotiff_chunks-1]` | `vrt/test_missing_sources.py::TestInvalidPolicyRejected::test_invalid_policy_raises_value_error_naming_value[1-dask]` | |

### Files NOT folded in (justified)

| File | Reason left in place |
|---|---|
| `test_vrt_missing_sources_default_raise_1843.py` | Different surface area: tests the *internal* `xrspatial.geotiff._vrt.read_vrt` entry point (not the public `xrspatial.geotiff.read_vrt`), plus the `XRSPATIAL_GEOTIFF_STRICT=1` env-var override. Neither is in the public-API matrix covered by `vrt/test_missing_sources.py`. A future PR that consolidates the strict-env-var coverage can fold this in then. |

## Verification

- Old eager byte coverage: 3 tests preserved (warn / raise / invalid).
- Old eager+dask float coverage: 16 parametrised cases preserved (default x2, explicit raise x2, warn x2, invalid 6 bad values x 2 readers).
- Net file delta: 3 files deleted (`_tiff_surgery.py`, `test_vrt_missing_sources_policy_1799.py`, `test_vrt_missing_sources_policy_2367.py`); 6 files added (`_helpers/{__init__,tiff_builders,tiff_surgery,markers}.py`, `vrt/{__init__,test_missing_sources}.py`). The two extra `__init__.py` files plus the audit file (deleted before merge) keep the net `test_*.py` count drop at exactly 2.
