"""GPU (cupy) backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 3 of the corpus plan. Mirrors the eager and dask parity layers
but reads each fixture through ``open_geotiff(str(path), gpu=True)``,
returning a CuPy-backed DataArray. The oracle's ``_candidate_pixels``
pulls the CuPy array back to host via ``.get()`` before comparing, so
the comparison machinery is unchanged.

The whole module ``pytest.importorskip``s cupy and skips if no CUDA
device is reachable. CI matrices without CUDA collect zero tests from
this module; runs with a GPU exercise every fixture the eager and dask
backends already do.

``_PARITY_GAPS`` is empty for the GPU backend now that integer nodata
masking and the citation-only CRS gap are closed at the codec / attrs
layer. ``_GPU_SKIPS`` holds GPU-only failures, currently the
JPEG-YCbCr fixture: the GPU decoder does not handle it and
``on_gpu_failure='strict'`` raises rather than falling back, so the
read fails before the oracle can compare. On eager / dask the same
fixture exposes the RGB axis-order divergence; on GPU strict mode it
never gets that far.

Resolved gaps (no longer xfail):

* Integer nodata masking -- closed by the oracle's
  ``_normalise_for_masked_nodata`` helper.
* ``crs_citation_only`` -- closed by ``_synthesize_user_defined_wkt``
  in ``_geotags.py``, which stamps a canonical
  ``attrs['crs_wkt']`` on user-defined geographic CRSes.

The GPU read is configured with ``on_gpu_failure='strict'`` so a codec
that would silently CPU-fall-back instead surfaces as an xfail / fail
the corpus can act on.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")
cupy = pytest.importorskip("cupy")

try:
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip(
            "no CUDA device available", allow_module_level=True
        )
except Exception as exc:  # pragma: no cover - CI without CUDA
    pytest.skip(
        f"cupy is importable but CUDA is not usable: {exc}",
        allow_module_level=True,
    )

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


# Integer-nodata masking used to live here (closed by the oracle's
# _normalise_for_masked_nodata helper), and crs_citation_only used to
# live here too (closed by _synthesize_user_defined_wkt in _geotags.py).
_PARITY_GAPS: dict[str, str] = {}

# GPU-only gaps. Empty in the current corpus: failures here would be
# fixtures the eager and dask backends decode cleanly but the GPU
# backend does not. Keep the table around as the documented home for
# such gaps if they appear.
_GPU_SKIPS: dict[str, str] = {}

# Fixtures whose overview-IFD code path is not yet wired into the
# xrspatial reader. The base-IFD parity check still runs; the
# overview-level loop driven by ``candidate_factory`` is skipped for
# these ids. See the eager module for the rationale.
_OVERVIEW_READER_GAPS: dict[str, str] = {
    "overview_external_ovr_uint16": (
        "External .ovr sidecar reader is not implemented in xrspatial."
    ),
}

# Fixtures whose codec is not implemented on the GPU read path and
# legitimately need to fall back to CPU. The test routes these through
# ``on_gpu_failure='auto'`` instead of ``'strict'``, which exercises
# the documented fallback contract: the GPU reader detects the
# unsupported codec, swaps to the CPU path, and returns a numpy-
# backed DataArray. The oracle then compares the CPU read against
# rasterio the same way the eager backend does.
_GPU_CPU_FALLBACK: dict[str, str] = {
    "compression_jpeg_uint8_ycbcr": (
        "JPEG-YCbCr decode is not implemented on the GPU read path; "
        "on_gpu_failure='auto' falls back to CPU. The oracle's "
        "_normalise_axis_order helper handles the (bands, y, x) vs "
        "(y, x, band) divergence so the CPU-fallback read compares "
        "cleanly against the rasterio reference."
    ),
}

_INTENTIONAL_SKIPS: dict[str, str] = {
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "#1797; rasterio leaves them raw. Covered by "
        "test_miniswhite_backend_parity_1797.py."
    ),
}


def _resolved_fixtures() -> list[dict]:
    manifest = generate.load_manifest()
    entries = generate.validate(manifest)
    entries.sort(key=lambda e: e["id"])
    return entries


def _fixture_path(entry: dict) -> pathlib.Path:
    return FIXTURES_DIR / f"{entry['id']}.tif"


def _is_lossy(entry: dict) -> bool:
    tol = entry.get("tolerance") or {}
    return bool(tol.get("lossy", False))


def _build_param(entry: dict) -> pytest.param:
    fid = entry["id"]
    if fid in _PARITY_GAPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.xfail(reason=_PARITY_GAPS[fid], strict=True),
        )
    if fid in _GPU_SKIPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.xfail(reason=_GPU_SKIPS[fid], strict=True),
        )
    if fid in _INTENTIONAL_SKIPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.skip(reason=_INTENTIONAL_SKIPS[fid]),
        )
    return pytest.param(entry, id=fid)


_FIXTURES = _resolved_fixtures()
_PARAMS = [_build_param(e) for e in _FIXTURES]


@pytest.mark.parametrize("manifest_entry", _PARAMS)
def test_gpu_parity(manifest_entry: dict) -> None:
    """``open_geotiff(path, gpu=True)`` agrees with the rasterio oracle.

    The GPU path uses nvCOMP for supported codecs. ``strict`` mode is
    the default so a silent CPU fallback surfaces as an exception
    rather than masking GPU coverage. Fixtures listed in
    ``_GPU_CPU_FALLBACK`` route through ``'auto'`` mode instead because
    their codec is genuinely not implemented on the GPU; the test
    asserts the documented CPU fallback produces oracle-equivalent
    output rather than xfailing the cell.
    """
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )
    on_gpu_failure = (
        "auto" if fixture_id in _GPU_CPU_FALLBACK else "strict"
    )
    candidate = open_geotiff(
        str(path), gpu=True, on_gpu_failure=on_gpu_failure
    )
    # Wire the oracle's overview-IFD check when the fixture carries
    # overviews. The factory inherits the GPU read options the base
    # candidate used so each overview level exercises the same nvCOMP /
    # CPU-fallback configuration.
    overviews = manifest_entry.get("overviews") or []
    factory = (
        (lambda lvl, p=path, on_fail=on_gpu_failure: open_geotiff(
            str(p), gpu=True, on_gpu_failure=on_fail, overview_level=lvl,
        ))
        if overviews and fixture_id not in _OVERVIEW_READER_GAPS
        else None
    )
    compare_to_oracle(
        path,
        candidate,
        lossy=_is_lossy(manifest_entry),
        candidate_factory=factory,
    )


def test_taxonomy_ids_are_in_manifest() -> None:
    """Every id in the parity-gap, GPU-skip, CPU-fallback, or
    intentional-skip tables must exist in the manifest.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = (
        set(_PARITY_GAPS)
        | set(_GPU_SKIPS)
        | set(_GPU_CPU_FALLBACK)
        | set(_OVERVIEW_READER_GAPS)
        | set(_INTENTIONAL_SKIPS)
    )
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )


def test_gpu_candidate_is_actually_on_device() -> None:
    """Sanity check: the GPU backend returns a CuPy-backed array.

    Catches the failure mode where ``gpu=True`` silently CPU-falls-back
    and ``open_geotiff`` returns a numpy array instead. ``strict`` mode
    should already raise rather than fall back, but this is a belt-and-
    braces check against a regression in the fallback policy.
    """
    plain_fixtures = [
        e for e in _FIXTURES
        if e["id"] not in _PARITY_GAPS
        and e["id"] not in _GPU_SKIPS
        and e["id"] not in _INTENTIONAL_SKIPS
        and _fixture_path(e).exists()
    ]
    if not plain_fixtures:
        pytest.skip("no eligible fixtures on disk")
    entry = plain_fixtures[0]
    da = open_geotiff(
        str(_fixture_path(entry)), gpu=True, on_gpu_failure="strict"
    )
    assert isinstance(da.data, cupy.ndarray), (
        f"expected a cupy.ndarray for {entry['id']!r}, "
        f"got {type(da.data).__name__}"
    )
