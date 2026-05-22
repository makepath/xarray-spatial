"""Dask+GPU backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 4 of the corpus plan. Reads each fixture through
``open_geotiff(str(path), gpu=True, chunks=32, on_gpu_failure='strict')``,
returning a dask-of-cupy DataArray. The oracle pulls pixels via
``.compute()`` then ``.get()`` so the comparison machinery is unchanged.

The module skips cleanly when no CUDA device is reachable. Strict on-
gpu-failure is set so a silent CPU fallback surfaces as an exception
rather than masking the dask+GPU coverage.

The shared codec / attrs parity gaps in ``_PARITY_GAPS`` carry over from
the eager / dask / GPU modules. ``_DASK_GPU_SKIPS`` is reserved for
gaps that surface only when chunked GPU reads stitch through nvCOMP
plus dask; it starts empty.

Resolved gaps (no longer xfail):

* Integer nodata masking -- closed by the oracle's
  ``_normalise_for_masked_nodata`` helper.
* ``crs_citation_only`` -- closed by ``_synthesize_user_defined_wkt``
  in ``_geotags.py``, which stamps a canonical
  ``attrs['crs_wkt']`` on user-defined geographic CRSes.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")
pytest.importorskip("dask")
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
from xrspatial.geotiff.tests.golden_corpus._oracle import compare_to_oracle  # noqa: E402

FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

CHUNK_SIZE = 32


# Integer-nodata masking used to live here too; the oracle's
# _normalise_for_masked_nodata helper (#2046) closes that gap so it is
# no longer xfailed on any backend. The multi-band axis-order gap for
# the JPEG-YCbCr fixture is also closed on the eager / dask paths
# (see ``_normalise_axis_order`` in ``_oracle.py``); on the dask+GPU
# pipeline the decode error wins first so the JPEG fixture lives in
# ``_INTENTIONAL_SKIPS`` below. crs_citation_only moved off this list
# once ``_synthesize_user_defined_wkt`` started stamping a canonical
# WKT for user-defined geographic CRSes.
_PARITY_GAPS: dict[str, str] = {}

# Empty: failures unique to the dask+GPU combo (eager / dask / pure-
# GPU all pass) would land here. Kept around as the documented home
# for such gaps.
_DASK_GPU_SKIPS: dict[str, str] = {}

# Fixtures whose overview-IFD code path is not yet wired into the
# xrspatial reader. The base-IFD parity check still runs; the
# overview-level loop driven by ``candidate_factory`` is skipped for
# these ids. See the eager module for the rationale.
_OVERVIEW_READER_GAPS: dict[str, str] = {
    "overview_external_ovr_uint16": (
        "External .ovr sidecar reader is not implemented in xrspatial."
    ),
}

_INTENTIONAL_SKIPS: dict[str, str] = {
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "#1797; rasterio leaves them raw. Covered by "
        "test_miniswhite_backend_parity_1797.py."
    ),
    "compression_jpeg_uint8_ycbcr": (
        "JPEG-YCbCr decode is not implemented on the GPU read path and "
        "the dask+GPU pipeline cannot fall back to CPU per chunk: the "
        "decode error surfaces at .compute() time regardless of "
        "on_gpu_failure mode. The pure-GPU backend handles the same "
        "fixture by routing through on_gpu_failure='auto' (CPU "
        "fallback yields a single numpy array, not a dask graph), but "
        "that escape hatch does not exist for the chunked path. Plain "
        "skip rather than xfail because no foreseeable fix exists: "
        "implementing nvCOMP JPEG-YCbCr decode is its own project."
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
    if fid in _DASK_GPU_SKIPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.xfail(reason=_DASK_GPU_SKIPS[fid], strict=True),
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
def test_dask_gpu_parity(manifest_entry: dict) -> None:
    """``open_geotiff(path, gpu=True, chunks=32)`` agrees with the rasterio oracle."""
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )
    candidate = open_geotiff(
        str(path), gpu=True, chunks=CHUNK_SIZE, on_gpu_failure="strict"
    )
    # Wire the oracle's overview-IFD check when the fixture carries
    # overviews. The factory inherits the dask+GPU read options so each
    # overview level exercises the same chunked nvCOMP pipeline the
    # full-resolution candidate did.
    overviews = manifest_entry.get("overviews") or []
    factory = (
        (lambda lvl, p=path: open_geotiff(
            str(p), gpu=True, chunks=CHUNK_SIZE,
            on_gpu_failure="strict", overview_level=lvl,
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
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = (
        set(_PARITY_GAPS)
        | set(_DASK_GPU_SKIPS)
        | set(_OVERVIEW_READER_GAPS)
        | set(_INTENTIONAL_SKIPS)
    )
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )


def test_dask_gpu_candidate_is_chunked_and_on_device() -> None:
    """Sanity check: result is a dask array with a real chunk grid
    whose chunks materialise to CuPy arrays.

    Catches three failure modes at once: ``chunks=`` silently dropped
    (would yield a plain CuPy array), ``chunks=`` accepted but stitched
    into a single chunk that covers the whole file (windowing logic
    never runs), and ``gpu=True`` silently CPU-fallen-back (would yield
    a dask-of-numpy array). Picks the first fixture in sorted order
    whose pixel extent is at least ``2 * CHUNK_SIZE`` along both axes.
    """
    eligible = [
        e for e in _FIXTURES
        if e["id"] not in _PARITY_GAPS
        and e["id"] not in _DASK_GPU_SKIPS
        and e["id"] not in _INTENTIONAL_SKIPS
        and _fixture_path(e).exists()
        and e["width"] >= 2 * CHUNK_SIZE
        and e["height"] >= 2 * CHUNK_SIZE
    ]
    if not eligible:
        pytest.skip(
            f"no eligible fixture is at least {2 * CHUNK_SIZE}x{2 * CHUNK_SIZE}"
        )
    entry = eligible[0]
    da = open_geotiff(
        str(_fixture_path(entry)),
        gpu=True,
        chunks=CHUNK_SIZE,
        on_gpu_failure="strict",
    )
    assert hasattr(da.data, "dask"), (
        f"expected a dask-backed DataArray for {entry['id']!r}, "
        f"got data of type {type(da.data).__name__}"
    )
    nb = da.data.numblocks
    assert len(nb) >= 2 and all(b >= 2 for b in nb[-2:]), (
        f"expected a chunk grid >= 2x2 along the spatial axes for "
        f"{entry['id']!r}, got numblocks={nb}"
    )
    computed = da.data.compute()
    assert isinstance(computed, cupy.ndarray), (
        f"dask chunks must materialise to cupy.ndarray for {entry['id']!r}, "
        f"got {type(computed).__name__}"
    )
