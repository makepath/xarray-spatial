"""Dask+numpy backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 2 of the corpus plan. Mirrors the eager numpy parity layer
(``test_golden_corpus_eager_numpy_1930.py``) but reads each fixture
through the dask path: ``open_geotiff(str(path), chunks=32)``. The
oracle pulls the candidate's pixels via ``.compute()`` under the hood
(``_candidate_pixels`` is dask-aware), so the comparison machinery is
the same. What this PR exercises is the windowed-decode plumbing inside
``read_geotiff_dask``: any divergence between the eager and dask reads
shows up here.

The skip / xfail taxonomy is intentionally identical to the eager
module's. The parity gaps that motivated them (integer-nodata masking,
citation CRS encoding, RGB band axis order, MinIsWhite inversion) live
in the codec / attrs layer, which both backends share. If a future
parity gap turns out to be dask-specific, add it to ``_DASK_SKIPS``
rather than ``_PARITY_GAPS`` so the difference between "shared
codec/attrs gap" and "dask plumbing gap" stays legible.

The chunk size is fixed at 32. Most corpus fixtures are 64x64 or
smaller, so 32 produces either a 2x2 chunk grid or a single chunk;
either way the dask path is exercised. A future PR can parametrise
over chunk sizes if a fixture surfaces a chunk-edge bug.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")
pytest.importorskip("dask")

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

CHUNK_SIZE = 32


_NODATA_MASKING_REASON = (
    "integer nodata masking: xrspatial masks sentinel pixels to NaN and "
    "upcasts to float64 per #1988 (attrs['masked_nodata']=True). The oracle "
    "compares raw integer pixels; needs an oracle extension that consults "
    "attrs['masked_nodata']."
)

# Shared parity gaps. These live in the codec / attrs layer, not in the
# dask plumbing, so the dask backend inherits them verbatim from the
# eager backend.
_PARITY_GAPS: dict[str, str] = {
    "compression_jpeg_uint8_ycbcr": (
        "RGB band axis order divergence: rasterio reads (bands, y, x) while "
        "xrspatial reads (y, x, band). The oracle does not yet normalise "
        "multi-band axis order."
    ),
    "crs_citation_only": (
        "citation-only CRS: xrspatial decodes the citation into deprecated "
        "attrs['geog_citation'] but does not emit a canonical attrs['crs'] "
        "or attrs['crs_wkt']. Real parity gap; needs a fix in _crs.py."
    ),
    "nodata_int_sentinel_uint16": _NODATA_MASKING_REASON,
    "stripped_le_uint16": _NODATA_MASKING_REASON,
    "stripped_be_uint16": _NODATA_MASKING_REASON,
    "tiled_le_uint16": _NODATA_MASKING_REASON,
    "tiled_be_uint16": _NODATA_MASKING_REASON,
}

# Dask-only gaps go here. Empty in the first pass; add an entry only when
# a fixture is dask-specific (i.e. eager passes, dask does not).
_DASK_SKIPS: dict[str, str] = {}

_INTENTIONAL_SKIPS: dict[str, str] = {
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "#1797; rasterio leaves them raw. Covered by "
        "test_miniswhite_backend_parity_1797.py."
    ),
}


def _resolved_fixtures() -> list[dict]:
    """Return manifest entries with defaults merged, sorted by id for stability."""
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
    """Wrap a fixture entry in a ``pytest.param`` with the right mark."""
    fid = entry["id"]
    if fid in _PARITY_GAPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.xfail(reason=_PARITY_GAPS[fid], strict=True),
        )
    if fid in _DASK_SKIPS:
        return pytest.param(
            entry,
            id=fid,
            marks=pytest.mark.xfail(reason=_DASK_SKIPS[fid], strict=True),
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
def test_dask_numpy_parity(manifest_entry: dict) -> None:
    """``open_geotiff(path, chunks=32)`` agrees with the rasterio oracle.

    The dask path windows over the same decode primitives the eager path
    uses, so parity should be tight. Any divergence indicates a bug in
    the dask plumbing (window stitching, chunk-edge handling, attrs
    propagation) and warrants a real fix rather than a new skip.
    """
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )
    candidate = open_geotiff(str(path), chunks=CHUNK_SIZE)
    compare_to_oracle(path, candidate, lossy=_is_lossy(manifest_entry))


def test_taxonomy_ids_are_in_manifest() -> None:
    """Every id in the parity-gap, dask-skip, or intentional-skip tables
    must exist in the manifest.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = set(_PARITY_GAPS) | set(_DASK_SKIPS) | set(_INTENTIONAL_SKIPS)
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )


def test_dask_candidate_is_actually_chunked() -> None:
    """Sanity check: the dask backend returns a chunked array with a
    real (>1) chunk count along each axis.

    Catches two failure modes at once: ``chunks=`` silently dropped (the
    eager path would return a numpy array, not dask) and ``chunks=``
    accepted but stitched into a single chunk that covers the whole
    file (the windowing logic would never run). Picks the first fixture
    in sorted order whose pixel extent is at least ``2 * CHUNK_SIZE``
    along both axes, so a clean 2x2 (or larger) chunk grid is expected.
    """
    eligible = [
        e for e in _FIXTURES
        if e["id"] not in _PARITY_GAPS
        and e["id"] not in _DASK_SKIPS
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
    da = open_geotiff(str(_fixture_path(entry)), chunks=CHUNK_SIZE)
    assert hasattr(da.data, "dask"), (
        f"expected a dask-backed DataArray for {entry['id']!r}, "
        f"got data of type {type(da.data).__name__}"
    )
    # ``numblocks`` is a tuple of int per axis. Both spatial axes must
    # have at least two blocks; the windowing logic only fires when the
    # array is actually split.
    nb = da.data.numblocks
    assert len(nb) >= 2 and all(b >= 2 for b in nb[-2:]), (
        f"expected a chunk grid >= 2x2 along the spatial axes for "
        f"{entry['id']!r}, got numblocks={nb}"
    )
