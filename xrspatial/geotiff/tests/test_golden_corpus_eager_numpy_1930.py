"""Eager numpy backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 1 of the corpus plan. The phase 2 smoke tests build a candidate
DataArray by reading each fixture through rasterio directly; they prove the
oracle agrees with itself but say nothing about whether xrspatial's reader
agrees with rasterio. This module is the first real parity layer: it opens
every shipped fixture with the eager numpy ``open_geotiff`` path and feeds
the result to ``compare_to_oracle``.

The fixture list is discovered from the manifest. Each entry that has a
``.tif`` on disk runs through the oracle. Lossy fixtures (the JPEG cell)
are routed through the oracle's ``lossy=True`` path so only shape, dtype,
transform, and CRS are checked.

Skip taxonomy
-------------
Cells listed in ``_KNOWN_SKIPS`` are skipped with a documented reason. Each
entry either points at a real parity gap that needs a fix in
``xrspatial/geotiff/`` (open issue suggested below) or at an oracle gap
that will close once a follow-up phase extends ``_oracle.py``. The skip
list is itself the gap analysis the corpus is designed to surface.

* ``compression_jpeg_uint8_ycbcr`` -- RGB band axis order divergence
  between rasterio (bands, y, x) and xrspatial (y, x, band). The oracle's
  ``_assert_shape_only`` does not yet normalise multi-band axis order;
  candidate for a follow-up oracle extension.
* ``crs_citation_only`` -- xrspatial decodes the citation into the
  deprecated ``attrs['geog_citation']`` but does not emit a canonical
  ``attrs['crs']`` or ``attrs['crs_wkt']``. Real parity gap the corpus
  surfaced; needs a fix in ``_crs.py`` to round-trip citation WKT.
* ``nodata_int_sentinel_uint16``, ``stripped_le_uint16``,
  ``stripped_be_uint16``, ``tiled_le_uint16``, ``tiled_be_uint16`` --
  integer nodata masking. xrspatial masks sentinel pixels to NaN and
  upcasts to float64 per #1988 (``attrs['masked_nodata']=True``). The
  oracle compares the raw integer pixel array. Needs a small oracle
  extension that consults ``attrs['masked_nodata']`` and applies the
  equivalent mask to the rasterio reference before comparing.
* ``nodata_miniswhite_uint8`` -- MinIsWhite photometric inversion.
  xrspatial inverts pixels per #1797; rasterio leaves them raw. The
  inversion is asserted by ``test_miniswhite_backend_parity_1797.py``.

Each backend gets its own module under ``xrspatial/geotiff/tests/``; this
file owns the eager numpy slice. Phase 3 PRs for dask, GPU, dask+GPU, HTTP,
and VRT add their own siblings.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


# Skip-with-reason taxonomy. Each entry documents a known parity gap the
# corpus surfaces. See the module docstring for the rationale on each.
_NODATA_MASKING_SKIP = (
    "integer nodata masking: xrspatial masks sentinel pixels to NaN and "
    "upcasts to float64 per #1988 (attrs['masked_nodata']=True). The oracle "
    "compares raw integer pixels; needs an oracle extension that consults "
    "attrs['masked_nodata']."
)

_KNOWN_SKIPS: dict[str, str] = {
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
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "#1797; rasterio leaves them raw. Covered by "
        "test_miniswhite_backend_parity_1797.py."
    ),
    "nodata_int_sentinel_uint16": _NODATA_MASKING_SKIP,
    "stripped_le_uint16": _NODATA_MASKING_SKIP,
    "stripped_be_uint16": _NODATA_MASKING_SKIP,
    "tiled_le_uint16": _NODATA_MASKING_SKIP,
    "tiled_be_uint16": _NODATA_MASKING_SKIP,
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
    return bool(entry.get("tolerance", {}).get("lossy", False))


_FIXTURES = _resolved_fixtures()
_FIXTURE_IDS = [e["id"] for e in _FIXTURES]


@pytest.fixture(params=_FIXTURES, ids=_FIXTURE_IDS)
def manifest_entry(request) -> dict:
    return request.param


def test_eager_numpy_parity(manifest_entry: dict) -> None:
    """``open_geotiff(path)`` agrees with the rasterio oracle.

    Eager numpy is the default ``open_geotiff`` dispatch (no ``chunks``,
    no ``gpu=True``). The oracle compares pixels (bit-exact, or shape-only
    when ``lossy``), dtype, transform, CRS, and nodata. Known parity gaps
    are skipped with a reason; see ``_KNOWN_SKIPS`` and the module
    docstring.
    """
    fixture_id = manifest_entry["id"]
    if fixture_id in _KNOWN_SKIPS:
        pytest.skip(_KNOWN_SKIPS[fixture_id])
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )
    candidate = open_geotiff(str(path))
    compare_to_oracle(path, candidate, lossy=_is_lossy(manifest_entry))


def test_known_skip_ids_are_in_manifest() -> None:
    """Every id in ``_KNOWN_SKIPS`` must be a real manifest entry.

    Guards against typos: a stale skip key would silently let a known-bad
    fixture through if its id were misspelled.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    stale = set(_KNOWN_SKIPS) - manifest_ids
    assert not stale, f"skip taxonomy references unknown fixture ids: {sorted(stale)}"
