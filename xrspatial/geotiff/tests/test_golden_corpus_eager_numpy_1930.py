"""Eager numpy backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 1 of the corpus plan. The phase 2 smoke tests build a candidate
DataArray by reading each fixture through rasterio directly; they prove the
oracle agrees with itself but say nothing about whether xrspatial's reader
agrees with rasterio. This module is the first real parity layer: it opens
every shipped fixture with the eager numpy ``open_geotiff`` path and feeds
the result to ``compare_to_oracle``.

The fixture list is discovered from the manifest at module-import time so
``pytest.mark.parametrize`` can attach per-fixture marks (``xfail`` for real
gaps, ``skip`` for intentional divergences). A broken manifest therefore
fails collection rather than test execution; the manifest validator tests
under ``test_golden_corpus_manifest_1930.py`` catch that case separately.

Skip / xfail taxonomy
---------------------
``_PARITY_GAPS`` flags real parity gaps the corpus surfaces. Each entry
becomes a ``pytest.mark.xfail(strict=True)``: the test stays red-flagged
while the gap exists, and the day the gap closes it flips to a hard failure
that forces a developer to remove the mark. ``_INTENTIONAL_SKIPS`` covers
divergences that are by-design (the MinIsWhite inversion), where ``xfail``
would never fire.

Real parity gaps (``xfail``):

* ``compression_jpeg_uint8_ycbcr`` -- RGB band axis order divergence
  between rasterio ``(bands, y, x)`` and xrspatial ``(y, x, band)``. The
  oracle's ``_assert_shape_only`` does not yet normalise multi-band axis
  order.
* ``crs_citation_only`` -- xrspatial decodes the citation into the
  deprecated ``attrs['geog_citation']`` but does not emit a canonical
  ``attrs['crs']`` or ``attrs['crs_wkt']``. Real parity gap; needs a fix
  in ``_crs.py`` to round-trip citation WKT.
* ``nodata_int_sentinel_uint16``, ``stripped_le_uint16``,
  ``stripped_be_uint16``, ``tiled_le_uint16``, ``tiled_be_uint16`` --
  integer nodata masking. xrspatial masks sentinel pixels to NaN and
  upcasts to float64 per #1988 (``attrs['masked_nodata']=True``); the
  oracle compares the raw integer pixel array. Needs a small oracle
  extension that consults ``attrs['masked_nodata']`` and applies the
  equivalent mask to the rasterio reference before comparing.

Intentional skip (``skip``):

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
from xrspatial.geotiff.tests.golden_corpus._marks import (  # noqa: E402
    fast_slow_marks_for,
)
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


_NODATA_MASKING_REASON = (
    "integer nodata masking: xrspatial masks sentinel pixels to NaN and "
    "upcasts to float64 per #1988 (attrs['masked_nodata']=True). The oracle "
    "compares raw integer pixels; needs an oracle extension that consults "
    "attrs['masked_nodata']."
)

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
    # ``tolerance`` may be missing entirely or explicitly null in a future
    # manifest; ``or {}`` collapses both cases to an empty dict so
    # ``.get('lossy')`` does not blow up.
    tol = entry.get("tolerance") or {}
    return bool(tol.get("lossy", False))


def _build_param(entry: dict) -> pytest.param:
    """Wrap a fixture entry in a ``pytest.param`` with the right marks.

    Real parity gaps get ``xfail(strict=True)`` so the test surfaces a hard
    failure the day the gap closes. The MinIsWhite cell gets a plain skip
    because the divergence is intentional. Non-fast fixtures additionally
    pick up ``pytest.mark.slow`` from the corpus helper.
    """
    fid = entry["id"]
    marks = list(fast_slow_marks_for(entry))
    if fid in _PARITY_GAPS:
        marks.append(
            pytest.mark.xfail(reason=_PARITY_GAPS[fid], strict=True)
        )
    elif fid in _INTENTIONAL_SKIPS:
        marks.append(pytest.mark.skip(reason=_INTENTIONAL_SKIPS[fid]))
    return pytest.param(entry, id=fid, marks=marks)


_FIXTURES = _resolved_fixtures()
_PARAMS = [_build_param(e) for e in _FIXTURES]


@pytest.mark.parametrize("manifest_entry", _PARAMS)
def test_eager_numpy_parity(manifest_entry: dict) -> None:
    """``open_geotiff(path)`` agrees with the rasterio oracle.

    Eager numpy is the default ``open_geotiff`` dispatch (no ``chunks``,
    no ``gpu=True``). The oracle compares pixels (bit-exact, or shape-only
    when ``lossy``), dtype, transform, CRS, and nodata. Known parity gaps
    are flagged via ``xfail`` (strict); intentional divergences via
    ``skip``. See the module docstring for the rationale on each entry.
    """
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )
    candidate = open_geotiff(str(path))
    compare_to_oracle(path, candidate, lossy=_is_lossy(manifest_entry))


def test_taxonomy_ids_are_in_manifest() -> None:
    """Every id in the parity-gap or intentional-skip tables must exist.

    Guards against typos: a stale entry would silently keep a known-bad
    fixture marked as expected-to-fail even after it was renamed or removed.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = set(_PARITY_GAPS) | set(_INTENTIONAL_SKIPS)
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )
