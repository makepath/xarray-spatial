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

None today. Every parity gap the corpus has surfaced so far has either
been fixed (see Resolved gaps below) or moved to a per-backend skip
table because it is not a shared decode/attrs gap.

Resolved gaps (no longer xfail):

* Integer nodata masking. The oracle now consults
  ``attrs['masked_nodata']`` and rewrites the rasterio reference to
  match the candidate's float-plus-NaN view (see
  ``_normalise_for_masked_nodata`` in ``_oracle.py``), so the five
  fixtures ``nodata_int_sentinel_uint16``, ``stripped_le_uint16``,
  ``stripped_be_uint16``, ``tiled_le_uint16``, ``tiled_be_uint16``
  pass directly.
* ``compression_jpeg_uint8_ycbcr`` -- RGB band axis order divergence
  between rasterio ``(bands, y, x)`` and xrspatial ``(y, x, band)``.
  The oracle's ``_normalise_axis_order`` helper now transposes the
  trailing-band candidate to leading-band before the shape and pixel
  checks run, so this fixture passes directly.
* ``crs_citation_only`` -- the GeoTIFF reader now synthesizes a
  canonical WKT for user-defined geographic CRSes from the ellipsoid
  / units GeoKeys and stamps it on ``attrs['crs_wkt']`` (see
  ``_synthesize_user_defined_wkt`` in ``_geotags.py``). The oracle's
  ``_candidate_crs`` picks it up via the WKT branch, and ``_crs_equal``
  compares the PROJ dicts because neither side has an EPSG code.

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


_PARITY_GAPS: dict[str, str] = {}

# Fixtures whose overview-IFD code path is not yet wired into the
# xrspatial reader. Empty as of issue #2112, which taught the eager
# numpy reader to discover and parse sibling ``.tif.ovr`` files. New
# entries should include a follow-up issue link so the gap stays visible.
_OVERVIEW_READER_GAPS: dict[str, str] = {}

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
    # When the fixture carries pyramid overviews, hand the oracle a
    # factory it can use to fetch each overview level via the same
    # backend (eager numpy). The oracle introspects the rasterio source
    # for overview IFDs and compares each one against the factory's
    # output; fixtures without overviews exercise the historical
    # base-IFD-only path. Fixtures the xrspatial reader cannot resolve
    # at level >= 1 (e.g. sidecar .ovr overviews) live in
    # ``_OVERVIEW_READER_GAPS`` and skip the overview iteration; the
    # base-IFD comparison still runs.
    overviews = manifest_entry.get("overviews") or []
    factory = (
        (lambda lvl, p=path: open_geotiff(str(p), overview_level=lvl))
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
    """Every id in the parity-gap or intentional-skip tables must exist.

    Guards against typos: a stale entry would silently keep a known-bad
    fixture marked as expected-to-fail even after it was renamed or removed.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = (
        set(_PARITY_GAPS)
        | set(_OVERVIEW_READER_GAPS)
        | set(_INTENTIONAL_SKIPS)
    )
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )
