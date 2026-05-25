"""fsspec backend cells against the golden-corpus oracle (issue #1930).

Final corpus-coverage backend per issue #1930's proposal text
("fsspec where applicable"). Mirrors the eager, dask, and GPU parity
layers but reads each fixture through an fsspec ``memory://`` URL:
the fixture's bytes are pushed into the in-process fsspec memory
filesystem, then ``open_geotiff('memory:///corpus/<id>.tif')`` walks
through ``_is_fsspec_uri`` and ``_CloudSource`` the same way an
``s3://`` or ``gs://`` URL would. The ``memory://`` scheme ships
with fsspec and runs without any external credentials or services,
so it works in plain CI while still exercising the cloud read path.

The cloud-eager read path returns a numpy-backed DataArray (the
``_CloudSource`` constructor downloads the bytes via ``read_all()``
within the ``max_cloud_bytes`` budget; see #1928). It does NOT chunk
into dask. ``test_fsspec_candidate_is_actually_numpy`` pins that
contract.

Skip / xfail taxonomy
---------------------
``_PARITY_GAPS`` lists shared codec / attrs gaps; ``_FSSPEC_SKIPS``
lists gaps that surface only on the fsspec path; ``_INTENTIONAL_SKIPS``
covers by-design divergences (MinIsWhite inversion). All three tables
start empty on this PR because every shared gap has been closed at
the oracle / codec layer and the fsspec read path piggybacks on the
eager decode primitives.

Resolved gaps (no longer xfail):

* Integer nodata masking -- closed by the oracle's
  ``_normalise_for_masked_nodata`` helper.
* RGB band axis order on the JPEG-YCbCr fixture -- closed by the
  oracle's ``_normalise_axis_order`` helper.
* ``crs_citation_only`` -- closed by ``_synthesize_user_defined_wkt``
  in ``_geotags.py``.

Intentional skip (``skip``):

* ``nodata_miniswhite_uint8`` -- MinIsWhite photometric inversion.
  xrspatial inverts pixels per #1797; rasterio leaves them raw.
  Covered by ``test_miniswhite_backend_parity_1797.py``.

Memory filesystem hygiene
-------------------------
fsspec's memory filesystem is a process-global singleton. The
``memory_fs_clean`` fixture wipes it before and after each test so
fixtures from earlier tests cannot leak into the read path and a
flaky test cannot poison its successors.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")
fsspec = pytest.importorskip("fsspec")

import numpy as np  # noqa: E402

from xrspatial.geotiff import open_geotiff

# PR 4 of epic #2340: corpus has experimental + jpeg entries.
_OPTIN = {"allow_experimental_codecs": True, "allow_internal_only_jpeg": True}
  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._marks import fast_slow_marks_for  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import compare_to_oracle  # noqa: E402

FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


# Shared codec / attrs gaps. Empty: every gap the corpus has surfaced
# has been closed at the oracle / codec layer.
_PARITY_GAPS: dict[str, str] = {}

# fsspec-only gaps. Empty in the first pass; add an entry only when
# a fixture is fsspec-specific (i.e. eager passes, fsspec does not).
_FSSPEC_SKIPS: dict[str, str] = {}

# Fixtures whose overview-IFD code path is not yet wired into the
# xrspatial reader. The base-IFD parity check still runs; the
# overview-level loop driven by ``candidate_factory`` is skipped for
# these ids. See the eager module for the rationale.
_OVERVIEW_READER_GAPS: dict[str, str] = {
    "overview_external_ovr_uint16": (
        "External .ovr sidecar reader is not implemented in xrspatial. "
        "The base IFD still parity-checks; the overview levels live in "
        "a sibling .tif.ovr that the reader does not open."
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
    """Return manifest entries with defaults merged, sorted by id."""
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
    """Wrap a fixture entry in a ``pytest.param`` with the right marks.

    Shared codec / attrs gaps and fsspec-only gaps both become strict
    ``xfail`` so the cell flips to a hard failure the day the gap
    closes. Intentional skips get a plain skip. Slow fixtures inherit
    ``pytest.mark.slow`` from the corpus helper so the fast / slow
    split (#2062) cooperates.
    """
    fid = entry["id"]
    marks = list(fast_slow_marks_for(entry))
    if fid in _PARITY_GAPS:
        marks.append(
            pytest.mark.xfail(reason=_PARITY_GAPS[fid], strict=True)
        )
    elif fid in _FSSPEC_SKIPS:
        marks.append(
            pytest.mark.xfail(reason=_FSSPEC_SKIPS[fid], strict=True)
        )
    elif fid in _INTENTIONAL_SKIPS:
        marks.append(pytest.mark.skip(reason=_INTENTIONAL_SKIPS[fid]))
    return pytest.param(entry, id=fid, marks=marks)


_FIXTURES = _resolved_fixtures()
_PARAMS = [_build_param(e) for e in _FIXTURES]


def _serve_via_memory(payload: bytes, fixture_id: str) -> str:
    """Push ``payload`` into the fsspec ``memory://`` filesystem.

    Returns the ``memory:///corpus/<fixture_id>.tif`` URL the reader
    should open. The three-slash form is required: ``memory://`` writes
    must use a path beginning with ``/`` or fsspec rejects them.
    """
    fs = fsspec.filesystem("memory")
    url = f"memory:///corpus/{fixture_id}.tif"
    # ``pipe`` is the idiomatic single-shot byte writer for fsspec
    # filesystems; ``test_cloud_read_byte_limit_1928.py`` uses the
    # same pattern.
    fs.pipe(f"/corpus/{fixture_id}.tif", payload)
    return url


@pytest.fixture
def memory_fs_clean():
    """Wipe the fsspec memory filesystem around each test.

    The memory filesystem is a process-global singleton, so leftover
    state from an earlier test (or the sanity-check test below) could
    otherwise show up in the read path. Clears before and after for
    belt-and-braces hygiene.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    try:
        yield fs
    finally:
        fs.store.clear()


@pytest.mark.parametrize("manifest_entry", _PARAMS)
def test_fsspec_parity(manifest_entry: dict, memory_fs_clean) -> None:
    """``open_geotiff('memory://...')`` agrees with the rasterio oracle.

    The fixture's bytes are written into the fsspec memory filesystem
    and read back through ``_CloudSource``. This is the same read path
    a real ``s3://`` / ``gs://`` URL would take, minus the network and
    credentials. The oracle reads from the local on-disk fixture path
    because rasterio does not know about the memory filesystem; both
    sides see the same bytes either way.
    """
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            f"to materialise the full corpus"
        )

    with open(path, "rb") as f:
        payload = f.read()
    url = _serve_via_memory(payload, fixture_id)

    candidate = open_geotiff(url, **_OPTIN)
    # When the fixture carries pyramid overviews, hand the oracle a
    # factory it can use to fetch each overview level via the same
    # cloud-eager path. The overview-IFD scan also goes through
    # ``_CloudSource``, so any divergence there shows up here.
    overviews = manifest_entry.get("overviews") or []
    factory = (
        (lambda level, u=url: open_geotiff(u, overview_level=level, **_OPTIN))
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
    """Every id in the parity-gap, fsspec-skip, overview-gap, or
    intentional-skip tables must exist in the manifest.

    Guards against typos: a stale entry would silently keep a
    known-bad fixture marked as expected-to-fail even after it was
    renamed or removed.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    tagged = (
        set(_PARITY_GAPS)
        | set(_FSSPEC_SKIPS)
        | set(_OVERVIEW_READER_GAPS)
        | set(_INTENTIONAL_SKIPS)
    )
    stale = tagged - manifest_ids
    assert not stale, (
        f"taxonomy references unknown fixture ids: {sorted(stale)}"
    )


def test_fsspec_candidate_is_actually_numpy(memory_fs_clean) -> None:
    """Sanity check: the fsspec read path returns a numpy-backed array.

    ``open_geotiff(memory_url)`` without ``chunks=`` should go through
    the eager ``_CloudSource`` path which downloads the object and
    decodes into a numpy array. If a regression accidentally wired the
    fsspec URI to a dask-chunked path, this test catches it.

    Picks the first eligible (not skipped, on disk) fixture in sorted
    id order so the assertion is stable across corpus growth.
    """
    eligible = [
        e for e in _FIXTURES
        if e["id"] not in _PARITY_GAPS
        and e["id"] not in _FSSPEC_SKIPS
        and e["id"] not in _INTENTIONAL_SKIPS
        and _fixture_path(e).exists()
    ]
    if not eligible:
        pytest.skip("no eligible fixture on disk")
    entry = eligible[0]
    path = _fixture_path(entry)
    with open(path, "rb") as f:
        payload = f.read()
    url = _serve_via_memory(payload, entry["id"])
    da = open_geotiff(url, **_OPTIN)
    assert isinstance(da.data, np.ndarray), (
        f"expected a numpy.ndarray for the fsspec eager path on "
        f"{entry['id']!r}, got {type(da.data).__name__}"
    )
