"""Release gate / audit checklist parity tests (issue #2321 sub-task 6).

Background
----------
``docs/source/reference/release_gate_geotiff.rst`` enumerates every
feature the GeoTIFF module promises in release notes, along with the
regression test that locks each behaviour. Two things can break the
checklist's audit value:

1. A cited regression test file is renamed or removed and the checklist
   silently points at nothing.
2. A new tier key shows up in ``SUPPORTED_FEATURES`` and the checklist
   forgets to add a row for it.

These tests pin both. They are intentionally light -- they parse the
``.rst`` source and the ``SUPPORTED_FEATURES`` dict, then cross-check.

What this test pins
-------------------
* Every test file cited in the release gate checklist exists on disk.
* Every key in :data:`xrspatial.geotiff.SUPPORTED_FEATURES` whose tier
  is ``stable`` or ``advanced`` is named at least once in the checklist
  prose, so a new public tier cannot land without a checklist row.
* The HTTP SSRF presence gate (the checklist's cross-cutting row that
  has no other home today) is locked here: an HTTP URL pointing at a
  loopback host raises :class:`UnsafeURLError` from ``open_geotiff``.
* The VRT presence gate: every test file cited in the "VRT supported
  subset" section of the checklist contains at least one ``def test_``
  function, so the row is not pointing at an empty file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from xrspatial.geotiff import SUPPORTED_FEATURES, UnsafeURLError, open_geotiff

# --------------------------------------------------------------------------- #
# Locate the checklist.                                                       #
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_CHECKLIST = (
    _REPO_ROOT / "docs" / "source" / "reference" / "release_gate_geotiff.rst"
)

# Match xrspatial/geotiff/tests/<name>.py inside the checklist body.
_TEST_PATH_RE = re.compile(
    r"`{0,2}(xrspatial/geotiff/tests/[\w/]+\.py)`{0,2}",
)


def _checklist_text() -> str:
    assert _CHECKLIST.is_file(), (
        f"release gate checklist missing: {_CHECKLIST}; the checklist must "
        "ship with the geotiff docs so release notes can cite it"
    )
    return _CHECKLIST.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Gate 1: cited test files exist.                                             #
# --------------------------------------------------------------------------- #


def _cited_test_files() -> set[str]:
    text = _checklist_text()
    # Drop the self-reference so the gate cannot succeed only because it
    # cites itself.
    self_ref = "xrspatial/geotiff/tests/test_release_gate_2321.py"
    matches = set(_TEST_PATH_RE.findall(text))
    matches.discard(self_ref)
    return matches


def test_release_gate_cites_only_existing_test_files() -> None:
    cited = _cited_test_files()
    assert cited, (
        "release gate checklist cites zero test files; either the regex "
        "is wrong or the checklist is empty"
    )
    missing = sorted(p for p in cited if not (_REPO_ROOT / p).is_file())
    assert not missing, (
        "release gate checklist cites tests that do not exist on disk; "
        "rename the checklist row to a real file or restore the test: "
        f"{missing}"
    )
    # Tighten: every cited path must point at a `test_*.py` file, not at
    # ``conftest.py`` or a helper module. The leaf-prefix check catches
    # typos like ``conftests.py`` and accidental citations of non-test
    # support files even though they happen to exist on disk.
    non_test = sorted(p for p in cited if not Path(p).name.startswith("test_"))
    assert not non_test, (
        "release gate checklist cites paths that do not start with "
        "``test_``; the checklist should point at regression tests, not "
        f"conftest or helper modules: {non_test}"
    )


# --------------------------------------------------------------------------- #
# Gate 2: every public tier key appears in the checklist.                     #
# --------------------------------------------------------------------------- #

# Tiers that release notes are allowed to make promises about. ``stable``
# and ``advanced`` features must show up in the checklist so a reader can
# tell what the release covers. ``experimental`` and ``internal_only``
# are deliberately excluded -- the checklist's prose tags them as
# not-promised, so a missing row for those tiers is not a release gate
# failure. Codec keys are handled together as a group in the
# local-read/write section, so the gate excludes them from the
# per-key enumeration.
_PROMISED_TIERS = {"stable", "advanced"}


def _checklist_mentions(text: str, key: str) -> bool:
    """``key`` is something like ``reader.local_file``. Match either the
    bare key or the key as a ``SUPPORTED_FEATURES['key']`` lookup."""
    if key in text:
        return True
    return f"SUPPORTED_FEATURES['{key}']" in text


def test_release_gate_lists_every_promised_supported_feature() -> None:
    text = _checklist_text()
    missing = []
    for key, tier in SUPPORTED_FEATURES.items():
        if tier not in _PROMISED_TIERS:
            continue
        if key.startswith("codec."):
            # Codecs are grouped, not enumerated per-row.
            continue
        if not _checklist_mentions(text, key):
            missing.append((key, tier))
    assert not missing, (
        "promised SUPPORTED_FEATURES entries are missing from the release "
        "gate checklist; add a row (or update SUPPORTED_FEATURES) so the "
        "release notes can quote the tier: "
        f"{missing}"
    )


# --------------------------------------------------------------------------- #
# Gate 3: HTTP SSRF presence gate.                                            #
# --------------------------------------------------------------------------- #


def test_release_gate_http_ssrf_rejects_loopback() -> None:
    """The checklist promises that HTTP URLs targeting loopback hosts
    raise :class:`UnsafeURLError`. Lock that promise here so the row in
    the checklist always points at a passing test rather than at prose.
    """
    with pytest.raises(UnsafeURLError):
        # No network call -- the SSRF check rejects before the socket
        # opens, so this test is offline-safe.
        open_geotiff("http://127.0.0.1/does-not-matter.tif")


@pytest.mark.xfail(
    reason=(
        "Locks in once sub-PR 5 of #2321 (PR #2326) lands. Until then, "
        "uppercase HTTP slips past the SSRF check and falls through to "
        "fsspec, which raises a generic ValueError. Once #2326 is merged, "
        "remove this xfail marker so the release gate enforces the promise."
    ),
    strict=False,
    # Narrow to the two known shapes today (fsspec ValueError) and the
    # post-#2326 shape (UnsafeURLError). A future regression that raises
    # anything else (RuntimeError, OSError from a real socket dial, etc.)
    # should NOT silently xfail -- it should fail loudly.
    raises=(ValueError, UnsafeURLError),
)
def test_release_gate_http_ssrf_rejects_loopback_uppercase_scheme() -> None:
    """Uppercase scheme (sub-PR 5 of #2321) must take the same SSRF
    path. If this test ever skips silently or routes through fsspec,
    the checklist's HTTP row is lying.
    """
    with pytest.raises(UnsafeURLError):
        open_geotiff("HTTP://127.0.0.1/does-not-matter.tif")


# --------------------------------------------------------------------------- #
# Gate 4: VRT rows point at non-empty test files.                             #
# --------------------------------------------------------------------------- #


def _vrt_section_test_files() -> set[str]:
    """Return the test files cited inside the "VRT supported subset"
    section of the checklist."""
    text = _checklist_text()
    start = text.find("VRT supported subset")
    assert start != -1, "checklist is missing the VRT supported subset section"
    end = text.find("Sidecar and overview interactions", start)
    if end == -1:
        end = len(text)
    section = text[start:end]
    return set(_TEST_PATH_RE.findall(section))


def test_release_gate_vrt_rows_point_at_real_test_functions() -> None:
    files = _vrt_section_test_files()
    assert files, "no VRT test files cited in the checklist"
    self_ref = "xrspatial/geotiff/tests/test_release_gate_2321.py"
    empty = []
    for rel in sorted(files):
        if rel == self_ref:
            continue
        path = _REPO_ROOT / rel
        if not path.is_file():
            # Caught by gate 1; skip here.
            continue
        body = path.read_text(encoding="utf-8")
        if "def test_" not in body:
            empty.append(rel)
    assert not empty, (
        "VRT checklist rows cite files with no test functions; either the "
        "file was emptied or the row should be removed: "
        f"{empty}"
    )
