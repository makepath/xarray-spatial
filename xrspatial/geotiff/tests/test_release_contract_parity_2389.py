"""Lock ``geotiff_release_contract.md`` against ``SUPPORTED_FEATURES``.

Background
----------
``docs/source/reference/geotiff_release_contract.md`` lists every public
GeoTIFF feature with its tier and claims:

    The tier strings here match the strings in
    ``xrspatial.geotiff.SUPPORTED_FEATURES`` at runtime.

Before this test, nothing in CI checked that claim. The sibling release
gate registry (``release_gates/test_stable_features.py``,
``Cross-cutting meta-gates`` section) parses
``release_gate_geotiff.rst``, not this ``.md`` contract page, so the
contract could (and did) silently drift the next time a key was
re-tiered in ``_attrs.py`` -- twice in two releases (#2381 and #2389).

What this test pins
-------------------
* Every row in the feature tier table parses cleanly into a
  ``(key, tier)`` pair.
* The key on every row is a real key in ``SUPPORTED_FEATURES``.
* The tier column on every row matches ``SUPPORTED_FEATURES[key]``
  byte-for-byte (so a future ``internal-only`` vs ``internal_only``
  drift fails the gate before the doc lands).

Out of scope
------------
* Locking the section-heading prose (``### experimental`` etc.) --
  those are human-readable labels, not runtime tier strings.
* Locking the contract page against ``release_gate_geotiff.rst`` --
  the gate page only enumerates ``stable`` and ``advanced`` tiers
  (``release_gates/test_stable_features.py`` already covers that side).
"""
from __future__ import annotations

import re
from pathlib import Path

from xrspatial.geotiff import SUPPORTED_FEATURES

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_CONTRACT = (
    _REPO_ROOT / "docs" / "source" / "reference" / "geotiff_release_contract.md"
)

# Match table rows of the form:
#   | `codec.none` | stable | Uncompressed... |
# The key column is always in backticks; the tier column is the bare
# tier label that should appear verbatim in SUPPORTED_FEATURES.
_ROW_RE = re.compile(
    r"^\|\s*`([a-z_]+\.[a-z0-9_]+)`\s*\|\s*([a-z_]+)\s*\|",
    re.MULTILINE,
)


def _contract_rows() -> list[tuple[str, tuple[str, str]]]:
    """Return ``(line_number_hint, (key, tier))`` for every table row.

    The line-number hint is the 1-based offset of the match inside the
    file so assertion failures can point a maintainer at the exact row.
    """
    text = _CONTRACT.read_text(encoding="utf-8")
    rows: list[tuple[str, tuple[str, str]]] = []
    for match in _ROW_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        rows.append((f"{_CONTRACT.name}:{line_no}", (match.group(1), match.group(2))))
    return rows


def test_contract_table_parses_into_rows() -> None:
    """The regex catches the table rows. If a future doc rewrite breaks
    the row shape, fail loudly here instead of silently passing the
    tier check on zero rows.
    """
    rows = _contract_rows()
    assert rows, (
        f"no contract rows parsed from {_CONTRACT}; the markdown table "
        "shape may have changed and this test's regex needs to follow."
    )
    # Sanity floor: the contract today lists roughly 28 keys. Use a
    # conservative lower bound so a sweeping accidental table truncation
    # fails the gate. The exact count is not pinned; tiers move.
    assert len(rows) >= 20, (
        f"only {len(rows)} contract rows parsed; the table may have been "
        "truncated or the row format changed."
    )


def test_contract_keys_are_real_supported_features() -> None:
    """Every key in the contract table exists in ``SUPPORTED_FEATURES``.
    A stray row left behind after a key is removed from ``_attrs.py``
    fails here.
    """
    bad: list[tuple[str, str]] = []
    for where, (key, _tier) in _contract_rows():
        if key not in SUPPORTED_FEATURES:
            bad.append((where, key))
    assert not bad, (
        "contract table lists keys that are not in SUPPORTED_FEATURES; "
        "either the key was removed from _attrs.py and the doc row was "
        "left behind, or the row's backticked text is wrong: "
        f"{bad}"
    )


def test_contract_tiers_match_supported_features() -> None:
    """Every row's tier column matches ``SUPPORTED_FEATURES[key]``.
    This is the gate that would have caught the #2381 / #2389 drift.
    """
    mismatches: list[tuple[str, str, str, str]] = []
    for where, (key, tier) in _contract_rows():
        if key not in SUPPORTED_FEATURES:
            # Reported by ``test_contract_keys_are_real_supported_features``;
            # skip here to keep this failure focused on tier drift.
            continue
        expected = SUPPORTED_FEATURES[key]
        if tier != expected:
            mismatches.append((where, key, tier, expected))
    assert not mismatches, (
        "contract page tier strings disagree with SUPPORTED_FEATURES; "
        "the contract page promises the two match verbatim. Update the "
        "tier column in geotiff_release_contract.md to the runtime tier "
        "(format: (where, key, doc_tier, runtime_tier)): "
        f"{mismatches}"
    )
