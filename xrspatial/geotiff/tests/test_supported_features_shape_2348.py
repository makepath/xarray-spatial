"""Structural invariants for ``SUPPORTED_FEATURES`` (issue #2348).

Background
----------
Epic #2340 reconciles ``SUPPORTED_FEATURES`` with the GeoTIFF release
contract. The tier label on each entry is the source of truth that the
docs page, the user-guide notebook table, and the writer gates all
read from. If a new entry lands without a tier, or with a tier outside
the closed set ``{stable, advanced, experimental, internal_only}``,
the downstream consumers silently mis-render or skip it.

The tier-coverage test (``test_supported_features_tiers_2137.py``)
exercises the codec gate. This file pins the structural invariants of
the mapping itself so a stray entry without a tier, or with a
free-form tier string, fails CI before it reaches the consumers.

What this test pins
-------------------
* The mapping is a dict, non-empty, keyed by ``"<group>.<name>"``
  strings.
* Every value is one of ``stable`` / ``advanced`` / ``experimental``
  / ``internal_only``. The tier set is closed.
* Every key is unique (defended against accidental dict-literal
  duplicates that Python silently dedupes).
* The wave-1 reconciliation under #2340 lands the expected
  promotions / demotions (``reader.windowed`` and ``reader.dask``
  added at ``stable``; ``reader.allow_rotated`` and
  ``reader.allow_unparseable_crs`` at ``experimental``).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xrspatial.geotiff import SUPPORTED_FEATURES

_VALID_TIERS = frozenset({'stable', 'advanced', 'experimental', 'internal_only'})


def test_supported_features_is_non_empty_dict():
    """Catches an accidental refactor that swaps the dict for a
    different container or empties it."""
    assert isinstance(SUPPORTED_FEATURES, dict)
    assert len(SUPPORTED_FEATURES) > 0


def test_every_entry_has_a_tier():
    """Every value is a non-empty string. Catches ``None`` /
    placeholder values that would otherwise silently disable the
    docs / notebook renderer for that row."""
    for name, tier in SUPPORTED_FEATURES.items():
        assert isinstance(tier, str), (name, type(tier).__name__)
        assert tier, name


def test_tier_set_is_closed():
    """Every tier value is one of the four documented labels. The
    set is closed; introducing a new tier requires updating the
    docs, the notebook, and this test together."""
    seen = set(SUPPORTED_FEATURES.values())
    extras = seen - _VALID_TIERS
    assert not extras, (
        f"SUPPORTED_FEATURES carries unrecognised tier labels {sorted(extras)!r}; "
        f"valid labels are {sorted(_VALID_TIERS)!r}. Adding a new tier requires "
        f"updating xrspatial.geotiff.__init__ docs, the user-guide notebook table, "
        f"and this test in the same commit."
    )


def test_every_tier_label_is_used():
    """Every documented tier label appears on at least one entry.
    Catches accidental drift where a tier is documented in the
    package docstring but no feature references it (which makes the
    label dead code)."""
    seen = set(SUPPORTED_FEATURES.values())
    missing = _VALID_TIERS - seen
    assert not missing, (
        f"the following tier labels are documented but unused: {sorted(missing)!r}. "
        f"Either remove the label from the docs / notebook or add an entry that uses it."
    )


def test_keys_follow_group_dot_name_shape():
    """Every key is ``"<group>.<name>"`` with a non-empty group and
    name. The renderer in the user-guide notebook splits on ``.``
    once to group rows; an entry without a dot would land in an
    "(unknown)" bucket."""
    for key in SUPPORTED_FEATURES:
        assert isinstance(key, str), type(key).__name__
        head, _, tail = key.partition('.')
        assert head and tail, key
        assert key.count('.') >= 1, key


def test_keys_are_unique_in_source():
    """Dict literals silently dedupe duplicate keys ('a': 'x', 'a':
    'y' -> {'a': 'y'}). Parse the source and assert no key appears
    twice so a future copy/paste typo fails CI instead of silently
    overwriting the earlier tier."""
    src = Path(__file__).resolve().parents[1] / '_attrs.py'
    tree = ast.parse(src.read_text())

    found_keys: list[str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets
        if (len(targets) == 1
                and isinstance(targets[0], ast.Name)
                and targets[0].id == 'SUPPORTED_FEATURES'
                and isinstance(node.value, ast.Dict)):
            found_keys = []
            for k in node.value.keys:
                # Dict-literal keys are ast.Constant on 3.8+.
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    found_keys.append(k.value)
            break

    assert found_keys is not None, (
        "could not locate the ``SUPPORTED_FEATURES = {...}`` literal in "
        "xrspatial/geotiff/_attrs.py; this test parses the source so a "
        "duplicate key cannot be hidden by Python's dict-literal dedup."
    )
    duplicates = [k for k in found_keys if found_keys.count(k) > 1]
    assert not duplicates, (
        f"duplicate keys in the SUPPORTED_FEATURES literal: "
        f"{sorted(set(duplicates))!r}. Python silently dedupes these so the "
        f"later tier wins; remove the duplicates or rename them."
    )


@pytest.mark.parametrize("key,tier", [
    ('reader.windowed', 'stable'),
    ('reader.dask', 'stable'),
    ('reader.allow_rotated', 'experimental'),
    ('reader.allow_unparseable_crs', 'experimental'),
])
def test_epic_2340_wave_1_reconciliation(key, tier):
    """Wave-1 reconciliation under epic #2340 lands the expected
    promotions and demotions. Pinned so a future revert that bumps
    these back to ``advanced`` or drops them entirely fails this
    test before it reaches the docs / release notes.
    """
    assert key in SUPPORTED_FEATURES, (
        f"{key!r} dropped from SUPPORTED_FEATURES; epic #2340 introduced "
        f"this entry at tier {tier!r}. Restore it or update the test if "
        f"the reconciliation has been revised."
    )
    assert SUPPORTED_FEATURES[key] == tier, (
        f"{key!r} expected tier {tier!r}, got {SUPPORTED_FEATURES[key]!r}. "
        f"Epic #2340 set this tier; a promotion / demotion needs to be "
        f"justified in the changelog and reflected here."
    )
