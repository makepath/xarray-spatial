"""Fast / slow pytest marker helper for the golden-corpus matrix.

Each manifest fixture carries a ``tags`` list. Fixtures tagged ``fast``
run in the PR CI fast lane; everything else is treated as slow and
gets ``pytest.mark.slow`` attached so PR CI can opt out via
``pytest -m "not slow"``. Nightly / release CI runs without the filter
and exercises everything.

Today most shipped fixtures carry ``fast``. The six ``compression_*``
fixtures in the manifest do not, so they land in the slow lane and
``pytest -m "not slow"`` deselects them. A one-line manifest edit per
fixture would move them to the fast lane if the team decides that is
the right calibration. Future heavier fixtures (large COGs,
multi-source VRTs, jpeg2000 cells) will drop in behind the same
boundary without each backend test module re-implementing it.

Usage from a backend test module::

    from xrspatial.geotiff.tests.golden_corpus._marks import (
        fast_slow_marks_for,
    )

    def _build_param(entry):
        marks = list(fast_slow_marks_for(entry))
        if entry["id"] in _PARITY_GAPS:
            marks.append(pytest.mark.xfail(...))
        return pytest.param(entry, id=entry["id"], marks=marks)

``fast_slow_marks_for`` is a generator, so chaining with other marks
is just a ``list(...) + [extra_mark]`` away.
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import pytest

_FAST_TAG = "fast"

# Optional Python packages required to decode specific TIFF compressions.
# Fixtures using these codecs must be skipped when the package is not
# importable; otherwise the read-path raises ``ImportError`` and the
# parity test fails for an environmental reason rather than a real bug.
# Keep this list in sync with the optional codec imports in
# ``xrspatial.geotiff._compression``.
_COMPRESSION_OPTIONAL_DEPS: dict[str, str] = {
    "lerc": "lerc",
}


def is_fast(entry: dict[str, Any]) -> bool:
    """Return True when the manifest entry is in the fast lane.

    The contract: a fixture is fast iff its ``tags`` list contains the
    literal string ``"fast"``. Missing or empty ``tags`` count as slow,
    on the theory that an untagged fixture is one a contributor forgot
    to triage rather than one that is intentionally cheap.
    """
    tags = entry.get("tags") or []
    return _FAST_TAG in tags


def fast_slow_marks_for(entry: dict[str, Any]) -> list[pytest.MarkDecorator]:
    """Return the slow mark (in a list) when the entry is not fast.

    Returns ``[pytest.mark.slow]`` for slow fixtures and ``[]`` for fast
    ones, so the caller can splat the result into its ``marks=`` list
    without an empty-mark guard or a generator-to-list conversion.
    """
    return [pytest.mark.slow] if not is_fast(entry) else []


def optional_dep_marks_for(entry: dict[str, Any]) -> list[pytest.MarkDecorator]:
    """Return a skipif mark when the fixture's codec needs a missing dep.

    Some compressions (currently LERC) rely on a Python package that is
    not a hard dependency of xrspatial. On a host without that package
    the read-path raises ``ImportError``; the parity test would then
    fail for an environmental reason rather than a real bug. This helper
    yields a ``pytest.mark.skipif`` mark in that case so the run stays
    green on minimal envs. Returns ``[]`` when the codec has no optional
    dep or when the dep is importable.
    """
    codec = entry.get("compression")
    dep = _COMPRESSION_OPTIONAL_DEPS.get(codec)
    if dep is None or find_spec(dep) is not None:
        return []
    return [
        pytest.mark.skipif(
            True,
            reason=(
                f"{dep!r} is not installed; fixture {entry['id']!r} uses "
                f"the {codec!r} codec which requires it"
            ),
        )
    ]
