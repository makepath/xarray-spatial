"""Fast / slow pytest marker helper for the golden-corpus matrix
(issue #1930, phase 4 PR 1).

Each manifest fixture carries a ``tags`` list. Fixtures tagged ``fast``
run in the PR CI fast lane; everything else is treated as slow and
gets ``pytest.mark.slow`` attached so PR CI can opt out via
``pytest -m "not slow"``. Nightly / release CI runs without the filter
and exercises everything.

Today every shipped fixture carries ``fast``; the helper is in place
so future heavier fixtures (large COGs, multi-source VRTs, jpeg2000
cells) drop in without each backend test module re-implementing the
fast/slow boundary.

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

from collections.abc import Iterator
from typing import Any

import pytest


_FAST_TAG = "fast"


def is_fast(entry: dict[str, Any]) -> bool:
    """Return True when the manifest entry is in the fast lane.

    The contract: a fixture is fast iff its ``tags`` list contains the
    literal string ``"fast"``. Missing or empty ``tags`` count as slow,
    on the theory that an untagged fixture is one a contributor forgot
    to triage rather than one that is intentionally cheap.
    """
    tags = entry.get("tags") or []
    return _FAST_TAG in tags


def fast_slow_marks_for(entry: dict[str, Any]) -> Iterator[pytest.MarkDecorator]:
    """Yield ``pytest.mark.slow`` when the entry is not in the fast lane.

    Yields nothing for fast fixtures so the caller can just splat the
    iterator into its ``marks=`` list without an empty-mark guard.
    """
    if not is_fast(entry):
        yield pytest.mark.slow
