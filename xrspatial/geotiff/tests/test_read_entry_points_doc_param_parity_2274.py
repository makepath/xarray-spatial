"""Regression test for #2274: every kwarg on the public read entry
points has a Parameters-section docstring entry.

The original gap: the four read entry points (``open_geotiff``,
``read_geotiff_dask``, ``read_geotiff_gpu``, ``read_vrt``) accept
``allow_rotated`` and ``allow_unparseable_crs``, but those kwargs were
only documented on ``open_geotiff`` (and only inline in the Tier prose
paragraph for ``allow_unparseable_crs``). The three direct backends
also accept several gated kwargs whose only purpose is to raise
``ValueError`` on the wrong backend so all four readers stay
error-symmetric; those kwargs had no Parameters entry on the backends
that reject them.

This test pins the fix and catches any future addition of a signature
kwarg without a matching Parameters entry on any of the four read
entry points.
"""
from __future__ import annotations

import inspect
import re

import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, read_geotiff_gpu, read_vrt

READ_ENTRY_POINTS = (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
)


# Numpy-style docstring parameter heading pattern. Matches lines like
# ``    name : type`` after ``inspect.getdoc`` has normalised the
# leading indentation to column zero.
_PARAM_HEADING = re.compile(r"^(\w+) : ", flags=re.MULTILINE)


def _signature_params(fn):
    return set(inspect.signature(fn).parameters)


def _documented_params(fn):
    doc = inspect.getdoc(fn) or ""
    return set(_PARAM_HEADING.findall(doc))


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_read_entry_point_kwargs_have_docstring_entries(fn):
    """Every signature kwarg appears in the Parameters section."""
    params = _signature_params(fn)
    documented = _documented_params(fn)
    missing = sorted(params - documented)
    assert missing == [], (
        f"{fn.__name__} has kwargs without Parameters-section entries: "
        f"{missing}. Add a numpy-style ``name : type`` heading for each "
        f"so the docstring agrees with the signature. The kwargs may be "
        f"gated (raise ValueError on the wrong backend) but they are "
        f"still on the public surface, and tools that read the "
        f"docstring (Sphinx, IDE help) cannot tell the kwarg exists "
        f"without an entry. See #2274."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_read_entry_point_docstring_does_not_invent_params(fn):
    """Every Parameters entry maps to a real signature kwarg.

    Catches the inverse drift: a kwarg removed from the signature but
    still listed in the Parameters section.
    """
    params = _signature_params(fn)
    documented = _documented_params(fn)
    extra = sorted(documented - params)
    assert extra == [], (
        f"{fn.__name__} has Parameters-section entries that do not "
        f"appear in the signature: {extra}. Either remove the entry "
        f"or restore the kwarg."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_allow_rotated_documented(fn):
    """``allow_rotated`` was the load-bearing #2274 gap on the backends.

    Pin it explicitly so a future commit that strips the Parameters
    entry while keeping the signature kwarg fails loudly.
    """
    assert "allow_rotated" in _signature_params(fn), (
        f"{fn.__name__} unexpectedly dropped allow_rotated from its "
        f"signature"
    )
    assert "allow_rotated" in _documented_params(fn), (
        f"{fn.__name__} accepts allow_rotated but does not document it "
        f"in its Parameters section (#2274)."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_allow_unparseable_crs_documented(fn):
    """``allow_unparseable_crs`` was the other shared #2274 gap.

    ``open_geotiff`` had the kwarg only in the Tier prose paragraph;
    the three backends did not mention it at all.
    """
    assert "allow_unparseable_crs" in _signature_params(fn), (
        f"{fn.__name__} unexpectedly dropped allow_unparseable_crs from "
        f"its signature"
    )
    assert "allow_unparseable_crs" in _documented_params(fn), (
        f"{fn.__name__} accepts allow_unparseable_crs but does not "
        f"document it in its Parameters section (#2274)."
    )
