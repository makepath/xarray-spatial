"""Regression: rasterize() public signature carries type annotations on every
parameter and on the return type.

Pins the fix for issue #2250 (api-consistency sweep, Cat 3 MEDIUM). Without
this test, future contributors can drop annotations on ``geometries``,
``columns``, or ``merge`` without surfacing the drift between docstring and
signature -- IDE hints, mypy, and sphinx autodoc all silently lose
information.
"""
from __future__ import annotations

import inspect

from xrspatial.rasterize import rasterize


def test_rasterize_every_param_is_annotated():
    sig = inspect.signature(rasterize)
    unannotated = [
        name for name, p in sig.parameters.items()
        if p.annotation is inspect.Parameter.empty
    ]
    assert not unannotated, (
        f"rasterize() must annotate every public parameter; "
        f"missing annotation(s): {unannotated}"
    )


def test_rasterize_return_is_annotated():
    sig = inspect.signature(rasterize)
    assert sig.return_annotation is not inspect.Signature.empty, (
        "rasterize() must declare a return annotation"
    )


def test_rasterize_2250_specific_params_annotated():
    """Pin the three params the sweep called out so a future regression names
    them directly in the failure message.
    """
    sig = inspect.signature(rasterize)
    for name in ('geometries', 'columns', 'merge'):
        assert sig.parameters[name].annotation is not inspect.Parameter.empty, (
            f"rasterize({name}=...) lost its type annotation; "
            f"this regressed issue #2250"
        )
