"""Regression: resample() public signature carries type annotations on every
parameter and on the return type.

Pins the fix for issue #2544 (api-consistency sweep, Cat 3 MEDIUM). Without
this test, future contributors can drop annotations on ``agg``,
``scale_factor``, ``target_resolution``, ``method``, ``nodata``, or ``name``
without surfacing the drift between docstring and signature -- IDE hints,
mypy, and sphinx autodoc all silently lose information.
"""
from __future__ import annotations

import inspect

from xrspatial.resample import resample


def test_resample_every_param_is_annotated():
    sig = inspect.signature(resample)
    unannotated = [
        name for name, p in sig.parameters.items()
        if p.annotation is inspect.Parameter.empty
    ]
    assert not unannotated, (
        f"resample() must annotate every public parameter; "
        f"missing annotation(s): {unannotated}"
    )


def test_resample_return_is_annotated():
    sig = inspect.signature(resample)
    assert sig.return_annotation is not inspect.Signature.empty, (
        "resample() must declare a return annotation"
    )


def test_resample_2544_specific_params_annotated():
    """Pin every public param so a future regression names them directly
    in the failure message.
    """
    sig = inspect.signature(resample)
    expected = ('agg', 'scale_factor', 'target_resolution',
                'method', 'nodata', 'name')
    for name in expected:
        assert sig.parameters[name].annotation is not inspect.Parameter.empty, (
            f"resample({name}=...) lost its type annotation; "
            f"this regressed issue #2544"
        )
