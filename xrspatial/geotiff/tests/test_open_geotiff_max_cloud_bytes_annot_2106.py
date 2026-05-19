"""Regression test for #2106: every kwarg on the public read/write
entry points carries a type annotation.

The original gap: ``open_geotiff(max_cloud_bytes=...)`` had no annotation
on its kwarg, while every other kwarg on the function -- and every kwarg
on every other public reader and writer in ``xrspatial.geotiff`` -- did.
``inspect.signature``, IDE autocomplete, Sphinx, and ``mypy --strict`` all
saw a bare parameter for the only fsspec-related kwarg on the public
read entry point, despite the docstring declaring ``int or None``.

This test fixes the immediate gap (``max_cloud_bytes``) and pins every
other public reader/writer kwarg to a non-empty annotation so a future
addition cannot reopen the surface without surfacing in CI.
"""
from __future__ import annotations

import inspect

import pytest

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


PUBLIC_ENTRY_POINTS = (
    open_geotiff,
    read_geotiff_gpu,
    read_geotiff_dask,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def test_open_geotiff_max_cloud_bytes_has_type_annotation():
    """Pin the #2106 fix: the kwarg the bug named carries ``int | None``."""
    sig = inspect.signature(open_geotiff)
    param = sig.parameters["max_cloud_bytes"]
    assert param.annotation is not inspect.Parameter.empty, (
        "open_geotiff(max_cloud_bytes=...) is missing a type annotation; "
        "the docstring declares ``int or None`` so the surface should match."
    )
    # Use ``str(...)`` rather than identity so the assertion survives the
    # ``from __future__ import annotations`` lazy-eval form ``open_geotiff``
    # itself uses (its annotations come back as strings).
    annotation_repr = str(param.annotation)
    assert "int" in annotation_repr and "None" in annotation_repr, (
        f"open_geotiff(max_cloud_bytes=...) annotation should mention "
        f"int and None; got {annotation_repr!r}"
    )


@pytest.mark.parametrize("fn", PUBLIC_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_public_entry_point_kwargs_have_type_annotations(fn):
    """Every kwarg on the public read/write surface carries an annotation.

    Catches future regressions of the same class as #2106: a kwarg added
    to one entry point without an annotation while the rest of the
    signature has them.
    """
    sig = inspect.signature(fn)
    missing = [
        name
        for name, param in sig.parameters.items()
        if param.annotation is inspect.Parameter.empty
    ]
    assert missing == [], (
        f"{fn.__name__} has kwargs without type annotations: {missing}. "
        f"Add ``annotation`` to each so inspect.signature, IDE "
        f"autocomplete, Sphinx, and mypy --strict all see the declared "
        f"type. The docstring already declares the type for the kwargs "
        f"in question (#2106 raised this for max_cloud_bytes)."
    )
