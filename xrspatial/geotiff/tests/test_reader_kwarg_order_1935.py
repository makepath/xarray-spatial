"""Regression test for #1935: public reader entry points share a canonical
keyword-only parameter order.

``open_geotiff`` is the canonical surface. The three backend readers
(``read_geotiff_gpu``, ``read_geotiff_dask``, ``read_vrt``) must list the
shared kwargs in the same relative order so ``inspect.signature``, IDE
autocomplete, and Sphinx-rendered docs do not drift.

Each per-backend signature carries its own subset of the canonical
parameter list (``read_vrt`` does not take ``overview_level``,
``read_geotiff_dask`` does not take ``gpu``/``on_gpu_failure``, etc.).
The test compares each reader's params with the slice of the canonical
order it actually accepts; backend-specific extras (``read_geotiff_gpu``'s
deprecated ``gpu`` alias) are checked at the tail.

Prior to #1935: ``read_geotiff_gpu`` had ``overview_level`` before
``window``, ``read_geotiff_dask`` placed ``chunks`` and ``name`` out of
the canonical position.
"""
from __future__ import annotations

import inspect

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
)


# Canonical order taken from ``open_geotiff``'s public signature.
_CANONICAL_ORDER = (
    "dtype",
    "window",
    "overview_level",
    "band",
    "name",
    "chunks",
    "gpu",
    "max_pixels",
    "max_cloud_bytes",
    "on_gpu_failure",
    "missing_sources",
    "allow_rotated",
    "allow_unparseable_crs",
    "mask_nodata",
)


def _kwonly_params(fn):
    """Return the keyword-only parameter names of *fn* in declaration order."""
    sig = inspect.signature(fn)
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _assert_canonical(fn, allowed_tail=()):
    """Assert *fn*'s kw-only params follow the canonical order.

    Parameters that appear in ``_CANONICAL_ORDER`` must show up in the
    same relative order. Extras (e.g. the deprecated ``gpu`` alias on
    ``read_geotiff_gpu``) are accepted at the tail when listed in
    ``allowed_tail`` and otherwise rejected so new kwargs cannot be
    quietly added in arbitrary positions.
    """
    params = _kwonly_params(fn)
    canonical = [p for p in params if p in _CANONICAL_ORDER]
    expected = [p for p in _CANONICAL_ORDER if p in canonical]
    assert canonical == expected, (
        f"{fn.__name__} kwarg order {canonical!r} does not match the "
        f"canonical subset {expected!r}"
    )
    tail = [p for p in params if p not in _CANONICAL_ORDER]
    unexpected = set(tail) - set(allowed_tail)
    assert not unexpected, (
        f"{fn.__name__} has unexpected kw-only params {sorted(unexpected)!r}; "
        f"add them to _CANONICAL_ORDER or to the test's allowed_tail."
    )


def test_open_geotiff_defines_canonical_order():
    """``open_geotiff``'s signature is the canonical reference."""
    params = _kwonly_params(open_geotiff)
    expected = list(_CANONICAL_ORDER)
    assert params == expected, (
        f"open_geotiff kw-only params {params!r} no longer match the "
        f"canonical order {expected!r}. Update both the function and the "
        f"_CANONICAL_ORDER constant together."
    )


def test_read_geotiff_gpu_matches_canonical_order():
    """``read_geotiff_gpu`` must list shared params in the canonical order."""
    # ``gpu`` here is the deprecated alias for ``on_gpu_failure`` (see
    # ``read_geotiff_gpu``'s docstring). It is not the boolean backend
    # selector that lives on ``open_geotiff`` / ``read_vrt``, so it sits
    # at the tail rather than in its canonical-order slot.
    params = _kwonly_params(read_geotiff_gpu)
    # ``gpu`` is the deprecated alias, intentionally last.
    assert params[-1] == "gpu", (
        f"read_geotiff_gpu must keep the deprecated 'gpu' alias as the last "
        f"kwarg; got {params!r}"
    )
    # Drop the alias and run the canonical-subset check on the rest.
    head = params[:-1]
    canonical_head = [p for p in _CANONICAL_ORDER if p in head]
    assert head == canonical_head, (
        f"read_geotiff_gpu kwarg order {head!r} does not match the canonical "
        f"subset {canonical_head!r}"
    )


def test_read_geotiff_dask_matches_canonical_order():
    """``read_geotiff_dask`` must list shared params in the canonical order."""
    _assert_canonical(read_geotiff_dask)


def test_read_vrt_matches_canonical_order():
    """``read_vrt`` must list shared params in the canonical order.

    ``band_nodata`` is the #1987 PR 5 opt-out for the mixed-band metadata
    check; it is VRT-specific (no analogue on the other readers) and so
    lives in the per-function tail rather than in the shared canonical
    order.
    """
    _assert_canonical(read_vrt, allowed_tail=('band_nodata',))


def test_no_pairwise_order_inversions():
    """For any pair of params shared by two readers, the order is consistent.

    ``read_geotiff_gpu``'s ``gpu`` kwarg is a deprecated alias for
    ``on_gpu_failure`` rather than the boolean backend selector that
    ``open_geotiff`` / ``read_vrt`` expose, so it is excluded from the
    cross-reader pair check.
    """
    readers = (open_geotiff, read_geotiff_gpu, read_geotiff_dask, read_vrt)
    orders = {}
    for fn in readers:
        params = _kwonly_params(fn)
        if fn is read_geotiff_gpu:
            # Drop the deprecated alias before cross-comparing with the other
            # readers' boolean ``gpu`` kwarg (different meaning, same name).
            params = [p for p in params if p != "gpu"]
        orders[fn.__name__] = params
    canonical_pairs = []
    for i, a in enumerate(_CANONICAL_ORDER):
        for b in _CANONICAL_ORDER[i + 1:]:
            canonical_pairs.append((a, b))
    for name, params in orders.items():
        for a, b in canonical_pairs:
            if a in params and b in params:
                assert params.index(a) < params.index(b), (
                    f"{name}: {a!r} must appear before {b!r}; got "
                    f"{params!r}"
                )
