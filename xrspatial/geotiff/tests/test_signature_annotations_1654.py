"""Regression test for #1654: public geotiff API parameter annotations.

The api-consistency sweep on 2026-05-12 flagged a MEDIUM type-annotation
drift across the public ``xrspatial.geotiff`` surface. The same parameter
was annotated on some sibling functions but missing on others:

* ``window``: annotated on ``read_geotiff_dask`` and ``read_geotiff_gpu``
  but missing on ``open_geotiff`` and ``read_vrt``.
* ``path``: annotated on ``write_vrt.vrt_path`` (str-only) but missing
  on ``to_geotiff`` and ``write_geotiff_gpu`` (str or binary file-like).
* ``on_gpu_failure`` (and the deprecated ``gpu`` alias on
  ``read_geotiff_gpu``): documented as ``{'auto', 'strict'}`` strings
  but no annotation. The sentinel default did not preclude annotating
  the user-visible value type.

This module pins each annotation so future signature changes do not
silently drop them.
"""
from __future__ import annotations

import inspect

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def _annotation(fn, param_name):
    """Return the stringified annotation for ``fn``'s ``param_name``.

    ``from __future__ import annotations`` keeps annotations as strings
    at runtime, so the comparison works against the source literal.
    """
    sig = inspect.signature(fn)
    p = sig.parameters[param_name]
    assert p.annotation is not inspect.Parameter.empty, (
        f"{fn.__name__}({param_name}=...) is missing a type annotation"
    )
    return str(p.annotation)


# --- window: 4-tuple (r0, c0, r1, c1) or None ---


def test_open_geotiff_window_annotated():
    assert _annotation(open_geotiff, 'window') == 'tuple | None'


def test_read_vrt_window_annotated():
    assert _annotation(read_vrt, 'window') == 'tuple | None'


def test_read_geotiff_dask_window_annotated():
    """Pre-existing annotation -- keep it pinned so it does not regress."""
    assert _annotation(read_geotiff_dask, 'window') == 'tuple | None'


def test_read_geotiff_gpu_window_annotated():
    """Pre-existing annotation -- keep it pinned so it does not regress."""
    assert _annotation(read_geotiff_gpu, 'window') == 'tuple | None'


# --- path: str or binary file-like (writer entry points) ---


def test_to_geotiff_path_annotated():
    """``to_geotiff(data, path, ...)`` ``path`` accepts str or BinaryIO."""
    ann = _annotation(to_geotiff, 'path')
    assert 'str' in ann
    assert 'BinaryIO' in ann


def test_write_geotiff_gpu_path_annotated():
    """``write_geotiff_gpu(data, path, ...)`` ``path`` mirrors ``to_geotiff``."""
    ann = _annotation(write_geotiff_gpu, 'path')
    assert 'str' in ann
    assert 'BinaryIO' in ann


def test_write_vrt_vrt_path_annotated():
    """``write_vrt(vrt_path, ...)`` stays str-only (VRT writes are
    path-only by design; no file-like buffer support)."""
    assert _annotation(write_vrt, 'vrt_path') == 'str'


# --- on_gpu_failure: 'auto' | 'strict' (GPU failure policy) ---


def test_open_geotiff_on_gpu_failure_annotated():
    assert _annotation(open_geotiff, 'on_gpu_failure') == 'str'


def test_read_geotiff_gpu_on_gpu_failure_annotated():
    assert _annotation(read_geotiff_gpu, 'on_gpu_failure') == 'str'


def test_read_geotiff_gpu_deprecated_gpu_alias_annotated():
    """The deprecated ``gpu=`` alias on ``read_geotiff_gpu`` carries the
    same ``str`` annotation as the new ``on_gpu_failure`` kwarg."""
    assert _annotation(read_geotiff_gpu, 'gpu') == 'str'


# --- Smoke: the new annotations do not break runtime call semantics ---


def test_open_geotiff_window_and_failure_kwargs_runtime():
    """The annotated kwargs still accept the documented values at runtime."""
    import os
    import tempfile

    import numpy as np
    import xarray as xr

    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        path = f.name
    try:
        to_geotiff(da, path)
        # window is a 4-tuple; on_gpu_failure must not be passed on
        # gpu=False, so just verify window kwarg roundtrip
        r = open_geotiff(path, window=(0, 0, 4, 4))
        assert r.shape == (4, 4)
    finally:
        os.unlink(path)
