"""Regression test for #1946: write_vrt accepts ``path`` for parity
with ``to_geotiff`` / ``write_geotiff_gpu``.

The api-consistency sweep on 2026-05-15 flagged that ``write_vrt`` was
the only writer in ``xrspatial.geotiff`` whose destination kwarg was
named ``vrt_path`` while the sibling writers use ``path``. The fix adds
``path`` as the canonical kwarg and keeps ``vrt_path`` as a deprecated
alias.

This module pins:

* Positional ``write_vrt(path, sources)`` works (back-compat with the
  previous ``write_vrt(vrt_path, sources)`` positional form).
* Keyword ``write_vrt(path=..., source_files=...)`` works (the new
  canonical form).
* Keyword ``write_vrt(vrt_path=...)`` still works and emits
  ``DeprecationWarning``.
* Passing both ``path`` and ``vrt_path`` raises ``TypeError``.
* The signature exposes ``path`` as the first positional, matching
  ``to_geotiff`` / ``write_geotiff_gpu``.
* The deprecation shim does NOT warn when ``path`` is used.
* Omitting both names raises ``TypeError`` (preserves the pre-#1946
  required-argument semantics).
"""
from __future__ import annotations

import inspect
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def _build_source_tif(tmp_path, name='src.tif'):
    """Create a small GeoTIFF used as the VRT's source file."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    p = str(tmp_path / name)
    to_geotiff(da, p)
    return p


def test_write_vrt_signature_first_arg_is_path():
    """Signature parity with to_geotiff / write_geotiff_gpu.

    The api-consistency sweep cares specifically about
    ``inspect.signature``: IDE autocomplete, mypy, and Sphinx-rendered
    docs all read the same source. Pinning the first param name here
    catches any future re-rename that re-introduces the drift.
    """
    sig = inspect.signature(write_vrt)
    params = list(sig.parameters)
    # ``path`` is the new canonical name, ``source_files`` follows.
    # ``vrt_path`` is kept as a keyword-only deprecated alias.
    assert params[0] == 'path'
    assert params[1] == 'source_files'
    assert 'vrt_path' in params
    # ``vrt_path`` is keyword-only (the alias should never be used
    # positionally going forward).
    assert sig.parameters['vrt_path'].kind == inspect.Parameter.KEYWORD_ONLY


def test_write_vrt_positional_path_works(tmp_path):
    """Positional ``write_vrt(path, sources)`` is unchanged.

    Existing callers ``write_vrt(some_path, sources)`` keep working
    after the rename because the new ``path`` parameter sits where
    ``vrt_path`` used to be. No deprecation warning should fire.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = write_vrt(out, [src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_path_kwarg_works(tmp_path):
    """Keyword ``write_vrt(path=..., source_files=...)`` works.

    A caller who passes everything by keyword (no positional args)
    cannot reach the function before #1946 because ``path`` did not
    exist; this is the path-symmetric counterpart to the existing
    ``write_vrt(vrt_path=...)`` test below.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = write_vrt(path=out, source_files=[src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_vrt_path_kwarg_emits_deprecation_warning(tmp_path):
    """``vrt_path=...`` works but emits ``DeprecationWarning``.

    Mirrors the existing ``crs_wkt`` deprecation in the same writer
    (#1715): old name still works, but caller sees a clear migration
    hint via the warning.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with pytest.warns(DeprecationWarning, match='vrt_path'):
        result = write_vrt(vrt_path=out, source_files=[src])
    assert result == out
    assert os.path.exists(out)


def test_write_vrt_path_and_vrt_path_together_raises(tmp_path):
    """Both names supplied is ambiguous; refuse to pick one.

    Mirrors the ``crs`` / ``crs_wkt`` rule documented in the existing
    write_vrt source: passing both is rejected with TypeError
    regardless of whether the two values happen to match.
    """
    src = _build_source_tif(tmp_path)
    out = str(tmp_path / 'out.vrt')
    with pytest.raises(TypeError, match="path.*vrt_path"):
        write_vrt(path=out, vrt_path=out, source_files=[src])


def test_write_vrt_no_path_raises(tmp_path):
    """Neither ``path`` nor ``vrt_path`` -> TypeError.

    Before the shim, omitting the first positional argument raised
    ``TypeError: missing 1 required positional argument`` from CPython.
    The shim adds a default of ``None`` so the kwarg-only positional
    no longer triggers that automatic check; the explicit raise inside
    the shim preserves the pre-#1946 error semantics.
    """
    src = _build_source_tif(tmp_path)
    with pytest.raises(TypeError, match='path'):
        write_vrt(source_files=[src])


def test_write_vrt_first_arg_name_matches_writer_trio():
    """Cross-sibling consistency: all three writers use the same
    destination kwarg name.

    The deep-sweep-api-consistency sweep keeps adding to the writer
    trio's parity contract. Pin the rule here so future re-renames
    that split the trio again will trip a test.
    """
    eager_first = list(
        inspect.signature(to_geotiff).parameters
    )[1]  # data, path -> index 1
    gpu_first = list(
        inspect.signature(write_geotiff_gpu).parameters
    )[1]
    vrt_first = list(
        inspect.signature(write_vrt).parameters
    )[0]  # path, source_files -> index 0
    assert eager_first == 'path'
    assert gpu_first == 'path'
    assert vrt_first == 'path'


def test_write_vrt_path_round_trip_matches_old(tmp_path):
    """The written VRT decodes the same regardless of which kwarg name
    the caller used.

    Smoke test that the shim does not silently drop or re-route any of
    the other kwargs while resolving ``path`` vs ``vrt_path``.
    """
    src = _build_source_tif(tmp_path)
    out_new = str(tmp_path / 'out_new.vrt')
    out_old = str(tmp_path / 'out_old.vrt')

    write_vrt(out_new, [src])
    with warnings.catch_warnings():
        # ignore the deprecation; we still need the legacy path to
        # produce a byte-identical mosaic.
        warnings.simplefilter('ignore', DeprecationWarning)
        write_vrt(vrt_path=out_old, source_files=[src])

    a_new = read_vrt(out_new)
    a_old = read_vrt(out_old)
    np.testing.assert_array_equal(np.asarray(a_new), np.asarray(a_old))
