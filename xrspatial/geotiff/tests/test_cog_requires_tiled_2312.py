"""``cog=True`` requires ``tiled=True`` (#2312).

The COG specification mandates a tiled internal layout. Before this
issue's fix, ``to_geotiff(..., cog=True, tiled=False)`` returned
successfully and wrote a strip-layout TIFF: ``cog=True`` was silently
ignored for the layout decision while the overview-pyramid and IFD-order
parts of the COG path still ran, producing a malformed hybrid that
violated the stable COG contract promoted in #2300.

These tests pin three things:

* The public ``to_geotiff`` wrapper rejects ``cog=True, tiled=False``
  with a typed, actionable error that names both fixes the caller can
  apply (``tiled=True`` or ``cog=False``). This is the user-visible
  rejection.
* The defense-in-depth gate inside ``_writer._write`` also rejects the
  combination. Direct callers of the array-level entry point (the GPU
  CPU-fallback path, tests, internal tools) cannot bypass the public
  wrapper to produce the malformed file.
* The tiled COG path (``cog=True``, default ``tiled=True``) still works
  end-to-end. A regression in the new gate that broke valid COG writes
  would be a worse outcome than the original bug.

Message-substring assertions mirror the style of
``test_cog_invalid_input_errors_2286.py`` (PR #2301): every gate pins
both the exception type and the actionable tokens (``tiled=True``,
``cog=False``, ``COG``) so a future rewrite cannot silently turn the
error into a vague one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._writer import write as _array_write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_da(shape=(64, 64)):
    """A small float32 DataArray suitable for COG writes."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=('y', 'x')
    )


# ---------------------------------------------------------------------------
# Public boundary: ``to_geotiff(cog=True, tiled=False)`` is refused.
# ---------------------------------------------------------------------------

def test_public_writer_rejects_cog_true_tiled_false(tmp_path):
    """The public entry point raises ``ValueError`` with a message that
    names the COG-spec constraint and both caller-side fixes."""
    da = _float_da()
    p = tmp_path / 'cog_tiled_false_2312.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, tiled=False)

    msg = str(exc.value)
    # The message must name the violated constraint.
    assert 'COG' in msg, msg
    assert 'tiled' in msg.lower(), msg
    # Both caller-side fixes must appear so the error is actionable.
    assert 'tiled=True' in msg, msg
    assert 'cog=False' in msg, msg


def test_public_writer_rejects_cog_true_tiled_false_with_tile_size(tmp_path):
    """Pinning the rejection survives a ``tile_size`` kwarg too.

    Before #2312, ``to_geotiff(..., cog=True, tiled=False,
    tile_size=128)`` emitted the "tile_size is ignored when tiled=False"
    warning and then wrote strips. The new gate has to fire before that
    warning so the caller never sees the misleading "tile_size is
    ignored" message under ``cog=True``.
    """
    da = _float_da()
    p = tmp_path / 'cog_tiled_false_with_tile_size_2312.tif'

    # ``pytest.warns(None)`` was removed; use the stdlib catch_warnings
    # recorder to assert the dead "tile_size is ignored" warning never
    # fires on the ``cog=True`` arm.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, str(p), cog=True, tiled=False, tile_size=128)

    msg = str(exc.value)
    assert 'COG' in msg, msg
    assert 'tiled=True' in msg, msg

    tile_size_warnings = [
        w for w in record
        if 'tile_size' in str(w.message)
        and 'is ignored when tiled=False' in str(w.message)
    ]
    assert not tile_size_warnings, [str(w.message) for w in tile_size_warnings]


# ---------------------------------------------------------------------------
# Defense in depth: ``_writer._write(cog=True, tiled=False)`` also raises.
# ---------------------------------------------------------------------------

def test_lowlevel_write_rejects_cog_true_tiled_false(tmp_path):
    """The array-level entry point ``_writer._write`` (re-exported as
    ``write``) carries its own gate so a caller that bypasses the public
    wrapper still gets the typed rejection.

    Without this, a direct caller could quietly produce the malformed
    strip-plus-overviews file the public boundary refuses.
    """
    arr = np.zeros((64, 64), dtype=np.float32)
    p = tmp_path / 'cog_tiled_false_lowlevel_2312.tif'

    with pytest.raises(ValueError) as exc:
        _array_write(
            arr,
            str(p),
            compression='deflate',
            tiled=False,
            cog=True,
        )

    msg = str(exc.value)
    assert 'COG' in msg, msg
    assert 'tiled=True' in msg, msg
    assert 'cog=False' in msg, msg


# ---------------------------------------------------------------------------
# Smoke test: the valid tiled COG path still works.
# ---------------------------------------------------------------------------

def test_tiled_cog_smoke_still_works(tmp_path):
    """A regression in the new gate that broke valid COG writes would
    be a worse outcome than the original bug. Pin the happy path
    end-to-end so the gate has to stay narrowly targeted at the
    ``cog=True, tiled=False`` combination it is meant to catch.
    """
    da = _float_da(shape=(128, 128))
    p = tmp_path / 'cog_tiled_smoke_2312.tif'

    rv = to_geotiff(da, str(p), cog=True, tiled=True, tile_size=64)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_tiled_cog_smoke_default_tiled(tmp_path):
    """``tiled`` defaults to ``True`` on ``to_geotiff``, so ``cog=True``
    on its own should also produce a valid COG. Pinned so a future
    change that flipped the default would not silently start hitting
    the new rejection gate.
    """
    da = _float_da(shape=(128, 128))
    p = tmp_path / 'cog_tiled_default_smoke_2312.tif'

    rv = to_geotiff(da, str(p), cog=True)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Negative control: ``cog=False, tiled=False`` is still a valid strip TIFF.
# ---------------------------------------------------------------------------

def test_strip_layout_without_cog_still_works(tmp_path):
    """``tiled=False`` without ``cog=True`` is the supported strip-TIFF
    path; the new gate must not regress it. Pinned so a stricter
    interpretation of ``cog=True implies tiled=True`` could not creep
    into the general ``tiled=False`` path.
    """
    da = _float_da(shape=(64, 64))
    p = tmp_path / 'strip_no_cog_2312.tif'

    rv = to_geotiff(da, str(p), cog=False, tiled=False)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0
