"""COG writer rejects non-positive ``tile_size`` regardless of ``tiled`` (#2311).

Before this fix, ``to_geotiff(..., cog=True, tiled=False, tile_size=<=0)``
hung the writer. ``tile_size`` validation only ran when ``tiled=True``, but
the COG path in ``_writer.py`` still used ``tile_size`` to auto-generate
overviews regardless of ``tiled``. With ``tile_size=-1`` the auto-overview
loop in ``_writer.py:490`` had ``oh > tile_size and ow > tile_size``
permanently true once ``oh, ow`` halved to 0, while the inner
``if oh > 0 and ow > 0`` guard prevented the level list from growing --
the loop never exited.

The fix lives in two places:

1. ``to_geotiff`` in ``_writers/eager.py`` now runs ``_validate_tile_size_arg``
   when ``tiled=True`` OR ``cog=True``. The COG path consumes ``tile_size``
   for overview generation regardless of strip-vs-tiled layout, so the
   public boundary must validate it in both cases.
2. The auto-overview loop in ``_writer.py`` has a defensive pre-check that
   raises if ``tile_size`` is not a positive int, plus a tightened loop
   condition that requires ``oh, ow > 0`` to continue. Together these mean
   the loop cannot run forever even if a future internal caller bypasses
   the public validator.

Each row below uses a SIGALRM-based timeout so a regression that brings
the hang back fails the test instead of locking up the run. SIGALRM is a
POSIX-only mechanism (CPython on Linux/macOS); the tests fall back to
plain execution on Windows, where the original hang is still a concern
but the watchdog is unavailable.
"""
from __future__ import annotations

import contextlib
import os
import signal

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


@contextlib.contextmanager
def _alarm_timeout(seconds: int):
    """Raise TimeoutError after ``seconds`` to bound test failure modes.

    No-op on platforms that lack SIGALRM (Windows). The window is large
    enough that a healthy raise path finishes well before the alarm
    fires; if the fix regresses the writer hangs and the alarm fires.
    """
    if not hasattr(signal, 'SIGALRM') or os.name == 'nt':
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(
            f'test exceeded {seconds}s watchdog; the writer likely '
            f'regressed into the #2311 infinite-loop hang.'
        )

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _float_da(shape=(64, 64)):
    """A small float32 DataArray large enough to trigger COG overview build."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=('y', 'x')
    )


# ---------------------------------------------------------------------------
# Public boundary: ``to_geotiff(..., cog=True, tile_size<=0)`` must raise.
# Covers both tiled=True and tiled=False, plus 0 and a negative value, so
# the validator gate stays on regardless of layout flag.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tiled', [True, False])
@pytest.mark.parametrize('tile_size', [-1, 0])
def test_to_geotiff_cog_non_positive_tile_size_raises(tmp_path, tiled, tile_size):
    """``cog=True`` with ``tile_size<=0`` raises ValueError up front,
    regardless of ``tiled``. Before #2311 this hung the writer when
    ``tiled=False``."""
    da = _float_da()
    p = tmp_path / f'cog_tile_size_hang_2311_t{int(tiled)}_ts{tile_size}.tif'

    with _alarm_timeout(5), pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, tiled=tiled, tile_size=tile_size)

    msg = str(exc.value)
    assert 'tile_size' in msg, msg
    # The shared validator says "positive int" -- pin the substring so a
    # message rewrite still keeps the actionable wording.
    assert 'positive' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Sanity: ``cog=False`` with ``tiled=False`` still accepts an unused
# ``tile_size`` (the existing "ignored" warning shape) -- the new gate
# must not fire when neither path will consume the value.
# ---------------------------------------------------------------------------

def test_to_geotiff_non_cog_strip_does_not_validate_tile_size(tmp_path):
    """When neither tiled output nor COG overview generation will use
    ``tile_size``, the validator gate stays off. The pre-existing
    "tile_size ignored" warning still fires (it carries its own
    non-default-value check, not a positivity check), but no error
    is raised."""
    da = _float_da()
    p = tmp_path / 'cog_tile_size_hang_2311_no_cog_strip.tif'

    # A negative tile_size with cog=False AND tiled=False is accepted
    # (with the "ignored" warning) because nothing consumes the value.
    # Use ``filterwarnings`` to swallow the warning so the test only
    # asserts no raise / no hang.
    import warnings
    with _alarm_timeout(5), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        to_geotiff(da, str(p), cog=False, tiled=False, tile_size=-1)

    assert p.exists(), 'writer should have produced a strip-layout file'


# ---------------------------------------------------------------------------
# Defense in depth: drive the inner writer directly with a bad tile_size
# and assert the auto-overview loop raises instead of hanging. Guards
# against future internal callers that bypass ``to_geotiff``'s public
# validator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tile_size', [-1, 0])
def test_writer_auto_overview_loop_rejects_non_positive_tile_size(
        tmp_path, tile_size):
    """``_write(..., cog=True, overview_levels=None)`` raises ValueError
    when ``tile_size`` is not a positive int, instead of spinning in the
    halving loop. The public ``to_geotiff`` already validates earlier;
    this is the inner-writer safety net (#2311)."""
    from xrspatial.geotiff._writer import _write

    # Minimal float32 array large enough for the auto-overview branch to
    # be entered. The exact pixel values do not matter -- the validator
    # check runs before any encoding work.
    data = np.zeros((64, 64), dtype=np.float32)
    out = tmp_path / f'cog_tile_size_hang_2311_inner_ts{tile_size}.tif'

    with _alarm_timeout(5), pytest.raises(ValueError) as exc:
        _write(data, str(out),
               compression='none',
               tiled=True,
               tile_size=tile_size,
               cog=True,
               overview_levels=None)

    assert 'tile_size' in str(exc.value), str(exc.value)
