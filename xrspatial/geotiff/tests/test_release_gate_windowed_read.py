"""Release gate: windowed reads (epic #2340).

``open_geotiff(path, window=...)`` is part of the stable surface. The
release contract:

* A ``(row_start, col_start, row_stop, col_stop)`` window returns the
  exact subset of the source pixels.
* The result keeps ``attrs['crs']`` and produces a transform whose
  origin shifts to the window's top-left pixel corner.
* Reading the full extent via ``window=(0, 0, H, W)`` matches an
  unwindowed read.

Out of bounds and degenerate windows are covered by
``test_window_out_of_bounds_1634.py``; the release-gate test only
locks the supported, in-bounds use case so a release engineer knows
the user-facing API behaves end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


_H = 10
_W = 10
# A distinctive per-pixel value (row * 100 + col) means any row / col
# stride confusion in the windowed path fails the equality check.
_PIXELS = (
    np.arange(_H, dtype=np.int32).reshape(-1, 1) * 100
    + np.arange(_W, dtype=np.int32).reshape(1, -1)
).astype(np.int32)
_ORIGIN_X = 500000.0
_ORIGIN_Y = 4000000.0
_PIXEL_W = 30.0
_PIXEL_H = -30.0


def _write_known_good(path: str) -> None:
    gt = GeoTransform(
        origin_x=_ORIGIN_X,
        origin_y=_ORIGIN_Y,
        pixel_width=_PIXEL_W,
        pixel_height=_PIXEL_H,
    )
    write(
        _PIXELS,
        path,
        geo_transform=gt,
        crs_epsg=32610,
        compression="none",
        tiled=False,
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_returns_subset(tmp_path) -> None:
    """A windowed read returns exactly the requested subset."""
    path = str(tmp_path / "release_gate_windowed_read_subset_2340.tif")
    _write_known_good(path)

    # Take an interior 4x5 window so the test fails if the window
    # logic confuses row- and column-order.
    row_start, col_start = 2, 3
    row_stop, col_stop = 6, 8
    out = open_geotiff(path, window=(row_start, col_start, row_stop, col_stop))

    expected = _PIXELS[row_start:row_stop, col_start:col_stop]
    assert out.shape == expected.shape, (
        f"release gate: windowed read shape {out.shape} does not match "
        f"the requested window shape {expected.shape}"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        expected,
        err_msg=(
            "release gate: windowed read returned different pixels than "
            "the same rows / cols of the source array; this would silently "
            "break every downstream caller that relies on window= for "
            "subsetting"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_preserves_crs(tmp_path) -> None:
    """A windowed read carries ``attrs['crs']`` over from the source."""
    path = str(tmp_path / "release_gate_windowed_read_crs_2340.tif")
    _write_known_good(path)

    out = open_geotiff(path, window=(1, 1, 5, 5))
    crs = out.attrs.get("crs")
    assert crs is not None and int(crs) == 32610, (
        f"release gate: windowed read dropped or drifted "
        f"``attrs['crs']``: got {crs!r}"
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_shifts_transform_origin(tmp_path) -> None:
    """The transform origin shifts to the window's top-left pixel.

    Concretely: for a window starting at ``(row, col) = (2, 3)`` on a
    grid with pixel width ``+30`` and pixel height ``-30``, the new
    origin is ``(origin_x + 3 * 30, origin_y + 2 * -30)``.
    """
    path = str(tmp_path / "release_gate_windowed_read_transform_2340.tif")
    _write_known_good(path)

    row_start, col_start = 2, 3
    out = open_geotiff(path, window=(row_start, col_start, 6, 8))
    transform = out.attrs.get("transform")
    assert transform is not None and len(transform) == 6, (
        f"release gate: windowed read dropped or reshaped transform: "
        f"{transform!r}"
    )
    # Pixel size must not change.
    assert transform[0] == pytest.approx(_PIXEL_W, abs=1e-9), (
        f"release gate: windowed read changed pixel_width: {transform!r}"
    )
    assert transform[4] == pytest.approx(_PIXEL_H, abs=1e-9), (
        f"release gate: windowed read changed pixel_height: {transform!r}"
    )
    expected_origin_x = _ORIGIN_X + col_start * _PIXEL_W
    expected_origin_y = _ORIGIN_Y + row_start * _PIXEL_H
    assert transform[2] == pytest.approx(expected_origin_x, abs=1e-9), (
        f"release gate: windowed read origin_x did not shift to the "
        f"window's left edge: got {transform[2]!r} expected "
        f"{expected_origin_x!r}"
    )
    assert transform[5] == pytest.approx(expected_origin_y, abs=1e-9), (
        f"release gate: windowed read origin_y did not shift to the "
        f"window's top edge: got {transform[5]!r} expected "
        f"{expected_origin_y!r}"
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_full_extent_matches_unwindowed(
    tmp_path,
) -> None:
    """``window=(0, 0, H, W)`` returns the same pixels as no window."""
    path = str(tmp_path / "release_gate_windowed_read_full_2340.tif")
    _write_known_good(path)

    full = open_geotiff(path)
    windowed = open_geotiff(path, window=(0, 0, _H, _W))
    assert windowed.shape == full.shape, (
        f"release gate: full-extent window shape drift: "
        f"{windowed.shape} vs {full.shape}"
    )
    np.testing.assert_array_equal(
        np.asarray(windowed.values),
        np.asarray(full.values),
        err_msg=(
            "release gate: full-extent window returned different pixels "
            "than the unwindowed read"
        ),
    )
