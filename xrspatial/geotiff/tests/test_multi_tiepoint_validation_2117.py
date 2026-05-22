"""Multi-tiepoint consistency check in ``_extract_transform`` (issue #2117).

A ``ModelTiepointTag`` may carry one or many ``(I, J, K, X, Y, Z)``
tuples. Before the fix, the reader sliced only ``tiepoint[0:6]`` and
silently dropped any additional records. That works when the extra
tuples encode the same affine at every corner (common case) but
silently produces wrong coordinates when the tuples encode a non-affine
GCP warp.

These tests pin the new behaviour:

* Single-tiepoint files continue to read the same way (no regression).
* Multi-tiepoint files whose tuples agree within tolerance still read.
* Multi-tiepoint files with inconsistent tuples raise
  ``NotImplementedError`` with a message that names the GCP case so the
  user has a path forward (``gdalwarp`` to rectify first).
* The tolerance scales with pixel size, so files in different units
  (degrees vs metres) are treated consistently.
"""
from __future__ import annotations

import pytest

from xrspatial.geotiff._geotags import (TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
                                        _extract_transform, _validate_tiepoint_consistency)
from xrspatial.geotiff._header import IFD, IFDEntry


def _make_ifd(tiepoint: tuple, scale: tuple | None = (10.0, 10.0, 0.0)) -> IFD:
    ifd = IFD()
    ifd.entries[TAG_MODEL_TIEPOINT] = IFDEntry(
        tag=TAG_MODEL_TIEPOINT, type_id=12,
        count=len(tiepoint), value=tiepoint,
    )
    if scale is not None:
        ifd.entries[TAG_MODEL_PIXEL_SCALE] = IFDEntry(
            tag=TAG_MODEL_PIXEL_SCALE, type_id=12,
            count=len(scale), value=scale,
        )
    return ifd


# A simple axis-aligned affine: origin (100, 200), pixel size 10 in both axes.
# Pixel (i, j) maps to world (100 + 10*i, 200 - 10*j).
_SX = 10.0
_SY = 10.0
_ORIGIN_X = 100.0
_ORIGIN_Y = 200.0


def _world_at(i: float, j: float) -> tuple[float, float]:
    return (_ORIGIN_X + i * _SX, _ORIGIN_Y - j * _SY)


def test_single_tiepoint_unchanged():
    ifd = _make_ifd((0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0))
    gt, has_georef = _extract_transform(ifd)
    assert has_georef is True
    assert gt.origin_x == _ORIGIN_X
    assert gt.origin_y == _ORIGIN_Y
    assert gt.pixel_width == _SX
    assert gt.pixel_height == -_SY


def test_multiple_consistent_tiepoints_pass():
    # Four corners of a 100x100 raster, all consistent with the same affine.
    corners = []
    for i, j in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        wx, wy = _world_at(i, j)
        corners.extend([float(i), float(j), 0.0, wx, wy, 0.0])
    ifd = _make_ifd(tuple(corners))
    gt, has_georef = _extract_transform(ifd)
    assert has_georef is True
    assert gt.origin_x == pytest.approx(_ORIGIN_X)
    assert gt.origin_y == pytest.approx(_ORIGIN_Y)
    assert gt.pixel_width == pytest.approx(_SX)
    assert gt.pixel_height == pytest.approx(-_SY)


def test_inconsistent_tiepoints_raise():
    # Second tuple disagrees by a full pixel: that is a GCP warp.
    tiepoint = (
        0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0,
        100.0, 0.0, 0.0, _ORIGIN_X + 100 * _SX + 5.0, _ORIGIN_Y, 0.0,
    )
    ifd = _make_ifd(tiepoint)
    with pytest.raises(NotImplementedError, match="ground-control-point"):
        _extract_transform(ifd)


def test_tolerance_scales_with_pixel_size():
    # A 1e-7 residual on a pixel_size=10 file is below tolerance.
    tiny_resid = 1e-7
    tiepoint = (
        0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0,
        100.0, 0.0, 0.0, _ORIGIN_X + 100 * _SX + tiny_resid, _ORIGIN_Y, 0.0,
    )
    ifd = _make_ifd(tiepoint)
    # Should not raise.
    _extract_transform(ifd)


def test_validate_helper_no_op_for_single_tuple():
    # 6 elements -> n == 1; nothing to validate.
    _validate_tiepoint_consistency(
        (0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0),
        _ORIGIN_X, _ORIGIN_Y, _SX, _SY,
    )


def test_validate_helper_rejects_disagreement():
    tiepoint = (
        0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0,
        50.0, 0.0, 0.0, _ORIGIN_X + 50 * _SX + 100.0, _ORIGIN_Y, 0.0,
    )
    with pytest.raises(NotImplementedError, match="tuple 1"):
        _validate_tiepoint_consistency(
            tiepoint, _ORIGIN_X, _ORIGIN_Y, _SX, _SY,
        )


def test_validate_helper_y_axis_sign():
    # Verify the y-axis sign convention: predicted_y = origin_y - j * sy.
    # A consistent tuple at (i=0, j=100) is (origin_x, origin_y - 100 * sy).
    tp_world_y = _ORIGIN_Y - 100.0 * _SY
    tiepoint = (
        0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0,
        0.0, 100.0, 0.0, _ORIGIN_X, tp_world_y, 0.0,
    )
    _validate_tiepoint_consistency(
        tiepoint, _ORIGIN_X, _ORIGIN_Y, _SX, _SY,
    )


def test_tiepoint_without_scale_also_validates():
    # When ModelPixelScale is absent, the reader falls back to unit pixel
    # size; the consistency check must still fire, and the error message
    # must blame the missing ModelPixelScale tag (not the GCP-warp case),
    # since a real multi-tiepoint file without ModelPixelScale is almost
    # certainly malformed rather than a deliberate GCP set.
    tiepoint = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        10.0, 0.0, 0.0, 50.0, 0.0, 0.0,  # predicts x=10.0, declares 50.0
    )
    ifd = _make_ifd(tiepoint, scale=None)
    with pytest.raises(NotImplementedError, match="no ModelPixelScale"):
        _extract_transform(ifd)


def test_validate_helper_honours_custom_rel_tol():
    # A residual that passes the default 1e-6 * pixel_size tolerance
    # (= 1e-5 here) can still be caught by a tighter caller-supplied
    # rel_tol. Surveying / high-precision geodetic callers that want to
    # flag near-affine GCP files can pass a smaller rel_tol.
    residual = 5e-6  # below default tol (1e-5) but above tight tol (1e-7)
    tiepoint = (
        0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0,
        100.0, 0.0, 0.0, _ORIGIN_X + 100 * _SX + residual, _ORIGIN_Y, 0.0,
    )
    # Default tolerance accepts it.
    _validate_tiepoint_consistency(
        tiepoint, _ORIGIN_X, _ORIGIN_Y, _SX, _SY,
    )
    # Tighter tolerance rejects it.
    with pytest.raises(NotImplementedError, match="tuple 1"):
        _validate_tiepoint_consistency(
            tiepoint, _ORIGIN_X, _ORIGIN_Y, _SX, _SY, rel_tol=1e-8,
        )


def test_short_tiepoint_is_treated_as_single_tuple():
    # A truncated tiepoint with fewer than 12 elements has n == 1 (truncated
    # second tuple is dropped by integer division). The reader should not
    # crash; it falls back to the existing single-tuple semantics.
    tiepoint = (0.0, 0.0, 0.0, _ORIGIN_X, _ORIGIN_Y, 0.0, 1.0)
    ifd = _make_ifd(tiepoint)
    gt, has_georef = _extract_transform(ifd)
    assert has_georef is True
    assert gt.origin_x == _ORIGIN_X
