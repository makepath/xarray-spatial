"""GeoTIFF and VRT rotated-transform reads raise the same typed error (#2267).

Before #2267, the geotag parser at
``_geotags._extract_transform_and_georef`` raised a bare
``NotImplementedError`` for rotated ``ModelTransformationTag`` inputs,
while the VRT path raised ``RotatedTransformError`` via
``_check_read_rotated_transform``. A caller that wrote
``except RotatedTransformError`` (or ``except
GeoTIFFAmbiguousMetadataError``) would only catch the VRT case.

These tests pin the parity contract:

* The GeoTIFF path now raises ``RotatedTransformError``.
* That subclass relationship still routes through ``ValueError`` and
  ``GeoTIFFAmbiguousMetadataError``, so existing broader ``except``
  clauses keep working.
* The error message preserves the previous content (mentions
  ``ModelTransformationTag`` and ``rotation``) and the
  ``allow_rotated=True`` opt-out is still wired through.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._errors import (
    GeoTIFFAmbiguousMetadataError,
    RotatedTransformError,
)
from xrspatial.geotiff._geotags import (
    TAG_MODEL_TRANSFORMATION,
    _extract_transform,
)
from xrspatial.geotiff._header import IFD, IFDEntry


_COS30 = 0.8660254037844387
_SIN30 = 0.5
_ROTATED_M = (
    10.0 * _COS30, -10.0 * _SIN30, 0.0, 100.0,
    10.0 * _SIN30, 10.0 * _COS30, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _make_rotated_ifd() -> IFD:
    """Build an IFD that carries only a rotated ModelTransformationTag."""
    ifd = IFD()
    ifd.entries[TAG_MODEL_TRANSFORMATION] = IFDEntry(
        tag=TAG_MODEL_TRANSFORMATION,
        type_id=12,  # DOUBLE
        count=16,
        value=_ROTATED_M,
    )
    return ifd


def _write_rotated_tiff(path, arr: np.ndarray) -> None:
    """Mirror the minimal rotated TIFF writer used in #2115's tests.

    Single-band, single-strip, uncompressed, with only a rotated
    ``ModelTransformationTag``. Self-contained so this test does not
    depend on the #2115 fixture module surviving in its current shape.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries = []
    entries.append((256, 3, 1, w))
    entries.append((257, 3, 1, h))
    entries.append((258, 3, 1, 16))
    entries.append((259, 3, 1, 1))
    entries.append((262, 3, 1, 1))
    entries.append((273, 4, 1, header_size))
    entries.append((277, 3, 1, 1))
    entries.append((278, 3, 1, h))
    entries.append((279, 4, 1, strip_size))
    entries.append((339, 3, 1, 1))
    entries.append((TAG_MODEL_TRANSFORMATION, 12, 16, transform_off))
    entries.sort(key=lambda e: e[0])

    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)

    with open(path, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *_ROTATED_M))
        f.write(ifd_bytes)


# ---------------------------------------------------------------------------
# Unit-level: the parser raises the typed error.
# ---------------------------------------------------------------------------


def test_extract_transform_rotated_raises_typed_error():
    """``_extract_transform`` raises ``RotatedTransformError`` on a
    rotated ``ModelTransformationTag``. This is the bug fix at the
    source: ``raise NotImplementedError`` -> ``raise
    RotatedTransformError`` at ``_geotags.py:679``.
    """
    ifd = _make_rotated_ifd()
    with pytest.raises(RotatedTransformError):
        _extract_transform(ifd)


def test_rotated_error_is_geotiff_ambiguous_subclass():
    """``RotatedTransformError`` keeps its membership in the
    ``GeoTIFFAmbiguousMetadataError`` family so callers using the
    family ``except`` still work. ``ValueError`` membership is the
    other contract (legacy code catching ``ValueError`` keeps working).
    """
    ifd = _make_rotated_ifd()
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        _extract_transform(ifd)
    # And ValueError still catches it because RotatedTransformError ->
    # GeoTIFFAmbiguousMetadataError -> ValueError.
    with pytest.raises(ValueError):
        _extract_transform(ifd)


def test_extract_transform_rotated_does_not_raise_notimplemented():
    """Regression guard: the parser must NOT raise the bare
    ``NotImplementedError`` it used to. ``RotatedTransformError`` is a
    ``ValueError``, not a ``NotImplementedError`` (which is
    ``RuntimeError``), so a caller using ``except NotImplementedError``
    no longer catches the rotated case. This is the breaking-but-
    intentional half of #2267.
    """
    ifd = _make_rotated_ifd()
    with pytest.raises(RotatedTransformError):
        try:
            _extract_transform(ifd)
        except NotImplementedError:  # pragma: no cover - regression guard
            pytest.fail(
                "rotated ModelTransformationTag must raise "
                "RotatedTransformError, not NotImplementedError"
            )


def test_extract_transform_message_preserved():
    """Message content should still name the tag and the rotation
    issue, so user-facing diagnostics survive the type change.
    """
    ifd = _make_rotated_ifd()
    with pytest.raises(RotatedTransformError) as exc:
        _extract_transform(ifd)
    msg = str(exc.value)
    assert 'ModelTransformationTag' in msg
    assert 'rotation' in msg.lower() or 'skew' in msg.lower()
    assert 'allow_rotated' in msg


# ---------------------------------------------------------------------------
# End-to-end: the public reader surface raises the typed error.
# ---------------------------------------------------------------------------


def test_open_geotiff_rotated_raises_typed_error(tmp_path):
    """``open_geotiff`` on a rotated GeoTIFF raises
    ``RotatedTransformError``. This is the user-facing parity case
    with the VRT path (covered in
    ``test_remaining_fail_closed_1987.test_read_rejects_rotated_vrt``).
    """
    src = tmp_path / "tmp_2267_open_geotiff_rotated.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    with pytest.raises(RotatedTransformError):
        open_geotiff(str(src))


def test_open_geotiff_rotated_allow_rotated_still_reads(tmp_path):
    """The opt-out is still wired through: ``allow_rotated=True`` reads
    the pixel grid without raising. This pins that switching the
    exception type did not break the existing escape hatch.
    """
    src = tmp_path / "tmp_2267_open_geotiff_rotated_optin.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    da = open_geotiff(str(src), allow_rotated=True)
    assert da.shape == arr.shape
    np.testing.assert_array_equal(da.values, arr)
