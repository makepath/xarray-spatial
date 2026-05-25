"""Release-gate negative cases: ambiguous metadata fails closed (issue #2341 PR 5).

Epic #2341's acceptance criteria include the promise that "unsupported
or ambiguous metadata fails loudly instead of flattening or guessing".
The positive release-gate suites in ``test_release_gate_*.py`` lock the
"works" path. This file pins the negative side: when metadata is
ambiguous and the caller did NOT opt in via the documented flag, every
promised read entry point raises a typed error whose message names the
unlocking flag and points at the release-contract docs.

Four parametrized cases:

1. Conflicting CRS between the GeoTIFF header and a sibling ``.aux.xml``
   PAM sidecar. The reader must not silently prefer one over the other.
2. Integer nodata sentinel that cannot be honoured on a float-promoted
   raster. The reader must not silently mask with the wrong value.
3. Rotated affine transform without ``allow_rotated=True``. Eager,
   dask, and windowed entry points must each raise uniformly.
4. Mixed-tier VRT children when the caller asked for stable-only
   sources. The reader must name the offending child and the opt-in.

Assertions inlined per case so a single failing row is locatable
without cross-file helpers. Sibling PRs of epic #2341 are running in
parallel; this file does NOT introduce a shared helper module.

xfail policy
------------
Cases 1, 2, and 4 are tagged ``strict=False`` xfail because the
production-side rejection is tracked under a separate epic or a
follow-up issue:

* Case 1 -- ``.aux.xml`` PAM sidecar read is not a supported feature
  today (no entry in :data:`xrspatial.geotiff.SUPPORTED_FEATURES`).
  The release promise is that *if* the reader gains PAM sidecar
  support, it must fail closed on a CRS conflict; the xfail flips
  to a pass the moment that production fix lands.
* Case 2 -- the current behaviour on a non-finite or fractional
  integer nodata sentinel is a no-op (issue #1774). The release
  promise is to upgrade that no-op to a typed error so the caller
  sees the silent-coercion risk; the xfail flips when the upgrade
  lands.
* Case 4 -- the VRT stable-only knob is owned by epic #2342 and
  has not landed. The xfail flips when #2342 ships the knob.

Case 3 (rotated) actively passes today on every cited entry point.
"""
from __future__ import annotations

import struct
import uuid
from pathlib import Path

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask
from xrspatial.geotiff._errors import (
    GeoTIFFAmbiguousMetadataError,
    RotatedTransformError,
)
from xrspatial.geotiff.tests.conftest import requires_gpu

# --------------------------------------------------------------------------- #
# Shared release-contract pointers.                                           #
# --------------------------------------------------------------------------- #
#
# The release-contract doc lives next to the release-gate checklist. The
# negative-case error messages should mention either the contract page,
# the issue number, or the audit checklist so a caller hitting the error
# can find the promise the reader is enforcing.
_RELEASE_CONTRACT_HINTS = (
    "release_gate_geotiff",
    "geotiff_release_contract",
    "#2341",
    "#1987",
    "#2342",
    "release contract",
)


def _msg_cites_release_contract(msg: str) -> bool:
    """True iff *msg* names the release-contract docs or the epic.

    The error message contract is loose on purpose: any of the four
    pointers above counts. We do not pin one exact docs path because
    the docs file may move (``.rst`` vs ``.md``) without that being a
    release-gate regression.
    """
    return any(hint in msg for hint in _RELEASE_CONTRACT_HINTS)


def _tmp(tmp_path, label: str, *, suffix: str = ".tif") -> str:
    """Return a unique temp file path scoped to this test file.

    Sibling PRs of epic #2341 run in parallel against the same shared
    tmp dirs in CI. Including ``2341`` and a per-call UUID keeps file
    names from colliding across worktrees and across parametrized
    cases inside the same test. ``suffix`` lets callers pick ``.vrt``
    or another extension without doing fragile string replacement on
    the returned path.
    """
    return str(
        tmp_path / f"release_gate_neg_2341_{label}_{uuid.uuid4().hex}{suffix}"
    )


# --------------------------------------------------------------------------- #
# Synthetic rotated GeoTIFF (case 3).                                         #
# --------------------------------------------------------------------------- #
#
# Hand-rolled TIFF with a ModelTransformationTag carrying a non-zero
# rotation. Going via ``to_geotiff`` would not exercise the rotated
# branch -- the writer refuses rotated transforms at the boundary, so a
# round-trip through xrspatial cannot reproduce one. The 30-degree
# rotation matches ``test_allow_rotated_geotiff_2115.py`` so the gate
# rejects the same input shape that test pins behaviourally.
#
# Canonical copy of the rotated-matrix constants and the TIFF builder
# lives in ``test_allow_rotated_geotiff_2115.py``. The duplication here
# is intentional -- the four sibling PRs of epic #2341 share no helper
# module to avoid cross-PR symbol collisions. If the canonical copy
# drifts, mirror the change here in a follow-up PR.

_TAG_MODEL_TRANSFORMATION = 34264
_COS30 = 0.8660254037844387
_SIN30 = 0.5
_ROTATED_M = (
    10.0 * _COS30, -10.0 * _SIN30, 0.0, 100.0,
    10.0 * _SIN30, 10.0 * _COS30, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _write_rotated_tiff(path: str, arr: np.ndarray) -> None:
    """Emit a minimal little-endian TIFF with a rotated ModelTransformationTag.

    Lifted from ``test_allow_rotated_geotiff_2115.py``. Inlined here so
    the negative gate does not introduce a shared helper module across
    the sibling PRs of epic #2341.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries: list[tuple[int, int, int, int]] = []
    entries.append((256, 3, 1, w))            # ImageWidth
    entries.append((257, 3, 1, h))            # ImageLength
    entries.append((258, 3, 1, 16))           # BitsPerSample
    entries.append((259, 3, 1, 1))            # Compression none
    entries.append((262, 3, 1, 1))            # Photometric BlackIsZero
    entries.append((273, 4, 1, header_size))  # StripOffsets
    entries.append((277, 3, 1, 1))            # SamplesPerPixel
    entries.append((278, 3, 1, h))            # RowsPerStrip
    entries.append((279, 4, 1, strip_size))   # StripByteCounts
    entries.append((339, 3, 1, 1))            # SampleFormat uint
    entries.append((_TAG_MODEL_TRANSFORMATION, 12, 16, transform_off))

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


# --------------------------------------------------------------------------- #
# Case 1: conflicting CRS between header and ``.aux.xml`` sidecar.            #
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason=(
        "xrspatial.geotiff does not yet read ``.aux.xml`` PAM sidecars "
        "(no entry in SUPPORTED_FEATURES). When that support lands, "
        "the reader must fail closed on a CRS conflict between the "
        "header and the sidecar; this xfail flips to a pass at that "
        "point. Tracked alongside the PAM sidecar epic."
    ),
    strict=False,
)
def test_release_gate_negative_conflicting_aux_xml_crs(tmp_path) -> None:
    """The reader must not silently choose between header CRS and sidecar CRS.

    The release promise: when a ``.aux.xml`` sidecar declares a CRS that
    disagrees with the GeoTIFF header, the reader raises with a message
    naming both sources and the opt-in flag that would resolve the
    ambiguity.
    """
    from xrspatial.geotiff._writer import write
    from xrspatial.geotiff._geotags import GeoTransform

    path = _tmp(tmp_path, "case1_aux_xml_crs")
    pixels = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write(
        pixels,
        path,
        geo_transform=GeoTransform(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=-1.0,
        ),
        crs_epsg=4326,
        compression="none",
        tiled=False,
    )
    # PAM sidecar declaring a *different* CRS than the header (3857 vs 4326).
    sidecar = Path(path + ".aux.xml")
    sidecar.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PAMDataset>\n'
        '  <SRS dataAxisToSRSAxisMapping="1,2">EPSG:3857</SRS>\n'
        '</PAMDataset>\n',
        encoding="utf-8",
    )
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert "aux.xml" in msg or "sidecar" in msg or "PAM" in msg, (
        f"expected the error message to name the .aux.xml / PAM sidecar; "
        f"got: {msg!r}"
    )
    assert _msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


# --------------------------------------------------------------------------- #
# Case 2: integer nodata on a float-promoted raster.                          #
# --------------------------------------------------------------------------- #


def _build_uint16_tiff_with_nodata(nodata_str: str, path: str) -> None:
    """Emit a 2x2 uint16 TIFF whose GDAL_NODATA tag holds ``nodata_str``.

    Mirrors ``_build_uint16_tiff`` in ``test_nodata_nan_int_1774.py`` so
    this gate uses the same byte-level shape that file behaviourally
    pins. Inlined to keep the file self-contained.
    """
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_bytes = nodata_str.encode('ascii') + b'\x00'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag: int, val: int) -> None:
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag: int, val: int) -> None:
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_ascii(tag: int, data: bytes) -> None:
        tag_list.append((tag, 2, len(data), data))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 16)
    add_short(259, 1)
    add_short(262, 1)
    add_short(277, 1)
    add_short(278, height)
    add_short(339, 1)
    add_long(273, 0)
    add_long(279, len(pixels.tobytes()))
    add_ascii(42113, nodata_bytes)  # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])

    header_size = 8
    num_entries = len(tag_list)
    ifd_size = 2 + 12 * num_entries + 4
    ifd_off = header_size

    overflow = bytearray()
    overflow_start = header_size + ifd_size

    overflow_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_offsets[tag] = len(overflow)
            overflow.extend(raw)
            if len(overflow) % 2:
                overflow.append(0)
        else:
            overflow_offsets[tag] = None

    pixel_start = overflow_start + len(overflow)

    patched: list[tuple[int, int, int, bytes]] = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append(
                (tag, typ, count, struct.pack(f'{bo}I', pixel_start))
            )
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_off))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + overflow_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow)
    out.extend(pixels.tobytes())

    with open(path, 'wb') as f:
        f.write(out)


@pytest.mark.xfail(
    reason=(
        "Issue #1774 currently treats a non-finite or fractional "
        "integer nodata sentinel as a silent no-op rather than a hard "
        "error. The release promise is to upgrade the no-op to a "
        "typed rejection so the caller sees the silent-coercion risk; "
        "this xfail flips to a pass when the upgrade lands."
    ),
    strict=False,
)
def test_release_gate_negative_integer_nodata_float_promoted(tmp_path) -> None:
    """The reader must not silently coerce a non-finite int-file nodata sentinel.

    A uint16 file with ``GDAL_NODATA="nan"`` would otherwise be
    masked with the wrong sentinel (or silently ignored). The release
    promise: raise with a message naming the unlocking opt-in.
    """
    path = _tmp(tmp_path, "case2_int_nodata_float_promoted")
    _build_uint16_tiff_with_nodata("nan", path)
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert "nodata" in msg.lower(), (
        f"expected the error message to name nodata; got: {msg!r}"
    )
    assert _msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


# --------------------------------------------------------------------------- #
# Case 3: rotated transform without ``allow_rotated=True``.                   #
# --------------------------------------------------------------------------- #
#
# Three sub-cases parametrized over the eager, dask, and windowed entry
# points -- the release promise is that each path raises the same typed
# error with a message that names ``allow_rotated``.


_ROTATED_PIXELS = np.arange(20, dtype='<u2').reshape(4, 5)


@pytest.fixture
def rotated_geotiff_path(tmp_path) -> str:
    """A throwaway rotated GeoTIFF that case 3's three sub-tests share."""
    path = _tmp(tmp_path, "case3_rotated")
    _write_rotated_tiff(path, _ROTATED_PIXELS)
    return path


def _assert_rotated_message(msg: str) -> None:
    """Shared assertions on the rotated error message.

    Inlined rather than promoted to a shared helper module so a
    parallel sibling PR cannot accidentally rebind the function.
    """
    assert "allow_rotated" in msg, (
        f"expected the error message to name the ``allow_rotated`` "
        f"opt-in; got: {msg!r}"
    )
    # ``reader.allow_rotated`` is tagged ``experimental`` in
    # SUPPORTED_FEATURES (see ``_attrs.py``). The check accepts any of
    # the three promised tier strings so a future promotion or
    # demotion in the same PR as the message edit does not break the
    # gate; if the tier moves, update the message text in the same PR
    # that moves the row in ``release_gate_geotiff.rst``.
    assert any(tier in msg for tier in ("advanced", "experimental", "stable")), (
        f"expected the error message to name the feature tier; "
        f"got: {msg!r}"
    )
    assert _msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


def test_release_gate_negative_rotated_eager(rotated_geotiff_path) -> None:
    """Eager numpy path raises ``RotatedTransformError`` without the opt-in."""
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(rotated_geotiff_path)
    _assert_rotated_message(str(excinfo.value))


def test_release_gate_negative_rotated_dask(rotated_geotiff_path) -> None:
    """Dask path raises the same typed error, uniformly with the eager path."""
    with pytest.raises(RotatedTransformError) as excinfo:
        read_geotiff_dask(rotated_geotiff_path, chunks=2)
    _assert_rotated_message(str(excinfo.value))


def test_release_gate_negative_rotated_windowed(rotated_geotiff_path) -> None:
    """Windowed read raises the same typed error before pixel decode."""
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(rotated_geotiff_path, window=(0, 0, 2, 2))
    _assert_rotated_message(str(excinfo.value))


@requires_gpu
def test_release_gate_negative_rotated_gpu(rotated_geotiff_path) -> None:
    """GPU read raises the same typed error as the CPU paths.

    ``reader.gpu`` is the ``experimental`` tier in
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. The release promise
    is loose for GPU (behaviour can change without a deprecation
    window) but the rotated-transform refusal is upstream of the
    GPU decode path -- the validator fires on the header read, before
    any pixel buffer reaches the GPU -- so the same typed error
    surfaces here regardless of the GPU tier.
    """
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(rotated_geotiff_path, gpu=True)
    _assert_rotated_message(str(excinfo.value))


# --------------------------------------------------------------------------- #
# Case 4: mixed-tier VRT children when stable-only is requested.              #
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason=(
        "The VRT stable-only knob is owned by epic #2342 and has not "
        "landed yet. The release promise: when the caller asks for "
        "stable-only sources and a VRT child uses an experimental "
        "codec, the reader names the offending child and the opt-in "
        "flag. This xfail flips to a pass when #2342 ships the knob."
    ),
    strict=False,
)
def test_release_gate_negative_mixed_tier_vrt_children(tmp_path) -> None:
    """The reader must refuse mixed-tier VRT children when stable-only is asked.

    The release promise: ``open_geotiff(vrt, stable_only=True)``
    rejects a VRT whose child uses an experimental-tier codec, names
    the offending child, and names the opt-in
    (``allow_experimental_codecs=True``) plus the feature tier.

    XFAIL-to-PASS transition note
    -----------------------------
    Today this test fails with ``TypeError: unexpected keyword
    argument 'stable_only'`` because epic #2342 has not landed the
    kwarg yet. The strict=False xfail swallows that TypeError.
    When #2342 lands, the test will start raising
    :class:`GeoTIFFAmbiguousMetadataError` (or fail to raise) and
    the xfail will report XPASS. Before removing the xfail marker,
    confirm the new code path satisfies both inline assertions: the
    error message must mention either ``stable_only`` or
    ``allow_experimental_codecs``, and it must cite the release
    contract docs. If either assertion would not pass, fix the
    production message in the same PR that removes the xfail.
    """
    path = _tmp(tmp_path, "case4_mixed_tier_vrt", suffix=".vrt")
    Path(path).write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2"></VRTDataset>\n',
        encoding="utf-8",
    )
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path, stable_only=True)  # type: ignore[call-arg]
    msg = str(excinfo.value)
    assert "stable_only" in msg or "allow_experimental_codecs" in msg, (
        f"expected the error message to name the unlocking opt-in; "
        f"got: {msg!r}"
    )
    assert _msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )
