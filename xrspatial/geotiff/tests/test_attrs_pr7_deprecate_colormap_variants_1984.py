"""Deprecation locking test for the matplotlib colormap-derived attrs.

Issue #1984, PR 7. The reader emits ``attrs['cmap']`` (when matplotlib
is importable) and ``attrs['colormap_rgba']`` whenever the source file
declares ``Photometric=3`` (palette). The writer never selects
``Photometric=3`` from attrs alone, so a write -> read cycle drops both
keys. PR 6 (#2004) locked that drop in the pass-through contract.

PR 7 deprecates the read-side emission for one release cycle: callers
who want a matplotlib ``ListedColormap`` should build one from the
canonical ``attrs['colormap']`` (raw uint16 RGB triples from TIFF tag
320) instead. The plain ``attrs['colormap']`` is the canonical tier
attr and still round-trips through ``_merge_friendly_extra_tags``;
this PR does not touch it.

This file pins three behaviours:

1. Reading a ``Photometric=3`` TIFF emits ``DeprecationWarning`` for
   ``colormap_rgba`` (always) and for ``cmap`` (only when matplotlib
   is importable).
2. Both attrs still ARE emitted on the resulting DataArray during the
   deprecation window. Removing emission would break callers who have
   not yet migrated; that removal is a follow-up PR.
3. The plain ``attrs['colormap']`` is unaffected: no warning fires on
   it, and it still lands in attrs.
"""
from __future__ import annotations

import importlib.util
import struct
import warnings

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff


_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _make_palette_uint8_tiff(path, pixels, palette_rgb16):
    """Write an 8-bit, 256-entry palette TIFF (Photometric=3) directly.

    Mirrors the helper in ``test_metadata_round_trip_1484.py``. The
    writer in xrspatial cannot emit Photometric=3, so the deprecation
    path is only reachable via a hand-built fixture like this one.
    """
    bo = '<'
    width = pixels.shape[1]
    height = pixels.shape[0]
    n_colors = 256
    assert len(palette_rgb16) == n_colors

    flat = pixels.ravel().astype(np.uint8)
    pixel_bytes = flat.tobytes()

    r_vals = [c[0] for c in palette_rgb16]
    g_vals = [c[1] for c in palette_rgb16]
    b_vals = [c[2] for c in palette_rgb16]
    cmap_values = r_vals + g_vals + b_vals

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals),
             struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 8)        # bits per sample
    add_short(259, 1)        # no compression
    add_short(262, 3)        # photometric = palette
    add_short(277, 1)        # samples per pixel = 1
    add_short(278, height)   # rows per strip
    add_long(273, 0)         # strip offsets placeholder
    add_long(279, len(pixel_bytes))
    add_shorts(320, cmap_values)  # ColorMap
    add_short(339, 1)        # sample format = uint

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    with open(path, 'wb') as f:
        f.write(bytes(out))


def _palette_fixture(tmp_path, name='palette_pr7_1984.tif'):
    """Build a 2x5 uint8 palette TIFF with a 256-entry RGB palette."""
    palette = [(i * 257, (255 - i) * 257, (i * 2) % 65536)
               for i in range(256)]
    pixels = np.array([[0, 1, 2, 254, 255],
                       [10, 20, 30, 40, 50]], dtype=np.uint8)
    path = str(tmp_path / name)
    _make_palette_uint8_tiff(path, pixels, palette)
    return path


def test_colormap_rgba_emits_deprecation_warning(tmp_path):
    """Reading a Photometric=3 TIFF triggers DeprecationWarning for
    ``attrs['colormap_rgba']``.

    Fires regardless of whether matplotlib is installed: the reader
    sets ``colormap_rgba`` on both the matplotlib branch and the
    ImportError fallback branch.
    """
    path = _palette_fixture(tmp_path, name='palette_rgba_warn.tif')

    with pytest.warns(DeprecationWarning) as record:
        open_geotiff(path)

    matched = [
        w for w in record
        if "attrs['colormap_rgba']" in str(w.message)
        and 'issue #1984' in str(w.message)
    ]
    assert matched, (
        "Expected a DeprecationWarning mentioning attrs['colormap_rgba'] "
        "and issue #1984; got: "
        f"{[str(w.message) for w in record]}"
    )


@pytest.mark.skipif(
    not _HAS_MATPLOTLIB, reason="matplotlib not installed"
)
def test_cmap_emits_deprecation_warning(tmp_path):
    """Reading a Photometric=3 TIFF triggers DeprecationWarning for
    ``attrs['cmap']`` when matplotlib is installed."""
    path = _palette_fixture(tmp_path, name='palette_cmap_warn.tif')

    with pytest.warns(DeprecationWarning) as record:
        open_geotiff(path)

    matched = [
        w for w in record
        if "attrs['cmap']" in str(w.message)
        and 'issue #1984' in str(w.message)
    ]
    assert matched, (
        "Expected a DeprecationWarning mentioning attrs['cmap'] and "
        "issue #1984; got: "
        f"{[str(w.message) for w in record]}"
    )


def test_deprecated_colormap_attrs_still_emitted(tmp_path):
    """During the deprecation window the matplotlib variants still
    land on ``DataArray.attrs``.

    Removing emission is a follow-up PR; this test pins the current
    contract that callers can still read the attrs while migrating.
    """
    path = _palette_fixture(tmp_path, name='palette_still_emitted.tif')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        da = open_geotiff(path)

    assert 'colormap_rgba' in da.attrs, (
        "attrs['colormap_rgba'] should still be emitted during the "
        "deprecation window; got attrs: "
        f"{sorted(da.attrs.keys())}"
    )
    if _HAS_MATPLOTLIB:
        assert 'cmap' in da.attrs, (
            "attrs['cmap'] should still be emitted during the deprecation "
            "window when matplotlib is installed; got attrs: "
            f"{sorted(da.attrs.keys())}"
        )


def test_plain_colormap_attr_not_deprecated(tmp_path):
    """The canonical ``attrs['colormap']`` (raw uint16 triples from
    tag 320) is unaffected: it still lands in attrs and does NOT carry
    a DeprecationWarning mentioning its key.

    The matplotlib-derived variants share the same Photometric=3 gate,
    so warnings for those two fire on the same read. The check below
    targets ``"attrs['colormap']"`` as a whole-token substring (with
    its closing bracket) so it does not match the deprecated variants.
    """
    path = _palette_fixture(tmp_path, name='palette_plain_colormap.tif')

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        da = open_geotiff(path)

    # Plain ``colormap`` (canonical tier) is present.
    assert 'colormap' in da.attrs, (
        "attrs['colormap'] must still be emitted; it round-trips via "
        "_merge_friendly_extra_tags and is the canonical replacement "
        "for the deprecated matplotlib variants. attrs: "
        f"{sorted(da.attrs.keys())}"
    )
    # Raw uint16 triples: 3 * 256 = 768 entries.
    assert len(da.attrs['colormap']) == 768

    # No DeprecationWarning targets the plain ``colormap`` key. The
    # deprecated variant warnings reference ``attrs['colormap']`` in
    # their migration hint, so we look for the "is deprecated" verb
    # immediately after the key to find a warning whose *subject* is
    # the plain key.
    bad = [
        w for w in record
        if issubclass(w.category, DeprecationWarning)
        and "attrs['colormap'] is deprecated" in str(w.message)
    ]
    assert not bad, (
        "attrs['colormap'] must not trigger a DeprecationWarning; only "
        "the matplotlib variants (cmap, colormap_rgba) are deprecated. "
        f"Got: {[str(w.message) for w in bad]}"
    )
