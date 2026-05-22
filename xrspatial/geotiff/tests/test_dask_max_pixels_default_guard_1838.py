"""``read_geotiff_dask(max_pixels=None)`` must honour the module default.

The eager (``read_to_array``) and VRT chunked paths both substitute
``MAX_PIXELS_DEFAULT`` for ``None`` before applying the up-front pixel
count guard. ``read_geotiff_dask`` previously gated the guard on
``max_pixels is not None``, so callers could build a lazy graph over a
region far larger than the module-wide safety limit -- individual chunk
reads still fail at compute time, but the cheap up-front error path was
skipped, contradicting the documented "None uses the default cap"
semantics. Issue #1838.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from xrspatial.geotiff import read_geotiff_dask  # noqa: E402
from xrspatial.geotiff._reader import MAX_PIXELS_DEFAULT  # noqa: E402


def _write_oversized(path, *, h: int, w: int) -> None:
    """Write a tiny tiled TIFF whose declared dimensions exceed the cap.

    ``tifffile`` will not let us materialise a multi-billion-pixel array,
    so we exploit the fact that the dask reader only consults the
    header's ImageLength / ImageWidth tags for the up-front guard. The
    physical file is one tile; the header advertises a much larger
    image. Reading any window-less chunk would fail at decode time, but
    that is acceptable because the up-front guard is supposed to fire
    long before chunk tasks run.
    """
    # Smallest possible single-tile file; declare a tiny image so the
    # file is valid, then patch the IFD's width/length tags to advertise
    # an oversized image.
    arr = np.zeros((16, 16), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, tile=(16, 16),
                     photometric="minisblack", compression="none")
    # Rewrite the ImageWidth (256) and ImageLength (257) tags in the IFD.
    # tifffile writes a classic TIFF; the IFD starts at offset 8 for a
    # small file. Parsing the offset properly requires reading the
    # header; do that with tifffile's own parser to stay robust.
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        ifd_offset = page.offset
    raw = bytearray(path.read_bytes())
    # Little-endian classic TIFF; first 2 bytes of IFD = entry count.
    n_entries = int.from_bytes(raw[ifd_offset:ifd_offset + 2], 'little')
    for i in range(n_entries):
        entry = ifd_offset + 2 + i * 12
        tag = int.from_bytes(raw[entry:entry + 2], 'little')
        if tag == 256:  # ImageWidth
            raw[entry + 8:entry + 12] = int(w).to_bytes(4, 'little')
        elif tag == 257:  # ImageLength
            raw[entry + 8:entry + 12] = int(h).to_bytes(4, 'little')
    path.write_bytes(bytes(raw))


def test_default_max_pixels_guard_fires_for_full_region(tmp_path):
    """``max_pixels=None`` must apply the module default cap at the
    up-front region guard, matching the eager / VRT paths.
    """
    path = tmp_path / "tmp_1838_oversized.tif"
    side = int((MAX_PIXELS_DEFAULT ** 0.5)) + 2
    _write_oversized(path, h=side, w=side)
    with pytest.raises(ValueError, match=r"max_pixels"):
        read_geotiff_dask(str(path))


def test_explicit_max_pixels_still_enforced(tmp_path):
    path = tmp_path / "tmp_1838_explicit_cap.tif"
    _write_oversized(path, h=2048, w=2048)
    with pytest.raises(ValueError, match=r"max_pixels"):
        read_geotiff_dask(str(path), max_pixels=1024)


def test_small_region_unaffected(tmp_path):
    """The default cap must not interfere with normal small reads."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "tmp_1838_small.tif"
    tifffile.imwrite(str(path), arr, tile=(16, 16),
                     photometric="minisblack", compression="none")
    da = read_geotiff_dask(str(path), chunks=8)
    np.testing.assert_array_equal(da.compute().values, arr)
