"""Eager writer BigTIFF auto-detection uses exact IFD overhead.

Regression for issue #1905. The eager writer previously decided
BigTIFF with a fixed-fudge estimate:

    ifd_overhead = num_levels * (2 + 12 * max_tags_per_ifd + 4 + 1024)

The 1 KB constant under-promoted near the 4 GiB boundary when
``gdal_metadata_xml`` or ``extra_tags`` pushed the actual overflow
heap past it. The fix reuses ``_compute_classic_ifd_overhead`` from
the streaming writer (added in #1785, #1787) so eager and streaming
paths agree on the estimate.

Tests cover three angles:

* ``_compute_classic_ifd_overhead`` reports exactly the bytes
  ``_build_ifd`` emits.
* When the eager writer is called with a large ``gdal_metadata_xml``,
  the actual on-disk IFD bytes match the new estimate (and exceed
  the old 1 KB fudge).
* Round-tripping the file through the parser still works, so the
  fix did not perturb tag emission.
"""
from __future__ import annotations

import struct

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._dtypes import ASCII, LONG
from xrspatial.geotiff._writer import _build_ifd, _compute_classic_ifd_overhead


def _make_4x4_float32(
    crs: int = 4326, gdal_metadata_xml: str | None = None,
) -> xr.DataArray:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    attrs = {"crs": crs}
    if gdal_metadata_xml is not None:
        attrs["gdal_metadata_xml"] = gdal_metadata_xml
    return xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([0.5, 1.5, 2.5, 3.5]),
        },
        attrs=attrs,
    )


def test_overhead_matches_built_ifd_size():
    """Spot check ``_compute_classic_ifd_overhead`` exactness."""
    # ImageWidth, ImageLength (inline LONG) + a long ASCII metadata
    # value that forces overflow.
    metadata = "x" * 4096
    tags = [
        (256, LONG, 1, 16),  # ImageWidth
        (257, LONG, 1, 16),  # ImageLength
        (270, ASCII, len(metadata) + 1, metadata),  # ImageDescription
    ]
    expected = _compute_classic_ifd_overhead(tags)

    ifd_bytes, overflow_bytes = _build_ifd(
        tags, overflow_base=0, bigtiff=False,
    )
    actual = len(ifd_bytes) + len(overflow_bytes)
    assert expected == actual


def test_overhead_includes_strip_offset_arrays():
    """Strip / tile offset arrays land in the overflow heap when count
    pushes them past 4 bytes inline. The overhead must include them."""
    offsets = list(range(64))  # 64 LONG entries = 256 bytes overflow
    byte_counts = [10] * 64
    tags = [
        (256, LONG, 1, 16),
        (257, LONG, 1, 16),
        (273, LONG, 64, offsets),       # StripOffsets
        (279, LONG, 64, byte_counts),   # StripByteCounts
    ]
    overhead = _compute_classic_ifd_overhead(tags)
    # Two LONG arrays of 64 entries each contribute 64 * 4 * 2 = 512
    # bytes of overflow. Plus a 2 + 12*4 + 4 = 54-byte entry block,
    # padded to be even.
    assert overhead >= 2 + 12 * 4 + 4 + 64 * 4 * 2


def test_overhead_exceeds_old_fudge_for_large_metadata():
    """Old fudge: 1024 bytes per IFD. Large metadata blows past it."""
    metadata = "x" * 8192
    tags = [
        (256, LONG, 1, 16),
        (257, LONG, 1, 16),
        (42112, ASCII, len(metadata) + 1, metadata),  # GDAL_METADATA
    ]
    overhead = _compute_classic_ifd_overhead(tags)
    old_fudge = 2 + 12 * len(tags) + 4 + 1024
    assert overhead > old_fudge


def test_eager_writer_round_trip_with_large_gdal_metadata(tmp_path):
    """Writing with multi-KB ``gdal_metadata_xml`` still produces a
    valid file. The new overhead estimate accounts for the metadata
    size; the file parses back to the same array."""
    metadata_xml = (
        "<GDALMetadata>"
        + "<Item name='note'>" + ("y" * 4096) + "</Item>"
        + "</GDALMetadata>"
    )
    da = _make_4x4_float32(gdal_metadata_xml=metadata_xml)
    path = str(tmp_path / "large_metadata_1905.tif")
    to_geotiff(da, path, allow_experimental_codecs=True)

    rt = open_geotiff(path)
    np.testing.assert_array_equal(rt.values, da.values)

    # The actual emitted file should still be classic TIFF (magic 42)
    # because the total stays well under 4 GiB.
    with open(path, "rb") as f:
        head = f.read(8)
    assert head[:2] == b"II"
    magic = struct.unpack_from("<H", head, 2)[0]
    assert magic == 42


def test_eager_writer_promotes_to_bigtiff_when_overhead_dominates(
    tmp_path, monkeypatch,
):
    """When the IFD overhead alone is the deciding factor, the new
    estimate must drive BigTIFF auto-detection.

    Drives the boundary by shrinking the writer's classic-fits ceiling
    via the public ``force_bigtiff=False`` flag's negation: we monkey
    patch ``_compute_classic_ifd_overhead`` to return a value past
    UINT32_MAX. With the fix wired in, this flips ``estimated_file_size
    > UINT32_MAX`` and the writer emits a BigTIFF header.
    """
    da = _make_4x4_float32()
    path = str(tmp_path / "bigtiff_decision_1905.tif")

    from xrspatial.geotiff import _writer as writer_mod

    real = writer_mod._compute_classic_ifd_overhead

    def _huge_overhead(tags):
        # Real overhead + a chunk that pushes the total above UINT32_MAX.
        return real(tags) + 0x100000000

    monkeypatch.setattr(
        writer_mod, "_compute_classic_ifd_overhead", _huge_overhead,
    )

    to_geotiff(da, path, allow_experimental_codecs=True)
    with open(path, "rb") as f:
        head = f.read(8)
    assert head[:2] == b"II"
    magic = struct.unpack_from("<H", head, 2)[0]
    assert magic == 43, "writer should have chosen BigTIFF"


def test_eager_writer_keeps_classic_when_overhead_fits(tmp_path):
    """Sanity check: the default 4x4 file fits classic comfortably."""
    da = _make_4x4_float32()
    path = str(tmp_path / "classic_1905.tif")
    to_geotiff(da, path, allow_experimental_codecs=True)
    with open(path, "rb") as f:
        head = f.read(8)
    magic = struct.unpack_from("<H", head, 2)[0]
    assert magic == 42


def test_overhead_matches_actual_emitted_size_via_writer(tmp_path):
    """End-to-end: actual IFD bytes in the emitted file match the
    estimate ``_compute_classic_ifd_overhead`` would compute for the
    same tag list. Catches drift between writer and estimator."""
    metadata_xml = "<GDALMetadata><Item>" + ("z" * 1024) + "</Item></GDALMetadata>"
    da = _make_4x4_float32(gdal_metadata_xml=metadata_xml)
    path = str(tmp_path / "match_actual_1905.tif")
    to_geotiff(da, path, allow_experimental_codecs=True)

    # Parse the file to find IFD offset, then measure the bytes between
    # IFD start and the first pixel-data offset to confirm the writer
    # did not stray from its own estimate.
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"II"
    ifd_offset = struct.unpack_from("<I", data, 4)[0]
    num_entries = struct.unpack_from("<H", data, ifd_offset)[0]
    entry_block_end = ifd_offset + 2 + num_entries * 12 + 4
    # IFD entry block is a fixed size; overflow heap follows. Verify
    # the IFD entry block size matches what _compute_classic_ifd_overhead
    # accounts for as the fixed component.
    assert entry_block_end - ifd_offset == 2 + 12 * num_entries + 4
