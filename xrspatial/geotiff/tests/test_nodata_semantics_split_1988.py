"""Regression tests for issue #1988.

``attrs['nodata']`` was historically overloaded as both "the file
declared this sentinel" and "the reader already replaced sentinel
pixels with NaN." Issue #1988 split those into two attrs:

* ``attrs['nodata']`` -- declared file sentinel, always set when the
  source declared one.
* ``attrs['masked_nodata']`` -- boolean, ``True`` iff the in-memory
  array has been NaN-masked. ``False`` iff the array still carries the
  literal integer sentinel value (e.g. an int raster whose sentinel
  did not match any decoded pixel, or a dask graph that stayed at the
  source integer dtype).

This module exercises the matrix:

    source-has-sentinel x backend x {float-NaN-masked, int-sentinel-preserved}

and asserts the contract on every cell.
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from xrspatial.geotiff import open_geotiff, read_geotiff_dask


_SENTINEL = -9999.0


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


def _write_float_tiff(path: str, *, with_sentinel: bool) -> None:
    """Write a 4x4 float32 TIFF with or without a declared nodata sentinel."""
    arr = np.array(
        [[1.0, 2.0, _SENTINEL, 4.0],
         [5.0, _SENTINEL, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    kw = dict(
        driver="GTiff", height=4, width=4, count=1, dtype="float32",
        transform=from_origin(0, 4, 1, 1), crs="EPSG:4326",
    )
    if with_sentinel:
        kw["nodata"] = _SENTINEL
    with rasterio.open(path, "w", **kw) as ds:
        ds.write(arr, 1)


def _write_int_tiff(path: str, *, with_sentinel_hit: bool) -> None:
    """Write a uint16 TIFF.

    ``with_sentinel_hit`` controls whether the sentinel matches any
    pixel: when True, the file contains pixels equal to the declared
    sentinel (so the reader will float-promote and mask); when False,
    the file declares a sentinel but no pixel matches (so the reader
    keeps the integer dtype and masked_nodata stays False).
    """
    if with_sentinel_hit:
        arr = np.array(
            [[10, 20, 65535, 40],
             [50, 65535, 70, 80],
             [90, 100, 110, 120],
             [130, 140, 150, 160]],
            dtype=np.uint16,
        )
    else:
        arr = np.array(
            [[10, 20, 30, 40],
             [50, 60, 70, 80],
             [90, 100, 110, 120],
             [130, 140, 150, 160]],
            dtype=np.uint16,
        )
    with rasterio.open(
        path, "w",
        driver="GTiff", height=4, width=4, count=1, dtype="uint16",
        transform=from_origin(0, 4, 1, 1), crs="EPSG:4326",
        nodata=65535,
    ) as ds:
        ds.write(arr, 1)


# ----------------------------------------------------------------------------
# Eager numpy backend
# ----------------------------------------------------------------------------


class TestEagerNumpy:
    """``open_geotiff`` (eager numpy backend)."""

    def test_float_source_with_sentinel(self, tmp_path):
        """Float source + declared sentinel -> nodata set, masked_nodata=True."""
        path = str(tmp_path / "tnss1988_float_sentinel.tif")
        _write_float_tiff(path, with_sentinel=True)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == _SENTINEL
        assert da.attrs["masked_nodata"] is True
        # The literal sentinel must have been replaced with NaN.
        assert np.isnan(da.values).sum() == 2

    def test_float_source_without_sentinel(self, tmp_path):
        """Float source + no sentinel declared -> neither attr set."""
        path = str(tmp_path / "tnss1988_float_no_sentinel.tif")
        _write_float_tiff(path, with_sentinel=False)
        da = open_geotiff(path)
        assert "nodata" not in da.attrs
        # ``masked_nodata`` is only meaningful when a sentinel was
        # declared; absence is the signal.
        assert "masked_nodata" not in da.attrs

    def test_int_source_with_sentinel_hit(self, tmp_path):
        """Int source + sentinel hit -> nodata set, masked_nodata=True (promoted)."""
        path = str(tmp_path / "tnss1988_int_hit.tif")
        _write_int_tiff(path, with_sentinel_hit=True)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == 65535
        # Eager numpy promotes integer to float64 on the first hit.
        assert da.dtype.kind == "f"
        assert da.attrs["masked_nodata"] is True
        assert np.isnan(da.values).sum() == 2

    def test_int_source_no_hit_keeps_sentinel(self, tmp_path):
        """Int source + sentinel declared but no hit -> nodata set, masked_nodata=False.

        The eager numpy path only promotes integer arrays to float on
        the first sentinel hit. When the sentinel is in-range but never
        matches a pixel, the array stays at the source integer dtype
        and ``masked_nodata`` is False so downstream code knows the
        literal sentinel is still in-band.
        """
        path = str(tmp_path / "tnss1988_int_no_hit.tif")
        _write_int_tiff(path, with_sentinel_hit=False)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == 65535
        assert da.dtype.kind in ("u", "i")
        assert da.attrs["masked_nodata"] is False


# ----------------------------------------------------------------------------
# Dask numpy backend
# ----------------------------------------------------------------------------


class TestDaskNumpy:
    """``read_geotiff_dask`` (lazy dask + numpy backend)."""

    def test_float_source_with_sentinel(self, tmp_path):
        path = str(tmp_path / "tnss1988_dask_float_sentinel.tif")
        _write_float_tiff(path, with_sentinel=True)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == _SENTINEL
        assert da.attrs["masked_nodata"] is True

    def test_float_source_without_sentinel(self, tmp_path):
        path = str(tmp_path / "tnss1988_dask_float_no_sentinel.tif")
        _write_float_tiff(path, with_sentinel=False)
        da = read_geotiff_dask(path, chunks=2)
        assert "nodata" not in da.attrs
        assert "masked_nodata" not in da.attrs

    def test_int_source_with_in_range_sentinel(self, tmp_path):
        """Dask declares float64 up front for any in-range integer sentinel.

        The dask backend cannot defer promotion to runtime the way the
        eager path does (each chunk reads independently and concat
        needs a fixed dtype). When any integer sentinel is in-range,
        the declared graph dtype is float64 and ``masked_nodata`` is
        True regardless of whether a chunk actually hits the sentinel.
        """
        path = str(tmp_path / "tnss1988_dask_int_in_range.tif")
        _write_int_tiff(path, with_sentinel_hit=False)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == 65535
        assert da.dtype.kind == "f"
        assert da.attrs["masked_nodata"] is True


# ----------------------------------------------------------------------------
# Cross-cutting: integer sentinel that is out-of-range
# ----------------------------------------------------------------------------


def _build_uint16_with_out_of_range_nodata(path: str) -> None:
    """Write a uint16 TIFF whose declared nodata is out of dtype range.

    Mirrors the corpus used by ``test_nodata_nan_int_1774.py``. The
    sentinel cannot match any pixel, so masking is a no-op and the
    array stays uint16. ``masked_nodata`` must be False so downstream
    code knows the literal value space is intact.
    """
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_str = "-9999"  # out of range for uint16
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
    add_short(258, 16)            # BitsPerSample
    add_short(259, 1)             # Compression (none)
    add_short(262, 1)             # PhotometricInterpretation (BlackIsZero)
    add_short(277, 1)             # SamplesPerPixel
    add_short(284, 1)             # PlanarConfiguration (chunky)
    add_short(339, 1)             # SampleFormat (unsigned)
    add_long(254, 0)              # NewSubfileType
    add_long(322, 256)            # TileWidth (placeholder; we use strips)

    # Strip the placeholder TileWidth: switch to strips.
    tag_list = [t for t in tag_list if t[0] != 322]

    rows_per_strip = height
    add_short(278, rows_per_strip)  # RowsPerStrip

    pixels_bytes = pixels.tobytes()
    add_long(279, len(pixels_bytes))  # StripByteCounts

    add_ascii(42113, nodata_bytes)   # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])

    header_size = 8
    ifd_entries = len(tag_list)
    # Strip offset placeholder; pixel data goes after the IFD.
    strip_offsets_tag = (273, 4, 1, b'\x00\x00\x00\x00')  # type=LONG
    tag_list.append(strip_offsets_tag)
    ifd_entries += 1
    tag_list.sort(key=lambda t: t[0])

    ifd_size = 2 + ifd_entries * 12 + 4

    # Layout: header (8) + IFD + tag-overflow data + pixels.
    ifd_offset = header_size
    overflow_offset = ifd_offset + ifd_size

    overflow_buffers: list[bytes] = []
    overflow_positions: dict[int, int] = {}

    new_entries: list[bytes] = []
    for (tag, type_, count, data) in tag_list:
        if len(data) > 4:
            overflow_positions[tag] = overflow_offset + sum(
                len(b) for b in overflow_buffers
            )
            overflow_buffers.append(data)
            value_field = struct.pack(f'{bo}I', overflow_positions[tag])
        else:
            value_field = data.ljust(4, b'\x00')
        new_entries.append(
            struct.pack(f'{bo}HHI', tag, type_, count) + value_field
        )

    pixel_offset = (overflow_offset
                    + sum(len(b) for b in overflow_buffers))

    # Patch StripOffsets value field.
    patched_entries = []
    for entry, (tag, *_) in zip(new_entries, tag_list):
        if tag == 273:
            entry = entry[:8] + struct.pack(f'{bo}I', pixel_offset)
        patched_entries.append(entry)

    with open(path, 'wb') as f:
        f.write(b'II' + struct.pack(f'{bo}HI', 42, ifd_offset))
        f.write(struct.pack(f'{bo}H', ifd_entries))
        for e in patched_entries:
            f.write(e)
        f.write(struct.pack(f'{bo}I', 0))  # next IFD = none
        for b in overflow_buffers:
            f.write(b)
        f.write(pixels_bytes)


def test_int_source_with_out_of_range_sentinel(tmp_path):
    """Out-of-range int sentinel -> nodata set, masked_nodata=False (eager).

    The sentinel cannot match any pixel so masking is a no-op and the
    array stays at the source integer dtype. ``masked_nodata`` must be
    False so downstream code knows the literal sentinel value is still
    a possible (but in this case unhit) pixel value in the array.
    """
    path = str(tmp_path / "tnss1988_int_oor.tif")
    _build_uint16_with_out_of_range_nodata(path)
    da = open_geotiff(path)
    assert da.attrs["nodata"] == -9999
    assert da.dtype.kind == "u"
    assert da.attrs["masked_nodata"] is False


# ----------------------------------------------------------------------------
# Helper unit tests
# ----------------------------------------------------------------------------


class TestSetNodataAttrsHelper:
    """Direct coverage of :func:`_set_nodata_attrs` in ``_attrs.py``."""

    def test_float_dtype_marks_masked(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, -9999, array_dtype=np.float64)
        assert attrs == {"nodata": -9999, "masked_nodata": True}

    def test_int_dtype_marks_unmasked(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, -9999, array_dtype=np.uint16)
        assert attrs == {"nodata": -9999, "masked_nodata": False}

    def test_none_nodata_is_noop(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, None, array_dtype=np.uint16)
        assert attrs == {}

    def test_accepts_dtype_string(self):
        """``array_dtype`` may be a numpy dtype object, type, or string."""
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, 0, array_dtype="float32")
        assert attrs["masked_nodata"] is True
        attrs = {}
        _set_nodata_attrs(attrs, 0, array_dtype="uint8")
        assert attrs["masked_nodata"] is False
