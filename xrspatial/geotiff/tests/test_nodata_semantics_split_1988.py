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

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, read_vrt

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


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

    def test_int_source_with_out_of_range_sentinel(self, tmp_path):
        """Dask + out-of-range int sentinel -> graph stays int, masked_nodata=False.

        Mirrors the eager-path ``test_int_source_with_out_of_range_sentinel``
        free function. The dask ``effective_dtype`` branch only promotes
        to float64 when the sentinel fits the source dtype range; an
        out-of-range sentinel (e.g. uint16 file with
        ``GDAL_NODATA="-9999"``) cannot match any pixel, so the declared
        graph dtype stays uint16 and ``masked_nodata`` must be False.
        """
        path = str(tmp_path / "tnss1988_dask_int_oor.tif")
        _build_uint16_with_out_of_range_nodata(path)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == -9999
        assert da.dtype.kind == "u"
        assert da.attrs["masked_nodata"] is False


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


# ----------------------------------------------------------------------------
# VRT backend (eager + chunked)
# ----------------------------------------------------------------------------


def _write_uint16_vrt_source(tmp_path, *, sentinel_hit: bool, filename: str):
    """Write a 2x2 uint16 source raster with declared sentinel 65535."""
    from xrspatial.geotiff._writer import write
    if sentinel_hit:
        band = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    else:
        band = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=65535, compression="none", tiled=False)
    return p


def _build_vrt(tmp_path, source_path, vrt_dtype, nodata_value,
               filename="tnss1988.vrt"):
    """Hand-roll a 2x2 VRT pointing at ``source_path``."""
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{vrt_dtype}" band="1">
    <NoDataValue>{nodata_value}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = str(tmp_path / filename)
    with open(p, "w") as f:
        f.write(vrt_xml)
    return p


class TestVRTEager:
    """``read_vrt`` (eager path) honours the split-attrs contract."""

    def test_float32_vrt_int_source_with_hit(self, tmp_path):
        """Float-typed VRT over int source with sentinel hit -> masked_nodata=True."""
        src = _write_uint16_vrt_source(
            tmp_path, sentinel_hit=True, filename="tnss1988_vrt_src_hit.tif",
        )
        vrt = _build_vrt(tmp_path, src, "Float32", 65535,
                         filename="tnss1988_vrt_hit.vrt")
        r = read_vrt(vrt)
        assert r.attrs["nodata"] == 65535.0
        assert r.dtype.kind == "f"
        assert r.attrs["masked_nodata"] is True

    def test_uint16_vrt_int_source_no_hit(self, tmp_path):
        """Int-typed VRT over int source, no sentinel pixel -> masked_nodata=False.

        A ``dataType="UInt16"`` VRT with no scale/offset keeps the
        source integer dtype. With no sentinel pixel in the source, the
        eager path produces a uint16 array carrying the literal
        sentinel value space, so ``masked_nodata`` must be False.
        """
        src = _write_uint16_vrt_source(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_src_nohit.tif",
        )
        vrt = _build_vrt(tmp_path, src, "UInt16", 65535,
                         filename="tnss1988_vrt_nohit.vrt")
        r = read_vrt(vrt)
        assert r.attrs["nodata"] == 65535.0
        assert r.dtype.kind in ("u", "i")
        assert r.attrs["masked_nodata"] is False

    def test_vrt_no_nodata_emits_neither_attr(self, tmp_path):
        """VRT band with no ``<NoDataValue>`` -> neither attr set."""
        src = _write_uint16_vrt_source(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_src_no_nd.tif",
        )
        # Build a VRT without a NoDataValue element.
        vrt_xml = """<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">""" + src + """</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
        vrt = str(tmp_path / "tnss1988_vrt_no_nd.vrt")
        with open(vrt, "w") as f:
            f.write(vrt_xml)
        r = read_vrt(vrt)
        assert "nodata" not in r.attrs
        assert "masked_nodata" not in r.attrs


class TestVRTChunked:
    """``read_vrt(..., chunks=N)`` honours the split-attrs contract."""

    def test_chunked_int_source_in_range_sentinel(self, tmp_path):
        """Chunked VRT declares float64 for in-range int sentinel -> masked_nodata=True."""
        src = _write_uint16_vrt_source(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_chunked_src.tif",
        )
        vrt = _build_vrt(tmp_path, src, "UInt16", 65535,
                         filename="tnss1988_vrt_chunked.vrt")
        r = read_vrt(vrt, chunks=2)
        assert r.attrs["nodata"] == 65535.0
        # Chunked path promotes to float64 declared dtype.
        assert r.dtype == np.float64
        assert r.attrs["masked_nodata"] is True


# ----------------------------------------------------------------------------
# GPU backend
# ----------------------------------------------------------------------------


@_gpu_only
class TestGPU:
    """``read_geotiff_gpu`` honours the split-attrs contract."""

    def test_int_source_with_hit(self, tmp_path):
        """Int source + sentinel hit on GPU -> masked_nodata=True (float)."""
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_int_hit.tif")
        _write_int_tiff(path, with_sentinel_hit=True)
        da = read_geotiff_gpu(path)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind == "f"
        assert da.attrs["masked_nodata"] is True

    def test_int_source_no_hit_keeps_sentinel(self, tmp_path):
        """Int source + sentinel no hit on GPU -> masked_nodata=False.

        Mirrors the eager-numpy contract: GPU masking only promotes int
        to float64 when at least one sentinel pixel is found.
        """
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_int_nohit.tif")
        _write_int_tiff(path, with_sentinel_hit=False)
        da = read_geotiff_gpu(path)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind in ("u", "i")
        assert da.attrs["masked_nodata"] is False

    def test_dask_gpu_in_range_sentinel(self, tmp_path):
        """Dask+GPU declares float64 graph for in-range int sentinel."""
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_dask_int.tif")
        _write_int_tiff(path, with_sentinel_hit=False)
        da = read_geotiff_gpu(path, chunks=2)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind == "f"
        assert da.attrs["masked_nodata"] is True


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


# ----------------------------------------------------------------------------
# Writer-side consultation of ``attrs['masked_nodata']``.
#
# The forward (read) step promotes the file sentinel to NaN and tags
# the result with ``attrs['masked_nodata'] = True``. The reverse
# (write) step rewrites NaN back to the sentinel. The reverse must
# only run when the forward ran. When ``masked_nodata=False`` the
# array did not go through the forward step, so any NaN present came
# from elsewhere and the writer must preserve it rather than silently
# converting to the integer sentinel.
# ----------------------------------------------------------------------------


class TestShouldRestoreNanSentinelHelper:
    """Direct coverage of :func:`_should_restore_nan_sentinel`."""

    def test_missing_attr_defaults_to_true(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel({}) is True
        # The default preserves pre-#1988 behaviour for any DataArray
        # that did not pass through xrspatial's reader.
        assert _should_restore_nan_sentinel({"nodata": -9999}) is True

    def test_masked_nodata_true_returns_true(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        attrs = {"nodata": -9999, "masked_nodata": True}
        assert _should_restore_nan_sentinel(attrs) is True

    def test_masked_nodata_false_returns_false(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        attrs = {"nodata": -9999, "masked_nodata": False}
        assert _should_restore_nan_sentinel(attrs) is False

    def test_none_attrs_defaults_to_true(self):
        """GPU writer's positional-cupy branch has no attrs to read."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel(None) is True

    def test_non_mapping_defaults_to_true(self):
        """A misuse that hands in a non-mapping must not crash the writer."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel("not a dict") is True

    def test_stray_truthy_value_is_true(self):
        """Only literal ``False`` disables. Stray ``0`` / ``''`` stays True."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        # Anything other than literal False should keep the default
        # behaviour. ``0`` is falsy but is not the contract value.
        assert _should_restore_nan_sentinel({"masked_nodata": 0}) is True
        assert _should_restore_nan_sentinel({"masked_nodata": None}) is True


class TestWriterRoundTripEager:
    """Round-trip through ``to_geotiff`` to verify the writer respects
    ``attrs['masked_nodata']``."""

    def test_masked_nodata_true_restores_sentinel(self, tmp_path):
        """Reader-style attrs (masked_nodata=True): NaN -> sentinel on write."""
        from xrspatial.geotiff import to_geotiff
        import xarray as xr

        path = tmp_path / "test_1988_writer_masked.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": True,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # NaN pixels should have been rewritten to the sentinel.
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2

    def test_masked_nodata_false_preserves_nan(self, tmp_path):
        """``masked_nodata=False`` -> NaN survives, no silent sentinel rewrite."""
        from xrspatial.geotiff import to_geotiff
        import xarray as xr

        path = tmp_path / "test_1988_writer_unmasked.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            # The GDAL_NODATA tag is still set, regardless of the
            # in-memory masking state. The two attrs carry independent
            # meanings (see issue #1988).
            assert ds.nodata == -9999.0
        # NaN pixels survive unchanged: the writer must NOT rewrite
        # them to the integer sentinel because the array did not pass
        # through the reader's sentinel-to-NaN promotion.
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0

    def test_missing_masked_nodata_attr_restores_sentinel(self, tmp_path):
        """External DataArrays without the attr keep pre-#1988 behaviour."""
        from xrspatial.geotiff import to_geotiff
        import xarray as xr

        path = tmp_path / "test_1988_writer_no_attr.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        # No ``masked_nodata`` attr -> default True.
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # Pre-#1988 behaviour: missing attr = treat as masked.
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2

    def test_round_trip_preserves_masked_nodata_true(self, tmp_path):
        """Read sentinel TIFF -> attrs say masked=True -> write -> reread.

        Closes the loop: a file with a declared sentinel reads back
        with NaN-masked pixels and ``masked_nodata=True``. Writing it
        out then reading again must produce the same sentinel-tagged
        file (the writer correctly inverts the read-side promotion).
        """
        from xrspatial.geotiff import to_geotiff

        src = tmp_path / "test_1988_round_trip_src.tif"
        _write_float_tiff(str(src), with_sentinel=True)

        da = open_geotiff(str(src))
        assert da.attrs["masked_nodata"] is True
        # The reader promoted the sentinel value to NaN.
        arr_in = np.asarray(da.data)
        assert np.isnan(arr_in).sum() == 2

        dst = tmp_path / "test_1988_round_trip_dst.tif"
        to_geotiff(da, str(dst), compression="none")

        with rasterio.open(str(dst)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == _SENTINEL
        # Sentinel values restored at the expected positions.
        assert (on_disk == _SENTINEL).sum() == 2
        assert not np.isnan(on_disk).any()

    def test_dask_streaming_path_respects_flag(self, tmp_path):
        """Dask + tiled streaming write must honour the gate too."""
        from xrspatial.geotiff import to_geotiff
        import dask.array as da_mod
        import xarray as xr

        path = tmp_path / "test_1988_writer_dask.tif"
        # 32x32 with NaN sprinkled in -- the tiled streaming writer
        # requires tile_size to be a positive multiple of 16.
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        arr[12, 19] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        da = xr.DataArray(
            dask_arr,
            dims=("y", "x"),
            coords={
                "y": np.arange(32, 0, -1, dtype=np.float64),
                "x": np.arange(32, dtype=np.float64),
            },
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 32.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        to_geotiff(
            da, str(path), compression="none",
            tile_size=16, tiled=True,
        )

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # NaN preserved through the streaming path.
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0


class TestWriteStreamingRestoreSentinelKwarg:
    """Direct coverage of ``restore_sentinel`` on the low-level
    ``write_streaming`` callable surface, where the gate actually
    suppresses an internal NaN-to-sentinel rewrite step.

    The non-streaming ``write`` function expects its caller (e.g.
    ``to_geotiff``) to have already performed the NaN-to-sentinel
    rewrite, so its own ``restore_sentinel`` flag only gates the
    overview-decimation rewrite (a no-op when ``cog=False``).
    """

    def test_streaming_restore_sentinel_true_rewrites(self, tmp_path):
        import dask.array as da_mod
        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_true.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
            restore_sentinel=True,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert (on_disk == -9999.0).sum() == 1
        assert not np.isnan(on_disk).any()

    def test_streaming_restore_sentinel_false_preserves_nan(self, tmp_path):
        import dask.array as da_mod
        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_false.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
            restore_sentinel=False,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        # GDAL_NODATA tag still set; only the array bytes change.
        assert ds.nodata == -9999.0
        assert np.isnan(on_disk).sum() == 1
        assert (on_disk == -9999.0).sum() == 0

    def test_streaming_default_is_true(self, tmp_path):
        """Default preserves pre-#1988 behaviour."""
        import dask.array as da_mod
        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_default.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert (on_disk == -9999.0).sum() == 1
        assert not np.isnan(on_disk).any()

    def test_streaming_strip_layout_restore_false_preserves_nan(self, tmp_path):
        """The strip-write branch in ``write_streaming`` must honour the gate."""
        import dask.array as da_mod
        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_strip_false.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=False,
            restore_sentinel=False,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert np.isnan(on_disk).sum() == 1
        assert (on_disk == -9999.0).sum() == 0


@_gpu_only
class TestWriterGPU:
    """GPU writer also gates on ``attrs['masked_nodata']``."""

    def test_masked_nodata_false_preserves_nan_gpu(self, tmp_path):
        import cupy
        import xarray as xr
        from xrspatial.geotiff import write_geotiff_gpu

        path = tmp_path / "test_1988_writer_gpu_unmasked.tif"
        arr_np = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        arr = cupy.asarray(arr_np)
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        write_geotiff_gpu(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0

    def test_masked_nodata_true_restores_sentinel_gpu(self, tmp_path):
        import cupy
        import xarray as xr
        from xrspatial.geotiff import write_geotiff_gpu

        path = tmp_path / "test_1988_writer_gpu_masked.tif"
        arr_np = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        arr = cupy.asarray(arr_np)
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": True,
            },
        )
        write_geotiff_gpu(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2
