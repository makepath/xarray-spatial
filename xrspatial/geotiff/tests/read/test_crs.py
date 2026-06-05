"""Rotated and dropped-CRS read-path matrix.

Covers the ``allow_rotated`` opt-in path and its CRS-drop contract:

* Rotated TIFFs without embedded CRS; ``allow_rotated=False`` (default)
  raises and ``allow_rotated=True`` reads the pixel grid on the eager and
  dask GeoTIFF backends.
* Rotated TIFFs whose GeoKey block carries a CRS; ``allow_rotated=True``
  drops ``crs`` / ``crs_wkt`` together with the transform on both
  backends, plus the unit-level ``_populate_attrs_from_geo_info``
  contract.
* The eager / dask / cupy / dask+cupy backend matrix plus the VRT eager +
  chunked paths, all asserting CRS drop under ``allow_rotated=True``.

The HTTP-server rotated read lives in the integration suite, not here.

Test ID convention: ``(scenario, ...)`` where ``scenario`` is one of
``rotated_no_crs``, ``rotated_with_crs``, ``axis_aligned_with_crs``.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._attrs import _populate_attrs_from_geo_info
from xrspatial.geotiff._errors import RotatedTransformError
from xrspatial.geotiff._geotags import (_NO_GEOREF_KEY, TAG_MODEL_TRANSFORMATION, GeoInfo,
                                        GeoTransform, _extract_transform)
from xrspatial.geotiff._header import IFD, IFDEntry

from .._helpers.markers import requires_gpu

# ---------------------------------------------------------------------------
# Synthetic rotated 4x4 ModelTransformationTag values.
# ---------------------------------------------------------------------------

# 30-degree rotation, pixel size 10, origin (100, 200).
_COS30 = 0.8660254037844387
_SIN30 = 0.5
_ROTATED_M_30DEG = (
    10.0 * _COS30, -10.0 * _SIN30, 0.0, 100.0,
    10.0 * _SIN30, 10.0 * _COS30, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

# Smaller-rotation variant with pixel_width 1, b=0.1, pixel_height -1.
_ROTATED_M_SMALL = (
    1.0, 0.1, 0.0, 100.0,
    0.0, -1.0, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

_AXIS_ALIGNED_M = (
    10.0, 0.0, 0.0, 100.0,
    0.0, -10.0, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

# Minimal GeoKeyDirectoryTag declaring EPSG:4326.
# (version, key_revision, minor_revision, n_keys, KeyID, Location, Count, Value)
_GEO_KEYS_4326 = (
    1, 1, 0, 3,
    1024, 0, 1, 2,    # GTModelTypeGeoKey = Geographic
    1025, 0, 1, 1,    # GTRasterTypeGeoKey = Area
    2048, 0, 1, 4326,  # GeographicTypeGeoKey
)


# ---------------------------------------------------------------------------
# IFD builders for unit tests of ``_extract_transform``.
# ---------------------------------------------------------------------------


def _make_ifd_with_transform(matrix_16: tuple) -> IFD:
    """Build a synthetic IFD that carries only a ModelTransformationTag.

    Other geo-related tags stay absent so the transform branch is the
    only one exercised; this keeps the unit tests independent of the
    full geo-key plumbing.
    """
    ifd = IFD()
    ifd.entries[TAG_MODEL_TRANSFORMATION] = IFDEntry(
        tag=TAG_MODEL_TRANSFORMATION,
        type_id=12,  # DOUBLE
        count=16,
        value=tuple(matrix_16),
    )
    return ifd


# ---------------------------------------------------------------------------
# Hand-rolled TIFF writers. ``_helpers/tiff_builders.make_minimal_tiff``
# does not support a full 4x4 ModelTransformationTag (it writes axis-
# aligned files via ModelTiepoint + ModelPixelScale), so the rotated
# variants stay file-local. They are unchanged from the originals.
# ---------------------------------------------------------------------------


def _write_rotated_tiff(path, arr: np.ndarray, *, matrix=_ROTATED_M_30DEG) -> None:
    """Write a minimal little-endian TIFF with a rotated ModelTransformationTag.

    Single-band, single-strip, uncompressed; no embedded CRS. Just
    enough to exercise the reader's rotation path without rasterio /
    GDAL.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries = [
        (256, 3, 1, w),              # ImageWidth
        (257, 3, 1, h),              # ImageLength
        (258, 3, 1, 16),             # BitsPerSample = 16
        (259, 3, 1, 1),              # Compression = none
        (262, 3, 1, 1),              # Photometric = BlackIsZero
        (273, 4, 1, header_size),    # StripOffsets
        (277, 3, 1, 1),              # SamplesPerPixel
        (278, 3, 1, h),              # RowsPerStrip
        (279, 4, 1, strip_size),     # StripByteCounts
        (339, 3, 1, 1),              # SampleFormat = unsigned int
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])

    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:  # SHORT
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)  # next IFD

    with open(path, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *matrix))
        f.write(ifd_bytes)


def _write_rotated_tiff_with_crs(path, arr: np.ndarray) -> None:
    """Rotated TIFF with both ModelTransformationTag AND EPSG:4326 GeoKeys.

    The CRS is encoded in the GeoKeyDirectoryTag rather than ProjectedCRS
    overflow, so the reader populates ``geo_info.crs_epsg`` alongside the
    rotated transform. Used to exercise the CRS-drop path.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    geo_keys_off = transform_off + transform_size
    geo_keys_size = len(_GEO_KEYS_4326) * 2  # SHORT = 2 bytes
    ifd_off = geo_keys_off + geo_keys_size

    entries = [
        (256, 3, 1, w),
        (257, 3, 1, h),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, header_size),
        (277, 3, 1, 1),
        (278, 3, 1, h),
        (279, 4, 1, strip_size),
        (339, 3, 1, 1),
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
        (34735, 3, len(_GEO_KEYS_4326), geo_keys_off),  # GeoKeyDirectoryTag
    ]
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
        f.write(struct.pack('<16d', *_ROTATED_M_SMALL))
        f.write(struct.pack(f'<{len(_GEO_KEYS_4326)}H', *_GEO_KEYS_4326))
        f.write(ifd_bytes)


def _write_rotated_tiff_with_geokeys_via_tifffile(path, arr, *, epsg=4326):
    """Rotated TIFF with GeoKey CRS, written via tifffile.

    Used by the end-to-end ``open_geotiff`` cases that want a different
    code path from the hand-rolled writer above (specifically the
    Geographic key without the Area / ModelType companion keys).
    """
    tifffile = pytest.importorskip("tifffile")
    geo_key_directory = (
        1, 1, 0, 1,
        2048, 0, 1, int(epsg),
    )
    extratags = [
        (34264, 12, 16, _ROTATED_M_30DEG, False),
        (34735, 3, 8, geo_key_directory, False),
    ]
    tifffile.imwrite(
        path, arr, photometric='minisblack',
        planarconfig='contig', extratags=extratags,
    )


def _write_axis_aligned_with_crs_via_tifffile(path, arr):
    """Axis-aligned TIFF with the full _GEO_KEYS_4326 block.

    Used by the regression case that confirms the CRS-drop gate keys on
    ``has_georef`` rather than blindly stripping CRS whenever the GeoKey
    block exists.
    """
    tifffile = pytest.importorskip("tifffile")
    m_aligned = (
        1.0, 0.0, 0.0, 100.0,
        0.0, -1.0, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [
        (34264, 12, 16, m_aligned),
        (34735, 3, len(_GEO_KEYS_4326), _GEO_KEYS_4326),
    ]
    tifffile.imwrite(str(path), arr, photometric='minisblack',
                     extratags=extratags)


def _write_minimal_aligned_tiff(path):
    """Tiny axis-aligned TIFF used as a VRT's source band."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(str(path), arr, photometric='minisblack')


def _write_rotated_vrt(vrt_path, source_filename):
    """VRT whose GeoTransform carries a non-zero rotation term plus a
    full EPSG:4326 SRS block.
    """
    wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
        '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",'
        '0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    )
    xml = (
        f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        f'  <SRS>{wkt}</SRS>\n'
        f'  <GeoTransform>0.0, 1.0, 0.5, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="UInt16" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_filename}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


def _write_axis_aligned_vrt(vrt_path, source_filename):
    """Axis-aligned VRT with the same EPSG:4326 SRS block.

    Mirrors ``_write_rotated_vrt`` but with the rotation term set to zero
    so the CRS-drop gate stays inactive.
    """
    wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
        '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",'
        '0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    )
    xml = (
        f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        f'  <SRS>{wkt}</SRS>\n'
        f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="UInt16" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_filename}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


# ===========================================================================
# Unit tests: ``_extract_transform`` rotated handling.
# ===========================================================================


def test_extract_transform_rotated_default_raises():
    """Default ``allow_rotated=False`` rejects a rotated transform."""
    ifd = _make_ifd_with_transform(_ROTATED_M_30DEG)
    with pytest.raises(RotatedTransformError, match="rotation"):
        _extract_transform(ifd)


def test_extract_transform_rotated_optin_returns_no_georef():
    """``allow_rotated=True`` returns has_georef=False and preserves the
    rotated 6-tuple on ``rotated_affine``."""
    ifd = _make_ifd_with_transform(_ROTATED_M_30DEG)
    gt, has_georef = _extract_transform(ifd, allow_rotated=True)
    assert isinstance(gt, GeoTransform)
    assert has_georef is False
    expected = (
        _ROTATED_M_30DEG[0], _ROTATED_M_30DEG[1], _ROTATED_M_30DEG[3],
        _ROTATED_M_30DEG[4], _ROTATED_M_30DEG[5], _ROTATED_M_30DEG[7],
    )
    assert gt.rotated_affine is not None
    assert gt.rotated_affine == pytest.approx(expected)


def test_extract_transform_axis_aligned_optin_passes_through():
    """Axis-aligned files are unchanged by the opt-in path."""
    ifd = _make_ifd_with_transform(_AXIS_ALIGNED_M)
    gt, has_georef = _extract_transform(ifd, allow_rotated=True)
    assert has_georef is True
    assert gt.pixel_width == 10.0
    assert gt.pixel_height == -10.0
    assert gt.origin_x == 100.0
    assert gt.origin_y == 200.0
    assert gt.rotated_affine is None


# ===========================================================================
# Unit tests: ``_populate_attrs_from_geo_info`` CRS-drop contract.
# ===========================================================================


def _rotated_geo_info(*, with_crs: bool = True) -> GeoInfo:
    """Build a GeoInfo that mimics the ``allow_rotated=True`` parser output."""
    t = GeoTransform(
        origin_x=0.0, origin_y=0.0,
        pixel_width=1.0, pixel_height=-1.0,
        rotated_affine=(8.66, -5.0, 100.0, 5.0, 8.66, 200.0),
    )
    return GeoInfo(
        transform=t,
        has_georef=False,
        crs_epsg=4326 if with_crs else None,
        crs_wkt='GEOGCS["WGS 84"]' if with_crs else None,
    )


@pytest.mark.parametrize("attr", ["crs", "crs_wkt", "transform"])
def test_populate_attrs_rotated_optin_drops_attr(attr):
    """Rotated opt-in path drops ``crs`` / ``crs_wkt`` / ``transform``."""
    gi = _rotated_geo_info()
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert attr not in attrs


def test_populate_attrs_plain_no_georef_keeps_crs():
    """Plain no-georef file (no transform tag) still emits ``crs``.

    The drop gate is specific to the rotated opt-in path (transform
    carries ``rotated_affine`` and ``has_georef`` is False). A file with
    no transform at all but a GeoKey CRS keeps its CRS so callers can
    display / reproject the array.
    """
    gi = GeoInfo(transform=GeoTransform(), has_georef=False, crs_epsg=4326)
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert attrs.get('crs') == 4326
    assert 'transform' not in attrs


def test_populate_attrs_axis_aligned_keeps_crs_and_transform():
    """Axis-aligned georeferenced files keep both attrs."""
    gi = GeoInfo(
        transform=GeoTransform(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
        ),
        has_georef=True,
        crs_epsg=4326,
    )
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)
    assert attrs.get('crs') == 4326
    assert attrs.get('transform') == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


# ===========================================================================
# End-to-end ``open_geotiff`` matrix.
#
# Parameter axes:
#   * scenario: ``rotated_no_crs`` | ``rotated_with_crs`` |
#               ``axis_aligned_with_crs``
#   * chunks: None (eager) | 2 (dask)
# ===========================================================================


def _chunks_label(chunks) -> str:
    """Filename-friendly label matching the pytest test ID for ``chunks``.

    Keeps ``tmp_path`` trees readable: ``..._eager.tif`` / ``..._dask.tif``
    instead of ``..._None.tif`` / ``..._2.tif``.
    """
    return "eager" if chunks is None else "dask"


def _build_rotated_no_crs(tmp_path, name):
    src = tmp_path / name
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(str(src), arr)
    return src, arr


def _build_rotated_with_crs(tmp_path, name):
    src = tmp_path / name
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_crs(str(src), arr)
    return src, arr


def _build_axis_aligned_with_crs(tmp_path, name):
    src = tmp_path / name
    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    _write_axis_aligned_with_crs_via_tifffile(src, arr)
    return src, arr


@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_open_geotiff_rotated_no_crs_default_raises(tmp_path, chunks):
    """``allow_rotated=False`` raises on rotated TIFFs without CRS.

    Pinned for both backends so a caller that does not know about the
    opt-out keeps the safety net.
    """
    src, _ = _build_rotated_no_crs(
        tmp_path, f"rotated_no_crs_default_{_chunks_label(chunks)}.tif"
    )
    kwargs = {} if chunks is None else {"chunks": chunks}
    with pytest.raises(RotatedTransformError, match="rotation"):
        open_geotiff(str(src), **kwargs)


@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_open_geotiff_rotated_no_crs_optin_reads_pixels(tmp_path, chunks):
    """``allow_rotated=True`` reads the pixel grid on eager and dask."""
    src, arr = _build_rotated_no_crs(
        tmp_path, f"rotated_no_crs_optin_{_chunks_label(chunks)}.tif"
    )
    kwargs = {"allow_rotated": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    da = open_geotiff(str(src), **kwargs)
    assert da.shape == arr.shape
    materialized = da.compute().values if chunks is not None else da.values
    np.testing.assert_array_equal(materialized, arr)
    # Without a usable georef the reader must not emit a transform that
    # pretends to be axis-aligned.
    assert da.attrs.get('transform') in (None, 'rotated')


@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_open_geotiff_rotated_with_crs_drops_crs(tmp_path, chunks):
    """``allow_rotated=True`` drops ``crs`` / ``crs_wkt`` together with
    the transform when the source carries an embedded CRS."""
    src, arr = _build_rotated_with_crs(
        tmp_path, f"rotated_with_crs_{_chunks_label(chunks)}.tif"
    )
    kwargs = {"allow_rotated": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    da = open_geotiff(str(src), **kwargs)
    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs
    # Pixel grid: integer y / x coords (no-georef contract).
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    if chunks is None:
        np.testing.assert_array_equal(da.values, arr)


@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_open_geotiff_rotated_with_crs_geokey_only_drops_crs(tmp_path, chunks):
    """Tifffile-written rotated GeoTIFF with a Geographic-only GeoKey
    block also drops ``crs`` / ``crs_wkt`` under ``allow_rotated=True``.

    Pre-fix, ``_populate_attrs_from_geo_info`` wrote ``crs`` before
    checking ``has_georef``; the opt-in path dropped the transform but
    surfaced the CRS, which mismatched the docstring and misled callers
    that gated on ``"crs" in da.attrs`` to mean "spatially meaningful".
    """
    src = tmp_path / f"rotated_with_crs_geokey_{_chunks_label(chunks)}.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_geokeys_via_tifffile(str(src), arr, epsg=4326)
    kwargs = {"allow_rotated": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    da = open_geotiff(str(src), **kwargs)
    assert 'crs' not in da.attrs, (
        f"allow_rotated=True must drop attrs['crs']; "
        f"got {da.attrs.get('crs')!r}"
    )
    assert 'crs_wkt' not in da.attrs, (
        f"allow_rotated=True must drop attrs['crs_wkt']; "
        f"got {da.attrs.get('crs_wkt')!r}"
    )
    assert da.attrs.get('transform') is None or da.attrs.get('transform') == 'rotated'


def test_open_geotiff_axis_aligned_with_crs_keeps_crs(tmp_path):
    """Regression guard: non-rotated reads still emit ``crs`` /
    ``crs_wkt`` / ``transform``.

    The CRS-drop gate keys on ``has_georef``, which stays True for an
    axis-aligned ModelTransformation. Without this guard, narrowing the
    CRS emission could regress the common case.
    """
    src, _ = _build_axis_aligned_with_crs(tmp_path, "axis_aligned_with_crs.tif")
    da = open_geotiff(str(src))
    assert da.attrs.get('crs') == 4326
    assert 'crs_wkt' in da.attrs
    assert 'transform' in da.attrs


# ---------------------------------------------------------------------------
# GPU backends. ``open_geotiff(..., gpu=True)`` and the dask+CuPy combo.
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_open_geotiff_rotated_with_crs_drops_crs_gpu(tmp_path, chunks):
    """``allow_rotated=True`` on the GPU eager + dask+CuPy paths drops
    ``crs`` / ``crs_wkt``.

    ``_read_geotiff_gpu`` must forward ``allow_rotated`` when it falls back
    to ``_read_to_array`` for the stripped layout (and the three tiled
    CPU-fallback sites); otherwise the CPU re-read raises. The dask+CuPy
    path routes through the dask backend with ``cupy.asarray`` mapped over
    each block, which forwards ``allow_rotated`` correctly; both are
    pinned here.
    """
    src, _ = _build_rotated_with_crs(
        tmp_path, f"rotated_gpu_{_chunks_label(chunks)}.tif"
    )
    kwargs = {"allow_rotated": True, "gpu": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    da = open_geotiff(str(src), **kwargs)
    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs


# ===========================================================================
# VRT eager + chunked paths.
# ===========================================================================


@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "dask"])
def test_vrt_rotated_with_crs_drops_crs(tmp_path, chunks):
    """VRT eager + chunked paths drop ``crs`` / ``crs_wkt`` /
    ``transform`` on rotated read under ``allow_rotated=True``.

    VRT rotated reads must match the eager non-VRT contract: int64
    pixel coords plus the ``_NO_GEOREF_KEY`` marker, so the writer
    round-trip treats the array as a no-georef pixel grid rather than a
    user-authored integer-coord raster.
    """
    label = _chunks_label(chunks)
    src = tmp_path / f"vrt_src_{label}.tif"
    _write_minimal_aligned_tiff(src)
    vrt = tmp_path / f"vrt_rotated_{label}.vrt"
    _write_rotated_vrt(vrt, src.name)

    kwargs = {"allow_rotated": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    da = open_geotiff(str(vrt), **kwargs)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs, sorted(da.attrs.keys())
    assert da.x.dtype == np.int64, da.x.dtype
    assert da.y.dtype == np.int64, da.y.dtype
    assert da.attrs.get(_NO_GEOREF_KEY) is True, sorted(da.attrs.keys())


def test_vrt_axis_aligned_with_crs_keeps_crs(tmp_path):
    """Regression guard for the VRT path: non-rotated VRTs still emit crs."""
    src = tmp_path / "vrt_src_aligned.tif"
    _write_minimal_aligned_tiff(src)
    vrt = tmp_path / "vrt_aligned.vrt"
    _write_axis_aligned_vrt(vrt, src.name)

    da = open_geotiff(str(vrt))

    assert da.attrs.get('crs') == 4326
    assert 'crs_wkt' in da.attrs
    assert 'transform' in da.attrs
