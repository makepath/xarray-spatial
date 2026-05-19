"""``allow_rotated=True`` drops attrs['crs'] / ['crs_wkt'] (issue #2122).

The ``open_geotiff`` docstring promises that ``allow_rotated=True``
reads a rotated GeoTIFF as a no-georef pixel grid: integer pixel coords
on ``x`` / ``y`` and ``attrs['crs']`` dropped. Before #2122,
``_populate_attrs_from_geo_info`` only gated ``transform`` on the
``has_georef`` flag and still surfaced ``crs`` / ``crs_wkt`` from the
GeoKey block. Downstream code that branches on ``'crs' in da.attrs`` to
detect a projected raster would treat the rotated-read pixel grid as
georeferenced and apply the wrong projection math against integer pixel
indices.

These tests pin the contract on the four GeoTIFF backends and on the
two VRT entry points (eager + chunked).
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._geotags import TAG_MODEL_TRANSFORMATION


# Rotated 4x4 ModelTransformation: pixel_width 1.0, b=0.1 (column-axis
# rotation), pixel_height -1.0, origin (100, 200). Identical structure
# to the fixture in ``test_allow_rotated_geotiff_2115.py`` but with the
# EPSG:4326 GeoKey block appended so the CRS-leak path is exercised.
_ROTATED_M_2122 = (
    1.0, 0.1, 0.0, 100.0,
    0.0, -1.0, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

# Minimal GeoKeyDirectoryTag declaring EPSG:4326.
# Layout: (version, key_revision, minor_revision, n_keys,
#          KeyID, TIFFTagLocation, Count, Value_Offset, ...)
_GEO_KEYS_4326 = (
    1, 1, 0, 3,
    1024, 0, 1, 2,    # GTModelTypeGeoKey = Geographic
    1025, 0, 1, 1,    # GTRasterTypeGeoKey = Area
    2048, 0, 1, 4326, # GeographicTypeGeoKey
)


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _write_rotated_tiff_with_crs(path, arr: np.ndarray) -> None:
    """Write a TIFF with both rotated ModelTransformationTag AND EPSG:4326 GeoKeys.

    Single-strip uncompressed uint16, written by hand so the fixture has
    no rasterio / GDAL dependency. Mirrors the helper in
    ``test_allow_rotated_geotiff_2115.py`` but adds the
    GeoKeyDirectoryTag so the reader populates ``geo_info.crs_epsg``
    and ``geo_info.crs_wkt`` alongside the rotated transform.
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
        (34735, 3, len(_GEO_KEYS_4326), geo_keys_off),  # GeoKeyDirectoryTag
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
        f.write(struct.pack('<16d', *_ROTATED_M_2122))
        f.write(struct.pack(f'<{len(_GEO_KEYS_4326)}H', *_GEO_KEYS_4326))
        f.write(ifd_bytes)


# ---------------------------------------------------------------------------
# GeoTIFF backends.
# ---------------------------------------------------------------------------


def test_eager_rotated_read_drops_crs(tmp_path):
    """``allow_rotated=True`` on the eager numpy path drops ``crs`` / ``crs_wkt``."""
    src = tmp_path / "rotated_2122_eager.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_crs(str(src), arr)

    da = open_geotiff(str(src), allow_rotated=True)

    assert 'crs' not in da.attrs, (
        f"crs should be dropped on rotated read but got "
        f"crs={da.attrs.get('crs')!r}; attrs={sorted(da.attrs.keys())}"
    )
    assert 'crs_wkt' not in da.attrs, (
        f"crs_wkt should be dropped on rotated read; attrs="
        f"{sorted(da.attrs.keys())}"
    )
    assert 'transform' not in da.attrs
    # Pixel grid: integer y/x coords.
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    np.testing.assert_array_equal(da.values, arr)


def test_dask_rotated_read_drops_crs(tmp_path):
    """``allow_rotated=True`` on the dask path drops ``crs`` / ``crs_wkt``."""
    src = tmp_path / "rotated_2122_dask.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_crs(str(src), arr)

    da = open_geotiff(str(src), allow_rotated=True, chunks=2)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64


@_gpu_only
@pytest.mark.xfail(
    reason="read_geotiff_gpu CPU-fallback _read_to_array calls drop "
           "allow_rotated; tracked as a sibling finding to #2122. "
           "Re-enable once the GPU path forwards allow_rotated.",
    raises=NotImplementedError,
    strict=True,
)
def test_cupy_rotated_read_drops_crs(tmp_path):
    """``allow_rotated=True`` on the GPU eager path drops ``crs`` / ``crs_wkt``.

    Currently xfails because ``read_geotiff_gpu`` falls back to
    ``_read_to_array`` for stripped TIFFs without forwarding
    ``allow_rotated``; the CPU re-read then raises
    ``NotImplementedError``. Tracked separately from #2122. Once the
    GPU plumbing is fixed, this xfail will flip to a pass and the
    ``strict=True`` will surface that fact in CI.
    """
    src = tmp_path / "rotated_2122_gpu.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_crs(str(src), arr)

    da = open_geotiff(str(src), allow_rotated=True, gpu=True)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs


@_gpu_only
def test_dask_cupy_rotated_read_drops_crs(tmp_path):
    """``allow_rotated=True`` on the dask+CuPy path drops ``crs`` / ``crs_wkt``.

    ``open_geotiff(..., gpu=True, chunks=...)`` routes through the dask
    backend and then maps ``cupy.asarray`` over each block (see
    ``_backends/gpu.py::_read_geotiff_gpu_chunked``), bypassing the
    eager GPU CPU-fallback paths. The dask path already forwards
    ``allow_rotated`` correctly, so the rotated read succeeds here even
    though the eager GPU path xfails on the sibling finding.
    """
    src = tmp_path / "rotated_2122_dask_gpu.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_with_crs(str(src), arr)

    da = open_geotiff(str(src), allow_rotated=True, gpu=True, chunks=2)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs


def test_axis_aligned_read_still_emits_crs(tmp_path):
    """Regression guard: non-rotated reads still emit ``crs`` / ``crs_wkt``.

    The #2122 gate keys on ``has_georef``, which stays True for an
    axis-aligned ModelTransformation. Without this guard, narrowing the
    CRS emission could regress the common case.
    """
    tifffile = pytest.importorskip("tifffile")

    src = tmp_path / "aligned_2122.tif"
    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    # Axis-aligned ModelTransformation (b = d = 0).
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
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     extratags=extratags)

    da = open_geotiff(str(src))

    assert da.attrs.get('crs') == 4326
    assert 'crs_wkt' in da.attrs
    assert 'transform' in da.attrs


# ---------------------------------------------------------------------------
# VRT backends.
# ---------------------------------------------------------------------------


def _write_minimal_aligned_tiff(path):
    """Tiny axis-aligned TIFF used as the VRT's source band."""
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(str(path), arr, photometric='minisblack')


def _write_rotated_vrt(vrt_path, source_filename):
    """Write a VRT whose GeoTransform carries a non-zero rotation term.

    The CRS block sets EPSG:4326 via the inline WKT so the #2122 gate is
    exercised on the VRT eager + chunked attrs builds.
    """
    wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
        '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",'
        '0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    )
    xml = (
        f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        f'  <SRS>{wkt}</SRS>\n'
        # GDAL ordering: (origin_x, pixel_width, b, origin_y, d, pixel_height)
        # b=0.5 forces rotation.
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


def test_vrt_eager_rotated_read_drops_crs(tmp_path):
    """Eager VRT path drops ``crs`` / ``crs_wkt`` / ``transform`` on
    rotated read under ``allow_rotated=True``."""
    src = tmp_path / "vrt_src_2122.tif"
    _write_minimal_aligned_tiff(src)
    vrt = tmp_path / "rotated_2122.vrt"
    _write_rotated_vrt(vrt, src.name)

    da = open_geotiff(str(vrt), allow_rotated=True)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs, sorted(da.attrs.keys())


def test_vrt_chunked_rotated_read_drops_crs(tmp_path):
    """Chunked VRT path drops ``crs`` / ``crs_wkt`` / ``transform`` on
    rotated read under ``allow_rotated=True``."""
    src = tmp_path / "vrt_src_2122_chunks.tif"
    _write_minimal_aligned_tiff(src)
    vrt = tmp_path / "rotated_2122_chunks.vrt"
    _write_rotated_vrt(vrt, src.name)

    da = open_geotiff(str(vrt), allow_rotated=True, chunks=2)

    assert 'crs' not in da.attrs, sorted(da.attrs.keys())
    assert 'crs_wkt' not in da.attrs, sorted(da.attrs.keys())
    assert 'transform' not in da.attrs, sorted(da.attrs.keys())


def test_vrt_axis_aligned_still_emits_crs(tmp_path):
    """Regression guard for the VRT path: non-rotated VRTs still emit crs."""
    src = tmp_path / "vrt_src_2122_ok.tif"
    _write_minimal_aligned_tiff(src)
    vrt = tmp_path / "aligned_2122.vrt"
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
        f'      <SourceFilename relativeToVRT="1">{src.name}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt, 'w') as f:
        f.write(xml)

    da = open_geotiff(str(vrt))

    assert da.attrs.get('crs') == 4326
    assert 'crs_wkt' in da.attrs
    assert 'transform' in da.attrs
