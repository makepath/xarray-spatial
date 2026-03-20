"""Virtual Raster Table (VRT) reader.

Parses GDAL VRT XML files and assembles a virtual raster from one or
more source GeoTIFF files using windowed reads.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Lazy imports to avoid circular dependency
_DTYPE_MAP = {
    'Byte': np.uint8,
    'UInt16': np.uint16,
    'Int16': np.int16,
    'UInt32': np.uint32,
    'Int32': np.int32,
    'Float32': np.float32,
    'Float64': np.float64,
    'Int8': np.int8,
}


@dataclass
class _Rect:
    """Pixel rectangle: (x_off, y_off, x_size, y_size)."""
    x_off: int
    y_off: int
    x_size: int
    y_size: int


@dataclass
class _Source:
    """A single source region within a VRT band."""
    filename: str
    band: int  # 1-based
    src_rect: _Rect
    dst_rect: _Rect
    nodata: float | None = None
    # ComplexSource extras
    scale: float | None = None
    offset: float | None = None


@dataclass
class _VRTBand:
    """A single band in a VRT dataset."""
    band_num: int  # 1-based
    dtype: np.dtype
    nodata: float | None = None
    sources: list[_Source] = field(default_factory=list)
    color_interp: str | None = None


@dataclass
class VRTDataset:
    """Parsed Virtual Raster Table."""
    width: int
    height: int
    crs_wkt: str | None = None
    geo_transform: tuple | None = None  # (origin_x, res_x, skew_x, origin_y, skew_y, res_y)
    bands: list[_VRTBand] = field(default_factory=list)


def _parse_rect(elem) -> _Rect:
    """Parse a SrcRect or DstRect element."""
    return _Rect(
        x_off=int(float(elem.get('xOff', 0))),
        y_off=int(float(elem.get('yOff', 0))),
        x_size=int(float(elem.get('xSize', 0))),
        y_size=int(float(elem.get('ySize', 0))),
    )


def _text(elem, tag, default=None):
    """Get text content of a child element."""
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def parse_vrt(xml_str: str, vrt_dir: str = '.') -> VRTDataset:
    """Parse a VRT XML string into a VRTDataset.

    Parameters
    ----------
    xml_str : str
        VRT XML content.
    vrt_dir : str
        Directory of the VRT file, for resolving relative source paths.

    Returns
    -------
    VRTDataset
    """
    root = ET.fromstring(xml_str)

    width = int(root.get('rasterXSize', 0))
    height = int(root.get('rasterYSize', 0))

    # CRS
    crs_wkt = _text(root, 'SRS')

    # GeoTransform: "origin_x, res_x, skew_x, origin_y, skew_y, res_y"
    gt_str = _text(root, 'GeoTransform')
    geo_transform = None
    if gt_str:
        parts = [float(x.strip()) for x in gt_str.split(',')]
        if len(parts) == 6:
            geo_transform = tuple(parts)

    # Bands
    bands = []
    for band_elem in root.findall('VRTRasterBand'):
        band_num = int(band_elem.get('band', 1))
        dtype_name = band_elem.get('dataType', 'Float32')
        dtype = np.dtype(_DTYPE_MAP.get(dtype_name, np.float32))
        nodata_str = _text(band_elem, 'NoDataValue')
        nodata = float(nodata_str) if nodata_str else None
        color_interp = _text(band_elem, 'ColorInterp')

        sources = []
        for src_elem in band_elem:
            tag = src_elem.tag
            if tag not in ('SimpleSource', 'ComplexSource'):
                continue

            filename = _text(src_elem, 'SourceFilename') or ''
            relative = src_elem.find('SourceFilename')
            is_relative = (relative is not None and
                           relative.get('relativeToVRT', '0') == '1')
            if is_relative and not os.path.isabs(filename):
                filename = os.path.join(vrt_dir, filename)

            src_band = int(_text(src_elem, 'SourceBand') or '1')

            src_rect_elem = src_elem.find('SrcRect')
            dst_rect_elem = src_elem.find('DstRect')
            if src_rect_elem is None or dst_rect_elem is None:
                continue

            src_rect = _parse_rect(src_rect_elem)
            dst_rect = _parse_rect(dst_rect_elem)

            src_nodata_str = _text(src_elem, 'NODATA')
            src_nodata = float(src_nodata_str) if src_nodata_str else None

            # ComplexSource extras
            scale = None
            offset = None
            if tag == 'ComplexSource':
                scale_str = _text(src_elem, 'ScaleOffset')
                offset_str = _text(src_elem, 'ScaleRatio')
                # Note: GDAL uses ScaleOffset=offset, ScaleRatio=scale
                if offset_str:
                    scale = float(offset_str)
                if scale_str:
                    offset = float(scale_str)

            sources.append(_Source(
                filename=filename,
                band=src_band,
                src_rect=src_rect,
                dst_rect=dst_rect,
                nodata=src_nodata,
                scale=scale,
                offset=offset,
            ))

        bands.append(_VRTBand(
            band_num=band_num,
            dtype=dtype,
            nodata=nodata,
            sources=sources,
            color_interp=color_interp,
        ))

    return VRTDataset(
        width=width,
        height=height,
        crs_wkt=crs_wkt,
        geo_transform=geo_transform,
        bands=bands,
    )


def read_vrt(vrt_path: str, *, window=None,
             band: int | None = None) -> tuple[np.ndarray, VRTDataset]:
    """Read a VRT file by assembling pixel data from its source files.

    Parameters
    ----------
    vrt_path : str
        Path to the .vrt file.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed read.
    band : int or None
        Band index (0-based). None returns all bands.

    Returns
    -------
    (np.ndarray, VRTDataset) tuple
    """
    from ._reader import read_to_array

    with open(vrt_path, 'r') as f:
        xml_str = f.read()

    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    vrt = parse_vrt(xml_str, vrt_dir)

    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(vrt.height, r1)
        c1 = min(vrt.width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, vrt.height, vrt.width

    out_h = r1 - r0
    out_w = c1 - c0

    # Select bands
    if band is not None:
        selected_bands = [vrt.bands[band]]
    else:
        selected_bands = vrt.bands

    # Allocate output
    if len(selected_bands) == 1:
        dtype = selected_bands[0].dtype
        result = np.full((out_h, out_w), np.nan if dtype.kind == 'f' else 0,
                         dtype=dtype)
    else:
        dtype = selected_bands[0].dtype
        result = np.full((out_h, out_w, len(selected_bands)),
                         np.nan if dtype.kind == 'f' else 0, dtype=dtype)

    for band_idx, vrt_band in enumerate(selected_bands):
        nodata = vrt_band.nodata

        for src in vrt_band.sources:
            # Compute overlap between source's destination rect and our window
            dr = src.dst_rect
            sr = src.src_rect

            # Destination rect in virtual raster coordinates
            dst_r0 = dr.y_off
            dst_c0 = dr.x_off
            dst_r1 = dr.y_off + dr.y_size
            dst_c1 = dr.x_off + dr.x_size

            # Clip to window
            clip_r0 = max(dst_r0, r0)
            clip_c0 = max(dst_c0, c0)
            clip_r1 = min(dst_r1, r1)
            clip_c1 = min(dst_c1, c1)

            if clip_r0 >= clip_r1 or clip_c0 >= clip_c1:
                continue  # no overlap

            # Map back to source coordinates
            # Scale factor: source pixels per destination pixel
            scale_y = sr.y_size / dr.y_size if dr.y_size > 0 else 1.0
            scale_x = sr.x_size / dr.x_size if dr.x_size > 0 else 1.0

            src_r0 = sr.y_off + int((clip_r0 - dst_r0) * scale_y)
            src_c0 = sr.x_off + int((clip_c0 - dst_c0) * scale_x)
            src_r1 = sr.y_off + int((clip_r1 - dst_r0) * scale_y)
            src_c1 = sr.x_off + int((clip_c1 - dst_c0) * scale_x)

            # Read from source file using windowed read
            try:
                src_arr, _ = read_to_array(
                    src.filename,
                    window=(src_r0, src_c0, src_r1, src_c1),
                    band=src.band - 1,  # convert 1-based to 0-based
                )
            except Exception:
                continue  # skip missing/unreadable sources

            # Handle source nodata
            src_nodata = src.nodata or nodata
            if src_nodata is not None and src_arr.dtype.kind == 'f':
                src_arr = src_arr.copy()
                src_arr[src_arr == np.float32(src_nodata)] = np.nan

            # Apply ComplexSource scaling
            if src.scale is not None and src.scale != 1.0:
                src_arr = src_arr.astype(np.float64) * src.scale
            if src.offset is not None and src.offset != 0.0:
                src_arr = src_arr.astype(np.float64) + src.offset

            # Place into output
            out_r0 = clip_r0 - r0
            out_c0 = clip_c0 - c0
            out_r1 = out_r0 + src_arr.shape[0]
            out_c1 = out_c0 + src_arr.shape[1]

            # Handle size mismatch from rounding
            actual_h = min(src_arr.shape[0], out_r1 - out_r0)
            actual_w = min(src_arr.shape[1], out_c1 - out_c0)

            if len(selected_bands) == 1:
                result[out_r0:out_r0 + actual_h,
                       out_c0:out_c0 + actual_w] = src_arr[:actual_h, :actual_w]
            else:
                result[out_r0:out_r0 + actual_h,
                       out_c0:out_c0 + actual_w,
                       band_idx] = src_arr[:actual_h, :actual_w]

    return result, vrt
