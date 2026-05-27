"""Coordinate handling on read and on the matching write surface.

This file groups every test that asks "do the y/x coords survive the
round trip?" -- both the reader-side reconstruction from GeoTIFF tags
and the writer-side validation that refuses to write a transform we
cannot represent.

Sections, in read -> write order:

1. ``coords_from_pixel_geometry`` / ``transform_tuple_from_pixel_geometry``
   / ``coords_from_geo_info`` -- the shared GeoTransform-to-(y, x)
   helpers each backend's read path now calls instead of keeping its
   own inline copy.
2. ``_extract_transform`` multi-tiepoint consistency. Single-tiepoint
   files pass; multi-tiepoint files whose tuples agree within tolerance
   pass; inconsistent tuples raise ``NotImplementedError`` naming the
   GCP-warp case.
3. Zero-denominator ``RATIONAL`` / ``SRATIONAL`` tag rejection. The
   reader used to coerce a zero denominator to ``0.0``; it now raises
   ``ValueError`` with the tag name and the denominator in the message.
4. Descending / ascending coord round trip. The writer emits
   ``ModelTransformationTag`` (34264) for non-standard axis directions;
   the reader has to rebuild the original direction from that tag.
5. ``_coords_to_transform`` writer-side validation: uniform-spacing
   check on 1D coords, 3D ``(y, x, band)`` / ``(band, y, x)`` layouts,
   and the alias-aware ``NonUniformCoordsError`` path.
6. Integer-coord round trip and the ``_NO_GEOREF_KEY`` placeholder
   marker. User-authored integer-coord grids must not silently drop
   georef; the reader stamps the marker on legitimate no-georef reads
   and the writer carries it forward.

Tests across the sections share the writer-side helper
``_make_da`` / ``_make_da_uint8`` but no runtime fixtures, because each
section pins a distinct failure mode rather than a shared invariant.
"""
from __future__ import annotations

import io
import os
import struct
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import NonUniformCoordsError, _coords_to_transform, open_geotiff, to_geotiff
from xrspatial.geotiff._coords import (_has_no_georef_marker, coords_from_geo_info,
                                       coords_from_pixel_geometry,
                                       transform_tuple_from_pixel_geometry)
from xrspatial.geotiff._dtypes import RATIONAL, SRATIONAL
from xrspatial.geotiff._geotags import (_NO_GEOREF_KEY, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT,
                                        TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
                                        TAG_MODEL_TRANSFORMATION, GeoTransform, _extract_transform,
                                        _validate_tiepoint_consistency)
from xrspatial.geotiff._header import (IFD, TAG_X_RESOLUTION, TAG_Y_RESOLUTION, IFDEntry,
                                       parse_all_ifds, parse_header)
from xrspatial.geotiff._runtime import _X_DIM_NAMES, _Y_DIM_NAMES, _resolve_spatial_coords

from ..conftest import requires_gpu, requires_integration

# ---------------------------------------------------------------------------
# Shared helpers
#
# ``_make_da`` builds a float32 raster with sequential pixel values; used
# by the round-trip / orientation tests in section 4 that need a writer
# input recognisable by the reader. ``_make_da_uint8`` builds a uint8
# zero raster; used by the section-5 ``_coords_to_transform`` regularity
# checks where only the coord axes matter and the pixel buffer can be
# anything.
# ---------------------------------------------------------------------------


def _ifd_tag_ids(path: str) -> set[int]:
    with open(path, 'rb') as fh:
        data = fh.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return set(ifds[0].entries.keys())


def _make_da(x_coords: np.ndarray, y_coords: np.ndarray) -> xr.DataArray:
    arr = np.arange(len(y_coords) * len(x_coords), dtype=np.float32)
    arr = arr.reshape(len(y_coords), len(x_coords))
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': y_coords, 'x': x_coords},
    )


def _make_da_uint8(x_coords, y_coords):
    arr = np.zeros((len(y_coords), len(x_coords)), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.asarray(y_coords), 'x': np.asarray(x_coords)},
    )


# ===========================================================================
# Section 1 -- coords_from_pixel_geometry / coords_from_geo_info helpers
# ===========================================================================
#
# The shared helpers extracted from ``__init__.py``. Each backend's read
# path calls these instead of keeping its own inline copy of the
# GeoTransform-to-(y, x) maths.


class TestCoordsFromPixelGeometry:
    def test_basic_area_north_up(self):
        # Standard north-up affine: origin at top-left, negative
        # pixel_height. PixelIsArea => coords shift to pixel centers.
        coords = coords_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=3, width=4,
        )
        expected_x = np.array([105.0, 115.0, 125.0, 135.0])
        expected_y = np.array([195.0, 185.0, 175.0])
        np.testing.assert_array_equal(coords['x'], expected_x)
        np.testing.assert_array_equal(coords['y'], expected_y)

    def test_windowed_area(self):
        # Window (r0=1, c0=2, r1=3, c1=5) over a virtual source. The
        # returned coords describe absolute pixel-center positions for
        # rows 1..2 and columns 2..4, not 0..height-1 / 0..width-1.
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=-1.0,
            height=2, width=3,
            window=(1, 2, 3, 5),
        )
        # PixelIsArea adds half-pixel; column 2 center at 2.5, row 1 at -1.5
        np.testing.assert_array_equal(coords['x'], np.array([2.5, 3.5, 4.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-1.5, -2.5]))

    def test_pixel_is_point_skips_half_pixel_shift(self):
        # PixelIsPoint: the tiepoint already sits at the pixel center,
        # so coords come back as origin + n * step with no offset.
        coords_area = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=2, width=2,
            is_point=False,
        )
        coords_point = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=2, width=2,
            is_point=True,
        )
        # Area coords have a +5 / -5 half-pixel shift relative to Point.
        np.testing.assert_array_equal(
            coords_area['x'] - 5.0, coords_point['x'])
        np.testing.assert_array_equal(
            coords_area['y'] + 5.0, coords_point['y'])

    def test_negative_y_resolution_north_up(self):
        # Real GeoTIFFs are normally north-up (origin at top, y decreases
        # with row index). Confirm y[0] > y[-1] and step matches.
        coords = coords_from_pixel_geometry(
            origin_x=500_000.0, origin_y=4_500_000.0,
            pixel_width=30.0, pixel_height=-30.0,
            height=5, width=1,
        )
        assert coords['y'][0] > coords['y'][-1]
        np.testing.assert_allclose(np.diff(coords['y']), -30.0)
        # Half-pixel shift applied for PixelIsArea
        assert coords['y'][0] == pytest.approx(4_500_000.0 - 15.0)

    def test_no_georef_returns_integer_pixel_coords(self):
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=1.0,
            height=3, width=4,
            has_georef=False,
        )
        np.testing.assert_array_equal(coords['x'], np.arange(4, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(3, dtype=np.int64))
        # Integer coords, not float
        assert coords['x'].dtype == np.int64
        assert coords['y'].dtype == np.int64

    def test_no_georef_windowed_returns_integer_window_indices(self):
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=1.0,
            height=2, width=2,
            window=(5, 7, 7, 9),
            has_georef=False,
        )
        np.testing.assert_array_equal(coords['x'], np.array([7, 8]))
        np.testing.assert_array_equal(coords['y'], np.array([5, 6]))
        assert coords['x'].dtype == np.int64


class TestTransformTupleFromPixelGeometry:
    def test_basic_tuple_ordering(self):
        # Rasterio order: (a, 0, c, 0, e, f)
        tup = transform_tuple_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
        )
        assert tup == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)

    def test_windowed_origin_shifts(self):
        # window=(r0, c0, ...) bumps the origin by c0*pixel_width /
        # r0*pixel_height.
        tup = transform_tuple_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
            window=(3, 4, 0, 0),
        )
        assert tup == (10.0, 0.0, 140.0, 0.0, -10.0, 170.0)


class TestCoordsFromGeoInfo:
    def _geo_info(self, *, transform, raster_type, has_georef=True):
        return SimpleNamespace(
            transform=transform,
            raster_type=raster_type,
            has_georef=has_georef,
        )

    def test_area_full_extent(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=0.0, origin_y=0.0,
                pixel_width=1.0, pixel_height=-1.0,
            ),
            raster_type=RASTER_PIXEL_IS_AREA,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.array([0.5, 1.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-0.5, -1.5]))

    def test_windowed(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=0.0, origin_y=0.0,
                pixel_width=1.0, pixel_height=-1.0,
            ),
            raster_type=RASTER_PIXEL_IS_AREA,
        )
        coords = coords_from_geo_info(
            gi, height=2, width=2, window=(3, 4, 5, 6),
        )
        np.testing.assert_array_equal(coords['x'], np.array([4.5, 5.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-3.5, -4.5]))

    def test_pixel_is_point(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=10.0, origin_y=20.0,
                pixel_width=2.0, pixel_height=-2.0,
            ),
            raster_type=RASTER_PIXEL_IS_POINT,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.array([10.0, 12.0]))
        np.testing.assert_array_equal(coords['y'], np.array([20.0, 18.0]))

    def test_no_georef_returns_integer_coords(self):
        gi = self._geo_info(
            transform=GeoTransform(),
            raster_type=RASTER_PIXEL_IS_AREA,
            has_georef=False,
        )
        coords = coords_from_geo_info(gi, height=3, width=3)
        np.testing.assert_array_equal(coords['x'], np.arange(3, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(3, dtype=np.int64))
        assert coords['x'].dtype == np.int64

    def test_none_transform_treated_as_no_georef(self):
        gi = self._geo_info(
            transform=None,
            raster_type=RASTER_PIXEL_IS_AREA,
            has_georef=True,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.arange(2, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(2, dtype=np.int64))


# ===========================================================================
# Section 2 -- Multi-tiepoint consistency in _extract_transform
# ===========================================================================
#
# A ModelTiepointTag may carry one or many (I, J, K, X, Y, Z) tuples.
# Slicing only tiepoint[0:6] silently drops the rest, which produces
# wrong coordinates on GCP-warped files.


# A simple axis-aligned affine: origin (100, 200), pixel size 10 in both axes.
# Pixel (i, j) maps to world (100 + 10*i, 200 - 10*j).
_TP_SX = 10.0
_TP_SY = 10.0
_TP_ORIGIN_X = 100.0
_TP_ORIGIN_Y = 200.0


def _world_at(i: float, j: float) -> tuple[float, float]:
    return (_TP_ORIGIN_X + i * _TP_SX, _TP_ORIGIN_Y - j * _TP_SY)


def _make_tiepoint_ifd(tiepoint: tuple,
                       scale: tuple | None = (10.0, 10.0, 0.0)) -> IFD:
    ifd = IFD()
    ifd.entries[TAG_MODEL_TIEPOINT] = IFDEntry(
        tag=TAG_MODEL_TIEPOINT, type_id=12,
        count=len(tiepoint), value=tiepoint,
    )
    if scale is not None:
        ifd.entries[TAG_MODEL_PIXEL_SCALE] = IFDEntry(
            tag=TAG_MODEL_PIXEL_SCALE, type_id=12,
            count=len(scale), value=scale,
        )
    return ifd


class TestMultiTiepointValidation:
    def test_single_tiepoint_unchanged(self):
        ifd = _make_tiepoint_ifd(
            (0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0)
        )
        gt, has_georef = _extract_transform(ifd)
        assert has_georef is True
        assert gt.origin_x == _TP_ORIGIN_X
        assert gt.origin_y == _TP_ORIGIN_Y
        assert gt.pixel_width == _TP_SX
        assert gt.pixel_height == -_TP_SY

    def test_multiple_consistent_tiepoints_pass(self):
        # Four corners of a 100x100 raster, all consistent with the same affine.
        corners = []
        for i, j in [(0, 0), (100, 0), (0, 100), (100, 100)]:
            wx, wy = _world_at(i, j)
            corners.extend([float(i), float(j), 0.0, wx, wy, 0.0])
        ifd = _make_tiepoint_ifd(tuple(corners))
        gt, has_georef = _extract_transform(ifd)
        assert has_georef is True
        assert gt.origin_x == pytest.approx(_TP_ORIGIN_X)
        assert gt.origin_y == pytest.approx(_TP_ORIGIN_Y)
        assert gt.pixel_width == pytest.approx(_TP_SX)
        assert gt.pixel_height == pytest.approx(-_TP_SY)

    def test_inconsistent_tiepoints_raise(self):
        # Second tuple disagrees by a full pixel: that is a GCP warp.
        tiepoint = (
            0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0,
            100.0, 0.0, 0.0,
            _TP_ORIGIN_X + 100 * _TP_SX + 5.0, _TP_ORIGIN_Y, 0.0,
        )
        ifd = _make_tiepoint_ifd(tiepoint)
        with pytest.raises(NotImplementedError, match="ground-control-point"):
            _extract_transform(ifd)

    def test_tolerance_scales_with_pixel_size(self):
        # A 1e-7 residual on a pixel_size=10 file is below tolerance.
        tiny_resid = 1e-7
        tiepoint = (
            0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0,
            100.0, 0.0, 0.0,
            _TP_ORIGIN_X + 100 * _TP_SX + tiny_resid, _TP_ORIGIN_Y, 0.0,
        )
        ifd = _make_tiepoint_ifd(tiepoint)
        # Should not raise.
        _extract_transform(ifd)

    def test_validate_helper_no_op_for_single_tuple(self):
        # 6 elements -> n == 1; nothing to validate.
        _validate_tiepoint_consistency(
            (0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0),
            _TP_ORIGIN_X, _TP_ORIGIN_Y, _TP_SX, _TP_SY,
        )

    def test_validate_helper_rejects_disagreement(self):
        tiepoint = (
            0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0,
            50.0, 0.0, 0.0,
            _TP_ORIGIN_X + 50 * _TP_SX + 100.0, _TP_ORIGIN_Y, 0.0,
        )
        with pytest.raises(NotImplementedError, match="tuple 1"):
            _validate_tiepoint_consistency(
                tiepoint, _TP_ORIGIN_X, _TP_ORIGIN_Y, _TP_SX, _TP_SY,
            )

    def test_validate_helper_y_axis_sign(self):
        # Verify the y-axis sign convention: predicted_y = origin_y - j * sy.
        # A consistent tuple at (i=0, j=100) is (origin_x, origin_y - 100 * sy).
        tp_world_y = _TP_ORIGIN_Y - 100.0 * _TP_SY
        tiepoint = (
            0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0,
            0.0, 100.0, 0.0, _TP_ORIGIN_X, tp_world_y, 0.0,
        )
        _validate_tiepoint_consistency(
            tiepoint, _TP_ORIGIN_X, _TP_ORIGIN_Y, _TP_SX, _TP_SY,
        )

    def test_tiepoint_without_scale_also_validates(self):
        # When ModelPixelScale is absent, the reader falls back to unit pixel
        # size; the consistency check must still fire, and the error message
        # must blame the missing ModelPixelScale tag (not the GCP-warp case),
        # since a real multi-tiepoint file without ModelPixelScale is almost
        # certainly malformed rather than a deliberate GCP set.
        tiepoint = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 50.0, 0.0, 0.0,  # predicts x=10.0, declares 50.0
        )
        ifd = _make_tiepoint_ifd(tiepoint, scale=None)
        with pytest.raises(NotImplementedError, match="no ModelPixelScale"):
            _extract_transform(ifd)

    def test_validate_helper_honours_custom_rel_tol(self):
        # A residual that passes the default 1e-6 * pixel_size tolerance
        # (= 1e-5 here) can still be caught by a tighter caller-supplied
        # rel_tol. Surveying / high-precision geodetic callers that want to
        # flag near-affine GCP files can pass a smaller rel_tol.
        residual = 5e-6  # below default tol (1e-5) but above tight tol (1e-7)
        tiepoint = (
            0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0,
            100.0, 0.0, 0.0,
            _TP_ORIGIN_X + 100 * _TP_SX + residual, _TP_ORIGIN_Y, 0.0,
        )
        # Default tolerance accepts it.
        _validate_tiepoint_consistency(
            tiepoint, _TP_ORIGIN_X, _TP_ORIGIN_Y, _TP_SX, _TP_SY,
        )
        # Tighter tolerance rejects it.
        with pytest.raises(NotImplementedError, match="tuple 1"):
            _validate_tiepoint_consistency(
                tiepoint, _TP_ORIGIN_X, _TP_ORIGIN_Y, _TP_SX, _TP_SY,
                rel_tol=1e-8,
            )

    def test_short_tiepoint_is_treated_as_single_tuple(self):
        # A truncated tiepoint with fewer than 12 elements has n == 1
        # (truncated second tuple is dropped by integer division). The
        # reader should not crash; it falls back to the existing
        # single-tuple semantics.
        tiepoint = (0.0, 0.0, 0.0, _TP_ORIGIN_X, _TP_ORIGIN_Y, 0.0, 1.0)
        ifd = _make_tiepoint_ifd(tiepoint)
        gt, has_georef = _extract_transform(ifd)
        assert has_georef is True
        assert gt.origin_x == _TP_ORIGIN_X


# ===========================================================================
# Section 3 -- Zero-denominator RATIONAL / SRATIONAL rejection
# ===========================================================================
#
# A RATIONAL or SRATIONAL tag with a zero denominator is malformed by the
# TIFF spec. Rather than coerce it to 0.0 silently, the reader raises
# ValueError naming the tag and value.


def _build_tiff_with_malformed_resolution(numerator: int, denominator: int,
                                          *, which: int = TAG_X_RESOLUTION,
                                          srational: bool = False) -> bytes:
    """Build a minimal little-endian TIFF whose ``which`` resolution tag
    is a single RATIONAL (or SRATIONAL) pointing at ``(numerator,
    denominator)``.

    The other resolution tag is filled with a valid 72/1 so the IFD is
    only malformed in the one place the test cares about.
    """
    if which not in (TAG_X_RESOLUTION, TAG_Y_RESOLUTION):
        raise ValueError(
            f"which must be TAG_X_RESOLUTION or TAG_Y_RESOLUTION, got {which}"
        )

    bo = '<'
    rat_type = SRATIONAL if srational else RATIONAL
    rat_fmt = f'{bo}ii' if srational else f'{bo}II'

    if which == TAG_X_RESOLUTION:
        xres = struct.pack(rat_fmt, numerator, denominator)
        yres = struct.pack(rat_fmt, 72, 1)
    else:
        xres = struct.pack(rat_fmt, 72, 1)
        yres = struct.pack(rat_fmt, numerator, denominator)

    out = bytearray()
    # Header: little-endian, classic TIFF, first IFD at offset 8.
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', 8))

    tags = [
        # (tag, type_id, count, raw_bytes)
        (256, 3, 1, struct.pack(f'{bo}H', 4)),    # ImageWidth
        (257, 3, 1, struct.pack(f'{bo}H', 4)),    # ImageLength
        (258, 3, 1, struct.pack(f'{bo}H', 8)),    # BitsPerSample
        (259, 3, 1, struct.pack(f'{bo}H', 1)),    # Compression
        (262, 3, 1, struct.pack(f'{bo}H', 1)),    # PhotometricInterpretation
        (273, 4, 1, b'\x00\x00\x00\x00'),         # StripOffsets (patched)
        (277, 3, 1, struct.pack(f'{bo}H', 1)),    # SamplesPerPixel
        (278, 3, 1, struct.pack(f'{bo}H', 4)),    # RowsPerStrip
        (279, 4, 1, struct.pack(f'{bo}I', 16)),   # StripByteCounts
        (TAG_X_RESOLUTION, rat_type, 1, xres),
        (TAG_Y_RESOLUTION, rat_type, 1, yres),
    ]
    tags.sort(key=lambda t: t[0])

    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, typ, count, raw in tags:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)
    patched = []
    for tag, typ, count, raw in tags:
        if tag == 273:
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tags = patched

    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tags:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    while len(out) < pixel_data_start:
        out.append(0)
    out.extend(b'\x00' * 16)

    return bytes(out)


class TestRationalZeroDenominator:
    """Zero-denominator rationals must fail loudly."""

    def test_rational_zero_denominator_surfaces_from_parse_all_ifds(self):
        data = _build_tiff_with_malformed_resolution(72, 0)
        header = parse_header(data)
        with pytest.raises(ValueError, match="XResolution"):
            parse_all_ifds(data, header)

    def test_rational_zero_denominator_message_includes_denominator(self):
        data = _build_tiff_with_malformed_resolution(150, 0)
        header = parse_header(data)
        with pytest.raises(ValueError) as exc:
            parse_all_ifds(data, header)
        message = str(exc.value)
        assert "Malformed RATIONAL" in message
        assert "XResolution" in message
        assert "denominator=0" in message
        assert "numerator=150" in message

    def test_srational_zero_denominator_surfaces_from_parse_all_ifds(self):
        data = _build_tiff_with_malformed_resolution(-5, 0, srational=True)
        header = parse_header(data)
        with pytest.raises(ValueError, match="Malformed SRATIONAL"):
            parse_all_ifds(data, header)

    def test_rational_zero_denominator_fails_open_geotiff(self):
        # The public read entry point should fail loudly too, not just
        # the low-level header parser.
        data = _build_tiff_with_malformed_resolution(72, 0)
        buf = io.BytesIO(data)
        with pytest.raises(ValueError, match="XResolution"):
            open_geotiff(buf)

    def test_yresolution_zero_denominator_named_in_error(self):
        # Same path, different tag.
        data = _build_tiff_with_malformed_resolution(
            72, 0, which=TAG_Y_RESOLUTION
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="YResolution"):
            parse_all_ifds(data, header)

    def test_tag_constants_present(self):
        # Sanity check: the helpers we use to assert tag ids actually exist.
        assert TAG_X_RESOLUTION == 282
        assert TAG_Y_RESOLUTION == 283


# ===========================================================================
# Section 4 -- Descending / ascending coord round trip
# ===========================================================================
#
# The writer emits ModelTransformationTag (34264) when the axis direction
# is non-standard. The reader has to rebuild the original direction from
# that tag, so the round trip checks both halves at once.


class TestDescendingCoordsRoundTrip:
    """Round-trip read of non-standard-orientation rasters."""

    def test_descending_x_roundtrip(self, tmp_path):
        """Descending x coords survive the round trip."""
        # x decreases left-to-right (unusual but valid)
        x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
        y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)  # north-up
        da = _make_da(x, y)

        out = tmp_path / 'desc_x.tif'
        to_geotiff(da, str(out), crs=4326)

        loaded = open_geotiff(str(out))
        np.testing.assert_allclose(loaded.coords['x'].values, x)
        np.testing.assert_allclose(loaded.coords['y'].values, y)
        np.testing.assert_array_equal(loaded.values, da.values)

    def test_ascending_y_roundtrip(self, tmp_path):
        """Ascending y coords survive the round trip."""
        x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
        # y increases top-to-bottom (south-up)
        y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        da = _make_da(x, y)

        out = tmp_path / 'asc_y.tif'
        to_geotiff(da, str(out), crs=4326)

        loaded = open_geotiff(str(out))
        np.testing.assert_allclose(loaded.coords['x'].values, x)
        np.testing.assert_allclose(loaded.coords['y'].values, y)
        np.testing.assert_array_equal(loaded.values, da.values)

    def test_descending_x_and_ascending_y_roundtrip(self, tmp_path):
        """Both axes flipped relative to north-up."""
        x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
        y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        da = _make_da(x, y)

        out = tmp_path / 'desc_x_asc_y.tif'
        to_geotiff(da, str(out), crs=4326)

        loaded = open_geotiff(str(out))
        np.testing.assert_allclose(loaded.coords['x'].values, x)
        np.testing.assert_allclose(loaded.coords['y'].values, y)
        np.testing.assert_array_equal(loaded.values, da.values)


class TestOrientationTagSelection:
    """The writer picks the right tags for the orientation; the reader
    has to be able to read either flavour."""

    def test_north_up_uses_pixel_scale_and_tiepoint(self, tmp_path):
        """North-up keeps ModelPixelScale + ModelTiepoint (no transformation)."""
        x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
        y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)
        da = _make_da(x, y)

        out = tmp_path / 'north_up.tif'
        to_geotiff(da, str(out), crs=4326)

        tag_ids = _ifd_tag_ids(str(out))
        assert TAG_MODEL_PIXEL_SCALE in tag_ids
        assert TAG_MODEL_TIEPOINT in tag_ids
        assert TAG_MODEL_TRANSFORMATION not in tag_ids

    def test_descending_x_uses_transformation_tag(self, tmp_path):
        """Non-standard orientation emits ModelTransformationTag."""
        x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
        y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)
        da = _make_da(x, y)

        out = tmp_path / 'desc_x_tags.tif'
        to_geotiff(da, str(out), crs=4326)

        tag_ids = _ifd_tag_ids(str(out))
        assert TAG_MODEL_TRANSFORMATION in tag_ids
        assert TAG_MODEL_PIXEL_SCALE not in tag_ids
        assert TAG_MODEL_TIEPOINT not in tag_ids

    def test_ascending_y_uses_transformation_tag(self, tmp_path):
        x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
        y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        da = _make_da(x, y)

        out = tmp_path / 'asc_y_tags.tif'
        to_geotiff(da, str(out), crs=4326)

        tag_ids = _ifd_tag_ids(str(out))
        assert TAG_MODEL_TRANSFORMATION in tag_ids
        assert TAG_MODEL_PIXEL_SCALE not in tag_ids
        assert TAG_MODEL_TIEPOINT not in tag_ids


# ===========================================================================
# Section 5 -- _coords_to_transform writer-side validation
# ===========================================================================
#
# Regularity check on 1D coords.
# 3D (y, x, band) / (band, y, x) layout handling.
# Alias-aware NonUniformCoordsError on every documented spatial alias
# (y/x, lat/lon, latitude/longitude, row/col).


class TestCoordsToTransformRegularity:
    """1D coord uniformity check."""

    def test_uniform_coords_ok(self):
        """Uniform coords write successfully (no regression)."""
        da = _make_da_uint8(
            np.linspace(500.0, 700.0, 20),
            np.linspace(100.0, 200.0, 10),
        )
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
        np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)

    def test_uniform_coords_roundtrip_to_geotiff(self, tmp_path):
        """End-to-end write succeeds on uniform coords."""
        da = _make_da_uint8(
            np.linspace(500.0, 700.0, 20),
            np.linspace(100.0, 200.0, 10),
        )
        to_geotiff(da, str(tmp_path / 'uniform.tif'))

    def test_non_uniform_x_raises(self):
        """Non-uniform x coords raise ValueError naming x."""
        # Mostly-uniform x with one stretched gap to expose the bug.
        x = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        y = np.linspace(0.0, 10.0, 11)
        da = _make_da_uint8(x, y)
        with pytest.raises(ValueError,
                           match=r"\bx coords are not uniformly spaced"):
            _coords_to_transform(da)

    def test_non_uniform_y_raises(self):
        """Non-uniform y coords raise ValueError naming y."""
        x = np.linspace(0.0, 10.0, 11)
        y = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        da = _make_da_uint8(x, y)
        with pytest.raises(ValueError,
                           match=r"\by coords are not uniformly spaced"):
            _coords_to_transform(da)

    def test_jitter_within_tolerance_ok(self):
        """Float jitter within 1e-6 relative tolerance writes successfully."""
        x = np.linspace(0.0, 100.0, 11)
        # Add jitter at ~1e-8 relative scale -- well below the 1e-6 threshold.
        rng = np.random.default_rng(0)
        x = x + rng.uniform(-1e-7, 1e-7, size=x.shape)
        y = np.linspace(0.0, 50.0, 6)
        da = _make_da_uint8(x, y)
        gt = _coords_to_transform(da)
        assert gt is not None

    def test_jitter_just_above_tolerance_raises(self):
        """Jitter just above the 1e-6 relative tolerance raises ValueError."""
        # Step size is 10; one diff is 10 + 1e-4 -> relative deviation ~1e-5,
        # which exceeds the 1e-6 threshold.
        x = np.array([0.0, 10.0, 20.0001, 30.0001, 40.0001])
        y = np.linspace(0.0, 50.0, 6)
        da = _make_da_uint8(x, y)
        with pytest.raises(ValueError, match=r"max relative deviation"):
            _coords_to_transform(da)

    def test_two_sample_coords_ok(self):
        """Two-sample coords (one diff) trivially pass the regularity check."""
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 5.0])
        da = _make_da_uint8(x, y)
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, 10.0)
        np.testing.assert_allclose(gt.pixel_height, 5.0)

    def test_constant_coords_raises(self):
        """Constant coords (step == 0) raise ValueError."""
        x = np.array([0.0, 0.0, 0.0, 0.0])
        y = np.linspace(0.0, 10.0, 4)
        da = _make_da_uint8(x, y)
        with pytest.raises(ValueError, match=r"x coords are constant"):
            _coords_to_transform(da)


def _make_geo_da_3d(dims):
    """3D DataArray with georeferenced y/x coords and a band axis."""
    shape = []
    for d in dims:
        if d in ('y',):
            shape.append(10)
        elif d in ('x',):
            shape.append(20)
        else:
            shape.append(3)
    arr = np.arange(int(np.prod(shape)), dtype=np.uint8).reshape(shape)
    coords = {
        'y': np.linspace(100.0, 200.0, 10),
        'x': np.linspace(500.0, 700.0, 20),
        'band': np.arange(3),
    }
    return xr.DataArray(arr, dims=list(dims), coords=coords)


class TestCoordsToTransform3D:
    """3D (y, x, band) and (band, y, x) layouts pick y/x spacing."""

    def test_yxband_returns_yx_spacing(self):
        """3D (y, x, band) picks y/x spacing rather than (x, band) spacing."""
        da = _make_geo_da_3d(('y', 'x', 'band'))
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
        np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)

    def test_bandyx_returns_yx_spacing(self):
        """3D (band, y, x) also returns the y/x transform."""
        da = _make_geo_da_3d(('band', 'y', 'x'))
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
        np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)

    @pytest.mark.parametrize('band_name', ['band', 'bands', 'channel'])
    def test_3d_band_name_variants(self, band_name):
        """All recognized band-dim names (band, bands, channel) are filtered
        out when picking the y/x spatial dims."""
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        da = xr.DataArray(
            arr,
            dims=['y', 'x', band_name],
            coords={
                'y': np.linspace(100.0, 200.0, 10),
                'x': np.linspace(500.0, 700.0, 20),
                band_name: np.arange(3),
            },
        )
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
        np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)

    def test_2d_unchanged(self):
        """2D (y, x) keeps its original behaviour."""
        da = xr.DataArray(
            np.zeros((10, 20), dtype=np.uint8),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(100.0, 200.0, 10),
                'x': np.linspace(500.0, 700.0, 20),
            },
        )
        gt = _coords_to_transform(da)
        assert gt is not None
        np.testing.assert_allclose(gt.pixel_width, (700.0 - 500.0) / 19)
        np.testing.assert_allclose(gt.pixel_height, (200.0 - 100.0) / 9)

    def test_to_geotiff_roundtrip_3d_yxband(self, tmp_path):
        """to_geotiff -> open_geotiff round-trip on 3D arrays preserves coords.

        Before the fix the on-disk transform was derived from (x, band)
        spacing, so the round-tripped y/x coords had wrong pixel size and
        origin. After the fix the 3D output matches the 2D output.
        """
        da_3d = _make_geo_da_3d(('y', 'x', 'band'))
        da_2d = xr.DataArray(
            np.zeros((10, 20), dtype=np.uint8),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(100.0, 200.0, 10),
                'x': np.linspace(500.0, 700.0, 20),
            },
        )

        p2 = str(tmp_path / 'roundtrip_3d_2d.tif')
        p3 = str(tmp_path / 'roundtrip_3d_yxband.tif')
        to_geotiff(da_2d, p2)
        to_geotiff(da_3d, p3)

        rt2 = open_geotiff(p2)
        rt3 = open_geotiff(p3)
        np.testing.assert_allclose(rt3.y.values, rt2.y.values)
        np.testing.assert_allclose(rt3.x.values, rt2.x.values)
        assert rt3.attrs.get('transform') == rt2.attrs.get('transform')

    def test_to_geotiff_roundtrip_3d_bandyx(self, tmp_path):
        """(band, y, x) input round-trips with the correct transform.

        ``to_geotiff`` remaps a (band, y, x) input to (y, x, band) before
        writing, but ``_coords_to_transform`` runs against the original
        dim order. The fix handles both 3D layouts.
        """
        da_3d = _make_geo_da_3d(('band', 'y', 'x'))
        da_2d = xr.DataArray(
            np.zeros((10, 20), dtype=np.uint8),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(100.0, 200.0, 10),
                'x': np.linspace(500.0, 700.0, 20),
            },
        )

        p2 = str(tmp_path / 'roundtrip_3d_2d_b.tif')
        p3 = str(tmp_path / 'roundtrip_3d_bandfirst.tif')
        to_geotiff(da_2d, p2)
        to_geotiff(da_3d, p3)

        rt2 = open_geotiff(p2)
        rt3 = open_geotiff(p3)
        np.testing.assert_allclose(rt3.y.values, rt2.y.values)
        np.testing.assert_allclose(rt3.x.values, rt2.x.values)

    def test_to_geotiff_3d_does_not_invent_unit_pixels(self, tmp_path):
        """Regression sanity: the bad transform was pixel_width=1.0 (band
        axis spacing). Assert the round-tripped pixel_width is finite,
        non-unit, and matches the source x spacing.
        """
        da = _make_geo_da_3d(('y', 'x', 'band'))
        p = str(tmp_path / 'roundtrip_3d_not_unit.tif')
        to_geotiff(da, p)
        rt = open_geotiff(p)
        pw = abs(float(rt.x.values[1] - rt.x.values[0]))
        # Source x spacing is (700-500)/19 = ~10.526. The buggy path would
        # have produced pw=1.0 (the band axis spacing).
        assert pw > 1.5, (
            f"round-tripped pixel_width={pw} suggests the band-axis spacing "
            f"leaked into the GeoTransform; expected ~10.526")

    @requires_gpu
    def test_write_geotiff_gpu_roundtrip_3d(self, tmp_path):
        """GPU writer shares ``_coords_to_transform`` with the CPU writer.

        Same regression on the GPU path: a 3D ``(y, x, band)`` cupy
        DataArray without ``attrs['transform']`` would previously
        round-trip through a unit pixel-width transform.
        """
        import cupy as cp

        from xrspatial.geotiff import write_geotiff_gpu

        np_arr = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
        da = xr.DataArray(
            cp.asarray(np_arr),
            dims=['y', 'x', 'band'],
            coords={
                'y': np.linspace(100.0, 200.0, 10),
                'x': np.linspace(500.0, 700.0, 20),
                'band': np.arange(3),
            },
        )
        p = str(tmp_path / 'roundtrip_3d_gpu.tif')
        write_geotiff_gpu(da, p)
        rt = open_geotiff(p)
        pw = abs(float(rt.x.values[1] - rt.x.values[0]))
        assert pw > 1.5, (
            f"GPU writer round-tripped pixel_width={pw}; expected ~10.526")
        ph = abs(float(rt.y.values[1] - rt.y.values[0]))
        assert ph > 1.5, (
            f"GPU writer round-tripped pixel_height={ph}; expected ~11.111")


def _da_with_alias_coords(y_name, x_name, *, y_coord=None, x_coord=None,
                          shape=(4, 4)):
    """Build a 2-D DataArray with alias-named y/x dims and coords."""
    data = np.zeros(shape, dtype=np.float32)
    if y_coord is None:
        y_coord = np.linspace(3.0, 0.0, shape[0], dtype=np.float64)
    if x_coord is None:
        x_coord = np.linspace(0.0, 3.0, shape[1], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=(y_name, x_name),
        coords={y_name: y_coord, x_name: x_coord},
    )


_ALIAS_PAIRS = [
    ('y', 'x'),                  # canonical
    ('lat', 'lon'),
    ('latitude', 'longitude'),
    ('row', 'col'),
]


class TestNonUniformCoordsAliasResolution:
    """``_resolve_spatial_coords`` picks the right coord arrays."""

    @pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
    def test_resolve_spatial_coords_finds_alias(self, y_name, x_name):
        """Each documented alias resolves to the matching coord array."""
        da = _da_with_alias_coords(y_name, x_name)
        coord_y, coord_x = _resolve_spatial_coords(da)
        assert coord_y is not None
        assert coord_x is not None
        np.testing.assert_array_equal(coord_y, da.coords[y_name].values)
        np.testing.assert_array_equal(coord_x, da.coords[x_name].values)

    def test_resolve_spatial_coords_picks_canonical_first(self):
        """When both ``y`` and an alias exist, canonical wins.

        The alias list places ``y`` / ``x`` first so an array that
        happens to carry both names (rare, but possible after a rename
        + retain) keeps matching exactly the canonical coord.
        """
        data = np.zeros((4, 4), dtype=np.float32)
        y_arr = np.linspace(3.0, 0.0, 4, dtype=np.float64)
        lat_arr = np.array([99.0, 88.0, 77.0, 66.0], dtype=np.float64)
        x_arr = np.linspace(0.0, 3.0, 4, dtype=np.float64)
        da = xr.DataArray(
            data,
            dims=('y', 'x'),
            coords={'y': y_arr, 'x': x_arr, 'lat': ('y', lat_arr)},
        )
        coord_y, _ = _resolve_spatial_coords(da)
        np.testing.assert_array_equal(coord_y, y_arr)

    def test_resolve_spatial_coords_missing_returns_none(self):
        """No matching coord on either axis returns ``(None, None)``."""
        data = np.zeros((4, 4), dtype=np.float32)
        da = xr.DataArray(data, dims=('foo', 'bar'))
        coord_y, coord_x = _resolve_spatial_coords(da)
        assert coord_y is None
        assert coord_x is None

    def test_resolve_spatial_coords_handles_none_input(self):
        """Passing an object with no ``coords`` attribute returns ``(None, None)``."""
        coord_y, coord_x = _resolve_spatial_coords(object())
        assert coord_y is None
        assert coord_x is None


class TestNonUniformCoordsAlias:
    """Non-uniform alias coords raise NonUniformCoordsError, not plain
    ``ValueError``."""

    @pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
    def test_non_uniform_y_alias_raises_typed(self, tmp_path, y_name, x_name):
        """Non-uniform y-axis coords trip ``NonUniformCoordsError`` for every alias.

        Without the fix, only ``y_name == 'y'`` raised the typed error;
        alias names slipped past the validator and surfaced a plain
        ``ValueError`` from the later transform-synthesis path.
        """
        da = _da_with_alias_coords(
            y_name, x_name,
            y_coord=np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
        )
        with pytest.raises(NonUniformCoordsError) as exc_info:
            to_geotiff(da, str(tmp_path / f'non_uniform_{y_name}.tif'))
        # The user-facing contract: ``isinstance(exc, NonUniformCoordsError)``
        # holds regardless of which alias was used.
        assert isinstance(exc_info.value, NonUniformCoordsError)
        # And the legacy ``except ValueError`` clause still catches it,
        # because NonUniformCoordsError subclasses ValueError.
        assert isinstance(exc_info.value, ValueError)

    @pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
    def test_non_uniform_x_alias_raises_typed(self, tmp_path, y_name, x_name):
        """Non-uniform x-axis coords trip ``NonUniformCoordsError`` for every alias."""
        da = _da_with_alias_coords(
            y_name, x_name,
            x_coord=np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64),
        )
        with pytest.raises(NonUniformCoordsError) as exc_info:
            to_geotiff(da, str(tmp_path / f'non_uniform_{x_name}.tif'))
        assert isinstance(exc_info.value, NonUniformCoordsError)
        assert isinstance(exc_info.value, ValueError)

    @pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
    def test_constant_y_alias_raises_typed(self, tmp_path, y_name, x_name):
        """Constant (zero-step) y-axis coords raise the typed error for every alias."""
        da = _da_with_alias_coords(
            y_name, x_name,
            y_coord=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        )
        with pytest.raises(NonUniformCoordsError) as exc_info:
            to_geotiff(da, str(tmp_path / f'constant_{y_name}.tif'))
        assert isinstance(exc_info.value, NonUniformCoordsError)

    @pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
    def test_uniform_alias_coords_write_successfully(self, tmp_path,
                                                     y_name, x_name):
        """Alias-named coords with uniform spacing keep writing cleanly."""
        da = _da_with_alias_coords(y_name, x_name)
        out = tmp_path / f'uniform_{y_name}_{x_name}.tif'
        to_geotiff(da, str(out))
        assert out.exists()

    def test_alias_pairs_cover_every_documented_name(self):
        """Pin that the parametrization above covers every documented alias.

        If a new alias is added to ``_Y_DIM_NAMES`` / ``_X_DIM_NAMES``,
        this assertion fails and forces the parametrization to be
        updated so the consistency guarantee actually holds for the new
        name.
        """
        parametrized_y = {pair[0] for pair in _ALIAS_PAIRS}
        parametrized_x = {pair[1] for pair in _ALIAS_PAIRS}
        assert parametrized_y == set(_Y_DIM_NAMES), (
            f"Y alias coverage drift: parametrized={parametrized_y}, "
            f"_Y_DIM_NAMES={set(_Y_DIM_NAMES)}"
        )
        assert parametrized_x == set(_X_DIM_NAMES), (
            f"X alias coverage drift: parametrized={parametrized_x}, "
            f"_X_DIM_NAMES={set(_X_DIM_NAMES)}"
        )

    def test_legacy_except_value_error_still_catches(self, tmp_path):
        """Callers using ``except ValueError`` keep working on the typed path.

        ``NonUniformCoordsError`` subclasses ``ValueError`` (via
        ``GeoTIFFAmbiguousMetadataError``), so the legacy try/except shape
        keeps working even though the concrete type changed.
        """
        da = _da_with_alias_coords(
            'y', 'x',
            y_coord=np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
        )
        try:
            to_geotiff(da, str(tmp_path / 'legacy_except.tif'))
        except ValueError as exc:
            assert isinstance(exc, NonUniformCoordsError)
        else:  # pragma: no cover - defensive
            pytest.fail("expected ValueError (NonUniformCoordsError)")


# ===========================================================================
# Section 6 -- Integer-coord round trip and the _NO_GEOREF_KEY marker
# ===========================================================================
#
# User-authored integer spatial coords must not silently drop georef.
# Treating any int64 ascending step-1 grid as the no-georef placeholder
# is wrong; the marker-based predicate ``_has_no_georef_marker`` replaces
# shape-based sniffing.
#
# The integration-marked test covers the integer-coord round-trip
# interaction with the ``require_transform_for_georeferenced`` guard. It
# only runs under the integration marker because it writes to the OS temp
# dir.


def _arange_int64_shape(coord: np.ndarray) -> bool:
    """Test-local predicate matching the read-side placeholder shape.

    ``coords_from_pixel_geometry`` emits ``np.arange(start, stop,
    dtype=np.int64)`` for the y/x coords whenever the source file
    carries no transform tags -- both for full reads (``start=0``) and
    windowed reads (``start=window_offset``). This helper exists only
    so a few legacy round-trip assertions can verify the on-disk shape
    came back unchanged; it is not the production no-georef signal.
    """
    if coord.dtype != np.int64:
        return False
    n = len(coord)
    if n < 1:
        return False
    return bool(np.array_equal(
        coord, np.arange(coord[0], coord[0] + n, dtype=np.int64)
    ))


class TestNoGeorefMarkerPredicate:
    """``_has_no_georef_marker`` is an identity-True check on the
    attribute, not a coord-shape heuristic."""

    @pytest.mark.parametrize(
        "attrs,expected",
        [
            ({_NO_GEOREF_KEY: True}, True),
            ({}, False),
            ({_NO_GEOREF_KEY: False}, False),
            ({_NO_GEOREF_KEY: 'yes'}, False),     # not identity-True
            ({_NO_GEOREF_KEY: 1}, False),         # truthy int, not True
            ({'other': True}, False),
        ],
    )
    def test_marker_predicate_identity_check(self, attrs, expected):
        da = xr.DataArray(
            np.zeros((2, 2), dtype=np.float32),
            coords={
                'y': np.arange(2, dtype=np.int64),
                'x': np.arange(2, dtype=np.int64),
            },
            dims=('y', 'x'),
            attrs=attrs,
        )
        assert _has_no_georef_marker(da) is expected

    @pytest.mark.parametrize(
        "coord",
        [
            np.arange(5, dtype=np.int64),         # full read
            np.arange(3, 8, dtype=np.int64),      # windowed read
            np.arange(0, 1, dtype=np.int64),      # degenerate 1-element
            np.array([10, 11, 12], dtype=np.int64),
        ],
    )
    def test_arange_int64_shape_helper_accepts(self, coord):
        assert _arange_int64_shape(coord)

    @pytest.mark.parametrize(
        "coord",
        [
            np.array([100, 101, 102], dtype=np.int32),     # int32, not int64
            np.array([100, 101, 102], dtype=np.float64),   # float
            np.array([200, 199], dtype=np.int64),          # descending
            np.array([0, 2, 4], dtype=np.int64),           # step != 1
            np.array([1, 2, 5], dtype=np.int64),           # non-uniform
            np.array([], dtype=np.int64),                  # empty
        ],
    )
    def test_arange_int64_shape_helper_rejects(self, coord):
        assert not _arange_int64_shape(coord)


class TestIntCoordRoundTrip:
    """User-authored integer-coord grids keep their georef."""

    def test_user_authored_int_grid_writes_real_transform(self, tmp_path):
        # User-authored projected grid with integer-spaced coords. ``y``
        # decreases top-to-bottom by convention, so it does not match
        # the ascending sentinel even before any other check.
        da = xr.DataArray(
            np.zeros((2, 3), dtype=np.float32),
            coords={
                'y': np.array([200, 199]),
                'x': np.array([100, 101, 102]),
            },
            dims=('y', 'x'),
        )
        path = str(tmp_path / "int_grid.tif")
        to_geotiff(da, path)

        out = open_geotiff(path)
        # Coord values round-trip exactly; dtype flips int -> float
        # because the file now carries a real transform and the reader
        # emits float pixel-center coords.
        assert out.coords['x'].dtype.kind == 'f'
        assert out.coords['y'].dtype.kind == 'f'
        np.testing.assert_array_equal(out.coords['x'].values,
                                      [100.0, 101.0, 102.0])
        np.testing.assert_array_equal(out.coords['y'].values, [200.0, 199.0])
        # Transform attr is present (the bug was that it wasn't).
        assert out.attrs.get('transform') is not None

    def test_both_axes_ascending_int64_step1_writes_real_transform(self,
                                                                   tmp_path):
        # Treating any int64 ascending step-1 grid as the no-georef
        # placeholder (because the reader emits coords of that shape) and
        # silently stripping georef bites real users whose projected grids
        # happen to start at integer offsets like ``x=[500, 501, 502],
        # y=[1000, 1001]``. The placeholder signal lives in
        # ``attrs[_NO_GEOREF_KEY]`` so the writer does not guess from coord
        # shape alone.
        da = xr.DataArray(
            np.zeros((3, 3), dtype=np.float32),
            coords={
                'y': np.array([200, 201, 202], dtype=np.int64),
                'x': np.array([100, 101, 102], dtype=np.int64),
            },
            dims=('y', 'x'),
        )
        path = str(tmp_path / "both_arange.tif")
        to_geotiff(da, path)
        out = open_geotiff(path)
        assert out.coords['x'].dtype.kind == 'f'
        assert out.coords['y'].dtype.kind == 'f'
        np.testing.assert_array_equal(out.coords['x'].values,
                                      [100.0, 101.0, 102.0])
        np.testing.assert_array_equal(out.coords['y'].values,
                                      [200.0, 201.0, 202.0])
        assert out.attrs.get('transform') is not None

    def test_user_authored_int_grid_with_explicit_transform(self, tmp_path):
        # Caller in the ambiguous-trade-off corner who wants georef
        # sets attrs['transform'] explicitly. The writer must use that
        # transform rather than the sentinel inference.
        da = xr.DataArray(
            np.zeros((3, 3), dtype=np.float32),
            coords={
                'y': np.array([200, 201, 202], dtype=np.int64),
                'x': np.array([100, 101, 102], dtype=np.int64),
            },
            dims=('y', 'x'),
            attrs={'transform': (1.0, 0.0, 99.5, 0.0, 1.0, 199.5)},
        )
        path = str(tmp_path / "explicit_transform.tif")
        to_geotiff(da, path)
        out = open_geotiff(path)
        assert out.attrs.get('transform') is not None
        np.testing.assert_array_equal(out.coords['x'].values,
                                      [100.0, 101.0, 102.0])

    def test_non_uniform_int_coords_raise(self, tmp_path):
        # Non-uniform integer spacing must not silently strip georef. The
        # write-metadata validator catches it (the integer-dtype exemption
        # was replaced with a marker-based one); the lower-level
        # ``coords_to_transform`` ("not uniformly spaced") check is a
        # backstop. Either message satisfies the contract: a non-uniform
        # write must raise rather than silently misrepresent the grid.
        da = xr.DataArray(
            np.zeros((3, 3), dtype=np.float32),
            coords={
                'y': np.array([10, 11, 12], dtype=np.int64),
                'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform
            },
            dims=('y', 'x'),
        )
        path = str(tmp_path / "non_uniform.tif")
        with pytest.raises(ValueError, match="non.?uniform"):
            to_geotiff(da, path)

    def test_int_x_float_y_writes_transform(self, tmp_path):
        # One axis integer, the other float: under the old sentinel any
        # integer axis defeated the transform-inference. Under the
        # tightened sentinel, the float y axis means the int x axis
        # falls through to ``coords_to_transform`` (which handles int
        # math fine) and a transform is written.
        da = xr.DataArray(
            np.zeros((2, 3), dtype=np.float32),
            coords={
                'y': np.array([50.5, 49.5], dtype=np.float64),
                'x': np.array([100, 101, 102], dtype=np.int64),
            },
            dims=('y', 'x'),
        )
        path = str(tmp_path / "mixed_dtypes.tif")
        to_geotiff(da, path)
        out = open_geotiff(path)
        assert out.attrs.get('transform') is not None
        np.testing.assert_array_equal(out.coords['x'].values,
                                      [100.0, 101.0, 102.0])

    def test_no_georef_roundtrip_preserved(self, tmp_path):
        # The no-georef round-trip starts from a real no-georef file:
        # the reader stamps ``attrs[_NO_GEOREF_KEY] = True`` together
        # with the int64 ``np.arange``-shaped coords, and the writer
        # carries the marker forward so the next ``to_geotiff`` does
        # not invent a transform. The marker is the only signal -- a user
        # constructing the same coord arrays from scratch without the
        # marker writes a real unit transform instead (see
        # ``test_both_axes_ascending_int64_step1_writes_real_transform``).
        src = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            coords={
                'y': np.arange(4, dtype=np.int64),
                'x': np.arange(4, dtype=np.int64),
            },
            dims=('y', 'x'),
            attrs={_NO_GEOREF_KEY: True},
        )
        path = str(tmp_path / "no_georef.tif")
        to_geotiff(src, path)

        out = open_geotiff(path)
        # Verify the read came back as no-georef.
        assert out.coords['x'].dtype == np.int64
        assert out.coords['y'].dtype == np.int64
        assert out.attrs.get('transform') is None
        assert out.attrs.get(_NO_GEOREF_KEY) is True

        # Round-trip: write again. No transform should be invented.
        path2 = str(tmp_path / "no_georef_rt.tif")
        to_geotiff(out, path2)
        out2 = open_geotiff(path2)
        assert out2.coords['x'].dtype == np.int64
        assert out2.attrs.get('transform') is None
        assert out2.attrs.get(_NO_GEOREF_KEY) is True

    def test_windowed_no_georef_roundtrip_with_marker(self, tmp_path):
        # When a caller explicitly opts into no-georef writes via
        # ``attrs[_NO_GEOREF_KEY] = True``, the windowed-offset arange
        # pattern that windowed reads return round-trips cleanly.
        # Without the marker, the same coord values write a real
        # transform (covered by
        # ``test_both_axes_ascending_int64_step1_writes_real_transform``).
        da = xr.DataArray(
            np.zeros((3, 4), dtype=np.float32),
            coords={
                'y': np.arange(10, 13, dtype=np.int64),
                'x': np.arange(20, 24, dtype=np.int64),
            },
            dims=('y', 'x'),
            attrs={_NO_GEOREF_KEY: True},
        )
        path = str(tmp_path / "windowed.tif")
        to_geotiff(da, path)
        out = open_geotiff(path)
        assert out.coords['x'].dtype == np.int64
        assert out.attrs.get('transform') is None
        assert out.attrs.get(_NO_GEOREF_KEY) is True


@requires_integration
class TestIntCoordRoundTripIntegration:
    """Integration coverage of the int-coord round trip.

    ``require_transform_for_georeferenced`` raises when both spatial dims
    are present in ``da.coords`` and no transform was resolved.
    ``coords_to_transform`` returns ``None`` for integer-dtype x/y coords
    as a no-georef sentinel. Their interaction once broke every writer
    call against an int-coord DataArray: the resolver returned ``None``,
    then the guard raised. The integration cases below write to the OS
    temp dir and only run when ``XRSPATIAL_RUN_INTEGRATION=1``.
    """

    def _tmp_path(self, name):
        return os.path.join(tempfile.gettempdir(), name)

    def test_int_coords_2d_round_trip(self):
        pixels = np.arange(20, dtype=np.float32).reshape(4, 5)
        da = xr.DataArray(
            pixels,
            dims=['y', 'x'],
            coords={
                'y': np.arange(4, dtype=np.int64),
                'x': np.arange(5, dtype=np.int64),
            },
            attrs={'long_name': 'int_coord_2d'},
        )
        path = self._tmp_path('coords_int_2d.tif')
        try:
            to_geotiff(da, path)
            da2 = open_geotiff(path)
            np.testing.assert_array_equal(da2.values, pixels)
            # Read-side should re-emit the int-coord no-georef
            # placeholders, confirming the no-georef contract held.
            assert da2.coords['x'].dtype.kind in ('i', 'u')
            assert da2.coords['y'].dtype.kind in ('i', 'u')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_int_coords_3d_band_y_x_round_trip(self):
        pixels = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
        da = xr.DataArray(
            pixels,
            dims=['band', 'y', 'x'],
            coords={
                'band': np.arange(2, dtype=np.int64),
                'y': np.arange(4, dtype=np.int64),
                'x': np.arange(5, dtype=np.int64),
            },
            attrs={'long_name': 'int_coord_3d'},
        )
        path = self._tmp_path('coords_int_3d.tif')
        try:
            to_geotiff(da, path)
            da2 = open_geotiff(path)
            # open_geotiff may emit (band, y, x) or (y, x, band) layout;
            # compare on a band-by-band basis instead of guessing axis order.
            arr = da2.values
            assert arr.shape in ((2, 4, 5), (4, 5, 2))
            if arr.shape == (2, 4, 5):
                np.testing.assert_array_equal(arr, pixels)
            else:
                np.testing.assert_array_equal(
                    np.moveaxis(arr, -1, 0), pixels
                )
        finally:
            if os.path.exists(path):
                os.remove(path)
