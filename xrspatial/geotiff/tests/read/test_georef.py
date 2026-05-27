"""Georef status, no-georef variants, and rotated-affine handling.

These tests cover one concern: how the reader and writer agree on
whether a file is georeferenced, what ``attrs['georef_status']``
records, and how rotated affines are surfaced or rejected.

Read side:

* Non-georeferenced reads and south-up orientation round trips.
* ``resolve_georef`` parity across reader / writer call sites.
* Canonical ``georef_status`` attr for the five reader states.
* No-georef marker migration from coord dtype to ``attrs``.
* int64 step-1 user grids keep their georef on round trip.
* Windowed reads emit the same integer pixel coords as full reads.
* ``allow_rotated=True`` surfaces ``attrs['rotated_affine']``.
* ``_transform_from_attr`` rejects rotated / sheared 6-tuples.
* GeoTIFF and VRT rotated reads raise the same typed error.

Write side:

* 1xN / Nx1 / 1x1 georeferenced writes preserve their transform.
* No-georef files keep integer coords across read -> write -> read.
* ``to_geotiff`` refuses to silently drop a rotated affine.

The write-side cases stay in this file rather than a sibling
``write/test_georef.py`` because they share the rotated-affine and
no-georef-marker fixtures with the read-side tests; splitting them
would scatter one concern across two modules.

``read/test_crs.py`` owns the rotated-CRS read surface and is not
touched here. The status tests below import its ``_write_rotated_tiff``
helper so the two suites share one rotated-TIFF byte layout.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (NonUniformCoordsError, _coords_to_transform, _transform_from_attr,
                               open_geotiff, read_vrt, to_geotiff, write_geotiff_gpu)
from xrspatial.geotiff._attrs import (_ATTRS_CONTRACT_VERSION, GEOREF_STATUS_CRS_ONLY,
                                      GEOREF_STATUS_FULL, GEOREF_STATUS_NONE,
                                      GEOREF_STATUS_ROTATED_DROPPED, GEOREF_STATUS_TRANSFORM_ONLY,
                                      _compute_georef_status, _compute_georef_status_from_parts,
                                      _populate_attrs_from_geo_info, attrs_to_metadata,
                                      geo_info_to_metadata)
from xrspatial.geotiff._coords import (_NO_GEOREF_KEY, GEOREF_STATUS_COORDS, GeorefResolution,
                                       _has_no_georef_marker, resolve_georef)
from xrspatial.geotiff._errors import GeoTIFFAmbiguousMetadataError, RotatedTransformError
from xrspatial.geotiff._geotags import (RASTER_PIXEL_IS_AREA, TAG_MODEL_TRANSFORMATION, GeoInfo,
                                        GeoTransform, _extract_transform, extract_geo_info)
from xrspatial.geotiff._header import IFD, IFDEntry, parse_all_ifds, parse_header
from xrspatial.geotiff._writer import write
from xrspatial.geotiff._writers.eager import _write_vrt_tiled

from .._helpers.markers import gpu_available
from .._helpers.markers import requires_gpu as _gpu_only
from .._helpers.tiff_builders import make_minimal_tiff
# Share the rotated-TIFF byte layout with the CRS suite.
from .test_crs import _write_rotated_tiff as _write_rotated_tiff_crs

tifffile = pytest.importorskip("tifffile")

_STATUS_KEY = 'georef_status'


# ===========================================================================
# read/test_crs.py owns rotated-CRS reads. The status suite below reuses its
# rotated-TIFF writer; everything else in this file is georef-status,
# no-georef, or drop-rotation.
# ===========================================================================


# ===========================================================================
# Non-georeferenced reads and south-up orientation.
# ===========================================================================


class TestNonGeoreferencedRead:
    """T-4: opening a TIFF with no GeoTIFF tags should not crash."""

    def test_open_plain_tiff(self, tmp_path):
        # make_minimal_tiff with no geo_transform / epsg = no GeoTIFF tags.
        data = make_minimal_tiff(4, 5, np.dtype('float32'))
        path = str(tmp_path / 'plain_1482.tif')
        with open(path, 'wb') as f:
            f.write(data)

        da = open_geotiff(path)

        assert da.shape == (5, 4)  # height=5, width=4
        # No CRS attribute should be set.
        assert 'crs' not in da.attrs
        assert 'crs_wkt' not in da.attrs

    def test_integer_pixel_coords(self, tmp_path):
        """Non-georef fallback: y=0..H-1, x=0..W-1."""
        data = make_minimal_tiff(6, 3, np.dtype('uint16'))
        path = str(tmp_path / 'plain_coords_1482.tif')
        with open(path, 'wb') as f:
            f.write(data)

        da = open_geotiff(path)
        np.testing.assert_array_equal(da.coords['y'].values, np.arange(3))
        np.testing.assert_array_equal(da.coords['x'].values, np.arange(6))

    def test_has_georef_flag_false(self, tmp_path):
        """The GeoInfo carries the explicit has_georef=False flag."""
        data = make_minimal_tiff(4, 4, np.dtype('float32'))
        header = parse_header(data)
        ifd = parse_all_ifds(data, header)[0]
        info = extract_geo_info(ifd, data, header.byte_order)
        assert info.has_georef is False

    def test_has_georef_flag_true_when_present(self, tmp_path):
        data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        header = parse_header(data)
        ifd = parse_all_ifds(data, header)[0]
        info = extract_geo_info(ifd, data, header.byte_order)
        assert info.has_georef is True

    def test_round_trip_preserves_pixels(self, tmp_path):
        """Read a plain TIFF, write it back, read again -- pixels match."""
        pixels = np.arange(20, dtype=np.float32).reshape(4, 5)
        data = make_minimal_tiff(5, 4, pixel_data=pixels)
        in_path = str(tmp_path / 'plain_in_1482.tif')
        out_path = str(tmp_path / 'plain_out_1482.tif')
        with open(in_path, 'wb') as f:
            f.write(data)

        da = open_geotiff(in_path)
        np.testing.assert_array_equal(da.values, pixels)

        # Round-trip through to_geotiff -- should not crash even without CRS
        to_geotiff(da, out_path)
        da2 = open_geotiff(out_path)
        np.testing.assert_array_equal(da2.values, pixels)


class TestDescendingYWrite:
    """T-3: writing a south-up DataArray (positive pixel_height).

    GeoTIFF's ModelPixelScale tag stores absolute scale values; sign
    information is lost when round-tripping through Tiepoint+Scale.  The
    pixel data orientation, however, is preserved exactly.  This test
    documents that contract.
    """

    def _make_south_up_da(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        # Ascending y values: row 0 is the south edge, y increases northward.
        y = np.array([0.5, 1.5, 2.5, 3.5])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        return xr.DataArray(
            data, dims=('y', 'x'), coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )

    def test_data_orientation_preserved(self, tmp_path):
        """Pixel ordering is identical on round-trip even for south-up input."""
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)

        # Pixel values land in the same row/col positions.
        np.testing.assert_array_equal(da2.values, da.values)

    def test_x_coords_preserved(self, tmp_path):
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_x_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        np.testing.assert_allclose(da2.coords['x'].values,
                                   da.coords['x'].values)

    def test_y_coords_known_limitation(self, tmp_path):
        """Known limitation: y sign is not preserved through Scale+Tiepoint.

        ModelPixelScale stores |pixel_height|; on read we always restore
        a negative pixel_height per GeoTIFF convention.  Document the
        magnitude is right and the sign is the GeoTIFF default.
        """
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_y_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        # Magnitudes match the original spacing.
        spacing = np.abs(np.diff(da2.coords['y'].values))
        assert np.allclose(spacing, 1.0)

    def test_descending_y_round_trip(self, tmp_path):
        """The standard north-up case (descending y) round-trips cleanly."""
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        # Descending y: row 0 at the north edge.
        y = np.array([3.5, 2.5, 1.5, 0.5])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        da = xr.DataArray(
            data, dims=('y', 'x'), coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        path = str(tmp_path / 'north_up_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        np.testing.assert_array_equal(da2.values, da.values)
        np.testing.assert_allclose(da2.coords['y'].values,
                                   da.coords['y'].values)
        np.testing.assert_allclose(da2.coords['x'].values,
                                   da.coords['x'].values)


# ===========================================================================
# resolve_georef parity across reader / writer call sites.
# ===========================================================================


def _make_axis_aligned_da(y_name: str, x_name: str) -> xr.DataArray:
    """Return a 4x5 DataArray with axis-aligned float pixel-center coords.

    Origin at (100.0, 200.0), 1.0 pixel width, -1.0 pixel height. No
    ``attrs['transform']`` so the resolver must derive the transform
    from the coords themselves. No CRS attr.
    """
    px = 1.0
    py = -1.0
    ox = 100.0
    oy = 200.0
    ys = oy + (np.arange(4) + 0.5) * py
    xs = ox + (np.arange(5) + 0.5) * px
    return xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=(y_name, x_name),
        coords={y_name: ys, x_name: xs},
        attrs={},
    )


def _make_no_georef_da() -> xr.DataArray:
    """Reader-style placeholder: int pixel coords + ``_NO_GEOREF_KEY``.

    Mirrors what the reader emits when a file has no GeoTIFF transform
    tags. ``row/col`` dim names are used because the no-georef path is
    independent of dim naming -- the marker is the only signal.
    """
    return xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        dims=('row', 'col'),
        coords={
            'row': np.arange(3, dtype=np.int64),
            'col': np.arange(3, dtype=np.int64),
        },
        attrs={_NO_GEOREF_KEY: True},
    )


def _make_transform_only_da() -> xr.DataArray:
    """``attrs['transform']`` set, no CRS, no coord arrays.

    The resolver should derive the transform straight from the attr
    and bucket as ``transform_only`` on the reader side, or ``coords``
    on the writer side (writer doesn't see the reader's
    ``has_georef`` bit, so any transform-without-CRS is just "coords").
    """
    return xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=('y', 'x'),
        attrs={
            'transform': (1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        },
    )


def _make_crs_only_da() -> xr.DataArray:
    """``attrs['crs']`` set with no transform and no coords.

    Resolver should land on ``crs_only`` with ``transform=None``.
    """
    return xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )


def _make_full_geo_info() -> GeoInfo:
    return GeoInfo(
        transform=GeoTransform(
            origin_x=100.0, origin_y=200.0,
            pixel_width=1.0, pixel_height=-1.0,
        ),
        has_georef=True,
        crs_epsg=4326,
        crs_wkt='GEOGCS["WGS 84"]',
        raster_type=RASTER_PIXEL_IS_AREA,
    )


def _make_transform_only_geo_info() -> GeoInfo:
    return GeoInfo(
        transform=GeoTransform(
            origin_x=100.0, origin_y=200.0,
            pixel_width=1.0, pixel_height=-1.0,
        ),
        has_georef=True,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=RASTER_PIXEL_IS_AREA,
    )


def _make_crs_only_geo_info() -> GeoInfo:
    return GeoInfo(
        transform=None,
        has_georef=False,
        crs_epsg=4326,
        crs_wkt=None,
        raster_type=RASTER_PIXEL_IS_AREA,
    )


def _make_none_geo_info() -> GeoInfo:
    return GeoInfo(
        transform=None,
        has_georef=False,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=RASTER_PIXEL_IS_AREA,
    )


def _make_rotated_dropped_geo_info() -> GeoInfo:
    return GeoInfo(
        transform=GeoTransform(
            rotated_affine=(1.0, 0.1, 100.0, 0.2, -1.0, 200.0),
        ),
        has_georef=False,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=RASTER_PIXEL_IS_AREA,
    )


@pytest.mark.parametrize(
    "geo_info_factory, expected_status, expected_marker, expected_has_transform",
    [
        (_make_full_geo_info, GEOREF_STATUS_FULL, False, True),
        (_make_transform_only_geo_info, GEOREF_STATUS_TRANSFORM_ONLY, False, True),
        (_make_crs_only_geo_info, GEOREF_STATUS_CRS_ONLY, True, False),
        (_make_none_geo_info, GEOREF_STATUS_NONE, True, False),
        (_make_rotated_dropped_geo_info, GEOREF_STATUS_ROTATED_DROPPED, True, False),
    ],
    ids=['full', 'transform_only', 'crs_only', 'none', 'rotated_dropped'],
)
def test_resolve_georef_reader_path(
    geo_info_factory, expected_status, expected_marker, expected_has_transform,
):
    """Reader-side: ``resolve_georef(geo_info=...)`` returns the expected bucket."""
    geo_info = geo_info_factory()
    result = resolve_georef(geo_info=geo_info)
    assert isinstance(result, GeorefResolution)
    assert result.georef_status == expected_status
    assert result.applied_no_georef_marker is expected_marker
    assert (result.transform is not None) is expected_has_transform


def test_reader_compute_georef_status_routes_through_resolver():
    """``_compute_georef_status`` is a thin wrapper over the resolver.

    Every supported ``GeoInfo`` bucket must map to the same string the
    resolver returns. If the two ever drift the read-path
    ``attrs['georef_status']`` and the resolver disagree about which
    state the read landed in.
    """
    for factory, expected in [
        (_make_full_geo_info, GEOREF_STATUS_FULL),
        (_make_transform_only_geo_info, GEOREF_STATUS_TRANSFORM_ONLY),
        (_make_crs_only_geo_info, GEOREF_STATUS_CRS_ONLY),
        (_make_none_geo_info, GEOREF_STATUS_NONE),
        (_make_rotated_dropped_geo_info, GEOREF_STATUS_ROTATED_DROPPED),
    ]:
        gi = factory()
        assert _compute_georef_status(gi) == expected
        assert resolve_georef(geo_info=gi).georef_status == expected


def test_reader_compute_georef_status_from_parts_matches_resolver():
    """The VRT shim ``_compute_georef_status_from_parts`` matches the resolver.

    The VRT branches don't build a ``GeoInfo``, they thread raw
    booleans. Build matching ``GeoInfo`` records for the same boolean
    combinations and assert the two helpers agree.
    """
    cases = [
        (True, True, False, GEOREF_STATUS_FULL),
        (True, False, False, GEOREF_STATUS_TRANSFORM_ONLY),
        (False, True, False, GEOREF_STATUS_CRS_ONLY),
        (False, False, False, GEOREF_STATUS_NONE),
        (False, False, True, GEOREF_STATUS_ROTATED_DROPPED),
        (True, True, True, GEOREF_STATUS_ROTATED_DROPPED),
    ]
    for has_t, has_c, rotated, expected in cases:
        from_parts = _compute_georef_status_from_parts(
            has_transform=has_t, has_crs=has_c, rotated_dropped=rotated,
        )
        assert from_parts == expected


@pytest.mark.parametrize(
    "y_name,x_name",
    [('y', 'x'), ('lat', 'lon'), ('latitude', 'longitude')],
    ids=['y_x', 'lat_lon', 'latitude_longitude'],
)
def test_resolve_georef_writer_axis_aligned_coords(y_name, x_name):
    """Dim-name parity: same coord shape, three dim conventions.

    The resolver picks the spatial dims as the last two regardless of
    name, so all three should produce identical transforms and the
    ``coords`` bucket (no CRS attr).
    """
    da = _make_axis_aligned_da(y_name, x_name)
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_COORDS
    assert result.applied_no_georef_marker is False
    assert result.transform is not None
    assert result.transform.pixel_width == pytest.approx(1.0)
    assert result.transform.pixel_height == pytest.approx(-1.0)
    assert result.transform.origin_x == pytest.approx(100.0)
    assert result.transform.origin_y == pytest.approx(200.0)


def test_resolve_georef_writer_no_georef_marker():
    """``attrs[_NO_GEOREF_KEY]=True`` forces no-georef regardless of coords.

    The reader-stamped placeholder int64 coord arrays would otherwise
    produce a synthetic unit transform. The marker is the gate that
    keeps that synthesis from running on round-trip.
    """
    da = _make_no_georef_da()
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_NONE
    assert result.applied_no_georef_marker is True
    assert result.transform is None


def test_resolve_georef_writer_transform_only_attr():
    """``attrs['transform']`` resolves to ``transform_only`` (no CRS).

    Writer side: the only georef signal is the attr, so the resolver
    lands on ``transform_only`` to distinguish it from the
    ``coords``-derived bucket (which fires when the transform was
    inferred from coord arrays). The transform is the rasterio
    6-tuple ``(a, b, c, d, e, f)`` reconstructed into a
    ``GeoTransform``.
    """
    da = _make_transform_only_da()
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_TRANSFORM_ONLY
    assert result.transform is not None
    assert result.transform.pixel_width == pytest.approx(1.0)
    assert result.transform.pixel_height == pytest.approx(-1.0)
    assert result.transform.origin_x == pytest.approx(100.0)
    assert result.transform.origin_y == pytest.approx(200.0)
    assert result.applied_no_georef_marker is False


def test_resolve_georef_writer_crs_only():
    """``attrs['crs']`` without transform / coords -> ``crs_only``."""
    da = _make_crs_only_da()
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_CRS_ONLY
    assert result.transform is None
    # No marker on the writer side: writer didn't stamp anything; the
    # caller just passed a CRS without a transform. The marker is
    # reader-stamped; writer paths don't apply it.
    assert result.applied_no_georef_marker is False


def test_resolve_georef_writer_full_round_trip():
    """DataArray with both transform attr AND CRS -> ``full`` bucket."""
    da = _make_transform_only_da()
    da = da.assign_attrs(crs=4326)
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_FULL
    assert result.transform is not None


def test_writer_and_reader_agree_on_full_case():
    """A reader-emitted DataArray and the underlying GeoInfo agree.

    Build a DataArray that mimics what the reader stamps for a fully
    georeferenced file (coord arrays + attrs['transform'] +
    attrs['crs']) and a ``GeoInfo`` carrying the same metadata.
    ``resolve_georef`` should land on the same status from either
    entry point (``full``).
    """
    da = _make_axis_aligned_da('y', 'x').assign_attrs(
        transform=(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        crs=4326,
    )
    gi = _make_full_geo_info()
    writer_result = resolve_georef(da)
    reader_result = resolve_georef(geo_info=gi)
    assert writer_result.georef_status == reader_result.georef_status
    assert writer_result.georef_status == GEOREF_STATUS_FULL
    # Both produced an axis-aligned transform with the same pixel
    # geometry. They are not the same object, but the values match.
    assert writer_result.transform.pixel_width == pytest.approx(
        reader_result.transform.pixel_width)
    assert writer_result.transform.pixel_height == pytest.approx(
        reader_result.transform.pixel_height)


def test_writer_and_reader_agree_on_no_georef_case():
    """No-georef placeholder DataArray and its underlying GeoInfo agree."""
    da = _make_no_georef_da()
    gi = _make_none_geo_info()
    writer_result = resolve_georef(da)
    reader_result = resolve_georef(geo_info=gi)
    assert writer_result.georef_status == GEOREF_STATUS_NONE
    assert reader_result.georef_status == GEOREF_STATUS_NONE
    assert writer_result.transform is None
    assert reader_result.transform is None
    assert writer_result.applied_no_georef_marker is True
    assert reader_result.applied_no_georef_marker is True


def test_writer_rotated_attr_transform_rejected():
    """A rotated 6-tuple in ``attrs['transform']`` raises through the resolver.

    The writers' ``transform_from_attr`` rejects rotated / sheared
    affines with a ``ValueError``; the resolver must preserve that
    diagnostic so writer callers see the same failure.
    """
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=('y', 'x'),
        attrs={'transform': (1.0, 0.1, 100.0, 0.2, -1.0, 200.0)},
    )
    with pytest.raises(ValueError, match="non-zero rotation"):
        resolve_georef(da)


def test_dropped_reason_populated_for_rotated_reader():
    """``rotated_dropped`` reader case records the drop reason."""
    gi = _make_rotated_dropped_geo_info()
    result = resolve_georef(geo_info=gi)
    assert result.dropped_reason == 'rotated_affine_dropped'
    assert result.applied_no_georef_marker is True


def test_dropped_reason_populated_for_no_transform_writer():
    """Writer-side: ``attrs[_NO_GEOREF_KEY]`` records ``no_georef_marker``."""
    da = _make_no_georef_da()
    result = resolve_georef(da)
    assert result.dropped_reason == 'no_georef_marker'


def test_bare_input_crs_only():
    """``resolve_georef(crs_epsg=...)`` with no transform returns ``crs_only``."""
    result = resolve_georef(crs_epsg=4326)
    assert result.georef_status == GEOREF_STATUS_CRS_ONLY
    assert result.transform is None


def test_bare_input_transform_only():
    """``transform_hint`` only -> ``transform_only`` bucket."""
    t = (1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    result = resolve_georef(transform_hint=t)
    assert result.georef_status == GEOREF_STATUS_TRANSFORM_ONLY
    assert result.transform is not None
    assert result.transform.pixel_width == pytest.approx(1.0)


def test_bare_input_none_signal():
    """No signals at all -> ``none`` with ``no_inputs`` drop reason."""
    result = resolve_georef()
    assert result.georef_status == GEOREF_STATUS_NONE
    assert result.transform is None
    assert result.dropped_reason == 'no_inputs'


# ===========================================================================
# Canonical georef_status attr for the five reader states.
# ===========================================================================


def test_contract_version_is_at_least_three():
    """The ``georef_status`` attr lands in v3; pin a lower bound so future
    contract bumps that keep the attr (e.g. a later ``rotated_affine``
    addition) do not regress this test."""
    assert _ATTRS_CONTRACT_VERSION >= 3


def test_public_constants_reexported():
    """The five status constants and ``GEOREF_STATUS_VALUES`` are part
    of the public surface. Downstream consumers should be able to import
    them from ``xrspatial.geotiff`` rather than reaching into the private
    ``_attrs`` module."""
    import xrspatial.geotiff as pkg
    assert pkg.GEOREF_STATUS_FULL == GEOREF_STATUS_FULL
    assert pkg.GEOREF_STATUS_TRANSFORM_ONLY == GEOREF_STATUS_TRANSFORM_ONLY
    assert pkg.GEOREF_STATUS_CRS_ONLY == GEOREF_STATUS_CRS_ONLY
    assert pkg.GEOREF_STATUS_NONE == GEOREF_STATUS_NONE
    assert pkg.GEOREF_STATUS_ROTATED_DROPPED == GEOREF_STATUS_ROTATED_DROPPED
    assert pkg.GEOREF_STATUS_VALUES == frozenset({
        'full', 'transform_only', 'crs_only', 'none', 'rotated_dropped',
    })


def test_compute_status_full():
    info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=1.0, pixel_height=-1.0),
        has_georef=True, crs_epsg=4326,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_FULL


def test_compute_status_transform_only():
    info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=1.0, pixel_height=-1.0),
        has_georef=True,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_TRANSFORM_ONLY


def test_compute_status_crs_only():
    info = GeoInfo(has_georef=False, crs_epsg=4326)
    assert _compute_georef_status(info) == GEOREF_STATUS_CRS_ONLY


def test_compute_status_crs_only_wkt_no_epsg():
    """``crs_wkt`` without an EPSG code is still ``crs_only``: the
    decision rule treats either CRS signal as present."""
    info = GeoInfo(has_georef=False, crs_wkt='GEOGCRS["custom"]')
    assert _compute_georef_status(info) == GEOREF_STATUS_CRS_ONLY


def test_compute_status_none():
    info = GeoInfo(has_georef=False)
    assert _compute_georef_status(info) == GEOREF_STATUS_NONE


def test_compute_status_rotated_dropped():
    """``rotated_affine`` set on the transform is the dropped-rotation
    signal; the value wins over CRS presence and ``has_georef``."""
    info = GeoInfo(
        transform=GeoTransform(rotated_affine=(1.0, 0.5, 0.0, 0.5, -1.0, 0.0)),
        has_georef=False, crs_epsg=4326,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_ROTATED_DROPPED


@pytest.mark.parametrize(
    "has_transform,has_crs,rotated,expected",
    [
        (True, True, False, GEOREF_STATUS_FULL),
        (True, False, False, GEOREF_STATUS_TRANSFORM_ONLY),
        (False, True, False, GEOREF_STATUS_CRS_ONLY),
        (False, False, False, GEOREF_STATUS_NONE),
        (False, False, True, GEOREF_STATUS_ROTATED_DROPPED),
        (True, True, True, GEOREF_STATUS_ROTATED_DROPPED),
    ],
)
def test_compute_status_from_parts(has_transform, has_crs, rotated, expected):
    """The VRT-side helper mirrors the GeoInfo decision exactly."""
    assert _compute_georef_status_from_parts(
        has_transform=has_transform, has_crs=has_crs,
        rotated_dropped=rotated,
    ) == expected


def _make_full_tiff(path):
    """Float coords + CRS -> ``full``. Goes through to_geotiff so the
    fixture exercises the same writer path real callers hit."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, path)


def _make_transform_only_tiff(path):
    """Float coords, no CRS -> ``transform_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, path)


def _make_crs_only_tiff(path):
    """No-georef marker + CRS -> ``crs_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, path)


def _make_none_tiff(path):
    """No-georef marker, no CRS -> ``none``. Equivalent to a plain image."""
    arr = np.zeros((4, 4), dtype=np.float32)
    # Write a bare TIFF with no GeoTIFF tags at all -- this is the
    # tightest 'none' fixture; it has no GeoKeyDirectory, no transform
    # tags, and no marker on disk. The reader stamps the marker on read.
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )


def test_full_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_full.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL
    assert 'transform' in rd.attrs
    assert rd.attrs.get('crs') == 4326


def test_transform_only_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_xform_only.tif")
    _make_transform_only_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_TRANSFORM_ONLY
    assert 'transform' in rd.attrs
    assert 'crs' not in rd.attrs
    # ``crs_wkt`` may be absent or present depending on whether the
    # writer emitted GeoKey citation text; either way the status is the
    # source of truth for "no CRS".


def test_crs_only_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_crs_only.tif")
    _make_crs_only_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_CRS_ONLY
    assert rd.attrs.get('crs') == 4326
    assert 'transform' not in rd.attrs
    # Reader still stamps the legacy marker for back-compat. The new
    # attr is additive; pinning both keys here so a future change that
    # drops one is caught at the same site.
    assert rd.attrs.get(_NO_GEOREF_KEY) is True


def test_none_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_none.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE
    assert 'transform' not in rd.attrs
    assert 'crs' not in rd.attrs
    assert rd.attrs.get(_NO_GEOREF_KEY) is True


def test_rotated_dropped_status(tmp_path):
    """Rotated ``ModelTransformationTag`` + ``allow_rotated=True`` is
    the only state where the attr disambiguates a case that all four
    other public signals were silent about."""
    path = str(tmp_path / "georef_status_2136_rotated.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_crs(path, arr)
    rd = open_geotiff(path, allow_rotated=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_ROTATED_DROPPED
    # ``transform`` is intentionally absent on the dropped path -- a
    # rotated affine cannot be expressed in the axis-aligned 6-tuple.
    assert 'transform' not in rd.attrs


def test_rotated_default_still_raises(tmp_path):
    """Default ``allow_rotated=False`` keeps the existing refusal so the
    rotated_dropped state is only reachable via the explicit opt-in.
    Pinned here so the status attr does not accidentally relax the
    rotated-read refusal contract."""
    path = str(tmp_path / "georef_status_2136_rotated_default.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_crs(path, arr)
    with pytest.raises(RotatedTransformError, match="rotation"):
        open_geotiff(path)


def test_full_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_full_dask.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


def test_none_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_none_dask.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


def test_rotated_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_rotated_dask.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_crs(path, arr)
    rd = open_geotiff(path, allow_rotated=True, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_ROTATED_DROPPED


@_gpu_only
def test_full_status_gpu(tmp_path):
    path = str(tmp_path / "georef_status_2136_full_gpu.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path, gpu=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


@_gpu_only
def test_none_status_gpu(tmp_path):
    path = str(tmp_path / "georef_status_2136_none_gpu.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path, gpu=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


def _write_vrt_georef_status(vrt_path, source_name, *, height, width,
                             crs_wkt=None, geo_transform=None):
    """Write a minimal VRT with optional CRS + geo_transform elements."""
    crs_elem = (
        f'  <SRS>{crs_wkt}</SRS>\n' if crs_wkt else ''
    )
    gt_elem = (
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        if geo_transform else ''
    )
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'{crs_elem}{gt_elem}'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_vrt_full_status(tmp_path):
    src = tmp_path / "georef_status_2136_vrt_full_src.tif"
    _make_full_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_full.vrt"
    _write_vrt_georef_status(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
        geo_transform='100.0, 1.0, 0.0, 200.5, 0.0, -1.0',
    )
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


def test_vrt_crs_only_status(tmp_path):
    """VRT with SRS but no GeoTransform element -> ``crs_only``."""
    src = tmp_path / "georef_status_2136_vrt_crsonly_src.tif"
    _make_none_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_crsonly.vrt"
    _write_vrt_georef_status(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
    )
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_CRS_ONLY


def test_vrt_none_status(tmp_path):
    """VRT with neither SRS nor GeoTransform -> ``none``."""
    src = tmp_path / "georef_status_2136_vrt_none_src.tif"
    _make_none_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_none.vrt"
    _write_vrt_georef_status(vrt, os.path.basename(src), height=4, width=4)
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


def test_vrt_full_status_chunked(tmp_path):
    src = tmp_path / "georef_status_2136_vrt_full_chunked_src.tif"
    _make_full_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_full_chunked.vrt"
    _write_vrt_georef_status(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
        geo_transform='100.0, 1.0, 0.0, 200.5, 0.0, -1.0',
    )
    rd = read_vrt(str(vrt), chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


@pytest.mark.parametrize(
    "make,expected",
    [
        (_make_full_tiff, GEOREF_STATUS_FULL),
        (_make_transform_only_tiff, GEOREF_STATUS_TRANSFORM_ONLY),
        (_make_crs_only_tiff, GEOREF_STATUS_CRS_ONLY),
        (_make_none_tiff, GEOREF_STATUS_NONE),
    ],
)
def test_roundtrip_preserves_status(tmp_path, make, expected):
    """Write a fixture of each writable state, read it, then write +
    read again. The status attr is stable across the cycle because the
    underlying CRS / transform decisions the writer makes are
    deterministic from the input attrs.

    The fixture set deliberately overlaps with the per-state tests
    above; the value of this parametrised test is the *cycle*, not the
    one-shot read.
    """
    p1 = str(tmp_path / f"georef_status_2136_rt_{expected}_1.tif")
    make(p1)
    rd1 = open_geotiff(p1)
    assert rd1.attrs[_STATUS_KEY] == expected

    p2 = str(tmp_path / f"georef_status_2136_rt_{expected}_2.tif")
    to_geotiff(rd1, p2)
    rd2 = open_geotiff(p2)
    assert rd2.attrs[_STATUS_KEY] == expected


# ===========================================================================
# No-georef marker migration from coord dtype to attrs.
# ===========================================================================


@pytest.fixture
def no_georef_path_2133(tmp_path):
    """4x4 float32 TIFF with no GeoTIFF tags."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = str(tmp_path / "no_georef_2133.tif")
    write(arr, path, compression='none', tiled=False)
    return path


def test_non_uniform_int_coords_without_marker_raise(tmp_path):
    """User-authored int coords with non-uniform spacing must not get a free pass.

    Exempting any integer dtype on the assumption that integer coords are
    the reader's placeholder is wrong: a non-uniform int-coord grid would
    slip past the validator and either reach the lower-level
    uniform-spacing check in ``coords_to_transform`` (raising a different
    error class) or, in some windowed paths, silently write a
    misrepresented transform. With the marker absent, the validator raises
    ``NonUniformCoordsError`` directly.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_non_uniform_int.tif")
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, path)


def test_non_uniform_int_coords_with_marker_pass(tmp_path):
    """Marker-stamped arrays skip uniformity entirely (they are placeholders).

    The marker contract: a DataArray carrying ``attrs[_NO_GEOREF_KEY]
    is True`` is treated as no-georef, the writer does not synthesise a
    transform, and uniformity of the placeholder coords is irrelevant.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform, but marked
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2133_non_uniform_int_marked.tif")
    # Must not raise.
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True


def test_non_uniform_float_coords_still_raise(tmp_path):
    """Pre-existing float-coord behaviour is unchanged."""
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10.0, 11.0, 12.0], dtype=np.float64),
            'x': np.array([1.0, 2.0, 5.0], dtype=np.float64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_non_uniform_float.tif")
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, path)


def test_uniform_int_coords_still_write(tmp_path):
    """A uniform int-coord grid without the marker writes a real transform.

    Regression guard: making the validator stricter must not break the
    common case of user-authored int-coord projected grids.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_uniform_int.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None


class TestMarkerOnRead:
    """A no-georef file must surface the marker on every read path."""

    def test_cpu_eager(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133)
        assert _has_no_georef_marker(da)

    def test_cpu_dask(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, chunks=2)
        assert _has_no_georef_marker(da)

    @_gpu_only
    def test_gpu_eager(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, gpu=True)
        assert _has_no_georef_marker(da)

    @_gpu_only
    def test_gpu_dask(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, gpu=True, chunks=2)
        assert _has_no_georef_marker(da)


def test_georef_file_has_no_marker(tmp_path):
    """Files with a transform must NOT carry the marker.

    Negative companion to the read-side tests: a real georeferenced
    write/read round-trip must leave the marker absent so downstream
    consumers can rely on its presence as a positive signal.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200.5, 201.5, 202.5], dtype=np.float64),
            'x': np.array([100.5, 101.5, 102.5], dtype=np.float64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_real_transform.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert not _has_no_georef_marker(out)
    assert _NO_GEOREF_KEY not in out.attrs


def test_marker_survives_copy():
    da = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        coords={'y': np.arange(2, dtype=np.int64),
                'x': np.arange(2, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    copied = da.copy()
    assert _has_no_georef_marker(copied)


def test_marker_survives_assign_attrs():
    da = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        coords={'y': np.arange(2, dtype=np.int64),
                'x': np.arange(2, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    # assign_attrs returns a new array with the merged attrs dict.
    # The marker must persist because attrs are passed through, not
    # filtered on underscore prefix.
    out = da.assign_attrs(extra='added')
    assert _has_no_georef_marker(out)
    assert out.attrs.get('extra') == 'added'


def test_3d_no_georef_round_trip(tmp_path):
    """A (band, y, x) no-georef array round-trips with the marker intact."""
    arr = np.zeros((3, 4, 4), dtype=np.float32)
    for i in range(3):
        arr[i] = i
    src = xr.DataArray(
        arr,
        coords={
            'band': np.arange(1, 4, dtype=np.int64),
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('band', 'y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2133_3d_no_georef.tif")
    to_geotiff(src, path)
    out = open_geotiff(path)
    assert _has_no_georef_marker(out)
    assert out.attrs.get('transform') is None
    # Spatial coords come back as int64 placeholders.
    assert out.coords['y'].dtype == np.int64
    assert out.coords['x'].dtype == np.int64
    # The reader returns ``(y, x, band)``; transpose before comparing
    # to the ``(band, y, x)`` source.
    band_dim = next(
        d for d in out.dims if d not in ('y', 'x')
    )
    np.testing.assert_array_equal(
        out.transpose(band_dim, 'y', 'x').values, src.values
    )


# ===========================================================================
# int64 step-1 user grids keep their georef on round trip.
# ===========================================================================


def test_int64_step1_user_grid_keeps_crs_on_round_trip(tmp_path):
    x = np.array([500, 501, 502], dtype=np.int64)
    y = np.array([1000, 1001], dtype=np.int64)
    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        coords={'y': y, 'x': x}, dims=('y', 'x'),
        attrs={'crs': 4326},
    )

    path = str(tmp_path / "tmp_2120_user_int64_grid.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)

    # Coord values round-trip exactly, dtype flips int -> float because
    # the file now carries a real transform.
    np.testing.assert_array_equal(out.coords['x'].values, [500.0, 501.0, 502.0])
    np.testing.assert_array_equal(out.coords['y'].values, [1000.0, 1001.0])
    assert out.attrs.get('crs') == 4326
    assert out.attrs.get('transform') is not None
    # The marker must not be set on a georef read.
    assert _NO_GEOREF_KEY not in out.attrs


def test_no_georef_file_carries_marker_and_round_trips(tmp_path):
    # Build a no-georef file via the explicit-marker write path.
    src = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.arange(4, dtype=np.int64),
                'x': np.arange(4, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2120_no_georef.tif")
    to_geotiff(src, path)

    out = open_geotiff(path)
    assert out.attrs.get(_NO_GEOREF_KEY) is True
    assert out.attrs.get('transform') is None
    assert out.coords['x'].dtype == np.int64

    # Round-trip preserves the marker and the absence of a transform.
    path2 = str(tmp_path / "tmp_2120_no_georef_rt.tif")
    to_geotiff(out, path2)
    out2 = open_geotiff(path2)
    assert out2.attrs.get(_NO_GEOREF_KEY) is True
    assert out2.attrs.get('transform') is None


def test_int64_step1_user_grid_without_marker_writes_transform(tmp_path):
    # Direct repro of the silent-strip the pre-fix code would emit.
    # x and y both match the reader's arange placeholder shape exactly,
    # but the marker is absent, so the writer treats this as a normal
    # georef grid and synthesises a unit transform.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2120_step1_no_marker.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None
    np.testing.assert_array_equal(out.coords['x'].values, [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(out.coords['y'].values, [200.0, 201.0, 202.0])


def test_marker_on_user_grid_skips_transform_synthesis(tmp_path):
    # A caller can opt into a no-georef write on an int-coord array by
    # setting the marker explicitly. The writer trusts the marker
    # rather than re-deriving a transform from the coords.
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.arange(3, dtype=np.int64),
            'x': np.arange(3, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2120_explicit_marker.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True


# ===========================================================================
# Windowed reads emit the same integer pixel coords as full reads.
# ===========================================================================


@pytest.fixture
def no_georef_path_1710(tmp_path):
    """8x8 float32 TIFF with NO GeoTIFF tags (no ModelPixelScale etc.)."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "no_georef_1710.tif")
    write(arr, path, compression='none', tiled=False)
    return path


class TestEagerWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710)
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))
        np.testing.assert_array_equal(da.x.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        """Windowed read at (2, 3) origin should give coords [2,3,4,5] / [3,4,5,6]."""
        da = open_geotiff(no_georef_path_1710, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_no_transform_attr_emitted(self, no_georef_path_1710):
        """Non-georef files should not advertise a fabricated transform."""
        da = open_geotiff(no_georef_path_1710)
        assert 'transform' not in da.attrs, (
            f"non-georef file should not carry a fabricated transform; "
            f"got attrs={dict(da.attrs)}"
        )
        # Same for windowed read
        da_w = open_geotiff(no_georef_path_1710, window=(0, 0, 4, 4))
        assert 'transform' not in da_w.attrs


class TestDaskWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4)
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, chunks=4, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))


@_gpu_only
class TestGpuWindowedCoords:

    def test_full_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True)
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(8))

    def test_windowed_read_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_dask_cupy_windowed_integer_coords(self, no_georef_path_1710):
        da = open_geotiff(no_georef_path_1710, gpu=True, chunks=4,
                          window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))

    def test_offset_window_integer_coords(self, no_georef_path_1710):
        """GPU windowed read at a non-zero origin: the stripped-GPU
        fallback in ``read_geotiff_gpu`` must check ``has_georef`` rather
        than ``t is None``; otherwise a non-georef TIFF emits
        ``[-0.5, -1.5, ...]`` via the unit
        ``GeoTransform`` placeholder. Pin the contract that the offset
        window produces file-relative integer coords identical to the
        eager numpy path.
        """
        da = open_geotiff(no_georef_path_1710, gpu=True,
                          window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_offset_window_no_transform_attr(self, no_georef_path_1710):
        """Non-georef GPU windowed read still must not advertise a
        fabricated ``attrs['transform']`` -- ``_populate_attrs_from_geo_info``
        gates on ``has_georef`` too, so flag the round-trip here.
        """
        da = open_geotiff(no_georef_path_1710, gpu=True,
                          window=(2, 3, 6, 7))
        assert 'transform' not in da.attrs, (
            f"non-georef GPU windowed read should not carry a "
            f"fabricated transform; got attrs={dict(da.attrs)}"
        )


class TestBackendParity:
    """Full read and windowed read must agree on coord dtype and values
    across every backend pair."""

    def test_dtype_parity_full(self, no_georef_path_1710):
        np_da = open_geotiff(no_georef_path_1710)
        dk_da = open_geotiff(no_georef_path_1710, chunks=4)
        assert np_da.y.dtype == dk_da.y.dtype == np.int64
        if gpu_available():
            gpu_da = open_geotiff(no_georef_path_1710, gpu=True)
            assert gpu_da.y.dtype == np.int64

    def test_dtype_parity_windowed(self, no_georef_path_1710):
        win = (0, 0, 4, 4)
        np_da = open_geotiff(no_georef_path_1710, window=win)
        dk_da = open_geotiff(no_georef_path_1710, chunks=4, window=win)
        assert np_da.y.dtype == dk_da.y.dtype == np.int64
        if gpu_available():
            gpu_da = open_geotiff(no_georef_path_1710, gpu=True, window=win)
            assert gpu_da.y.dtype == np.int64

    def test_full_and_windowed_first_n_match(self, no_georef_path_1710):
        """Coords for a window starting at (0, 0) should equal the first N
        coords of the full read."""
        win = (0, 0, 4, 4)
        full = open_geotiff(no_georef_path_1710)
        win_da = open_geotiff(no_georef_path_1710, window=win)
        np.testing.assert_array_equal(full.y.values[:4], win_da.y.values)
        np.testing.assert_array_equal(full.x.values[:4], win_da.x.values)


class TestGeorefStillWorks:
    """The fix must not regress georeferenced reads.

    Files with real GeoTIFF tags still get float64 coords with the
    half-pixel shift for PixelIsArea.
    """

    def test_georef_full_read_keeps_float64(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_1710.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path)
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        assert da.x.values[0] == pytest.approx(500015.0)
        # transform attr SHOULD be present for georef files
        assert 'transform' in da.attrs

    def test_georef_windowed_read_keeps_float64(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_win_1710.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, window=(0, 0, 4, 4))
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        assert 'transform' in da.attrs


# ===========================================================================
# _transform_from_attr rejects rotated / sheared 6-tuples.
# ===========================================================================


def _make_da_1764(transform_tuple):
    arr = np.zeros((4, 5), dtype=np.float32)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(4, dtype=np.float64),
            "x": np.arange(5, dtype=np.float64),
        },
        attrs={"transform": transform_tuple, "crs": 4326},
    )


class TestTransformFromAttrRejection:
    """Direct tests on the helper."""

    def test_rotation_b_nonzero_raises(self):
        # b=0.5 -> meaningful rotation
        t = (1.0, 0.5, 100.0, 0.0, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_shear_d_nonzero_raises(self):
        # d=0.25 -> meaningful shear
        t = (1.0, 0.0, 100.0, 0.25, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_both_b_and_d_nonzero_raises(self):
        t = (1.0, 0.7, 100.0, -0.3, -1.0, 200.0)
        with pytest.raises(ValueError) as exc_info:
            _transform_from_attr(t)
        msg = str(exc_info.value)
        # Error message names the offending values
        assert "0.7" in msg
        assert "-0.3" in msg

    def test_axis_aligned_passes(self):
        t = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
        gt = _transform_from_attr(t)
        assert isinstance(gt, GeoTransform)
        assert gt.pixel_width == 30.0
        assert gt.pixel_height == -30.0
        assert gt.origin_x == 500000.0
        assert gt.origin_y == 4000000.0

    def test_float_noise_at_zero_accepted(self):
        # 1e-15 is well below the 1e-12 tolerance: should pass
        t = (1.0, 1e-15, 100.0, -1e-15, -1.0, 200.0)
        gt = _transform_from_attr(t)
        assert isinstance(gt, GeoTransform)
        assert gt.origin_x == 100.0

    def test_tolerance_boundary_rejected(self):
        # 1e-10 is well above the 1e-12 tolerance: should be rejected
        t = (1.0, 1e-10, 100.0, 0.0, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_geotransform_instance_passes_through(self):
        gt_in = GeoTransform(
            origin_x=1.0, origin_y=2.0, pixel_width=0.1, pixel_height=-0.1,
        )
        gt_out = _transform_from_attr(gt_in)
        assert gt_out is gt_in

    def test_none_returns_none(self):
        assert _transform_from_attr(None) is None

    def test_wrong_length_returns_none(self):
        # Preserve existing behaviour: invalid shape is silently ignored
        assert _transform_from_attr((1.0, 0.0, 100.0)) is None


class TestToGeotiffRejectsRotated:
    """End-to-end: to_geotiff surfaces the ValueError."""

    def test_to_geotiff_rotated_raises(self, tmp_path):
        da = _make_da_1764((1.0, 0.5, 100.0, 0.0, -1.0, 200.0))
        out = str(tmp_path / "rotated_1764.tif")
        with pytest.raises(ValueError, match="rotation/shear"):
            to_geotiff(da, out, compression="none")

    def test_to_geotiff_skewed_raises(self, tmp_path):
        da = _make_da_1764((1.0, 0.0, 100.0, 0.4, -1.0, 200.0))
        out = str(tmp_path / "skewed_1764.tif")
        with pytest.raises(ValueError, match="rotation/shear"):
            to_geotiff(da, out, compression="none")

    def test_to_geotiff_axis_aligned_still_writes(self, tmp_path):
        da = _make_da_1764((30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0))
        out = str(tmp_path / "axis_aligned_1764.tif")
        to_geotiff(da, out, compression="none")
        # Round-trip: read back and confirm the transform survived
        result = open_geotiff(out)
        a, b, c, d, e, f = result.attrs["transform"]
        assert b == 0.0 and d == 0.0
        assert a == pytest.approx(30.0)
        assert e == pytest.approx(-30.0)
        assert c == pytest.approx(500000.0)
        assert f == pytest.approx(4000000.0)


# ===========================================================================
# allow_rotated=True surfaces attrs['rotated_affine'].
# ===========================================================================

_ROTATED_TUPLE_2129 = (8.66, -5.0, 100.0, 5.0, 8.66, 200.0)


def _rotated_geo_info_2129(*, with_crs: bool = True) -> GeoInfo:
    """Build a GeoInfo that mimics the ``allow_rotated=True`` parser output."""
    t = GeoTransform(
        origin_x=0.0, origin_y=0.0,
        pixel_width=1.0, pixel_height=-1.0,
        rotated_affine=_ROTATED_TUPLE_2129,
    )
    return GeoInfo(
        transform=t,
        has_georef=False,
        crs_epsg=4326 if with_crs else None,
        crs_wkt='GEOGCS["WGS 84"]' if with_crs else None,
    )


def test_rotated_optin_emits_rotated_affine_tuple():
    gi = _rotated_geo_info_2129()
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert attrs.get('rotated_affine') == _ROTATED_TUPLE_2129
    # Sanity: rotated path still drops crs / transform (existing
    # contract). Re-checked here so a regression to either branch
    # surfaces in the same test file as the new attr.
    assert 'crs' not in attrs
    assert 'transform' not in attrs


def test_rotated_optin_emits_rotated_affine_without_crs():
    gi = _rotated_geo_info_2129(with_crs=False)
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert attrs.get('rotated_affine') == _ROTATED_TUPLE_2129


def test_plain_no_georef_omits_rotated_affine():
    # Plain no-georef file: no transform tag, no rotated matrix. The new
    # attr must NOT appear on this path -- it is reserved for the
    # ``allow_rotated=True`` opt-in.
    gi = GeoInfo(transform=GeoTransform(), has_georef=False, crs_epsg=4326)
    attrs: dict = {}
    _populate_attrs_from_geo_info(attrs, gi)

    assert 'rotated_affine' not in attrs


def test_axis_aligned_read_omits_rotated_affine():
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

    assert 'rotated_affine' not in attrs


def test_rotated_affine_is_tuple_not_list():
    # ``GeoTransform.rotated_affine`` is documented as a tuple; the
    # public attr should respect the same type so downstream code can
    # rely on ``isinstance(attrs['rotated_affine'], tuple)``. The
    # parser tuple-casts the field even if a future change stores it
    # as a list or numpy sequence.
    gi = _rotated_geo_info_2129()
    # Replace with a list to simulate a parser change.
    gi.transform.rotated_affine = list(_ROTATED_TUPLE_2129)
    md = geo_info_to_metadata(gi)

    assert isinstance(md.rotated_affine, tuple)
    assert md.rotated_affine == _ROTATED_TUPLE_2129


def test_attrs_to_metadata_drops_rotated_affine():
    """The write-side boundary parser intentionally does not carry the
    rotated 6-tuple forward; the writer would otherwise need a
    ``ModelTransformationTag`` emit path. Keeping it off the record
    ensures ``to_geotiff`` keeps writing a plain no-georef file until
    that follow-up lands."""
    attrs = {
        'rotated_affine': _ROTATED_TUPLE_2129,
        '_xrspatial_no_georef': True,
        '_xrspatial_geotiff_contract': _ATTRS_CONTRACT_VERSION,
    }
    md = attrs_to_metadata(attrs)

    assert md.rotated_affine is None
    assert md.has_georef is False
    assert md.transform is None


def _write_rotated_tiff_2129(path, arr, *, epsg=None):
    """Write a rotated GeoTIFF with an optional GeoKey-encoded CRS.

    The 4x4 ``ModelTransformationTag`` matrix encodes a 30-degree
    rotation with 10-unit pixel spacing. Mirrors the helper in
    ``read/test_crs.py`` so the two suites share the same synthetic
    file shape.
    """
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [
        # ModelTransformationTag (34264) -- DOUBLE, count=16.
        (34264, 12, 16, m, False),
    ]
    if epsg is not None:
        # GeoKeyDirectory: header (4 shorts) + one key for GeographicTypeGeoKey.
        geo_key_directory = (
            1, 1, 0, 1,
            2048, 0, 1, int(epsg),
        )
        extratags.append((34735, 3, 8, geo_key_directory, False))
    tifffile.imwrite(
        path, arr, photometric='minisblack',
        planarconfig='contig', extratags=extratags,
    )
    return m


def test_open_geotiff_rotated_emits_rotated_affine(tmp_path):
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    src = tmp_path / "tmp_2129_rotated_attr_eager.tif"
    m = _write_rotated_tiff_2129(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True)

    expected = (m[0], m[1], m[3], m[4], m[5], m[7])
    assert da.attrs.get('rotated_affine') == expected
    assert isinstance(da.attrs['rotated_affine'], tuple)
    # CRS attrs stay dropped on this path.
    assert 'crs' not in da.attrs
    assert 'transform' not in da.attrs


def test_open_geotiff_rotated_emits_rotated_affine_dask(tmp_path):
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    src = tmp_path / "tmp_2129_rotated_attr_dask.tif"
    m = _write_rotated_tiff_2129(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True, chunks=4)

    expected = (m[0], m[1], m[3], m[4], m[5], m[7])
    assert da.attrs.get('rotated_affine') == expected


def test_open_geotiff_plain_no_georef_omits_rotated_affine(tmp_path):
    """A plain TIFF with no transform tags must not grow a
    ``rotated_affine`` attr. The opt-in is rotation-specific."""
    arr = np.arange(12, dtype='<u2').reshape(3, 4)
    src = tmp_path / "tmp_2129_plain_no_georef.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' not in da.attrs


def _write_rotated_vrt_2129(tmp_path, src_path, *, gt_str, size=4):
    """Build a VRT pointing at ``src_path`` with a custom ``<GeoTransform>``.

    GDAL ``geo_transform`` ordering is
    ``(origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height)``;
    non-zero ``rot_x`` / ``rot_y`` is what the reader treats as a
    rotated VRT (see ``_vrt_is_rotated`` in ``_backends/vrt.py``).
    """
    vrt_xml = (
        f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n'
        f'  <GeoTransform>{gt_str}</GeoTransform>\n'
        f'  <VRTRasterBand dataType="UInt16" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt = tmp_path / "tmp_2129_rotated.vrt"
    vrt.write_text(vrt_xml)
    return str(vrt)


def test_open_geotiff_rotated_vrt_emits_rotated_affine(tmp_path):
    """A VRT with a rotated ``<GeoTransform>`` opened with
    ``allow_rotated=True`` lands in ``georef_status='rotated_dropped'``
    and must surface ``rotated_affine`` so callers can recover the
    mapping. Mirrors the non-VRT ``ModelTransformationTag`` path."""
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_rotated_vrt_src.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    # GDAL geo_transform: origin_x, res_x, rot_x, origin_y, rot_y, res_y.
    # Non-zero rotation terms (positions 2 and 4) trigger the rotated
    # path. rasterio Affine ordering: (a, b, c, d, e, f)
    # = (res_x, rot_x, origin_x, rot_y, res_y, origin_y).
    gt_str = "100.0, 10.0, 5.0, 200.0, -5.0, -10.0"
    expected = (10.0, 5.0, 100.0, -5.0, -10.0, 200.0)

    vrt_path = _write_rotated_vrt_2129(tmp_path, str(src), gt_str=gt_str)
    da = open_geotiff(vrt_path, allow_rotated=True)

    assert da.attrs.get('rotated_affine') == expected
    assert isinstance(da.attrs['rotated_affine'], tuple)
    # Same drops as the non-VRT path.
    assert 'crs' not in da.attrs
    assert 'transform' not in da.attrs


def test_open_geotiff_rotated_vrt_emits_rotated_affine_dask(tmp_path):
    """Chunked VRT read path mirrors the eager path: ``rotated_affine``
    rides on the metadata record at the chunked build site too."""
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_rotated_vrt_src_dask.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    gt_str = "100.0, 10.0, 5.0, 200.0, -5.0, -10.0"
    expected = (10.0, 5.0, 100.0, -5.0, -10.0, 200.0)

    vrt_path = _write_rotated_vrt_2129(tmp_path, str(src), gt_str=gt_str)
    da = open_geotiff(vrt_path, allow_rotated=True, chunks=2)

    assert da.attrs.get('rotated_affine') == expected


def test_open_geotiff_axis_aligned_vrt_omits_rotated_affine(tmp_path):
    """An axis-aligned VRT (zero rotation terms) round-trips via
    ``attrs['transform']`` and must NOT grow a ``rotated_affine``."""
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    src = tmp_path / "tmp_2129_axis_vrt_src.tif"
    tifffile.imwrite(str(src), arr, photometric='minisblack',
                     planarconfig='contig')

    gt_str = "100.0, 10.0, 0.0, 200.0, 0.0, -10.0"
    vrt_path = _write_rotated_vrt_2129(tmp_path, str(src), gt_str=gt_str)

    da = open_geotiff(vrt_path, allow_rotated=True)
    assert 'rotated_affine' not in da.attrs


def test_open_geotiff_axis_aligned_omits_rotated_affine(tmp_path):
    """An axis-aligned ``ModelTransformationTag`` round-trips via
    ``attrs['transform']``; the rotated attr must stay absent so
    downstream code does not branch on it for non-rotated reads."""
    arr = np.arange(12, dtype='<u2').reshape(3, 4)
    src = tmp_path / "tmp_2129_axis_aligned.tif"
    m = (
        10.0, 0.0, 0.0, 100.0,
        0.0, -10.0, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    tifffile.imwrite(
        str(src), arr, photometric='minisblack', planarconfig='contig',
        extratags=[(34264, 12, 16, m, False)],
    )

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' not in da.attrs
    assert da.attrs.get('transform') == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


# ===========================================================================
# GeoTIFF and VRT rotated reads raise the same typed error.
# ===========================================================================

_COS30 = 0.8660254037844387
_SIN30 = 0.5
_ROTATED_M_2267 = (
    10.0 * _COS30, -10.0 * _SIN30, 0.0, 100.0,
    10.0 * _SIN30, 10.0 * _COS30, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _make_rotated_ifd_2267() -> IFD:
    """Build an IFD that carries only a rotated ModelTransformationTag."""
    ifd = IFD()
    ifd.entries[TAG_MODEL_TRANSFORMATION] = IFDEntry(
        tag=TAG_MODEL_TRANSFORMATION,
        type_id=12,  # DOUBLE
        count=16,
        value=_ROTATED_M_2267,
    )
    return ifd


def _write_rotated_tiff_2267(path, arr: np.ndarray) -> None:
    """Minimal rotated TIFF writer.

    Single-band, single-strip, uncompressed, with only a rotated
    ``ModelTransformationTag``. Self-contained so this test does not
    depend on any external fixture module surviving in its current shape.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries = []
    entries.append((256, 3, 1, w))
    entries.append((257, 3, 1, h))
    entries.append((258, 3, 1, 16))
    entries.append((259, 3, 1, 1))
    entries.append((262, 3, 1, 1))
    entries.append((273, 4, 1, header_size))
    entries.append((277, 3, 1, 1))
    entries.append((278, 3, 1, h))
    entries.append((279, 4, 1, strip_size))
    entries.append((339, 3, 1, 1))
    entries.append((TAG_MODEL_TRANSFORMATION, 12, 16, transform_off))
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
        f.write(struct.pack('<16d', *_ROTATED_M_2267))
        f.write(ifd_bytes)


def test_extract_transform_rotated_raises_typed_error():
    """``_extract_transform`` raises ``RotatedTransformError`` on a
    rotated ``ModelTransformationTag``. This is the bug fix at the
    source: ``raise NotImplementedError`` -> ``raise
    RotatedTransformError`` at ``_geotags.py:679``.
    """
    ifd = _make_rotated_ifd_2267()
    with pytest.raises(RotatedTransformError):
        _extract_transform(ifd)


def test_rotated_error_is_geotiff_ambiguous_subclass():
    """``RotatedTransformError`` keeps its membership in the
    ``GeoTIFFAmbiguousMetadataError`` family so callers using the
    family ``except`` still work. ``ValueError`` membership is the
    other contract (legacy code catching ``ValueError`` keeps working).
    """
    ifd = _make_rotated_ifd_2267()
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        _extract_transform(ifd)
    # And ValueError still catches it because RotatedTransformError ->
    # GeoTIFFAmbiguousMetadataError -> ValueError.
    with pytest.raises(ValueError):
        _extract_transform(ifd)


def test_rotated_error_not_a_notimplemented_subclass():
    """Type-hierarchy invariant: ``RotatedTransformError`` must NOT be
    a subclass of ``NotImplementedError`` (which descends from
    ``RuntimeError``). This is a deliberate contract: a caller using
    ``except NotImplementedError`` does not catch the rotated case, which
    is the intended breaking behaviour. Pinning the relationship here
    makes any
    accidental re-parenting of the typed-error hierarchy fail loudly
    instead of silently restoring the old contract.
    """
    assert not issubclass(RotatedTransformError, NotImplementedError)
    # And the inverse: NotImplementedError must not become a
    # GeoTIFFAmbiguousMetadataError either (would let unrelated callers
    # accidentally catch unrelated errors).
    assert not issubclass(NotImplementedError, GeoTIFFAmbiguousMetadataError)


def test_extract_transform_message_preserved():
    """Message content should still name the tag and the rotation
    issue, so user-facing diagnostics survive the type change.
    """
    ifd = _make_rotated_ifd_2267()
    with pytest.raises(RotatedTransformError) as exc:
        _extract_transform(ifd)
    msg = str(exc.value)
    assert 'ModelTransformationTag' in msg
    assert 'rotation' in msg.lower() or 'skew' in msg.lower()
    assert 'allow_rotated' in msg


def test_open_geotiff_rotated_raises_typed_error(tmp_path):
    """``open_geotiff`` on a rotated GeoTIFF raises
    ``RotatedTransformError``. This is the user-facing parity case
    with the VRT path (covered in
    ``unit/test_metadata.py::test_read_rejects_rotated_vrt``).
    """
    src = tmp_path / "tmp_2267_open_geotiff_rotated.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_2267(str(src), arr)
    with pytest.raises(RotatedTransformError):
        open_geotiff(str(src))


def test_open_geotiff_rotated_allow_rotated_still_reads(tmp_path):
    """The opt-out is still wired through: ``allow_rotated=True`` reads
    the pixel grid without raising. This pins that switching the
    exception type did not break the existing escape hatch.
    """
    src = tmp_path / "tmp_2267_open_geotiff_rotated_optin.tif"
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff_2267(str(src), arr)
    da = open_geotiff(str(src), allow_rotated=True)
    assert da.shape == arr.shape
    np.testing.assert_array_equal(da.values, arr)


# ===========================================================================
# write side: degenerate georeferenced writes preserve their transform.
# ===========================================================================

_PIXEL_1945 = 1.0
_X0_1945 = -120.0
_Y0_1945 = 45.0


def _strip_1xN(raster_type: str = "area") -> xr.DataArray:
    # The borrow-from-other-axis fallback is opt-in. These tests document
    # the borrow path, so they set the opt-in flag. Round-trip semantics
    # are unchanged once the flag is set.
    attrs = {"crs": 4326, "assume_square_pixels_for_degenerate_axis": True}
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(1, 8),
        dims=("y", "x"),
        coords={
            "x": _X0_1945 + np.arange(8, dtype="float64") * _PIXEL_1945,
            "y": np.array([_Y0_1945], dtype="float64"),
        },
        attrs=attrs,
    )


def _strip_Nx1(raster_type: str = "area") -> xr.DataArray:
    attrs = {"crs": 4326, "assume_square_pixels_for_degenerate_axis": True}
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(8, 1),
        dims=("y", "x"),
        coords={
            "x": np.array([_X0_1945], dtype="float64"),
            "y": _Y0_1945 - np.arange(8, dtype="float64") * _PIXEL_1945,
        },
        attrs=attrs,
    )


def _pixel_1x1_with_transform() -> xr.DataArray:
    # rasterio-style 6-tuple: (a, b, c, d, e, f) = (px, 0, ox, 0, py, oy)
    transform = (_PIXEL_1945, 0.0, _X0_1945 - _PIXEL_1945 * 0.5,
                 0.0, -_PIXEL_1945, _Y0_1945 + _PIXEL_1945 * 0.5)
    return xr.DataArray(
        np.array([[42.0]], dtype="float32"),
        dims=("y", "x"),
        coords={"x": np.array([_X0_1945]), "y": np.array([_Y0_1945])},
        attrs={"crs": 4326, "transform": transform},
    )


def _pixel_1x1_no_transform() -> xr.DataArray:
    return xr.DataArray(
        np.array([[42.0]], dtype="float32"),
        dims=("y", "x"),
        coords={"x": np.array([_X0_1945]), "y": np.array([_Y0_1945])},
        attrs={"crs": 4326},
    )


def _assert_round_trip_1945(path: str, expected: xr.DataArray) -> None:
    """Read back ``path`` and check coords/transform survived the write."""
    r = open_geotiff(path)

    # Coordinates must be float (georeferenced), not integer pixel indices.
    assert np.issubdtype(r.coords["x"].dtype, np.floating), (
        f"x coord came back as {r.coords['x'].dtype}; "
        "writer stripped georeferencing"
    )
    assert np.issubdtype(r.coords["y"].dtype, np.floating), (
        f"y coord came back as {r.coords['y'].dtype}; "
        "writer stripped georeferencing"
    )

    np.testing.assert_allclose(
        r.coords["x"].values, expected.coords["x"].values, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        r.coords["y"].values, expected.coords["y"].values, rtol=0, atol=1e-9
    )

    assert r.attrs.get("transform") is not None, "transform missing from attrs"

    np.testing.assert_array_equal(np.asarray(r.values), np.asarray(expected.values))


class TestEagerWriterDegenerateGeoref:

    def test_1xN_round_trip_preserves_transform(self, tmp_path):
        da = _strip_1xN()
        p = str(tmp_path / "georef_1xN_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)

    def test_Nx1_round_trip_preserves_transform(self, tmp_path):
        da = _strip_Nx1()
        p = str(tmp_path / "georef_Nx1_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)

    def test_1x1_with_transform_attr_round_trips(self, tmp_path):
        da = _pixel_1x1_with_transform()
        p = str(tmp_path / "georef_1x1_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)

    def test_1x1_without_transform_raises(self, tmp_path):
        """1x1 cannot recover pixel size from coords alone; raise rather
        than silently strip georeferencing."""
        da = _pixel_1x1_no_transform()
        p = str(tmp_path / "georef_1x1_no_tx_eager_1945.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


class TestDaskNumpyWriterDegenerateGeoref:

    def test_1xN_dask_round_trip(self, tmp_path):
        da = _strip_1xN().chunk({"x": 4, "y": 1})
        p = str(tmp_path / "georef_1xN_dask_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, _strip_1xN())

    def test_Nx1_dask_round_trip(self, tmp_path):
        da = _strip_Nx1().chunk({"x": 1, "y": 4})
        p = str(tmp_path / "georef_Nx1_dask_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, _strip_Nx1())


@_gpu_only
class TestGpuWriterDegenerateGeoref:

    def test_1xN_round_trip_preserves_transform(self, tmp_path):
        import cupy
        da_cpu = _strip_1xN()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1xN_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip_1945(p, da_cpu)

    def test_Nx1_round_trip_preserves_transform(self, tmp_path):
        import cupy
        da_cpu = _strip_Nx1()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_Nx1_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip_1945(p, da_cpu)

    def test_1x1_with_transform_attr_round_trips(self, tmp_path):
        import cupy
        da_cpu = _pixel_1x1_with_transform()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1x1_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip_1945(p, da_cpu)

    def test_1x1_without_transform_raises(self, tmp_path):
        import cupy
        da_cpu = _pixel_1x1_no_transform()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1x1_no_tx_gpu_1945.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            write_geotiff_gpu(da_gpu, p)


@_gpu_only
class TestDaskCupyWriterDegenerateGeoref:

    def test_1xN_dask_cupy_round_trip(self, tmp_path):
        import cupy
        da_cpu = _strip_1xN()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu = da_gpu.chunk({"x": 4, "y": 1})
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1xN_dask_cupy_1945.tif")
        to_geotiff(da_gpu, p)
        _assert_round_trip_1945(p, da_cpu)

    def test_Nx1_dask_cupy_round_trip(self, tmp_path):
        import cupy
        da_cpu = _strip_Nx1()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu = da_gpu.chunk({"x": 1, "y": 4})
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_Nx1_dask_cupy_1945.tif")
        to_geotiff(da_gpu, p)
        _assert_round_trip_1945(p, da_cpu)


class TestVrtTiledWriterDegenerateGeoref:
    """``to_geotiff(da, '*.vrt')`` dispatches to ``_write_vrt_tiled``."""

    def test_1xN_vrt_round_trip(self, tmp_path):
        da = _strip_1xN()
        p = str(tmp_path / "georef_1xN_1945.vrt")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)

    def test_Nx1_vrt_round_trip(self, tmp_path):
        da = _strip_Nx1()
        p = str(tmp_path / "georef_Nx1_1945.vrt")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)

    def test_1x1_vrt_without_transform_raises(self, tmp_path):
        da = _pixel_1x1_no_transform()
        p = str(tmp_path / "georef_1x1_no_tx_1945.vrt")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


class TestEagerWriterPointRasterDegenerate:

    @pytest.mark.parametrize(
        "factory, name",
        [(_strip_1xN, "1xN"), (_strip_Nx1, "Nx1")],
    )
    def test_point_raster_round_trip(self, tmp_path, factory, name):
        da = factory(raster_type="point")
        p = str(tmp_path / f"georef_{name}_point_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip_1945(p, da)


class TestCoordsToTransformBorrowSignPinning:
    """Pin the sign convention of the borrowed (degenerate) axis."""

    def test_y_ascending_Nx1_borrows_positive_pixel_width(self):
        from xrspatial.geotiff._coords import coords_to_transform

        da = xr.DataArray(
            np.arange(8, dtype="float32").reshape(8, 1),
            dims=("y", "x"),
            coords={
                "x": np.array([_X0_1945], dtype="float64"),
                # y ascending: 38, 39, 40, ... so pixel_height = +1.0
                "y": 38.0 + np.arange(8, dtype="float64") * _PIXEL_1945,
            },
            # Opt in to the borrow-from-other-axis path.
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        # Non-degenerate axis (y) keeps its sign.
        assert t.pixel_height == pytest.approx(1.0)
        # Borrowed pixel_width takes ``abs(pixel_height)``: always positive,
        # regardless of y's sign. This pins the current convention.
        assert t.pixel_width == pytest.approx(1.0)

    def test_x_descending_1xN_borrows_negative_pixel_height(self):
        from xrspatial.geotiff._coords import coords_to_transform

        da = xr.DataArray(
            np.arange(8, dtype="float32").reshape(1, 8),
            dims=("y", "x"),
            coords={
                # x descending: pixel_width = -1.0
                "x": -113.0 - np.arange(8, dtype="float64") * _PIXEL_1945,
                "y": np.array([_Y0_1945], dtype="float64"),
            },
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        # Non-degenerate axis (x) keeps its sign.
        assert t.pixel_width == pytest.approx(-1.0)
        # Borrowed pixel_height takes ``-abs(pixel_width)``: always negative,
        # regardless of x's sign. This pins the current convention.
        assert t.pixel_height == pytest.approx(-1.0)


# ===========================================================================
# write side: no-georef files keep int coords across read -> write -> read.
# ===========================================================================


def test_coords_to_transform_returns_transform_for_int_y_float_x():
    """Mixed int / float coords are user-authored, not the read-side
    sentinel; the sentinel requires both axes to match. The transform
    inference path runs and produces a real GeoTransform."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.float64),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    assert gt.pixel_width == pytest.approx(1.0)
    assert gt.pixel_height == pytest.approx(1.0)


def test_coords_to_transform_returns_transform_for_float_y_int_x():
    """Symmetric check on the other axis."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.float64),
            'x': np.arange(5, dtype=np.int64),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    assert gt.pixel_width == pytest.approx(1.0)
    assert gt.pixel_height == pytest.approx(1.0)


@pytest.mark.parametrize("kind", [np.int32, np.int16, np.uint32])
def test_coords_to_transform_returns_transform_for_non_int64_kinds(kind):
    """The tightened sentinel only matches ``int64``. int32 / int16 /
    uint32 are not the read-side placeholder (the reader explicitly
    uses ``np.int64``), so a user authoring an integer-coord grid in
    those dtypes gets a real transform instead of silent georef loss."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=kind),
            'x': np.arange(5, dtype=kind),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None, (
        f"expected a real GeoTransform for {kind.__name__} coords; "
        f"the sentinel only catches int64 ascending arange")


def test_coords_to_transform_float_coords_unchanged():
    """Float coords still produce a GeoTransform (sanity baseline)."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    assert gt.pixel_width == pytest.approx(10.0)
    assert gt.pixel_height == pytest.approx(1.0)


def test_coords_to_transform_3d_yxband_with_marker_returns_none():
    """3D (y, x, band) carrying the no-georef marker short-circuits.

    The marker ``attrs[_NO_GEOREF_KEY] = True`` is the only no-georef
    signal; the same coord shape without the marker synthesises a real
    transform rather than short-circuiting on coord shape alone. The
    helper picks the y/x dims (filtering out 'band'); the marker check
    applies regardless of the band-axis layout.
    """
    da = xr.DataArray(
        np.zeros((4, 5, 3), dtype=np.float32),
        dims=['y', 'x', 'band'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.int64),
            'band': np.arange(3),
        },
        attrs={_NO_GEOREF_KEY: True},
    )
    assert _coords_to_transform(da) is None


def test_coords_to_transform_3d_yxband_without_marker_synthesises_transform():
    """3D (y, x, band) without the marker now writes a real transform.

    A shape-based check would treat the int64 step-1 coords as the
    no-georef placeholder and return ``None``, silently stripping georef
    from user-authored 3D arrays with integer-aligned coords.
    """
    da = xr.DataArray(
        np.zeros((4, 5, 3), dtype=np.float32),
        dims=['y', 'x', 'band'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.int64),
            'band': np.arange(3),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    assert gt.pixel_width == pytest.approx(1.0)
    assert gt.pixel_height == pytest.approx(1.0)


def _make_no_georef_tiff_1949(path, shape=(4, 5), seed=1949):
    """Write a TIFF with no GeoTIFF tags."""
    arr = np.random.default_rng(seed=seed).random(shape).astype(np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )
    return arr


def test_round_trip_preserves_int_coords_eager(tmp_path):
    """Eager numpy round-trip: y/x coord dtype stays int64 and no
    transform attr is injected."""
    src = str(tmp_path / "no_georef_src_1949.tif")
    _make_no_georef_tiff_1949(src)

    da = open_geotiff(src)
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    assert 'transform' not in da.attrs

    out = str(tmp_path / "no_georef_out_1949.tif")
    to_geotiff(da, out, compression='none', tiled=False)

    da2 = open_geotiff(out)
    assert da2.coords['y'].dtype == np.int64, (
        f"round-trip flipped y coord dtype from int64 to "
        f"{da2.coords['y'].dtype}: {da2.coords['y'].values}")
    assert da2.coords['x'].dtype == np.int64
    assert 'transform' not in da2.attrs, (
        f"round-trip injected fake transform attr: "
        f"{da2.attrs.get('transform')}")
    np.testing.assert_array_equal(
        da2.coords['y'].values, da.coords['y'].values)
    np.testing.assert_array_equal(
        da2.coords['x'].values, da.coords['x'].values)


def test_round_trip_preserves_int_coords_3d(tmp_path):
    """Same round-trip on a 3D multi-band file."""
    src = str(tmp_path / "no_georef_src_3d_1949.tif")
    arr = np.random.default_rng(seed=1949).random(
        (4, 5, 3)).astype(np.float32)
    tifffile.imwrite(src, arr, photometric='minisblack',
                     planarconfig='contig', metadata=None)

    da = open_geotiff(src)
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    assert 'transform' not in da.attrs

    out = str(tmp_path / "no_georef_out_3d_1949.tif")
    to_geotiff(da, out, compression='none')

    da2 = open_geotiff(out)
    assert da2.coords['y'].dtype == np.int64
    assert da2.coords['x'].dtype == np.int64
    assert 'transform' not in da2.attrs


def test_round_trip_dask_streaming_preserves_int_coords(tmp_path):
    """Same contract via the dask streaming writer path."""
    src = str(tmp_path / "no_georef_src_dask_1949.tif")
    arr = np.random.default_rng(seed=1949).random(
        (64, 64)).astype(np.float32)
    tifffile.imwrite(src, arr, photometric='minisblack',
                     planarconfig='contig', tile=(32, 32),
                     compression='deflate', metadata=None)

    da_dk = open_geotiff(src, chunks=32)
    assert da_dk.coords['y'].dtype == np.int64
    assert da_dk.coords['x'].dtype == np.int64
    assert 'transform' not in da_dk.attrs

    out = str(tmp_path / "no_georef_out_dask_1949.tif")
    to_geotiff(da_dk, out, compression='deflate', tile_size=32)

    da_rt = open_geotiff(out)
    assert da_rt.coords['y'].dtype == np.int64
    assert da_rt.coords['x'].dtype == np.int64
    assert 'transform' not in da_rt.attrs


def test_double_round_trip_stable(tmp_path):
    """Two consecutive read -> write -> read cycles produce identical
    coord arrays and attrs sets."""
    src = str(tmp_path / "no_georef_double_1949.tif")
    _make_no_georef_tiff_1949(src, shape=(6, 8))

    da1 = open_geotiff(src)
    out1 = str(tmp_path / "no_georef_double_out1_1949.tif")
    to_geotiff(da1, out1, compression='none', tiled=False)

    da2 = open_geotiff(out1)
    out2 = str(tmp_path / "no_georef_double_out2_1949.tif")
    to_geotiff(da2, out2, compression='none', tiled=False)

    da3 = open_geotiff(out2)
    np.testing.assert_array_equal(
        da1.coords['y'].values, da3.coords['y'].values)
    np.testing.assert_array_equal(
        da1.coords['x'].values, da3.coords['x'].values)
    assert da3.coords['y'].dtype == da1.coords['y'].dtype
    assert da3.coords['x'].dtype == da1.coords['x'].dtype
    # ``transform`` should never appear in either attrs set
    assert 'transform' not in da1.attrs
    assert 'transform' not in da3.attrs


def test_explicit_transform_attr_still_writes(tmp_path):
    """When the user explicitly sets ``attrs['transform']`` on a
    DataArray with int coords (an odd combination, but allowed), the
    writer still honours that explicit value -- the fix only catches the
    implicit-from-int-coords path."""
    arr = np.zeros((4, 5), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.int64),
        },
        attrs={'transform': (30.0, 0.0, 100000.0, 0.0, -30.0, 4000000.0),
               'crs': 32610},
    )
    out = str(tmp_path / "explicit_transform_1949.tif")
    to_geotiff(da, out, compression='none', tiled=False)
    da_rt = open_geotiff(out)
    assert 'transform' in da_rt.attrs
    assert da_rt.attrs['transform'][0] == 30.0
    assert da_rt.attrs['crs'] == 32610


@_gpu_only
def test_gpu_writer_preserves_int_coords(tmp_path):
    """The GPU writer path goes through the same ``_coords_to_transform``
    fallback; the fix has to extend there too."""
    src = str(tmp_path / "no_georef_src_gpu_1949.tif")
    _make_no_georef_tiff_1949(src, shape=(64, 64))

    da_gpu = open_geotiff(src, gpu=True)
    assert da_gpu.coords['y'].dtype == np.int64
    assert da_gpu.coords['x'].dtype == np.int64
    assert 'transform' not in da_gpu.attrs

    out = str(tmp_path / "no_georef_out_gpu_1949.tif")
    to_geotiff(da_gpu, out, compression='deflate')

    da_rt = open_geotiff(out)
    assert da_rt.coords['y'].dtype == np.int64
    assert da_rt.coords['x'].dtype == np.int64
    assert 'transform' not in da_rt.attrs


# ===========================================================================
# write side: to_geotiff refuses to silently drop a rotated affine.
# ===========================================================================

_ROTATED_TUPLE_2216 = (8.66, -5.0, 100.0, 5.0, 8.66, 200.0)


def _write_rotated_tiff_2216(path, arr, *, epsg=None):
    """Write a synthetic rotated GeoTIFF (30-degree rotation)."""
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [(34264, 12, 16, m, False)]
    if epsg is not None:
        geo_key_directory = (
            1, 1, 0, 1,
            2048, 0, 1, int(epsg),
        )
        extratags.append((34735, 3, 8, geo_key_directory, False))
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        extratags=extratags,
    )
    return m


def _rotated_dataarray_2216():
    """Build a 2D DataArray that mimics ``open_geotiff(..., allow_rotated=True)``.

    Avoids depending on the synthetic-TIFF helper for the parsing-side
    tests so the rejection logic is exercised on the boundary regardless
    of whether the synthetic file produced the exact attrs the reader
    would emit.
    """
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(4), 'x': np.arange(5)},
        attrs={'rotated_affine': _ROTATED_TUPLE_2216},
    )


def test_to_geotiff_rejects_rotated_affine_without_opt_in(tmp_path):
    da = _rotated_dataarray_2216()
    out = tmp_path / "tmp_2216_reject.tif"

    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    # The error names the opt-in so the caller learns the flag.
    with pytest.raises(ValueError, match="drop_rotation=True"):
        to_geotiff(da, str(out))

    # Nothing got written.
    assert not out.exists()


def test_to_geotiff_error_message_points_at_issue(tmp_path):
    """The rejection message embeds the ``#2216`` issue reference. The
    check is on that token, not on surrounding phrasing, so the wording
    can evolve without breaking the test."""
    da = _rotated_dataarray_2216()
    out = tmp_path / "tmp_2216_issue_ref.tif"

    with pytest.raises(ValueError, match="#2216"):
        to_geotiff(da, str(out))


def test_to_geotiff_drop_rotation_writes_axis_aligned_file(tmp_path):
    """``drop_rotation=True`` lets the write proceed and the output is a
    plain axis-aligned (non-rotated) TIFF -- the round-trip reader sees
    no ``rotated_affine`` attr."""
    da = _rotated_dataarray_2216()
    out = tmp_path / "tmp_2216_drop.tif"

    to_geotiff(da, str(out), drop_rotation=True)

    # Sanity: the file exists and re-opens.
    assert out.exists()
    da2 = open_geotiff(str(out))

    # The rotated attr is gone on the round-trip (the writer has no
    # ModelTransformationTag emit path, so the on-disk file is
    # axis-aligned; the reader's normal path therefore sees no
    # rotated-tag and emits no rotated_affine attr).
    assert 'rotated_affine' not in da2.attrs


def test_to_geotiff_drop_rotation_preserves_pixel_values(tmp_path):
    """The opt-in only drops the rotated *georeferencing*; the pixel
    grid itself round-trips unchanged."""
    da = _rotated_dataarray_2216()
    out = tmp_path / "tmp_2216_pixels.tif"

    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))

    np.testing.assert_array_equal(
        np.asarray(da2.data, dtype=np.float32),
        np.asarray(da.data, dtype=np.float32),
    )


def test_to_geotiff_normal_raster_unchanged(tmp_path):
    """A DataArray with no ``rotated_affine`` attr writes the same way
    it always did -- the new gate must not change behaviour for the
    common path."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    da = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(3), 'x': np.arange(4)},
        attrs={},
    )
    out = tmp_path / "tmp_2216_normal.tif"

    to_geotiff(da, str(out))
    assert out.exists()

    # And explicitly setting ``drop_rotation=True`` on a non-rotated
    # input is a no-op; the kwarg only affects the rotated path.
    out2 = tmp_path / "tmp_2216_normal_optin.tif"
    to_geotiff(da, str(out2), drop_rotation=True)
    assert out2.exists()


def test_to_geotiff_rotated_affine_none_does_not_trigger(tmp_path):
    """A literal ``attrs['rotated_affine'] = None`` does not trigger
    the gate. The check is on the attr's truthiness so a future read
    path that pre-allocates the key with ``None`` does not break the
    common write path."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    da = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'rotated_affine': None},
    )
    out = tmp_path / "tmp_2216_none.tif"

    to_geotiff(da, str(out))
    assert out.exists()


def test_round_trip_rotated_tiff_requires_opt_in(tmp_path):
    """Read a rotated TIFF with ``allow_rotated=True``, then attempt to
    write it back. Without the opt-in the write raises; with the opt-in
    it succeeds and the round-trip output is axis-aligned."""
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    src = tmp_path / "tmp_2216_round_trip_src.tif"
    _write_rotated_tiff_2216(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' in da.attrs

    # Fail closed without opt-in.
    out = tmp_path / "tmp_2216_round_trip_out.tif"
    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    # Opt-in succeeds; the read-back file is plain axis-aligned.
    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))
    assert 'rotated_affine' not in da2.attrs


def test_round_trip_dask_rotated_tiff_requires_opt_in(tmp_path):
    """Same contract on the dask read path -- ``allow_rotated=True``
    with ``chunks=`` lands the attr on the lazy DataArray, and the
    writer's gate fires before any tile streaming begins."""
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    src = tmp_path / "tmp_2216_round_trip_dask_src.tif"
    _write_rotated_tiff_2216(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True, chunks=4)
    assert 'rotated_affine' in da.attrs

    out = tmp_path / "tmp_2216_round_trip_dask_out.tif"
    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))
    assert 'rotated_affine' not in da2.attrs


def test_to_geotiff_vrt_path_rejects_rotated_affine(tmp_path):
    """The VRT branch of ``to_geotiff`` runs the same gate so the
    silent-loss surface is uniform across single-file and tiled-VRT
    outputs."""
    da = _rotated_dataarray_2216()
    vrt_out = tmp_path / "tmp_2216_vrt.vrt"

    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(vrt_out))

    # Underlying tiles directory should not exist on the failed write.
    tiles_dir = tmp_path / "tmp_2216_vrt_tiles"
    assert not tiles_dir.exists() or not list(tiles_dir.iterdir())


def test_write_vrt_tiled_direct_call_rejects_rotated_affine(tmp_path):
    """A direct ``_write_vrt_tiled`` call bypasses the ``to_geotiff``
    wrapper but must still refuse rotated inputs. Without the explicit
    gate inside ``_write_vrt_tiled`` the helper would silently produce
    identity-affine tiles."""
    da = _rotated_dataarray_2216()
    vrt_out = tmp_path / "tmp_2216_vrt_direct.vrt"

    with pytest.raises(ValueError, match="rotated_affine"):
        _write_vrt_tiled(da, str(vrt_out))

    # The error names the function actually running the check, not the
    # public wrapper, so a direct caller of the private helper learns
    # which entry point fired.
    with pytest.raises(ValueError, match="_write_vrt_tiled"):
        _write_vrt_tiled(da, str(vrt_out))
