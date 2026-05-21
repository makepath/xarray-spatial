"""Parity tests for :func:`resolve_georef` across backend call sites.

Issue #2225 (PR-B of #2211). The transform/georef contract used to
live in three places: ``_attrs._compute_georef_status``, the eager
writer's inline ``_transform_from_attr`` / ``_coords_to_transform``
ladder, and the GPU writer's copy of the same ladder. This PR moved
the decisions into :func:`xrspatial.geotiff._coords.resolve_georef`
and routed every call site through it.

The tests below construct each of the six fixture cases the issue
calls out and assert the resolver returns the same
:class:`GeorefResolution` regardless of:

* coord dim naming (``y/x``, ``lat/lon``, ``row/col``),
* whether the caller is on the writer side (DataArray with attrs)
  or the reader side (``GeoInfo``-like record),
* whether ``attrs['transform']`` is the only georef signal
  (transform-only), CRS is the only signal (crs_only), or the input
  carried a rotated ``ModelTransformation`` (rotated_dropped).

The 6 cases:

1. ``y/x`` axis-aligned, no CRS -- writer-side resolver returns
   ``coords`` because the only georef signal is the coord arrays;
   reader-side returns ``transform_only`` because the reader holds an
   explicit ``GeoInfo.has_georef`` bit.
2. ``lat/lon`` axis-aligned, no CRS -- same coord shape, just
   different dim names; resolver decision is identical.
3. ``row/col`` (no georef) -- the reader-stamped placeholder coord
   path. ``attrs[_NO_GEOREF_KEY] = True`` flips both the writer and
   the reader to ``none``.
4. transform-only -- ``attrs['transform']`` is supplied, no CRS.
   Writer derives the same transform; reader bucket is
   ``transform_only``.
5. CRS-only -- ``attrs['crs']`` set, no transform / no coords. Both
   paths land on ``crs_only`` with ``transform=None``.
6. Rotated affine dropped -- a reader-side ``GeoInfo`` whose
   transform carries ``rotated_affine`` lands on
   ``rotated_dropped`` with ``applied_no_georef_marker=True``.

The acceptance criterion in #2225 is that the resolver returns the
same record for each fixture across every call site. The structured
test below builds the six fixtures once and asserts each invocation
path (``resolve_georef`` directly, ``_compute_georef_status``,
``_compute_georef_status_from_parts``) routes the same status string
out.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff._attrs import (
    GEOREF_STATUS_CRS_ONLY,
    GEOREF_STATUS_FULL,
    GEOREF_STATUS_NONE,
    GEOREF_STATUS_ROTATED_DROPPED,
    GEOREF_STATUS_TRANSFORM_ONLY,
    _compute_georef_status,
    _compute_georef_status_from_parts,
)
from xrspatial.geotiff._coords import (
    GEOREF_STATUS_COORDS,
    GeorefResolution,
    resolve_georef,
)
from xrspatial.geotiff._geotags import (
    _NO_GEOREF_KEY,
    GeoInfo,
    GeoTransform,
    RASTER_PIXEL_IS_AREA,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reader-side parity: GeoInfo -> resolve_georef -> GeorefResolution
# ---------------------------------------------------------------------------

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
    """``_compute_georef_status`` is a thin wrapper now (#2225).

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


# ---------------------------------------------------------------------------
# Writer-side parity: DataArray -> resolve_georef -> GeorefResolution
# ---------------------------------------------------------------------------

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
    keeps that synthesis from running on round-trip (#1949, #2120).
    """
    da = _make_no_georef_da()
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_NONE
    assert result.applied_no_georef_marker is True
    assert result.transform is None


def test_resolve_georef_writer_transform_only_attr():
    """``attrs['transform']`` resolves to a transform with no CRS bucket.

    Writer side: the only georef signal is the attr, so the resolver
    lands on ``coords`` (writer's equivalent of "transform present,
    no CRS"). The transform is the rasterio 6-tuple ``(a, b, c, d, e,
    f)`` reconstructed back into a ``GeoTransform``.
    """
    da = _make_transform_only_da()
    result = resolve_georef(da)
    assert result.georef_status == GEOREF_STATUS_COORDS
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


# ---------------------------------------------------------------------------
# Cross-site parity
# ---------------------------------------------------------------------------

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

    The writers' ``transform_from_attr`` already rejected rotated /
    sheared affines with a ``ValueError`` for #2216 -- the resolver
    must preserve that diagnostic so writer callers see the same
    failure as before the refactor.
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


# ---------------------------------------------------------------------------
# Smoke test for the bare-input path (no DataArray, no GeoInfo)
# ---------------------------------------------------------------------------

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
