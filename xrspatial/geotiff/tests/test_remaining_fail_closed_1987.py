"""Fail-closed checks for ambiguous geospatial metadata (issue #1987).

PR 1 of the #1987 series wired ``ConflictingCRSError`` (PR 6 in the
issue's PR numbering). This file pins the remaining four checks that
land in the bundled follow-up:

* ``UnparseableCRSError`` (#1987 PR 2): read + write paths refuse a
  CRS string that pyproj cannot parse. The existing write-side
  ``_validate_crs_fallback`` raise was retyped from ``ValueError`` to
  the new subclass; the read-side check is new.
* ``RotatedTransformError`` (#1987 PR 3): the four read entry points
  refuse a non-axis-aligned affine transform. Opt-out via
  ``allow_rotated=True``.
* ``NonUniformCoordsError`` (#1987 PR 4): the writer refuses coords
  that imply a non-uniform pixel grid. The int-dtype sentinel
  exemption from #1969 still applies.
* ``ConflictingNodataError`` (#1987 PR 7): the writer refuses a
  DataArray whose ``attrs['nodata']`` disagrees with every concrete
  entry in ``attrs['nodatavals']``. Opt-out via explicit ``nodata=``
  kwarg.

``MixedBandMetadataError`` (#1987 PR 5) is intentionally NOT
registered in this PR (see the corresponding ``# register_...`` comment
in ``_validation.py``); the migration cost on existing VRT fixtures is
its own follow-up.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    ConflictingNodataError,
    GeoTIFFAmbiguousMetadataError,
    NonUniformCoordsError,
    RotatedTransformError,
    UnparseableCRSError,
    open_geotiff,
    to_geotiff,
)
from xrspatial.geotiff._validation import (
    _check_read_rotated_transform,
    _check_read_unparseable_crs,
    _check_write_conflicting_nodata,
    _check_write_non_uniform_coords,
    _registered_read_metadata_checks,
    _registered_write_metadata_checks,
)

pyproj = pytest.importorskip("pyproj")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _da(*, coords=None, attrs=None, shape=(4, 4)):
    """Build a minimal 2-D DataArray. Caller supplies coords + attrs."""
    data = np.zeros(shape, dtype=np.float32)
    if coords is None:
        coords = {
            'y': np.linspace(3.0, 0.0, shape[0], dtype=np.float64),
            'x': np.linspace(0.0, 3.0, shape[1], dtype=np.float64),
        }
    return xr.DataArray(
        data, dims=('y', 'x'), coords=coords, attrs=dict(attrs or {}),
    )


def _write_minimal_tiff_with_wkt(path: str, wkt: str) -> None:
    """Hand-build a 4x4 float32 TIFF that stashes ``wkt`` in
    ``GeoAsciiParams`` (tag 34737), referenced from
    ``GeoKeyDirectory`` (tag 34735) as ``GTCitationGeoKey`` (id 1026)."""
    bo = '<'
    pixels = np.zeros((4, 4), dtype=np.float32).tobytes()
    ascii_buf = bytearray((wkt + '|').encode('ascii'))
    # GeoKeyDirectory: header + one entry pointing into GeoAsciiParams.
    gkd = [
        1, 1, 0, 1,
        1026,                  # GTCitationGeoKey
        34737,                 # location = GeoAsciiParams tag
        len(wkt) + 1,          # count (incl. '|')
        0,                     # offset into GeoAsciiParams
    ]
    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals),
                         struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_doubles(tag, vals):
        tag_list.append((tag, 12, len(vals),
                         struct.pack(f'{bo}{len(vals)}d', *vals)))

    def add_ascii(tag, raw_bytes):
        if not raw_bytes.endswith(b'\x00'):
            raw_bytes = raw_bytes + b'\x00'
        tag_list.append((tag, 2, len(raw_bytes), raw_bytes))

    add_short(256, 4)
    add_short(257, 4)
    add_short(258, 32)
    add_short(259, 1)
    add_short(262, 1)
    add_short(277, 1)
    add_short(339, 3)
    add_short(278, 4)
    add_long(273, 0)
    add_long(279, len(pixels))
    add_doubles(33550, [1.0, 1.0, 0.0])
    add_doubles(33922, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add_shorts(34735, gkd)
    add_ascii(34737, bytes(ascii_buf))
    tag_list.sort(key=lambda t: t[0])

    n = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * n + 4
    overflow_start = ifd_start + ifd_size
    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None
    pixel_data_start = overflow_start + len(overflow_buf)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            raw = struct.pack(f'{bo}I', pixel_data_start)
        patched.append((tag, typ, count, raw))
    tag_list = patched
    out = bytearray(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', n))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{bo}I', overflow_start + tag_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixels)
    Path(path).write_bytes(bytes(out))


def _write_rotated_vrt(
    path: Path,
    source_basename: str,
    *,
    geo_transform: str = '0.0, 1.0, 0.5, 0.0, 0.0, -1.0',
) -> None:
    """Build a VRT carrying ``geo_transform``. Default value rotates on
    the x-axis (GDAL ``GT[2] = 0.5``); pass a different string to
    exercise the row-axis (``GT[4]``) branch."""
    Path(path).write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS>EPSG:4326</SRS>\n'
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_basename}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


# ---------------------------------------------------------------------------
# Registry sanity.
# ---------------------------------------------------------------------------


def test_four_checks_registered():
    """Four of the five #1987 PR 2-7 checks are active by default. The
    mixed-band check is intentionally deferred to a follow-up that also
    migrates the legacy VRT test fixtures."""
    write_names = {c.__name__ for c in _registered_write_metadata_checks()}
    read_names = {c.__name__ for c in _registered_read_metadata_checks()}
    assert _check_write_conflicting_nodata.__name__ in write_names
    assert _check_write_non_uniform_coords.__name__ in write_names
    assert _check_read_unparseable_crs.__name__ in read_names
    assert _check_read_rotated_transform.__name__ in read_names


def test_error_class_hierarchy():
    for cls in (
        UnparseableCRSError,
        RotatedTransformError,
        NonUniformCoordsError,
        ConflictingNodataError,
    ):
        assert issubclass(cls, GeoTIFFAmbiguousMetadataError)
        assert issubclass(cls, ValueError)


# ---------------------------------------------------------------------------
# UnparseableCRSError (write side + read side).
# ---------------------------------------------------------------------------


def test_write_rejects_garbage_crs_kwarg(tmp_path):
    """``to_geotiff(..., crs="not a CRS")`` refuses to emit the
    unparseable string into ``GTCitationGeoKey``."""
    da = _da()
    with pytest.raises(UnparseableCRSError):
        to_geotiff(da, str(tmp_path / 'bad_crs.tif'), crs='not a CRS')


def test_write_garbage_crs_kwarg_opt_in_passes(tmp_path):
    """``allow_unparseable_crs=True`` keeps the pre-#1929 citation-only
    behaviour for callers who need it."""
    da = _da()
    to_geotiff(
        da, str(tmp_path / 'bad_crs_opt_in.tif'),
        crs='not a CRS', allow_unparseable_crs=True,
    )


def test_read_rejects_garbage_crs_wkt():
    """The read-side check raises on a string pyproj cannot parse.
    Driven directly through ``validate_read_metadata`` because building
    a TIFF whose stored WKT actually surfaces as ``attrs['crs_wkt']``
    (vs ``attrs['crs_name']``, which carries the GeoKey citation) is a
    full-stack fixture exercise rather than a unit one."""
    from xrspatial.geotiff._validation import validate_read_metadata

    with pytest.raises(UnparseableCRSError):
        validate_read_metadata({'crs_wkt': 'NOT A REAL WKT STRING'})


def test_read_garbage_crs_wkt_opt_in_passes():
    """``allow_unparseable_crs=True`` short-circuits the check."""
    from xrspatial.geotiff._validation import validate_read_metadata

    validate_read_metadata({
        'crs_wkt': 'NOT A REAL WKT STRING',
        'allow_unparseable_crs': True,
    })


def test_read_pyproj_parseable_non_wkt_passes():
    """``EPSG:4326`` is not WKT but pyproj parses it via
    ``from_user_input``. The check tolerates pyproj-parseable
    placeholders (e.g. GDAL's ``<SRS>EPSG:4326</SRS>`` convention)."""
    from xrspatial.geotiff._validation import validate_read_metadata

    validate_read_metadata({'crs_wkt': 'EPSG:4326'})


# ---------------------------------------------------------------------------
# RotatedTransformError.
# ---------------------------------------------------------------------------


def test_read_rejects_rotated_vrt(tmp_path):
    """A VRT whose GeoTransform carries non-zero rotation/shear terms is
    refused on read."""
    src = tmp_path / 'flat.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated.vrt'
    _write_rotated_vrt(vrt, os.path.basename(src))

    with pytest.raises(RotatedTransformError):
        open_geotiff(str(vrt))


def test_read_rotated_vrt_opt_in_passes(tmp_path):
    """``allow_rotated=True`` skips the check and returns the pixel grid
    without the axis-aligned-grid assumption."""
    src = tmp_path / 'flat.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated_ok.vrt'
    _write_rotated_vrt(vrt, os.path.basename(src))

    da = open_geotiff(str(vrt), allow_rotated=True)
    assert da.shape == (4, 4)


def test_read_rejects_row_axis_rotated_vrt(tmp_path):
    """The ``d`` term (GDAL ``GT[4]``, rasterio ``Affine`` index 3) is
    the row-axis rotation; the sibling test above only sets the column
    rotation. Pin that the row-axis branch raises too."""
    src = tmp_path / 'flat_d.tif'
    _write_minimal_tiff_with_wkt(str(src), 'EPSG:4326')
    vrt = tmp_path / 'rotated_d.vrt'
    _write_rotated_vrt(
        vrt, os.path.basename(src),
        geo_transform='0.0, 1.0, 0.0, 0.0, 0.5, -1.0',
    )

    with pytest.raises(RotatedTransformError):
        open_geotiff(str(vrt))


# ---------------------------------------------------------------------------
# NonUniformCoordsError.
# ---------------------------------------------------------------------------


def test_write_rejects_non_uniform_y_coords(tmp_path):
    """A DataArray whose ``y`` coords are not uniformly spaced is refused.
    The writer would otherwise pick the first two values as the pixel
    size and silently misrepresent the rest of the axis."""
    coords = {
        'y': np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, str(tmp_path / 'non_uniform_y.tif'))


def test_write_rejects_non_uniform_x_coords(tmp_path):
    coords = {
        'y': np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, str(tmp_path / 'non_uniform_x.tif'))


def test_write_accepts_uniform_coords(tmp_path):
    """Float coords that are uniformly spaced pass."""
    coords = {
        'y': np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    to_geotiff(da, str(tmp_path / 'uniform.tif'))


def test_write_accepts_integer_coord_sentinel(tmp_path):
    """Int-dtype coords (the no-georef sentinel from #1969) bypass the
    uniformity check because they don't represent geographic positions."""
    coords = {
        'y': np.array([0, 1, 2, 3], dtype=np.int64),
        'x': np.array([0, 1, 2, 3], dtype=np.int64),
    }
    da = _da(coords=coords)
    to_geotiff(da, str(tmp_path / 'int_sentinel.tif'))


def test_write_rejects_constant_float_coords(tmp_path):
    """A float coord array whose first two values are equal makes the
    derived pixel step zero. The check refuses rather than emitting a
    zero-step GeoTransform."""
    coords = {
        'y': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        'x': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    da = _da(coords=coords)
    with pytest.raises(NonUniformCoordsError, match='constant'):
        to_geotiff(da, str(tmp_path / 'constant_y.tif'))


# ---------------------------------------------------------------------------
# ConflictingNodataError.
# ---------------------------------------------------------------------------


def test_write_rejects_conflicting_nodata_attrs(tmp_path):
    """``attrs['nodata']`` disagreeing with every concrete entry in
    ``attrs['nodatavals']`` raises."""
    da = _da(attrs={'nodata': -8888.0, 'nodatavals': (-9999.0,)})
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, str(tmp_path / 'conflict_nodata.tif'))


def test_write_accepts_agreeing_nodata_attrs(tmp_path):
    """When the tuple includes the canonical value (e.g. per-band
    rioxarray convention) the check passes."""
    da = _da(attrs={'nodata': -9999.0, 'nodatavals': (-9999.0, -9999.0)})
    to_geotiff(da, str(tmp_path / 'agree_nodata.tif'))


def test_write_nan_canonical_with_concrete_alias_raises(tmp_path):
    """``nodata=NaN`` paired with a concrete numeric in ``nodatavals``
    means "NaN is the sentinel" and "some integer is the sentinel"
    at the same time. Refuse the ambiguity."""
    da = _da(attrs={'nodata': float('nan'), 'nodatavals': (-9999.0,)})
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, str(tmp_path / 'nan_vs_concrete.tif'))


def test_write_explicit_nodata_kwarg_bypasses_check(tmp_path):
    """``to_geotiff(..., nodata=X)`` overrides both attrs and bypasses
    the conflict check entirely."""
    da = _da(attrs={'nodata': -8888.0, 'nodatavals': (-9999.0,)})
    to_geotiff(da, str(tmp_path / 'kwarg_override.tif'), nodata=-1.0)


def test_write_none_in_nodatavals_tuple_is_skipped(tmp_path):
    """``None`` entries in the rioxarray tuple are "no sentinel on this
    band" and don't count as disagreement."""
    da = _da(attrs={'nodata': -9999.0, 'nodatavals': (None, -9999.0)})
    to_geotiff(da, str(tmp_path / 'none_skipped.tif'))


# ---------------------------------------------------------------------------
# Round-trip safety: a written-then-read DataArray with both attrs set
# (which the reader does emit by default) still writes again cleanly.
# ---------------------------------------------------------------------------


def test_read_then_write_round_trip_does_not_raise(tmp_path):
    """``to_geotiff -> open_geotiff -> to_geotiff`` is the typical
    pipeline. The reader emits both ``crs`` and ``crs_wkt`` whenever the
    file has a CRS, so the second write hits both attrs populated; they
    must agree (they came from the same file)."""
    src = _da(attrs={'crs': 4326, 'nodata': -9999.0})
    first = str(tmp_path / 'rt_first.tif')
    second = str(tmp_path / 'rt_second.tif')
    to_geotiff(src, first)
    da = open_geotiff(first)
    to_geotiff(da, second)
