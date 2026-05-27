"""CRS write-side tests.

Covers the CRS write paths:

* conflicting ``attrs['crs']`` vs ``attrs['crs_wkt']`` fail-closed.
* bool / unresolvable EPSG rejection at all writer entry points.
* citation-only fall-through guard for unparseable CRS strings.
* numpy integer EPSG round-trip through eager / vrt-tiled / GPU writers.
* user-defined WKT promotion on read and round-trip.
* ``UserWarning`` on the WKT-only write path.

``read/test_crs.py`` owns the read side; write-side coverage lives here.
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (ConflictingCRSError, GeoTIFFAmbiguousMetadataError,
                               NonRepresentableEPSGCRSError, open_geotiff, to_geotiff)
from xrspatial.geotiff._crs import _WKT_ROOT_KEYWORDS
from xrspatial.geotiff._crs import _looks_like_wkt as _looks_like_wkt_1929
from xrspatial.geotiff._crs import (_reject_non_representable_epsg, _validate_crs_arg,
                                    _validate_crs_fallback)
from xrspatial.geotiff._geotags import GeoTransform, _looks_like_wkt, build_geo_tags
from xrspatial.geotiff._validation import (_check_write_conflicting_crs,
                                           _registered_write_metadata_checks)

from .._helpers.markers import requires_gpu

pyproj = pytest.importorskip("pyproj")


# =============================================================================
# Section: Conflicting CRS write check
# =============================================================================
#
# The writer historically consulted both ``attrs['crs']`` and
# ``attrs['crs_wkt']`` and gave priority to the EPSG-style ``crs``
# attr. When the two encoded different CRSes, the writer silently
# emitted the EPSG and dropped the WKT -- the on-disk CRS no longer
# matched what the DataArray advertised.
# ``xrspatial.geotiff._validation._check_write_conflicting_crs`` is wired
# into the write entry points so the disagreement now raises
# ``ConflictingCRSError`` (subclass of ``ValueError``) at the boundary.


def _da_1987(*, attrs):
    """Build a minimal 2-D georeferenced DataArray with the given attrs."""
    data = np.zeros((2, 2), dtype=np.float32)
    return xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={
            'y': np.array([1.0, 0.0], dtype=np.float64),
            'x': np.array([0.0, 1.0], dtype=np.float64),
        },
        attrs=dict(attrs),
    )


_WKT_4326_1987 = pyproj.CRS.from_epsg(4326).to_wkt()
_WKT_32633_1987 = pyproj.CRS.from_epsg(32633).to_wkt()


def test_check_registered_at_import_1987():
    """``_check_write_conflicting_crs`` is in the write-side registry."""
    registered = _registered_write_metadata_checks()
    assert _check_write_conflicting_crs in registered, (
        f"_check_write_conflicting_crs missing from registry; "
        f"registered checks: {[c.__name__ for c in registered]}"
    )


def test_error_class_hierarchy_1987():
    assert issubclass(ConflictingCRSError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(ConflictingCRSError, ValueError)


def test_disagreeing_crs_and_crs_wkt_raises_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_32633_1987})
    out = str(tmp_path / 'disagree.tif')

    with pytest.raises(ConflictingCRSError) as excinfo:
        to_geotiff(da, out)

    msg = str(excinfo.value)
    assert "attrs['crs']" in msg and "attrs['crs_wkt']" in msg, msg
    assert '#1987' in msg, msg


def test_disagreeing_attrs_raise_value_error_too_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_32633_1987})
    out = str(tmp_path / 'disagree_value_error.tif')

    with pytest.raises(ValueError):
        to_geotiff(da, out)


def test_disagreeing_attrs_raise_ambiguous_metadata_error_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_32633_1987})
    out = str(tmp_path / 'disagree_base.tif')

    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        to_geotiff(da, out)


def test_matching_crs_int_and_crs_wkt_passes_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_4326_1987})
    to_geotiff(da, str(tmp_path / 'matching_int.tif'))


def test_matching_crs_string_and_crs_wkt_passes_1987(tmp_path):
    da = _da_1987(attrs={'crs': 'EPSG:4326', 'crs_wkt': _WKT_4326_1987})
    to_geotiff(da, str(tmp_path / 'matching_str.tif'))


def test_only_crs_passes_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326})
    to_geotiff(da, str(tmp_path / 'only_crs.tif'))


def test_only_crs_wkt_passes_1987(tmp_path):
    da = _da_1987(attrs={'crs_wkt': _WKT_4326_1987})
    to_geotiff(da, str(tmp_path / 'only_wkt.tif'))


def test_neither_attr_set_passes_1987(tmp_path):
    da = _da_1987(attrs={})
    to_geotiff(da, str(tmp_path / 'neither.tif'))


def test_explicit_crs_kwarg_bypasses_check_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_32633_1987})
    to_geotiff(da, str(tmp_path / 'explicit_kwarg.tif'), crs=4326)


def test_explicit_crs_kwarg_zero_does_not_bypass_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': _WKT_32633_1987})

    with pytest.raises(ValueError):
        to_geotiff(da, str(tmp_path / 'kwarg_zero.tif'), crs=0)


def test_unparseable_crs_wkt_does_not_fire_conflicting_1987(tmp_path):
    da = _da_1987(attrs={'crs': 4326, 'crs_wkt': 'NOT A VALID WKT STRING'})
    out = str(tmp_path / 'unparseable_wkt.tif')

    try:
        to_geotiff(da, out)
    except ConflictingCRSError as e:
        pytest.fail(
            "Unparseable WKT should not surface as ConflictingCRSError; "
            f"that case is owned by UnparseableCRSError. Got: {e}"
        )
    except Exception:
        pass


def test_read_back_roundtrip_does_not_raise_1987(tmp_path):
    first = str(tmp_path / 'roundtrip_first.tif')
    second = str(tmp_path / 'roundtrip_second.tif')

    src = _da_1987(attrs={'crs': 4326})
    to_geotiff(src, first)

    da = open_geotiff(first)
    assert 'crs' in da.attrs and 'crs_wkt' in da.attrs, (
        "reader should emit both crs and crs_wkt on a CRS-bearing file; "
        f"got attrs={sorted(da.attrs)}"
    )

    to_geotiff(da, second)


# =============================================================================
# Section: CRS argument validation
# =============================================================================
#
# ``bool`` is an int subclass, so ``crs=True`` used to slip through
# ``isinstance(crs, int)`` and write EPSG=1 to the file (with EPSG=0
# for ``crs=False``). Integer EPSG codes were also written without a
# pyproj round-trip, so any int that does not resolve as a CRS
# produced a file with garbage in ``ProjectedCSType`` /
# ``GeographicType`` and only a ``GeoTIFFFallbackWarning`` to flag it.


def _square_1971(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("bad_crs", [True, False])
def test_validate_crs_arg_rejects_bool_1971(bad_crs):
    with pytest.raises(ValueError, match="bool"):
        _validate_crs_arg(bad_crs)


def test_validate_crs_arg_rejects_unresolvable_epsg_1971():
    with pytest.raises(ValueError, match="EPSG"):
        _validate_crs_arg(1)


def test_validate_crs_arg_accepts_valid_epsg_1971():
    _validate_crs_arg(4326)


def test_validate_crs_arg_accepts_none_1971():
    _validate_crs_arg(None)


def test_validate_crs_arg_accepts_str_1971():
    _validate_crs_arg("EPSG:4326")
    _validate_crs_arg('PROJCS["foo",GEOGCS["bar"]]')


def test_validate_crs_arg_rejects_non_int_non_str_1971():
    with pytest.raises(TypeError, match="crs must be int"):
        _validate_crs_arg(4326.0)


@pytest.mark.parametrize("np_int", [np.int64(4326), np.int32(4326)])
def test_validate_crs_arg_accepts_numpy_integer_1971(np_int):
    _validate_crs_arg(np_int)


def test_validate_crs_arg_rejects_numpy_bool_1971():
    val = np.bool_(True)
    with pytest.raises((ValueError, TypeError)):
        _validate_crs_arg(val)


@pytest.mark.parametrize("bad_crs", [True, False])
def test_to_geotiff_rejects_bool_crs_1971(bad_crs):
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(_square_1971(), buf, crs=bad_crs)


def test_to_geotiff_rejects_unresolvable_epsg_1971():
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(_square_1971(), buf, crs=1)


def test_to_geotiff_accepts_valid_epsg_1971():
    buf = io.BytesIO()
    to_geotiff(_square_1971(), buf, crs=4326)
    assert buf.getbuffer().nbytes > 0


def test_to_geotiff_accepts_numpy_int_epsg_1971():
    buf = io.BytesIO()
    to_geotiff(_square_1971(), buf, crs=np.int64(4326))
    assert buf.getbuffer().nbytes > 0


def test_to_geotiff_attrs_crs_bool_bypass_1971(tmp_path):
    da = _square_1971()
    da.attrs['crs'] = True
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(da, buf)


def test_to_geotiff_attrs_crs_unresolvable_epsg_bypass_1971():
    da = _square_1971()
    da.attrs['crs'] = 1
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(da, buf)


def test_to_geotiff_vrt_path_rejects_bool_crs_1971(tmp_path):
    vrt_path = str(tmp_path / "tmp_1971_vrt_tiled.vrt")
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(_square_1971(), vrt_path, crs=True)


def test_to_geotiff_vrt_path_rejects_unresolvable_epsg_1971(tmp_path):
    vrt_path = str(tmp_path / "tmp_1971_vrt_bad_epsg.vrt")
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(_square_1971(), vrt_path, crs=1)


def test_to_geotiff_vrt_path_attrs_crs_bool_bypass_1971(tmp_path):
    da = _square_1971()
    da.attrs['crs'] = True
    vrt_path = str(tmp_path / "tmp_1971_vrt_attrs_bool.vrt")
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(da, vrt_path)


# =============================================================================
# Section: CRS fail-closed citation guard
# =============================================================================
#
# The CRS resolution path used to swallow ``pyproj`` parse failures
# with only a warning, then write the original unvalidatable string
# verbatim into ``GTCitationGeoKey``. The fix adds
# ``_validate_crs_fallback`` in ``_crs.py``. It refuses to land any
# non-WKT-shaped string in the citation when the caller has not opted
# in via ``allow_unparseable_crs=True``.


def _make_da_1929() -> xr.DataArray:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


class TestLooksLikeWkt_1929:
    """Structural WKT recognition. Cheap and pyproj-free."""

    @pytest.mark.parametrize("root", _WKT_ROOT_KEYWORDS)
    def test_each_root_keyword_recognised(self, root):
        assert _looks_like_wkt_1929(f'{root}["WGS 84", ...]')

    def test_leading_whitespace_tolerated(self):
        assert _looks_like_wkt_1929('   PROJCS["UTM"]')

    def test_case_insensitive(self):
        assert _looks_like_wkt_1929('projcs["UTM"]')
        assert _looks_like_wkt_1929('GeOgCs["WGS 84"]')

    def test_epsg_token_rejected(self):
        assert not _looks_like_wkt_1929("EPSG:4326")

    def test_proj_string_rejected(self):
        assert not _looks_like_wkt_1929("+proj=utm +zone=10")

    def test_garbage_rejected(self):
        assert not _looks_like_wkt_1929("not a CRS at all")

    def test_empty_string_rejected(self):
        assert not _looks_like_wkt_1929("")

    def test_non_string_rejected(self):
        assert not _looks_like_wkt_1929(None)
        assert not _looks_like_wkt_1929(4326)


class TestValidateCrsFallback_1929:
    """Direct unit tests on the validator helper."""

    def test_none_fallback_returns(self):
        _validate_crs_fallback(None, allow_unparseable_crs=False)

    def test_wkt_shaped_returns(self):
        _validate_crs_fallback(
            'PROJCS["test", ...]', allow_unparseable_crs=False
        )

    def test_non_wkt_raises(self):
        with pytest.raises(ValueError, match="GTCitationGeoKey"):
            _validate_crs_fallback("EPSG:4326", allow_unparseable_crs=False)

    def test_opt_in_allows_non_wkt(self):
        _validate_crs_fallback("EPSG:4326", allow_unparseable_crs=True)

    def test_message_names_the_offending_string(self):
        with pytest.raises(ValueError) as exc:
            _validate_crs_fallback("frobnicate", allow_unparseable_crs=False)
        assert "frobnicate" in str(exc.value)
        assert "allow_unparseable_crs" in str(exc.value)


class TestToGeotiffCrsFailClosed_1929:

    def test_epsg_int_unchanged(self, tmp_path):
        out = str(tmp_path / "epsg_int_1929.tif")
        to_geotiff(_make_da_1929(), out, crs=4326)
        assert os.path.exists(out)

    def test_valid_wkt_unchanged(self, tmp_path):
        out = str(tmp_path / "wkt_shaped_1929.tif")
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
            '6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]]'
        )
        to_geotiff(_make_da_1929(), out, crs=wkt)
        assert os.path.exists(out)

    def test_epsg_token_string_via_kwarg(self, tmp_path):
        out = str(tmp_path / "epsg_token_1929.tif")
        pytest.importorskip("pyproj")
        to_geotiff(_make_da_1929(), out, crs="EPSG:4326")
        assert os.path.exists(out)

    def test_garbage_string_kwarg_raises(self, tmp_path):
        out = str(tmp_path / "garbage_kwarg_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(_make_da_1929(), out, crs="absolute-garbage")

    def test_opt_in_allows_garbage(self, tmp_path):
        out = str(tmp_path / "opt_in_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_da_1929(), out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)

    def test_garbage_string_attr_raises(self, tmp_path):
        out = str(tmp_path / "garbage_attr_1929.tif")
        da = _make_da_1929()
        da.attrs["crs"] = "still-garbage"
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(da, out)

    def test_no_crs_at_all_unchanged(self, tmp_path):
        out = str(tmp_path / "no_crs_1929.tif")
        to_geotiff(_make_da_1929(), out)
        assert os.path.exists(out)

    def test_message_recommends_alternatives(self, tmp_path):
        out = str(tmp_path / "msg_check_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError) as exc:
                to_geotiff(_make_da_1929(), out, crs="bogus")
        msg = str(exc.value)
        assert "EPSG" in msg
        assert "WKT" in msg
        assert "allow_unparseable_crs" in msg


# =============================================================================
# Section: Numpy integer CRS round-trip
# =============================================================================
#
# ``_validate_crs_arg`` accepts ``numbers.Integral`` (incl. ``np.int32``
# / ``np.int64`` / ``np.uint16``), but the eager and GPU writers used
# to gate EPSG assignment on ``isinstance(crs, int)``. Numpy integers
# fail that check, so the value passed validation, fell through the
# writer branch, and the file landed on disk with no EPSG and no
# ``crs_wkt`` tag.


def _square_2082(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize(
    "np_int",
    [np.int32(4326), np.int64(4326), np.uint16(4326)],
)
def test_to_geotiff_numpy_int_crs_roundtrips_2082(tmp_path, np_int):
    path = str(tmp_path / f"tmp_2082_eager_{type(np_int).__name__}.tif")
    to_geotiff(_square_2082(), path, crs=np_int)
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326


@pytest.mark.parametrize(
    "np_int",
    [np.int32(3857), np.int64(3857)],
)
def test_to_geotiff_numpy_int_crs_roundtrips_bytesio_2082(np_int):
    buf = io.BytesIO()
    to_geotiff(_square_2082(), buf, crs=np_int)
    buf.seek(0)
    out = open_geotiff(buf)
    assert out.attrs.get('crs') == 3857


def test_to_geotiff_vrt_tiled_numpy_int_crs_roundtrips_2082(tmp_path):
    vrt_path = str(tmp_path / "tmp_2082_vrt_tiled.vrt")
    to_geotiff(_square_2082(), vrt_path, crs=np.int64(4326))
    tile_path = str(tmp_path / "tmp_2082_vrt_tiled_tiles" / "tile_00_00.tif")
    out = open_geotiff(tile_path)
    assert out.attrs.get('crs') == 4326


def test_to_geotiff_python_int_crs_still_works_2082(tmp_path):
    path = str(tmp_path / "tmp_2082_plain_int.tif")
    to_geotiff(_square_2082(), path, crs=4326)
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326


def test_to_geotiff_numpy_int_crs_writes_real_int_to_attrs_2082(tmp_path):
    path = str(tmp_path / "tmp_2082_attrs_int_type.tif")
    to_geotiff(_square_2082(), path, crs=np.int64(4326))
    out = open_geotiff(path)
    crs_val = out.attrs.get('crs')
    assert int(crs_val) == 4326
    assert not isinstance(crs_val, np.integer)


# GPU path
_cupy_spec = pytest.importorskip("cupy", reason="GPU writer needs cupy")


@requires_gpu
def test_write_geotiff_gpu_numpy_int_crs_roundtrips_2082(tmp_path):
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu
    arr = cupy.zeros((4, 4), dtype=cupy.float32)
    da = xr.DataArray(
        arr,
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2082_gpu.tif")
    write_geotiff_gpu(da, path, crs=np.int64(4326))
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326


# =============================================================================
# Section: User-defined CRS WKT promotion
# =============================================================================
#
# Files with a user-defined CRS (no EPSG, WKT stored in the GeoTIFF
# citation under ``GEOKEY_*_CS_TYPE == 32767``) used to round-trip
# with ``attrs['crs_name']`` set but ``attrs['crs_wkt']`` and
# ``attrs['crs']`` unset. ``to_geotiff`` only consults the latter two,
# so a read -> write cycle silently dropped the projection.
tifffile_1632 = pytest.importorskip("tifffile")

_USER_DEFINED_WKT_1632 = (
    'PROJCS["User defined LCC",'
    'GEOGCS["NAD83",'
    'DATUM["North American Datum 1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Lambert Conformal Conic 2SP"],'
    'PARAMETER["central_meridian",-100],'
    'PARAMETER["latitude_of_origin",40],'
    'PARAMETER["standard_parallel_1",30],'
    'PARAMETER["standard_parallel_2",50],'
    'UNIT["metre",1]]'
)


def _write_user_defined_crs_tif_1632(path):
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs_wkt': _USER_DEFINED_WKT_1632},
    )
    to_geotiff(da, path, compression='none')
    return da


@pytest.mark.parametrize("text", [
    'PROJCS["foo"]',
    'GEOGCS["foo"]',
    'PROJCRS["foo"]',
    'GEOGCRS["foo"]',
    'COMPD_CS["foo"]',
    'COMPOUNDCRS["foo"]',
    'BOUNDCRS["foo"]',
    'VERT_CS["foo"]',
    'VERTCRS["foo"]',
    'LOCAL_CS["foo"]',
    '  PROJCS["leading whitespace"]',
    'projcs["lowercase"]',
])
def test_looks_like_wkt_positive_1632(text):
    assert _looks_like_wkt(text)


def test_looks_like_wkt_requires_bracket_1632():
    assert not _looks_like_wkt("PROJCS")
    assert not _looks_like_wkt("PROJCS no bracket here")


@pytest.mark.parametrize("text", [
    None,
    '',
    'NAD83 / UTM Zone 12N',
    'epsg:4326',
    'WGS 84',
    'Some random string',
    'PROJ string +proj=longlat +datum=WGS84',
    42,
    b'PROJCS["bytes input"]',
])
def test_looks_like_wkt_negative_1632(text):
    assert not _looks_like_wkt(text)


def test_eager_emits_crs_wkt_for_user_defined_crs_1632(tmp_path):
    p = str(tmp_path / "user_defined_crs.tif")
    _write_user_defined_crs_tif_1632(p)

    rd = open_geotiff(p)
    assert rd.attrs.get("crs") is None
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")
    assert "crs_name" not in rd.attrs


def test_dask_emits_crs_wkt_for_user_defined_crs_1632(tmp_path):
    p = str(tmp_path / "user_defined_crs_dask.tif")
    _write_user_defined_crs_tif_1632(p)

    rd = open_geotiff(p, chunks=4)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


@requires_gpu
def test_cupy_emits_crs_wkt_for_user_defined_crs_1632(tmp_path):
    p = str(tmp_path / "user_defined_crs_gpu.tif")
    _write_user_defined_crs_tif_1632(p)

    rd = open_geotiff(p, gpu=True)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


@requires_gpu
def test_dask_cupy_emits_crs_wkt_for_user_defined_crs_1632(tmp_path):
    p = str(tmp_path / "user_defined_crs_dask_gpu.tif")
    _write_user_defined_crs_tif_1632(p)

    rd = open_geotiff(p, gpu=True, chunks=4)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


def test_user_defined_crs_round_trips_through_to_geotiff_1632(tmp_path):
    src = str(tmp_path / "round_trip_src.tif")
    _write_user_defined_crs_tif_1632(src)

    rd = open_geotiff(src)
    dst = str(tmp_path / "round_trip_dst.tif")
    to_geotiff(rd, dst, compression='none')

    rd2 = open_geotiff(dst)
    assert rd2.attrs.get("crs_wkt") == rd.attrs.get("crs_wkt")

    with tifffile_1632.TiffFile(dst) as tif:
        keys = tif.pages[0].tags.get(34735)
        ascii_tag = tif.pages[0].tags.get(34737)
        assert keys is not None
        assert len(keys.value) > 4
        assert ascii_tag is not None
        assert "PROJCS[" in ascii_tag.value


def test_epsg_crs_unchanged_by_fix_1632(tmp_path):
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / "epsg.tif")
    to_geotiff(da, p, compression='none')

    rd = open_geotiff(p)
    assert rd.attrs.get("crs") == 4326
    assert rd.attrs.get("crs_wkt") is not None
    assert "crs_name" not in rd.attrs


def test_human_readable_crs_name_not_promoted_to_crs_wkt_1632(tmp_path):
    assert not _looks_like_wkt("NAD83 / UTM Zone 12N")
    assert not _looks_like_wkt("WGS 84")
    assert not _looks_like_wkt("")
    assert not _looks_like_wkt(None)


def test_synthesize_user_defined_wkt_sphere_1632():
    from xrspatial.geotiff._geotags import MODEL_TYPE_GEOGRAPHIC, _synthesize_user_defined_wkt

    wkt = _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_GEOGRAPHIC,
        semi_major=6378137.0,
        semi_minor=6378137.0,
        inv_flattening=0.0,
    )
    assert isinstance(wkt, str) and wkt
    crs = pyproj.CRS.from_wkt(wkt)
    proj_dict = crs.to_dict()
    assert proj_dict.get("proj") == "longlat"
    assert proj_dict.get("R") == 6378137.0


def test_synthesize_user_defined_wkt_oblate_ellipsoid_1632():
    from xrspatial.geotiff._geotags import MODEL_TYPE_GEOGRAPHIC, _synthesize_user_defined_wkt

    wkt = _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_GEOGRAPHIC,
        semi_major=6378137.0,
        semi_minor=None,
        inv_flattening=298.257223563,
    )
    assert isinstance(wkt, str) and wkt
    crs = pyproj.CRS.from_wkt(wkt)
    assert crs.ellipsoid.semi_major_metre == pytest.approx(6378137.0)
    assert crs.ellipsoid.inverse_flattening == pytest.approx(298.257223563)


def test_synthesize_user_defined_wkt_projected_returns_none_1632():
    from xrspatial.geotiff._geotags import MODEL_TYPE_PROJECTED, _synthesize_user_defined_wkt

    assert _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_PROJECTED,
        semi_major=6378137.0,
        semi_minor=6378137.0,
        inv_flattening=0.0,
    ) is None


def test_synthesize_user_defined_wkt_geocentric_returns_none_1632():
    from xrspatial.geotiff._geotags import MODEL_TYPE_GEOCENTRIC, _synthesize_user_defined_wkt

    assert _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_GEOCENTRIC,
        semi_major=6378137.0,
        semi_minor=6378137.0,
        inv_flattening=0.0,
    ) is None
    assert _synthesize_user_defined_wkt(
        model_type=0,
        semi_major=6378137.0,
        semi_minor=6378137.0,
        inv_flattening=0.0,
    ) is None


def test_synthesize_user_defined_wkt_missing_ellipsoid_returns_none_1632():
    from xrspatial.geotiff._geotags import MODEL_TYPE_GEOGRAPHIC, _synthesize_user_defined_wkt

    assert _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_GEOGRAPHIC,
        semi_major=None,
        semi_minor=None,
        inv_flattening=None,
    ) is None
    assert _synthesize_user_defined_wkt(
        model_type=MODEL_TYPE_GEOGRAPHIC,
        semi_major=6378137.0,
        semi_minor=None,
        inv_flattening=None,
    ) is None


# =============================================================================
# Section: WKT-only CRS warning
# =============================================================================
#
# ``build_geo_tags`` writes a user-defined CRS (no EPSG, WKT only) by
# setting ``ProjectedCSType`` / ``GeographicType`` = 32767 and storing
# the raw WKT in ``GTCitationGeoKey``. libgeotiff and GDAL can recover
# the CRS by parsing the citation, but many other GeoTIFF readers
# treat the citation as a free-form name and silently drop the
# projection. The fix emits a ``UserWarning`` so the interop
# limitation is visible at write time.
tifffile_1768 = pytest.importorskip("tifffile")

_USER_DEFINED_WKT_1768 = (
    'PROJCS["User defined LCC 1768",'
    'GEOGCS["NAD83",'
    'DATUM["North American Datum 1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Lambert Conformal Conic 2SP"],'
    'PARAMETER["central_meridian",-100],'
    'PARAMETER["latitude_of_origin",40],'
    'PARAMETER["standard_parallel_1",30],'
    'PARAMETER["standard_parallel_2",50],'
    'UNIT["metre",1]]'
)


def test_build_geo_tags_warns_on_wkt_only_path_1768():
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with pytest.warns(UserWarning, match=r"user-defined CRS via WKT only"):
        tags = build_geo_tags(t, crs_wkt=_USER_DEFINED_WKT_1768)

    assert 34735 in tags
    assert 34737 in tags
    assert _USER_DEFINED_WKT_1768 in tags[34737]


def test_build_geo_tags_no_warning_on_epsg_path_1768():
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tags = build_geo_tags(t, crs_epsg=4326)
    assert 34735 in tags


def test_build_geo_tags_no_warning_when_no_crs_1768():
    t = GeoTransform(origin_x=0.0, origin_y=10.0,
                     pixel_width=1.0, pixel_height=-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_geo_tags(t)


def _wkt_only_da_1768():
    arr = np.ones((4, 4), dtype=np.float32)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs_wkt': _USER_DEFINED_WKT_1768},
    )


def test_to_geotiff_warns_on_wkt_only_crs_1768(tmp_path):
    da = _wkt_only_da_1768()
    p = str(tmp_path / "wkt_only_1768.tif")

    with pytest.warns(UserWarning, match=r"user-defined CRS via WKT only"):
        to_geotiff(da, p, compression='none')

    rd = open_geotiff(p)
    assert rd.attrs.get("crs_wkt") is not None
    assert rd.attrs["crs_wkt"].startswith("PROJCS[")


def test_to_geotiff_no_warning_on_epsg_crs_1768(tmp_path):
    arr = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.linspace(50.0, 47.0, 4),
                'x': np.linspace(10.0, 13.0, 4)},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / "epsg_only_1768.tif")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_geotiff(da, p, compression='none')

    wkt_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "user-defined CRS via WKT only" in str(w.message)
    ]
    assert wkt_warnings == [], (
        f"EPSG path unexpectedly emitted the WKT-only warning: "
        f"{[str(w.message) for w in wkt_warnings]}"
    )


def test_wkt_only_citation_bytes_unchanged_after_fix_1768(tmp_path):
    da = _wkt_only_da_1768()
    p = str(tmp_path / "wkt_only_bytes_1768.tif")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        to_geotiff(da, p, compression='none')

    with tifffile_1768.TiffFile(p) as tif:
        keys = tif.pages[0].tags.get(34735)
        ascii_tag = tif.pages[0].tags.get(34737)
        assert keys is not None
        assert len(keys.value) >= 20

        entries = list(keys.value[4:])
        user_defined_crs_marker = False
        for i in range(0, len(entries), 4):
            key_id = entries[i]
            location = entries[i + 1]
            value = entries[i + 3]
            if key_id in (2048, 3072) and location == 0 and value == 32767:
                user_defined_crs_marker = True
                break
        assert user_defined_crs_marker, (
            "GeoKeyDirectory must mark the CRS as user-defined (32767) "
            "via ProjectedCSType (3072) or GeographicType (2048)"
        )

        assert ascii_tag is not None
        assert _USER_DEFINED_WKT_1768 in ascii_tag.value

# ===========================================================================
# Compound EPSG reject on stable writer path (#2418)
# Source: test_compound_crs_reject_2418.py
# ===========================================================================


pyproj = pytest.importorskip("pyproj")
rasterio = pytest.importorskip("rasterio")


# Compound EPSG codes that previously slipped through the validator.
# 6349: NAD83(2011) + NAVD88 height (compound, horizontal=geographic)
# 5498: NAD83 + NAVD88 height (compound, horizontal=geographic)
# 7415: Amersfoort/RD New + NAP height (compound, horizontal=projected)
# 8360: ETRS89 + EVRF2000 Austria height (compound, horizontal=geographic)
COMPOUND_EPSG_CODES = [6349, 5498, 7415, 8360]


def _square(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_validate_crs_arg_rejects_compound_epsg(epsg):
    """``_validate_crs_arg`` raises for compound EPSG codes."""
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        _validate_crs_arg(epsg)


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_to_geotiff_rejects_compound_epsg_kwarg(epsg):
    """End-to-end ``to_geotiff(crs=<compound>)`` raises."""
    buf = io.BytesIO()
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        to_geotiff(_square(), buf, crs=epsg)


@pytest.mark.parametrize("epsg", COMPOUND_EPSG_CODES)
def test_to_geotiff_rejects_compound_epsg_attrs(epsg):
    """End-to-end ``DataArray.attrs['crs'] = <compound>`` raises."""
    arr = _square()
    arr.attrs['crs'] = epsg
    buf = io.BytesIO()
    with pytest.raises(NonRepresentableEPSGCRSError, match="compound"):
        to_geotiff(arr, buf)


def test_reject_non_representable_epsg_helper():
    """The helper raises directly on a compound pyproj CRS object."""
    from pyproj import CRS
    crs_obj = CRS.from_epsg(6349)
    assert crs_obj.is_compound is True
    with pytest.raises(NonRepresentableEPSGCRSError, match="6349"):
        _reject_non_representable_epsg(6349, crs_obj)


def test_reject_non_representable_epsg_passes_horizontal():
    """The helper does not raise for plain horizontal CRSs."""
    from pyproj import CRS
    _reject_non_representable_epsg(4326, CRS.from_epsg(4326))  # geographic 2D
    _reject_non_representable_epsg(32633, CRS.from_epsg(32633))  # projected
    _reject_non_representable_epsg(3857, CRS.from_epsg(3857))  # web mercator


@pytest.mark.parametrize("epsg", [4326, 32633, 3857, 4269])
def test_horizontal_crs_still_round_trips_through_rasterio(epsg, tmp_path):
    """Regression: scoped fix doesn't break the normal stable-EPSG path."""
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / f"tmp_2418_{epsg}.tif"
    to_geotiff(arr, str(path), crs=epsg)
    with rasterio.open(str(path)) as src:
        assert src.crs is not None
        assert src.crs.to_epsg() == epsg


def test_compound_wkt_takes_user_defined_path(tmp_path):
    """A compound CRS passed as WKT writes via the citation path (#1768).

    ``_wkt_to_epsg`` returns None for a compound CRS so the writer
    treats it as a user-defined CRS rather than emitting a compound
    EPSG into the horizontal-CRS GeoKey slot.
    """
    from pyproj import CRS
    wkt = CRS.from_epsg(6349).to_wkt()
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_compound_wkt.tif"
    # The WKT-only path emits a UserWarning per #1768; we don't care
    # about that here, just that the write succeeds and the CRS info
    # survives in xrspatial's own attrs.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_geotiff(arr, str(path), crs=wkt)
    da = open_geotiff(str(path))
    # ``crs_wkt`` carries the full compound CRS string back through the
    # xrspatial reader (the WKT-only path's contract).
    assert da.attrs.get('crs_wkt') is not None
    assert 'COMPOUND' in da.attrs['crs_wkt'].upper()


def test_compound_epsg_corruption_surfaces_when_validator_bypassed(
    tmp_path, monkeypatch,
):
    """If the validator is bypassed, the file IS corrupted (regression).

    Monkey-patches ``_validate_crs_arg`` and
    ``_model_type_from_epsg``'s compound check to no-ops, then writes
    EPSG:6349 through the integer path. Asserts the file rasterio
    reads back is not EPSG:6349 -- the exact corruption the fix
    prevents. If a future refactor drops either guard, this test
    starts failing instead of the writer silently corrupting files.
    """
    from xrspatial.geotiff import _crs as _crs_mod
    from xrspatial.geotiff import _geotags as _geotags_mod

    # Bypass the validator at both gates so we exercise the broken
    # write path the production code now refuses.
    monkeypatch.setattr(
        _crs_mod, "_reject_non_representable_epsg", lambda *a, **kw: None
    )

    def _model_type_without_compound_check(crs_epsg):
        # Reproduce the pre-fix behaviour: only branch on is_geographic.
        from pyproj import CRS
        crs = CRS.from_epsg(crs_epsg)
        if crs.is_geographic:
            return _geotags_mod.MODEL_TYPE_GEOGRAPHIC
        return _geotags_mod.MODEL_TYPE_PROJECTED

    monkeypatch.setattr(
        _geotags_mod, "_model_type_from_epsg", _model_type_without_compound_check
    )

    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_bypass_demo.tif"
    to_geotiff(arr, str(path), crs=6349)

    # Pin the corruption: rasterio either fails to recover the EPSG
    # entirely (``to_epsg() is None``) or recovers the wrong one
    # (the horizontal sub-CRS, with the vertical component dropped).
    # Both outcomes are caught by ``rasterio_epsg != 6349``.
    with rasterio.open(str(path)) as src:
        rasterio_epsg = src.crs.to_epsg() if src.crs is not None else None
    assert rasterio_epsg != 6349, (
        f"rasterio recovered EPSG:6349 from a write that bypassed the "
        f"validator: got {rasterio_epsg}. Either the writer is now "
        "correctly emitting compound GeoKeys (good, drop the test) or "
        "the bypass no longer reaches the broken path (unlikely)."
    )

    # Sanity: when validation is back in place, the same write raises.
    monkeypatch.undo()
    with pytest.raises(NonRepresentableEPSGCRSError):
        to_geotiff(arr, str(path), crs=6349)


def test_geographic_3d_crs_still_works(tmp_path):
    """EPSG:4979 (3D WGS84) is geographic, not compound: still allowed.

    The fix is narrow -- it only rejects compound CRSs, which are the
    demonstrably-broken case. 3D-geographic and other non-2D-horizontal
    EPSGs that round-trip cleanly through rasterio remain allowed.
    """
    from pyproj import CRS
    crs_obj = CRS.from_epsg(4979)
    assert crs_obj.is_compound is False
    # Should not raise from the validator.
    _validate_crs_arg(4979)
    arr = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={'y': np.array([3.5, 2.5, 1.5, 0.5]),
                'x': np.array([0.5, 1.5, 2.5, 3.5])},
        dims=('y', 'x'),
    )
    path = tmp_path / "tmp_2418_4979.tif"
    to_geotiff(arr, str(path), crs=4979)
    with rasterio.open(str(path)) as src:
        # rasterio resolves EPSG:4979 correctly.
        assert src.crs is not None
        assert src.crs.to_epsg() == 4979


def test_compound_epsg_error_message_points_to_workaround():
    """Error message names the compound CRS and offers a route forward."""
    with pytest.raises(NonRepresentableEPSGCRSError) as exc:
        _validate_crs_arg(6349)
    msg = str(exc.value)
    assert "6349" in msg
    assert "compound" in msg.lower()
    # The message should point at both the WKT fallback and the
    # horizontal-sub-CRS workaround so the caller can act on it.
    assert "WKT" in msg or "wkt" in msg
    assert "horizontal" in msg.lower()
