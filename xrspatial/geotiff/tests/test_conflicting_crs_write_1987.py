"""Conflicting ``attrs['crs']`` and ``attrs['crs_wkt']`` write check.

Issue #1987, PR 1 of the ambiguous-metadata fail-closed series.

The writer historically consulted both ``attrs['crs']`` and
``attrs['crs_wkt']`` and gave priority to the EPSG-style ``crs`` attr.
When the two encoded different CRSes, the writer silently emitted the
EPSG and dropped the WKT -- the on-disk CRS no longer matched what the
DataArray advertised. PR 1 wires
``xrspatial.geotiff._validation._check_write_conflicting_crs`` into the
write entry points so the disagreement now raises
``ConflictingCRSError`` (subclass of ``ValueError``) at the boundary.

This file pins:

* The check fires on disagreeing attrs (``test_disagreeing_*``).
* The check is silent on consistent attrs, even when both are set --
  this matches the read-back state where the reader always emits both
  (``test_matching_*``).
* The check is silent when only one attr is set (``test_only_*``).
* The check is bypassed when the caller passes ``crs=`` explicitly --
  the kwarg overrides both attrs (``test_explicit_crs_kwarg_*``).
* Subclass relationships hold: ``ConflictingCRSError`` is a
  ``GeoTIFFAmbiguousMetadataError`` and a ``ValueError``
  (``test_error_class_hierarchy``).
* The check is registered in the write-side registry at module load
  (``test_check_registered_at_import``).
* Unparseable inputs fall through without firing this check -- they
  are the subject of a sibling PR's ``UnparseableCRSError``
  (``test_unparseable_attrs_dont_raise_conflicting``).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import ConflictingCRSError, GeoTIFFAmbiguousMetadataError, to_geotiff
from xrspatial.geotiff._validation import (_check_write_conflicting_crs,
                                           _registered_write_metadata_checks)

pyproj = pytest.importorskip("pyproj")


def _da(*, attrs):
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


_WKT_4326 = pyproj.CRS.from_epsg(4326).to_wkt()
_WKT_32633 = pyproj.CRS.from_epsg(32633).to_wkt()


# ---------------------------------------------------------------------------
# Registry sanity: the check is wired in by import-time registration.
# ---------------------------------------------------------------------------


def test_check_registered_at_import():
    """``_check_write_conflicting_crs`` is in the write-side registry.

    Confirms the registration call at the bottom of ``_validation.py``
    fired on first import. A regression here would mean the check is
    silently inert and every write path stopped enforcing the rule.
    """
    registered = _registered_write_metadata_checks()
    assert _check_write_conflicting_crs in registered, (
        f"_check_write_conflicting_crs missing from registry; "
        f"registered checks: {[c.__name__ for c in registered]}"
    )


def test_error_class_hierarchy():
    """``ConflictingCRSError`` subclasses both the ambiguous-metadata
    family and ``ValueError``.

    Catching ``GeoTIFFAmbiguousMetadataError`` handles the whole #1987
    family. Catching ``ValueError`` keeps legacy ``except ValueError``
    blocks working through the deprecation of guess-and-continue.
    """
    assert issubclass(ConflictingCRSError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(ConflictingCRSError, ValueError)


# ---------------------------------------------------------------------------
# Disagreeing attrs: the check raises.
# ---------------------------------------------------------------------------


def test_disagreeing_crs_and_crs_wkt_raises(tmp_path):
    """``attrs['crs']=4326`` plus ``attrs['crs_wkt']=<WKT for 32633>``
    raises ``ConflictingCRSError`` at write."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_32633})
    out = str(tmp_path / 'disagree.tif')

    with pytest.raises(ConflictingCRSError) as excinfo:
        to_geotiff(da, out)

    msg = str(excinfo.value)
    assert "attrs['crs']" in msg and "attrs['crs_wkt']" in msg, msg
    assert '#1987' in msg, msg


def test_disagreeing_attrs_raise_value_error_too(tmp_path):
    """Legacy ``except ValueError`` still catches the conflict because
    ``ConflictingCRSError`` is a ``ValueError`` subclass."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_32633})
    out = str(tmp_path / 'disagree_value_error.tif')

    with pytest.raises(ValueError):
        to_geotiff(da, out)


def test_disagreeing_attrs_raise_ambiguous_metadata_error(tmp_path):
    """The whole #1987 family is catchable via the base class."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_32633})
    out = str(tmp_path / 'disagree_base.tif')

    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        to_geotiff(da, out)


# ---------------------------------------------------------------------------
# Agreeing attrs: the check is silent. This is the round-trip read-back
# state, which always carries both keys.
# ---------------------------------------------------------------------------


def test_matching_crs_int_and_crs_wkt_passes(tmp_path):
    """``attrs['crs']=4326`` plus ``attrs['crs_wkt']=<WKT for 4326>``
    writes without error -- this is the normal read-back state."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_4326})
    out = str(tmp_path / 'matching_int.tif')

    to_geotiff(da, out)


def test_matching_crs_string_and_crs_wkt_passes(tmp_path):
    """``attrs['crs']='EPSG:4326'`` plus the matching WKT canonicalises
    to the same CRS via pyproj and writes without error."""
    da = _da(attrs={'crs': 'EPSG:4326', 'crs_wkt': _WKT_4326})
    out = str(tmp_path / 'matching_str.tif')

    to_geotiff(da, out)


# ---------------------------------------------------------------------------
# Only one attr set: the check has nothing to compare against.
# ---------------------------------------------------------------------------


def test_only_crs_passes(tmp_path):
    da = _da(attrs={'crs': 4326})
    to_geotiff(da, str(tmp_path / 'only_crs.tif'))


def test_only_crs_wkt_passes(tmp_path):
    da = _da(attrs={'crs_wkt': _WKT_4326})
    to_geotiff(da, str(tmp_path / 'only_wkt.tif'))


def test_neither_attr_set_passes(tmp_path):
    da = _da(attrs={})
    to_geotiff(da, str(tmp_path / 'neither.tif'))


# ---------------------------------------------------------------------------
# Explicit ``crs=`` kwarg overrides both attrs, so the check is bypassed.
# ---------------------------------------------------------------------------


def test_explicit_crs_kwarg_bypasses_check(tmp_path):
    """``to_geotiff(da, ..., crs=4326)`` overrides attrs entirely,
    so the disagreement in attrs does not affect this write."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_32633})

    to_geotiff(da, str(tmp_path / 'explicit_kwarg.tif'), crs=4326)


def test_explicit_crs_kwarg_zero_does_not_bypass(tmp_path):
    """``crs=0`` is not a valid EPSG and the writer's own kwarg
    validation rejects it; the conflicting-CRS check fires too.

    The bypass branch checks ``is not None``, so any non-None value
    triggers it. ``0`` would slip past the bypass and reach the
    downstream EPSG validator (issue #1971). This test pins that the
    conflicting-CRS check does NOT mask the downstream rejection in
    this corner: pass ``crs=0`` plus disagreeing attrs and ensure
    *some* ``ValueError`` fires (whichever sibling check raises
    first).
    """
    da = _da(attrs={'crs': 4326, 'crs_wkt': _WKT_32633})

    with pytest.raises(ValueError):
        to_geotiff(da, str(tmp_path / 'kwarg_zero.tif'), crs=0)


# ---------------------------------------------------------------------------
# Unparseable inputs: this check defers to the sibling
# ``UnparseableCRSError`` PR, which is not yet wired.
# ---------------------------------------------------------------------------


def test_unparseable_crs_wkt_does_not_fire_conflicting(tmp_path):
    """A ``crs_wkt`` that pyproj cannot parse should not surface as
    ``ConflictingCRSError``. The sibling ``UnparseableCRSError`` PR
    will handle it; until then the writer's existing WKT-string fallback
    decides what happens. This test pins that *our* check stays out of
    the way."""
    da = _da(attrs={'crs': 4326, 'crs_wkt': 'NOT A VALID WKT STRING'})
    out = str(tmp_path / 'unparseable_wkt.tif')

    try:
        to_geotiff(da, out)
    except ConflictingCRSError as e:
        pytest.fail(
            "Unparseable WKT should not surface as ConflictingCRSError; "
            f"that case is owned by UnparseableCRSError. Got: {e}"
        )
    except Exception:
        # Any other exception from the downstream WKT-fallback path is
        # acceptable -- this test only asserts that we do not falsely
        # claim a "conflict" for a value that pyproj cannot parse.
        pass


# ---------------------------------------------------------------------------
# Round-trip safety: the typical read-back DataArray (both attrs set,
# agreeing) writes cleanly without firing the check.
# ---------------------------------------------------------------------------


def test_read_back_roundtrip_does_not_raise(tmp_path):
    """End-to-end: write a 4326 file, open it (reader emits both attrs),
    write again. The second write must not trip the conflict check."""
    from xrspatial.geotiff import open_geotiff

    first = str(tmp_path / 'roundtrip_first.tif')
    second = str(tmp_path / 'roundtrip_second.tif')

    src = _da(attrs={'crs': 4326})
    to_geotiff(src, first)

    da = open_geotiff(first)
    assert 'crs' in da.attrs and 'crs_wkt' in da.attrs, (
        "reader should emit both crs and crs_wkt on a CRS-bearing file; "
        f"got attrs={sorted(da.attrs)}"
    )

    to_geotiff(da, second)
