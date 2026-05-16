"""Unit tests for the golden-corpus oracle harness (issue #1930, Phase 1.2).

These tests exercise the oracle in isolation: they synthesise a tiny TIFF
in a tmp_path with rasterio, build an ``xarray.DataArray`` by hand that
mirrors what an xrspatial reader would emit, and verify the oracle accepts
the matching case and rejects each property mismatch individually.

The tests do NOT depend on the Phase 1 PR 1 manifest or generator; the
oracle takes raw filesystem paths, so unit tests for the oracle only need
a writable temp directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip('rasterio')

from rasterio.transform import Affine, from_origin  # noqa: E402

from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write_tiff(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: str | int | None = 'EPSG:4326',
    nodata: float | int | None = None,
) -> Path:
    """Write a single-band TIFF to ``path`` using rasterio."""
    if transform is None:
        transform = from_origin(0.0, data.shape[0], 1.0, 1.0)
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'transform': transform,
    }
    if crs is not None:
        profile['crs'] = rasterio.crs.CRS.from_user_input(crs)
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
    return path


def _build_candidate(
    data: np.ndarray,
    *,
    transform: Affine,
    crs: int | None = 4326,
    crs_wkt: str | None = None,
    nodata: float | int | None = None,
    drop_band_axis: bool = True,
) -> xr.DataArray:
    """Build an xrspatial-shaped DataArray from in-memory data."""
    if drop_band_axis and data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    height = data.shape[-2]
    width = data.shape[-1]
    # pixel-centre coords matching xrspatial's coords_from_pixel_geometry
    pw = float(transform.a)
    ph = float(transform.e)
    ox = float(transform.c)
    oy = float(transform.f)
    x = ox + (np.arange(width) + 0.5) * pw
    y = oy + (np.arange(height) + 0.5) * ph
    attrs: dict = {
        'transform': (pw, 0.0, ox, 0.0, ph, oy),
    }
    if crs is not None:
        attrs['crs'] = crs
    if crs_wkt is not None:
        attrs['crs_wkt'] = crs_wkt
    if nodata is not None:
        attrs['nodata'] = nodata
    return xr.DataArray(
        data,
        dims=('y', 'x') if data.ndim == 2 else ('band', 'y', 'x'),
        coords={'y': y, 'x': x},
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_compare_to_oracle_success_int(tmp_path: Path) -> None:
    """Matching int raster: oracle accepts."""
    data = np.arange(16, dtype=np.int16).reshape(4, 4)
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'ok_int.tif', data, transform=transform)
    cand = _build_candidate(data, transform=transform, nodata=None)
    compare_to_oracle(fixture, cand)


def test_compare_to_oracle_success_float_with_nan(tmp_path: Path) -> None:
    """Float raster with NaN sentinels: oracle treats NaN as equal."""
    data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'ok_nan.tif', data, transform=transform,
        nodata=float('nan'),
    )
    cand = _build_candidate(
        data.copy(), transform=transform, nodata=float('nan'),
    )
    compare_to_oracle(fixture, cand)


def test_compare_to_oracle_success_via_crs_wkt(tmp_path: Path) -> None:
    """A candidate carrying crs_wkt (no EPSG int) still compares equal."""
    data = np.zeros((3, 3), dtype=np.uint8)
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'crs_wkt.tif', data, transform=transform)
    wkt = rasterio.crs.CRS.from_epsg(4326).to_wkt()
    cand = _build_candidate(
        data, transform=transform, crs=None, crs_wkt=wkt,
    )
    compare_to_oracle(fixture, cand)


def test_compare_to_oracle_lossy_skips_pixel_check(tmp_path: Path) -> None:
    """``lossy=True`` skips bit-exact pixel comparison."""
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'lossy.tif', data, transform=transform)
    # Different pixel values, same shape/dtype/transform/CRS:
    perturbed = data + 5
    cand = _build_candidate(perturbed.astype(np.uint8), transform=transform)
    # Strict mode rejects:
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand)
    # Lossy mode accepts:
    compare_to_oracle(fixture, cand, lossy=True)


# ---------------------------------------------------------------------------
# Mismatched-property failures
# ---------------------------------------------------------------------------

def test_dtype_mismatch_fails(tmp_path: Path) -> None:
    data = np.ones((4, 4), dtype=np.int16)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'dtype.tif', data, transform=transform)
    cand = _build_candidate(data.astype(np.int32), transform=transform)
    with pytest.raises(AssertionError, match='dtype mismatch'):
        compare_to_oracle(fixture, cand)


def test_transform_mismatch_fails(tmp_path: Path) -> None:
    data = np.zeros((4, 4), dtype=np.int16)
    fixture_transform = from_origin(0.0, 4.0, 1.0, 1.0)
    candidate_transform = from_origin(99.0, 4.0, 1.0, 1.0)  # shifted origin
    fixture = _write_tiff(
        tmp_path / 'tx.tif', data, transform=fixture_transform,
    )
    cand = _build_candidate(data, transform=candidate_transform)
    with pytest.raises(AssertionError, match='transform mismatch'):
        compare_to_oracle(fixture, cand)


def test_crs_mismatch_fails(tmp_path: Path) -> None:
    data = np.zeros((3, 3), dtype=np.int16)
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'crs.tif', data, transform=transform, crs='EPSG:4326',
    )
    cand = _build_candidate(data, transform=transform, crs=3857)
    with pytest.raises(AssertionError, match='CRS mismatch'):
        compare_to_oracle(fixture, cand)


def test_nodata_nan_vs_zero_fails(tmp_path: Path) -> None:
    """A float fixture with NaN nodata must not pass when candidate uses 0."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'nodata.tif', data, transform=transform,
        nodata=float('nan'),
    )
    cand = _build_candidate(data, transform=transform, nodata=0.0)
    with pytest.raises(AssertionError, match='nodata mismatch'):
        compare_to_oracle(fixture, cand)


def test_nodata_missing_on_candidate_fails(tmp_path: Path) -> None:
    """rasterio reports an int nodata, candidate forgets to set it."""
    data = np.full((3, 3), 7, dtype=np.int16)
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'nodata_none.tif', data, transform=transform, nodata=-1,
    )
    cand = _build_candidate(data, transform=transform, nodata=None)
    with pytest.raises(AssertionError, match='nodata mismatch'):
        compare_to_oracle(fixture, cand)


def test_pixel_mismatch_fails(tmp_path: Path) -> None:
    data = np.arange(9, dtype=np.int16).reshape(3, 3)
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'pix.tif', data, transform=transform)
    bad = data.copy()
    bad[1, 1] = 999
    cand = _build_candidate(bad, transform=transform)
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand)


def test_pixel_nan_vs_zero_fails(tmp_path: Path) -> None:
    """A NaN in the fixture must not silently match a 0 in the candidate."""
    data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'pix_nan.tif', data, transform=transform,
        nodata=float('nan'),
    )
    bad = data.copy()
    bad[0, 1] = 0.0
    cand = _build_candidate(
        bad, transform=transform, nodata=float('nan'),
    )
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand)


# ---------------------------------------------------------------------------
# EPSG-equivalent-WKT handling
# ---------------------------------------------------------------------------

def test_crs_epsg_equivalent_wkt_compares_equal(tmp_path: Path) -> None:
    """rasterio EPSG:4326 vs a WKT for the same CRS must compare equal."""
    data = np.zeros((3, 3), dtype=np.int16)
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'crs_eq.tif', data, transform=transform, crs='EPSG:4326',
    )
    # Candidate uses a hand-written WGS84 WKT instead of the EPSG int.
    wkt = rasterio.crs.CRS.from_epsg(4326).to_wkt()
    cand = _build_candidate(data, transform=transform, crs=None, crs_wkt=wkt)
    compare_to_oracle(fixture, cand)


# ---------------------------------------------------------------------------
# Lossy-mode failure modes
# ---------------------------------------------------------------------------

def test_lossy_still_rejects_dtype_mismatch(tmp_path: Path) -> None:
    data = np.ones((4, 4), dtype=np.uint8)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(tmp_path / 'lossy_dtype.tif', data, transform=transform)
    cand = _build_candidate(data.astype(np.uint16), transform=transform)
    with pytest.raises(AssertionError, match='dtype mismatch'):
        compare_to_oracle(fixture, cand, lossy=True)


def test_lossy_still_rejects_transform_mismatch(tmp_path: Path) -> None:
    data = np.zeros((4, 4), dtype=np.uint8)
    fixture_transform = from_origin(0.0, 4.0, 1.0, 1.0)
    candidate_transform = from_origin(0.0, 4.0, 2.0, 2.0)  # wrong pixel size
    fixture = _write_tiff(
        tmp_path / 'lossy_tx.tif', data, transform=fixture_transform,
    )
    cand = _build_candidate(data, transform=candidate_transform)
    with pytest.raises(AssertionError, match='transform mismatch'):
        compare_to_oracle(fixture, cand, lossy=True)


def test_lossy_rejects_shape_mismatch(tmp_path: Path) -> None:
    """In lossy mode, the shape check still trips when sizes disagree.

    This is the only test that exercises ``_assert_shape_only``; strict
    mode goes through ``_assert_pixels`` instead, which catches shape
    mismatches as part of the bit-exact comparison.
    """
    # Same pixel size + CRS + origin on both sides, but the fixture is 4x4
    # while the candidate is 5x5 -- only shape disagrees.
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'lossy_shape_only.tif',
        np.zeros((4, 4), dtype=np.uint8),
        transform=transform,
    )
    bigger = np.zeros((5, 5), dtype=np.uint8)
    cand = _build_candidate(bigger, transform=transform)
    with pytest.raises(AssertionError, match='shape mismatch'):
        compare_to_oracle(fixture, cand, lossy=True)


# ---------------------------------------------------------------------------
# Missing-file handling
# ---------------------------------------------------------------------------

def test_identity_transform_with_crs_still_compared(tmp_path: Path) -> None:
    """Regression: an identity-shaped transform on a real raster (CRS set)
    must still go through the transform comparison.

    Earlier the oracle treated any identity-equal ref transform as
    "no georef" and short-circuited. That hid bugs where the candidate
    drifted from the fixture's identity transform on real rasters
    written at origin (0, 0) with pixel size 1.0. The fix keys "no
    georef" off ``src.crs is None and src.transform.is_identity``
    together, not transform alone.
    """
    from rasterio.transform import Affine
    data = np.arange(4, dtype=np.int16).reshape(2, 2)
    fixture = _write_tiff(
        tmp_path / 'id_with_crs.tif',
        data,
        transform=Affine.identity(),
        crs='EPSG:4326',
    )
    # Candidate carries a SHIFTED transform; oracle must catch it.
    shifted = from_origin(99.0, 2.0, 1.0, 1.0)
    cand = _build_candidate(data, transform=shifted)
    with pytest.raises(AssertionError, match='transform mismatch'):
        compare_to_oracle(fixture, cand)


def test_no_georef_fixture_tolerates_missing_candidate_transform(
    tmp_path: Path,
) -> None:
    """A fixture with no CRS *and* identity transform may match a candidate
    that drops the transform attr entirely (xrspatial #1710 behaviour).
    """
    from rasterio.transform import Affine
    data = np.zeros((2, 2), dtype=np.int16)
    fixture = _write_tiff(
        tmp_path / 'no_georef.tif',
        data,
        transform=Affine.identity(),
        crs=None,
    )
    # Build a candidate by hand with no transform attr, no CRS attr,
    # integer-pixel coords.
    cand = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(2), 'x': np.arange(2)},
        attrs={},
    )
    compare_to_oracle(fixture, cand)


def test_missing_fixture_raises_filenotfounderror(tmp_path: Path) -> None:
    cand = _build_candidate(
        np.zeros((2, 2), dtype=np.int16),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    )
    with pytest.raises(FileNotFoundError):
        compare_to_oracle(tmp_path / 'does_not_exist.tif', cand)


# ---------------------------------------------------------------------------
# Phase 2 PR 8 CRS-variant fixtures
#
# Smoke tests for the three CRS-representation fixtures added in PR 8 of
# issue #1930. Each test reads the on-disk fixture with rasterio to pin
# the bytes-on-disk behaviour, then drives the oracle with a hand-built
# candidate to verify the comparison path the fixture is meant to
# exercise. Phase 3 will wire real backends to these same files.
# ---------------------------------------------------------------------------

_CRS_FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'


def _read_crs_fixture(name: str):
    """Open a fixture and return its rasterio metadata plus pixel data."""
    path = _CRS_FIXTURE_DIR / f'{name}.tif'
    with rasterio.open(path) as src:
        return (
            path,
            src.crs,
            src.transform,
            src.read(1),  # single-band uint8
        )


def test_crs_epsg_3857_fixture_reports_epsg() -> None:
    """``crs_epsg_3857`` fixture: rasterio reports CRS.from_epsg(3857).

    The straight-EPSG path. Oracle accepts a candidate that carries the
    EPSG int under ``attrs['crs']``.
    """
    path, ref_crs, transform, data = _read_crs_fixture('crs_epsg_3857')
    assert ref_crs == rasterio.crs.CRS.from_epsg(3857)
    assert ref_crs.to_epsg() == 3857

    cand = _build_candidate(data, transform=transform, crs=3857)
    compare_to_oracle(path, cand)


def test_crs_wkt_utm10n_fixture_resolves_to_epsg_via_fallback() -> None:
    """``crs_wkt_utm10n``: WKT-only on disk, but resolves to EPSG:32610.

    The fixture's WKT has no AUTHORITY tags, so it is not byte-identical
    to what ``CRS.from_epsg(32610).to_wkt()`` emits. PROJ still recognises
    it as UTM 10N and assigns it EPSG:32610 on read, which is the
    fallback path ``_crs_equal`` was built for. A candidate that carries
    the bare EPSG int must compare equal to the rasterio-read WKT CRS.
    """
    path, ref_crs, transform, data = _read_crs_fixture('crs_wkt_utm10n')
    assert ref_crs.to_epsg() == 32610

    # Candidate carries only the EPSG int. The oracle reaches the
    # EPSG-fallback branch of _crs_equal because ref's WKT and the
    # canonical EPSG:32610 WKT are not structurally equal.
    cand = _build_candidate(data, transform=transform, crs=32610)
    compare_to_oracle(path, cand)


def test_crs_citation_only_fixture_oracle_accepts_via_proj_dict() -> None:
    """``crs_citation_only``: GeoKey citation, no AUTHORITY.

    Neither side has an EPSG code, and libgeotiff mutates the WKT on
    round-trip (axis order, UNIT AUTHORITY) so structural ``CRS.__eq__``
    fails. The oracle falls back to comparing ``to_dict()`` (PROJ form),
    which is stable across that round-trip. Pinned here so any future
    refactor of ``_crs_equal`` that drops the PROJ-dict branch trips a
    test instead of silently regressing.
    """
    path, ref_crs, transform, data = _read_crs_fixture('crs_citation_only')
    assert ref_crs is not None
    assert ref_crs.to_epsg() is None

    # Candidate carries the WKT under crs_wkt; oracle's _candidate_crs
    # picks it up via from_user_input.
    cand = _build_candidate(
        data, transform=transform, crs=None, crs_wkt=ref_crs.to_wkt(),
    )
    compare_to_oracle(path, cand)


def test_crs_citation_only_fixture_rejects_unrelated_crs() -> None:
    """Negative pin: the PROJ-dict fallback must still reject mismatches.

    EPSG:4326 has the same coarse ``proj=longlat`` family as the
    citation-only CRS but a different ellipsoid (WGS84 vs the fixture's
    unknown sphere). ``to_dict()`` differs, so the oracle must raise.
    """
    path, _ref_crs, transform, data = _read_crs_fixture('crs_citation_only')
    cand = _build_candidate(data, transform=transform, crs=4326)
    with pytest.raises(AssertionError, match='CRS mismatch'):
        compare_to_oracle(path, cand)
