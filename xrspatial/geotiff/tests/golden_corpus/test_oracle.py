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


def _write_multiband_tiff(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: str | int | None = 'EPSG:4326',
) -> Path:
    """Write a multi-band TIFF from a ``(B, H, W)`` array."""
    assert data.ndim == 3, 'multi-band writer expects (B, H, W) input'
    bands, height, width = data.shape
    if transform is None:
        transform = from_origin(0.0, float(height), 1.0, 1.0)
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': bands,
        'dtype': data.dtype,
        'transform': transform,
    }
    if crs is not None:
        profile['crs'] = rasterio.crs.CRS.from_user_input(crs)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
    return path


def _build_candidate(
    data: np.ndarray,
    *,
    transform: Affine,
    crs: int | None = 4326,
    crs_wkt: str | None = None,
    nodata: float | int | None = None,
    drop_band_axis: bool = True,
    band_axis: str = 'leading',
) -> xr.DataArray:
    """Build an xrspatial-shaped DataArray from in-memory data.

    Parameters
    ----------
    data
        2-D ``(H, W)`` or 3-D array. For 3-D inputs ``band_axis`` chooses
        which dimension carries the band axis: ``'leading'`` matches
        rasterio's ``(B, H, W)`` shape (the default), ``'trailing'``
        matches xrspatial's multi-band ``(H, W, B)`` layout. The latter
        is what the JPEG-YCbCr fixture exercises in the corpus.
    drop_band_axis
        When ``True`` and ``data`` is shape ``(1, H, W)``, squeeze the
        leading length-1 axis to a 2-D array before building the
        DataArray. Has no effect on ``(H, W, B)`` inputs (the band axis
        is trailing there).
    """
    if drop_band_axis and data.ndim == 3 and data.shape[0] == 1 and band_axis == 'leading':
        data = data[0]
    if data.ndim == 3:
        if band_axis == 'leading':
            height = data.shape[1]
            width = data.shape[2]
            dims: tuple[str, ...] = ('band', 'y', 'x')
        elif band_axis == 'trailing':
            height = data.shape[0]
            width = data.shape[1]
            dims = ('y', 'x', 'band')
        else:
            raise ValueError(
                f'band_axis must be "leading" or "trailing", got {band_axis!r}'
            )
    else:
        height = data.shape[-2]
        width = data.shape[-1]
        dims = ('y', 'x')
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
        dims=dims,
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


def test_crs_wkt_utm10n_fixture_accepts_wkt_attr() -> None:
    """``crs_wkt_utm10n`` also accepts a candidate that carries crs_wkt.

    Complements the EPSG-int test by exercising the WKT branch of
    ``_candidate_crs`` (``attrs['crs_wkt']`` -> ``from_user_input``).
    Both paths must reach the same verdict.
    """
    path, ref_crs, transform, data = _read_crs_fixture('crs_wkt_utm10n')
    cand = _build_candidate(
        data, transform=transform, crs=None, crs_wkt=ref_crs.to_wkt(),
    )
    compare_to_oracle(path, cand)


def test_crs_equal_rejects_empty_proj_dict() -> None:
    """``_crs_equal`` must refuse to declare two LOCAL_CS-style CRSes equal.

    Regression pin for the PROJ-dict fallback added in this PR. PROJ
    returns ``{}`` from ``to_dict()`` for LOCAL_CS WKTs; an unguarded
    fallback would treat any two such CRSes as equal, which is a
    silent-false-positive in the oracle. The fallback must short-circuit
    on empty dicts.
    """
    from xrspatial.geotiff.tests.golden_corpus._oracle import _crs_equal

    # Two LOCAL_CS WKTs with different UNIT blocks so rasterio's own
    # ``CRS.__eq__`` reports them as unequal (otherwise the early-return
    # in _crs_equal would short-circuit before the fallback runs).
    a = rasterio.crs.CRS.from_wkt(
        'LOCAL_CS["a",UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
        'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    b = rasterio.crs.CRS.from_wkt(
        'LOCAL_CS["b",UNIT["foot",0.3048],'
        'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    # Sanity: structurally unequal, neither has EPSG, both have empty
    # PROJ-dict. Without the guard, the fallback would return True.
    assert a != b
    assert a.to_epsg() is None
    assert b.to_epsg() is None
    assert a.to_dict() == {} == b.to_dict()
    assert _crs_equal(a, b) is False


# ---------------------------------------------------------------------------
# Masked-nodata contract (issue #1988)
# ---------------------------------------------------------------------------

def _masked_nodata_pair(
    tmp_path: Path, sentinel: int = 0
) -> tuple[Path, xr.DataArray, np.ndarray]:
    """Build a uint16 fixture with an integer nodata sentinel and a
    float candidate that has masked the sentinel pixels to NaN.

    Returns ``(fixture_path, candidate, candidate_float_array)``.
    """
    raw = np.array(
        [
            [1, 2, sentinel, 4],
            [5, sentinel, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.uint16,
    )
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'masked_nodata.tif',
        raw,
        transform=transform,
        nodata=sentinel,
    )
    masked = raw.astype(np.float64)
    masked[raw == sentinel] = np.nan
    cand = _build_candidate(
        masked, transform=transform, nodata=sentinel,
    )
    cand.attrs['masked_nodata'] = True
    return fixture, cand, masked


def test_masked_nodata_success(tmp_path: Path) -> None:
    """Candidate that masks the integer sentinel to NaN passes the oracle.

    The oracle rewrites the rasterio reference to match the candidate's
    float-plus-NaN view before comparing, so dtype shift + pixel mask
    no longer registers as a mismatch.
    """
    fixture, cand, _ = _masked_nodata_pair(tmp_path)
    compare_to_oracle(fixture, cand)


def test_masked_nodata_off_keeps_strict_dtype_check(tmp_path: Path) -> None:
    """Without ``masked_nodata=True``, the oracle keeps strict dtype parity.

    Drop the flag and the candidate's float dtype no longer matches the
    rasterio uint16 reference; the dtype assertion fires.
    """
    fixture, cand, _ = _masked_nodata_pair(tmp_path)
    del cand.attrs['masked_nodata']
    with pytest.raises(AssertionError, match='dtype mismatch'):
        compare_to_oracle(fixture, cand)


def test_masked_nodata_candidate_left_sentinel_pixel_fails(
    tmp_path: Path,
) -> None:
    """If the candidate forgot to mask a sentinel pixel, the pixel
    comparison fires.

    The reference is rewritten to have NaN at the sentinel positions;
    a candidate that still carries the raw integer value (here 0,
    cast to float) at one of those positions cannot pass NaN-aware
    equality.
    """
    fixture, cand, masked = _masked_nodata_pair(tmp_path)
    # Restore the first sentinel pixel to the raw value, simulating
    # a reader that mis-masked.
    bad = masked.copy()
    bad[0, 2] = 0.0
    cand2 = _build_candidate(
        bad, transform=from_origin(0.0, 4.0, 1.0, 1.0), nodata=0,
    )
    cand2.attrs['masked_nodata'] = True
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand2)


def test_masked_nodata_wrong_nodata_attr_fails(tmp_path: Path) -> None:
    """``attrs['nodata']`` must still match the fixture even with masking.

    The oracle masks the reference using the *reference's* nodata, so
    a wrong sentinel in the candidate's attrs is caught by the
    independent ``_assert_nodata`` check before pixels are compared.
    """
    fixture, cand, _ = _masked_nodata_pair(tmp_path)
    cand.attrs['nodata'] = 999  # not the real sentinel
    with pytest.raises(AssertionError, match='nodata mismatch'):
        compare_to_oracle(fixture, cand)


def test_masked_nodata_does_not_engage_for_float_nan_fixture(
    tmp_path: Path,
) -> None:
    """Float-NaN-nodata fixtures take the plain pixel path unchanged.

    The masked-nodata normaliser only fires when the candidate reports
    ``masked_nodata=True``. A float fixture whose nodata is NaN never
    needs the dtype shift, and the new path must not muck with it.
    """
    data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'float_nan.tif', data, transform=transform,
        nodata=float('nan'),
    )
    cand = _build_candidate(
        data.copy(), transform=transform, nodata=float('nan'),
    )
    # No masked_nodata flag -- the candidate is float to begin with.
    compare_to_oracle(fixture, cand)


def test_masked_nodata_with_non_integer_sentinel_passes_through(
    tmp_path: Path,
) -> None:
    """A NaN sentinel in the rasterio reference does not trip the integer
    masking path even when ``masked_nodata`` is set.

    The normaliser is gated on a finite sentinel via ``np.isfinite``.
    A NaN-nodata fixture should compare like any other float-NaN
    fixture: candidate keeps its own NaN pixels, reference keeps
    its own NaN pixels, and NaN-aware equality does the rest.
    """
    data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'masked_nan.tif', data, transform=transform,
        nodata=float('nan'),
    )
    cand = _build_candidate(
        data.copy(), transform=transform, nodata=float('nan'),
    )
    # Set the flag to confirm the normaliser early-exits cleanly on
    # NaN sentinels rather than trying to do something silly.
    cand.attrs['masked_nodata'] = True
    compare_to_oracle(fixture, cand)


def test_masked_nodata_fractional_sentinel_does_not_mask(
    tmp_path: Path,
) -> None:
    """A non-integer sentinel cannot match an integer pixel; the
    normaliser declines to mask.

    Without this guard the oracle would cast ``3.5`` to ``3`` and
    silently mask every 3-valued pixel. The upstream xrspatial reader
    never sets ``attrs['masked_nodata']`` for a fractional sentinel,
    so the oracle's stricter check mirrors that contract. Outcome:
    the oracle stays on the raw-pixel path and the dtype mismatch
    surfaces normally.
    """
    raw = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'frac_sentinel.tif', raw, transform=transform,
        nodata=3.5,
    )
    masked = raw.astype(np.float64)
    cand = _build_candidate(
        masked, transform=transform, nodata=3.5,
    )
    cand.attrs['masked_nodata'] = True
    with pytest.raises(AssertionError, match='dtype mismatch'):
        compare_to_oracle(fixture, cand)


def test_masked_nodata_out_of_range_sentinel_does_not_mask() -> None:
    """A sentinel outside the source integer range cannot match any pixel.

    Without the range guard, ``np.uint16(-1.0)`` wraps to ``65535`` and
    masks the dtype-max pixel. The reader rejects such sentinels via
    ``info.min <= nodata_int <= info.max``; the oracle mirrors that
    check.

    Rasterio refuses to write an out-of-range nodata at the writer
    level, so we cannot reach this code path through a real fixture.
    The test calls the helper directly with synthesised inputs to
    confirm the guard fires.
    """
    from xrspatial.geotiff.tests.golden_corpus._oracle import (
        _normalise_for_masked_nodata,
    )

    ref_pixels = np.array(
        [[1, 2, 65535], [4, 5, 6]], dtype=np.uint16,
    )
    ref_dtype = ref_pixels.dtype
    cand = _build_candidate(
        ref_pixels.astype(np.float64),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-1,
    )
    cand.attrs['masked_nodata'] = True

    out_pixels, out_dtype = _normalise_for_masked_nodata(
        ref_pixels, ref_dtype, -1, cand,
    )
    # Out-of-range sentinel must abort the rewrite. The helper returns
    # the inputs unchanged; the unsigned wraparound that would otherwise
    # mask the dtype-max pixel does not happen.
    assert out_dtype == ref_dtype
    assert out_pixels is ref_pixels


# ---------------------------------------------------------------------------
# Multi-band axis-order normalisation (issue #1930)
#
# rasterio reads every TIFF as ``(bands, H, W)``; xrspatial reads multi-band
# rasters as ``(H, W, B)``. The convention difference is documented, not a
# bug, so the oracle normalises before comparing. The corpus exposes this
# via the JPEG-YCbCr fixture, which is 3 bands of uint8.
# ---------------------------------------------------------------------------

def _multiband_pixels() -> np.ndarray:
    """Distinct per-band uint8 pixel data of shape ``(3, 4, 5)``.

    Each band carries a different ramp so a misalignment between bands
    would surface as a pixel mismatch rather than a silent pass.
    """
    h, w = 4, 5
    bands = np.stack(
        [
            np.arange(h * w, dtype=np.uint8).reshape(h, w),
            np.arange(h * w, dtype=np.uint8).reshape(h, w) + 50,
            np.arange(h * w, dtype=np.uint8).reshape(h, w) + 100,
        ],
        axis=0,
    )
    assert bands.shape == (3, h, w)
    return bands


def test_multiband_axis_order_success_strict(tmp_path: Path) -> None:
    """Multi-band ``(B, H, W)`` ref vs ``(H, W, B)`` candidate passes.

    rasterio writes and reads the fixture in band-first layout; xrspatial
    presents the same logical image with the band axis trailing. The
    oracle normalises before the bit-exact comparison so identical
    pixels in either layout compare equal.
    """
    ref = _multiband_pixels()
    transform = from_origin(0.0, float(ref.shape[1]), 1.0, 1.0)
    fixture = _write_multiband_tiff(
        tmp_path / 'multiband_ok.tif', ref, transform=transform,
    )
    # xrspatial-shaped candidate: (H, W, B)
    cand_data = np.moveaxis(ref, 0, -1)
    assert cand_data.shape == (ref.shape[1], ref.shape[2], ref.shape[0])
    cand = _build_candidate(
        cand_data, transform=transform, band_axis='trailing',
    )
    compare_to_oracle(fixture, cand)


def test_multiband_axis_order_pixel_mismatch_fails(tmp_path: Path) -> None:
    """A real pixel mismatch across the two layouts still trips ``_assert_pixels``.

    The axis-order normalisation must not paper over a genuine divergence.
    Flip one pixel in one band; after the oracle transposes the candidate
    to ``(B, H, W)`` the comparison must still raise.
    """
    ref = _multiband_pixels()
    transform = from_origin(0.0, float(ref.shape[1]), 1.0, 1.0)
    fixture = _write_multiband_tiff(
        tmp_path / 'multiband_pixmm.tif', ref, transform=transform,
    )
    cand_data = np.moveaxis(ref, 0, -1).copy()
    cand_data[2, 3, 1] = 99  # perturb band 1 at (y=2, x=3)
    cand = _build_candidate(
        cand_data, transform=transform, band_axis='trailing',
    )
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand)


def test_multiband_axis_order_lossy_shape_only(tmp_path: Path) -> None:
    """``lossy=True`` shape-only comparison handles the axis order.

    Mirrors the JPEG-YCbCr corpus cell: rasterio sees ``(3, H, W)``,
    xrspatial sees ``(H, W, 3)``, and the lossy path checks only shape,
    dtype, transform, and CRS. The normalisation must align the two
    before the shape check, so identical logical shapes pass.
    """
    ref = _multiband_pixels()
    transform = from_origin(0.0, float(ref.shape[1]), 1.0, 1.0)
    fixture = _write_multiband_tiff(
        tmp_path / 'multiband_lossy.tif', ref, transform=transform,
    )
    # Lossy: perturb the pixels but keep the same shape.
    perturbed = np.moveaxis(ref, 0, -1).copy() + 5
    cand = _build_candidate(
        perturbed.astype(np.uint8),
        transform=transform,
        band_axis='trailing',
    )
    # Strict comparison rejects (pixel values differ):
    with pytest.raises(AssertionError, match='pixel arrays differ'):
        compare_to_oracle(fixture, cand)
    # Lossy accepts because the shape lines up after axis normalisation:
    compare_to_oracle(fixture, cand, lossy=True)


def test_multiband_axis_order_lossy_shape_mismatch_fails(
    tmp_path: Path,
) -> None:
    """Genuine multi-band shape mismatches still trip the lossy path.

    If the spatial extent disagrees, the normalisation should not
    transpose (the H/W axes would not line up), so the assertion fires
    with the raw shapes.
    """
    ref = _multiband_pixels()  # (3, 4, 5)
    transform = from_origin(0.0, float(ref.shape[1]), 1.0, 1.0)
    fixture = _write_multiband_tiff(
        tmp_path / 'multiband_lossy_shape.tif', ref, transform=transform,
    )
    # Candidate has the wrong height (5 instead of 4); (H, W, B)
    # cannot transpose cleanly to (B, H, W) with H/W matching.
    wrong = np.zeros((5, 5, 3), dtype=np.uint8)
    cand = _build_candidate(
        wrong, transform=transform, band_axis='trailing',
    )
    with pytest.raises(AssertionError, match='shape mismatch'):
        compare_to_oracle(fixture, cand, lossy=True)


def test_singleband_axis_order_still_squeezed(tmp_path: Path) -> None:
    """Regression: the existing ``(1, H, W)`` vs ``(H, W)`` squeeze path
    still works after the multi-band branch was added.

    The single-band case must continue to compare equal: rasterio reads
    every fixture as 3-D with a leading band axis, xrspatial drops it
    for the single-band layout, and the oracle squeezes so the two
    compare as 2-D. The multi-band code path is gated on ``B > 1`` so
    this case still lands in the squeeze branch.
    """
    data = np.arange(20, dtype=np.int16).reshape(4, 5)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'singleband_squeeze.tif', data, transform=transform,
    )
    cand = _build_candidate(data, transform=transform)
    compare_to_oracle(fixture, cand)


def test_normalise_axis_order_helper_directly() -> None:
    """Unit-level coverage of ``_normalise_axis_order``.

    Pins the four supported cases so a future refactor of the helper has
    to spell out the contract:

    1. (1, H, W) vs (H, W) -- squeezes ref.
    2. (H, W) vs (1, H, W) -- squeezes cand.
    3. (B, H, W) vs (H, W, B) with B > 1 -- transposes cand to leading.
    4. (H, W, B) vs (B, H, W) with B > 1 -- transposes ref to leading.

    A genuine 3-D mismatch (e.g. different band counts on either side)
    falls through unchanged.
    """
    from xrspatial.geotiff.tests.golden_corpus._oracle import (
        _normalise_axis_order,
    )

    a2 = np.arange(12).reshape(3, 4)
    a3_lead_single = a2[np.newaxis]  # (1, 3, 4)
    out_ref, out_cand = _normalise_axis_order(a3_lead_single, a2)
    assert out_ref.shape == (3, 4)
    assert out_cand.shape == (3, 4)
    out_ref, out_cand = _normalise_axis_order(a2, a3_lead_single)
    assert out_ref.shape == (3, 4)
    assert out_cand.shape == (3, 4)

    bands_lead = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (B=2, H=3, W=4)
    bands_trail = np.moveaxis(bands_lead, 0, -1)        # (H=3, W=4, B=2)
    out_ref, out_cand = _normalise_axis_order(bands_lead, bands_trail)
    assert out_ref.shape == (2, 3, 4)
    assert out_cand.shape == (2, 3, 4)
    assert np.array_equal(out_ref, out_cand)
    out_ref, out_cand = _normalise_axis_order(bands_trail, bands_lead)
    assert out_ref.shape == (2, 3, 4)
    assert out_cand.shape == (2, 3, 4)
    assert np.array_equal(out_ref, out_cand)

    # Genuine mismatch: 3 bands vs 2 bands. No transpose applies; the
    # helper returns inputs unchanged so the caller's shape assertion
    # raises with the real shapes.
    mismatch_ref = np.zeros((3, 3, 4))  # (B=3, H=3, W=4)
    mismatch_cand = np.zeros((3, 4, 2))  # (H=3, W=4, B=2)
    out_ref, out_cand = _normalise_axis_order(mismatch_ref, mismatch_cand)
    assert out_ref.shape == (3, 3, 4)
    assert out_cand.shape == (3, 4, 2)
