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


def test_crs_citation_only_open_geotiff_stamps_canonical_wkt() -> None:
    """``open_geotiff`` on the citation fixture stamps ``attrs['crs_wkt']``.

    The user-defined geographic CRS in ``crs_citation_only.tif`` has no
    EPSG code and no WKT in the citation; only the ellipsoid radius,
    inverse flattening, and angular-units GeoKeys are populated. The
    reader synthesizes a canonical WKT from those parameters via
    :func:`xrspatial.geotiff._geotags._synthesize_user_defined_wkt`,
    which closes the Phase 3 ``crs_citation_only`` parity gap.

    Pinned here so any future refactor that drops the synthesis branch
    re-opens the gap visibly. The companion oracle test
    (``test_crs_citation_only_xrspatial_round_trips_through_oracle``)
    drives the same fixture through ``compare_to_oracle`` to confirm the
    stamp lands somewhere the parity check accepts.
    """
    from xrspatial.geotiff import open_geotiff

    path = _CRS_FIXTURE_DIR / 'crs_citation_only.tif'
    da = open_geotiff(str(path))

    # The fixture has no EPSG, so the canonical EPSG attr stays absent.
    assert da.attrs.get('crs') is None
    wkt = da.attrs.get('crs_wkt')
    assert isinstance(wkt, str) and wkt, (
        f"open_geotiff must stamp a non-empty crs_wkt on the "
        f"citation-only fixture; got {wkt!r}"
    )
    # PROJ structural sanity: the synthesized WKT must parse back
    # through rasterio's CRS.from_wkt (which delegates to PROJ).
    parsed = rasterio.crs.CRS.from_wkt(wkt)
    assert parsed is not None
    # And it must share PROJ-dict shape with the rasterio reference.
    with rasterio.open(path) as src:
        ref = src.crs
    assert parsed.to_dict() == ref.to_dict(), (
        f"synthesized WKT must round-trip to the same PROJ dict as "
        f"the rasterio reference; got {parsed.to_dict()} vs "
        f"{ref.to_dict()}"
    )


def test_crs_citation_only_xrspatial_round_trips_through_oracle() -> None:
    """``compare_to_oracle`` accepts the xrspatial-stamped citation CRS.

    Drives the citation fixture through ``open_geotiff`` (the exact
    code path the Phase 3 backend parametrizations use) and runs the
    result through ``compare_to_oracle``. This is the end-to-end
    parity check that flips the corpus from xfail to pass once
    ``_synthesize_user_defined_wkt`` is wired in.
    """
    from xrspatial.geotiff import open_geotiff

    path = _CRS_FIXTURE_DIR / 'crs_citation_only.tif'
    da = open_geotiff(str(path))
    compare_to_oracle(path, da)


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

    # 2-D / 2-D pass-through: identical shapes need no normalisation.
    plain = np.arange(12).reshape(3, 4)
    out_ref, out_cand = _normalise_axis_order(plain, plain.copy())
    assert out_ref.shape == (3, 4)
    assert out_cand.shape == (3, 4)
    assert np.array_equal(out_ref, out_cand)

    # H == W == B ambiguity: the shape-equality short-circuit returns
    # the arrays untouched so two same-shape (3, 3, 3) cubes compare
    # directly rather than getting silently transposed.
    cube_a = np.arange(27).reshape(3, 3, 3)
    cube_b = cube_a.copy()
    out_ref, out_cand = _normalise_axis_order(cube_a, cube_b)
    assert out_ref is cube_a
    assert out_cand is cube_b


# ---------------------------------------------------------------------------
# Overview-level parity (issue #1930)
#
# ``compare_to_oracle`` accepts an optional ``candidate_factory`` so a
# caller can plumb the same backend through every overview level the
# fixture exposes. The base-IFD path stays exactly as before when no
# factory is given, which keeps the public single-level signature
# backward compatible.
# ---------------------------------------------------------------------------


def _write_tiff_with_overviews(
    path: Path,
    data: np.ndarray,
    *,
    factors: list[int],
    transform: Affine | None = None,
    crs: str | int | None = 'EPSG:4326',
    nodata: float | int | None = None,
) -> Path:
    """Write a single-band TIFF and build internal overviews.

    The base raster is written via ``_write_tiff``; the overview
    pyramid is then built in-place with nearest-neighbour resampling
    (the default for the corpus's overview fixtures, see manifest
    entries ``overview_internal_uint16`` and friends). Returns the
    same path.
    """
    from rasterio.enums import Resampling

    _write_tiff(path, data, transform=transform, crs=crs, nodata=nodata)
    with rasterio.open(path, 'r+') as dst:
        dst.build_overviews(factors, Resampling.nearest)
    return path


def _candidate_for_level(
    base_data: np.ndarray,
    base_transform: Affine,
    level: int,
    factors: list[int],
) -> xr.DataArray:
    """Build a hand-shaped candidate that matches the nearest-neighbour
    overview rasterio would write at ``level`` from ``base_data``.

    ``level=0`` returns the base raster. ``level>=1`` decimates by
    ``factors[level - 1]`` using nearest-neighbour subsampling
    (``data[::f, ::f]``), which agrees bit-for-bit with how rasterio
    builds the nearest-resampling pyramid for these small fixtures.
    The transform's pixel size scales by the same factor; the origin
    stays put.
    """
    if level == 0:
        return _build_candidate(base_data, transform=base_transform)
    factor = factors[level - 1]
    decimated = base_data[::factor, ::factor].copy()
    pw = float(base_transform.a) * factor
    ph = float(base_transform.e) * factor
    ox = float(base_transform.c)
    oy = float(base_transform.f)
    scaled = Affine(pw, 0.0, ox, 0.0, ph, oy)
    return _build_candidate(decimated, transform=scaled)


def test_compare_to_oracle_overview_success(tmp_path: Path) -> None:
    """A 2-level pyramid compares cleanly when the factory matches.

    Synthesises a uint16 raster with overviews at decimations [2, 4],
    hands the oracle a factory that returns the matching subsampled
    candidate at every level, and confirms the comparison accepts both
    overview levels in addition to the base.
    """
    factors = [2, 4]
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    transform = from_origin(0.0, 64.0, 1.0, 1.0)
    fixture = _write_tiff_with_overviews(
        tmp_path / 'ovr_ok_1930.tif',
        base,
        factors=factors,
        transform=transform,
    )
    base_candidate = _candidate_for_level(base, transform, 0, factors)

    def factory(level: int) -> xr.DataArray:
        return _candidate_for_level(base, transform, level, factors)

    compare_to_oracle(fixture, base_candidate, candidate_factory=factory)


def test_compare_to_oracle_overview_level1_mismatch_names_level(
    tmp_path: Path,
) -> None:
    """A pixel mismatch at level 1 only fails with a message that names
    the level.

    Level 0 and level 2 match the rasterio-built pyramid; level 1
    perturbs one pixel. The oracle must keep going past the matching
    base level, fail at level 1, and surface ``overview level 1`` in
    the assertion message so the failing level is greppable in CI logs.
    """
    factors = [2, 4]
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    transform = from_origin(0.0, 64.0, 1.0, 1.0)
    fixture = _write_tiff_with_overviews(
        tmp_path / 'ovr_lvl1_bad_1930.tif',
        base,
        factors=factors,
        transform=transform,
    )
    base_candidate = _candidate_for_level(base, transform, 0, factors)

    def factory(level: int) -> xr.DataArray:
        cand = _candidate_for_level(base, transform, level, factors)
        if level == 1:
            cand = cand.copy()
            cand.data[0, 0] = 65535  # corrupt one pixel at level 1 only
        return cand

    with pytest.raises(AssertionError, match=r'overview level 1'):
        compare_to_oracle(fixture, base_candidate, candidate_factory=factory)


def test_compare_to_oracle_no_overviews_skips_factory(tmp_path: Path) -> None:
    """A fixture without overviews ignores ``candidate_factory`` entirely.

    The base-IFD comparison runs as before; the factory must not be
    invoked even when supplied (the count check short-circuits the
    loop). A factory that explodes if called is a clean way to pin
    this -- if the loop ever runs against a fixture with zero
    overviews, the test fails with the explosion.
    """
    data = np.arange(16, dtype=np.int16).reshape(4, 4)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'no_ovr_1930.tif', data, transform=transform,
    )
    cand = _build_candidate(data, transform=transform)

    def boom(_level: int) -> xr.DataArray:
        raise AssertionError('factory must not be called when fixture '
                             'has no overviews')

    compare_to_oracle(fixture, cand, candidate_factory=boom)


def test_compare_to_oracle_external_ovr_sidecar(tmp_path: Path) -> None:
    """The sidecar ``.tif.ovr`` route is exercised end-to-end.

    rasterio routes ``OVERVIEW_LEVEL=N`` to whichever IFD chain holds
    the overviews -- the in-file chain for internal pyramids, the
    sibling ``.tif.ovr`` for external sidecars. The oracle does not
    care which storage the fixture uses; it just opens the source at
    ``OVERVIEW_LEVEL`` and reads. This test pins that contract: a
    hand-built fixture that writes overviews into a ``.ovr`` sidecar
    compares cleanly when the factory returns the matching candidate.

    Internal-IFD overview coverage is already pinned by
    ``test_compare_to_oracle_overview_success``; this test guards the
    external-sidecar route against a future regression in either
    rasterio or the oracle's source introspection.
    """
    from rasterio.enums import Resampling

    factors = [2, 4]
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    transform = from_origin(0.0, 64.0, 1.0, 1.0)
    fixture = _write_tiff(
        tmp_path / 'ovr_sidecar_1930.tif',
        base,
        transform=transform,
    )
    # Build overviews into a sidecar .tif.ovr by opening with
    # TIFF_USE_OVR=YES so rasterio writes them externally rather than
    # appending to the in-file IFD chain.
    with rasterio.Env(TIFF_USE_OVR='YES'):
        with rasterio.open(fixture, 'r+') as dst:
            dst.build_overviews(factors, Resampling.nearest)
    # Confirm the sidecar landed on disk before driving the oracle so
    # this test stays self-pinning if the env var changes meaning.
    assert (tmp_path / 'ovr_sidecar_1930.tif.ovr').exists(), (
        'rasterio did not write the sidecar .ovr; the env var may '
        'have stopped opting into external overviews'
    )

    base_candidate = _candidate_for_level(base, transform, 0, factors)

    def factory(level: int) -> xr.DataArray:
        return _candidate_for_level(base, transform, level, factors)

    compare_to_oracle(fixture, base_candidate, candidate_factory=factory)


def test_compare_to_oracle_overview_lossy_skips_pixel_check_per_level(
    tmp_path: Path,
) -> None:
    """``lossy=True`` applies to every overview level when a factory is given.

    Pins the docstring claim: a lossy comparison skips bit-exact pixel
    checks at the base AND at every overview level. The factory
    perturbs pixels at level 1 but keeps shape / dtype / transform /
    CRS intact; strict mode would fail there, lossy mode must accept.
    """
    factors = [2, 4]
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    transform = from_origin(0.0, 64.0, 1.0, 1.0)
    fixture = _write_tiff_with_overviews(
        tmp_path / 'ovr_lossy_1930.tif',
        base,
        factors=factors,
        transform=transform,
    )
    base_candidate = _candidate_for_level(base, transform, 0, factors)

    def factory(level: int) -> xr.DataArray:
        cand = _candidate_for_level(base, transform, level, factors)
        if level == 1:
            # Perturb pixels but keep shape / dtype / transform / CRS
            # intact so only the bit-exact comparison would fail.
            cand = cand.copy()
            cand.data[...] = cand.data + 1
        return cand

    # Strict mode rejects the level-1 pixel perturbation:
    with pytest.raises(AssertionError, match=r'overview level 1'):
        compare_to_oracle(
            fixture, base_candidate, candidate_factory=factory,
        )
    # Lossy mode accepts the same input because the shape-only check
    # passes at every level:
    compare_to_oracle(
        fixture, base_candidate, lossy=True, candidate_factory=factory,
    )


def test_compare_to_oracle_overview_corpus_external_ovr_fixture() -> None:
    """The shipped ``overview_external_ovr_uint16`` fixture parity-checks.

    Drives the on-disk corpus fixture through a hand-built factory
    that mirrors what rasterio reads at each level (no xrspatial
    reader involvement). Pins that the oracle can walk an external
    ``.ovr`` chain end-to-end on a real corpus fixture, independent
    of whether the xrspatial reader has caught up to sidecar
    overviews yet. The per-backend test modules skip the factory for
    this fixture because the reader cannot resolve
    ``overview_level >= 1`` against the sidecar; this oracle-level
    test still proves the oracle side of the contract works.
    """
    path = (
        Path(__file__).resolve().parent
        / 'fixtures' / 'overview_external_ovr_uint16.tif'
    )
    if not path.exists():
        pytest.skip(
            "overview_external_ovr_uint16 fixture not generated"
        )
    with rasterio.open(path) as src:
        base = src.read(1)
        base_transform = src.transform
        factors = src.overviews(1)
    assert factors, (
        'external .ovr fixture must report overview factors; got '
        f'{factors!r}'
    )

    def factory(level: int) -> xr.DataArray:
        # Read each overview level directly from rasterio so this
        # test exercises only the oracle's overview-iteration logic.
        # It deliberately bypasses ``open_geotiff`` (which does not
        # yet handle sidecar .ovr files, see _OVERVIEW_READER_GAPS in
        # the per-backend modules).
        with rasterio.open(path, OVERVIEW_LEVEL=level - 1) as src:
            arr = src.read(1)
            t = src.transform
        return _build_candidate(arr, transform=t)

    base_candidate = _build_candidate(base, transform=base_transform)
    compare_to_oracle(path, base_candidate, candidate_factory=factory)


# ---------------------------------------------------------------------------
# Phase 4 fixtures: planar-separate + sparse tiles (#1930)
#
# Bytes-on-disk pins for the two fixtures added by this PR. The smoke
# tests open each one with rasterio (planar) or tifffile (sparse) and
# assert the on-disk property the fixture is meant to expose. Phase 3
# backend parametrisations run the same files through compare_to_oracle.
# ---------------------------------------------------------------------------

_PHASE4_FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'


def test_planar_separate_fixture_reports_planarconfig_2() -> None:
    """``planar_separate_uint8_rgb`` is written with PLANARCONFIG=2 (BSQ).

    rasterio surfaces the layout as ``interleave='band'`` in the profile;
    tifffile reports the raw ``PlanarConfiguration`` tag as 2. Either way
    the bytes-on-disk axis under test is the planar config.
    """
    import tifffile

    path = _PHASE4_FIXTURE_DIR / 'planar_separate_uint8_rgb.tif'
    assert path.exists(), f'missing fixture {path}'
    with rasterio.open(path) as src:
        assert src.profile['interleave'] == 'band'
        assert src.count == 3
        assert src.dtypes[0] == 'uint8'
        assert src.width == 32 and src.height == 32
    with tifffile.TiffFile(path) as t:
        # tifffile maps PLANARCONFIG_SEPARATE to enum value 2.
        assert int(t.pages[0].planarconfig) == 2


def test_sparse_tiled_fixture_has_zero_tilebytecounts() -> None:
    """``sparse_tiled_uint16`` has at least one zero entry in TileByteCounts.

    SPARSE_OK=TRUE plus a uniform-zero pixel pattern lets GDAL elide
    every tile. The on-disk TileByteCounts array reads zero for elided
    tiles; on read the decoder reconstructs the implicit zeros.
    """
    import tifffile

    path = _PHASE4_FIXTURE_DIR / 'sparse_tiled_uint16.tif'
    assert path.exists(), f'missing fixture {path}'
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        byte_counts = page.tags['TileByteCounts'].value
    # 64x64 with 16-px tiles is a 4x4 = 16-tile grid; uniform 0 + deflate
    # + SPARSE_OK=TRUE elides every tile.
    assert len(byte_counts) == 16
    assert any(bc == 0 for bc in byte_counts), (
        f'sparse fixture must have at least one elided tile, '
        f'got TileByteCounts={byte_counts}'
    )


def test_sparse_tiled_fixture_reads_back_as_zeros() -> None:
    """The sparse fixture decodes to an all-zero raster of the right shape.

    The reader has to materialise zeros for every elided tile; this pins
    the round-trip so a future regression in the decode path surfaces
    here rather than as a silent corruption.
    """
    path = _PHASE4_FIXTURE_DIR / 'sparse_tiled_uint16.tif'
    with rasterio.open(path) as src:
        assert src.width == 64 and src.height == 64
        assert src.dtypes[0] == 'uint16'
        data = src.read(1)
    assert data.shape == (64, 64)
    assert data.dtype == np.uint16
    assert (data == 0).all()


def test_planar_separate_fixture_size_budget() -> None:
    path = _PHASE4_FIXTURE_DIR / 'planar_separate_uint8_rgb.tif'
    assert path.stat().st_size < 12 * 1024, (
        f'{path.name} exceeds the 12 KB per-fixture budget '
        f'({path.stat().st_size} bytes)'
    )


def test_sparse_tiled_fixture_size_budget() -> None:
    path = _PHASE4_FIXTURE_DIR / 'sparse_tiled_uint16.tif'
    assert path.stat().st_size < 12 * 1024, (
        f'{path.name} exceeds the 12 KB per-fixture budget '
        f'({path.stat().st_size} bytes)'
    )
