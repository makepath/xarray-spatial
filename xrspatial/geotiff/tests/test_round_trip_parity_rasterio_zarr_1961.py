"""Cross-library accuracy test (issue #1961).

Read the same GeoTIFF three ways and assert the results agree:

1. ``rasterio.open(path)`` -> numpy array, transform, CRS, nodata.
2. ``xrspatial.geotiff.open_geotiff(path)`` -> ``xr.DataArray``.
3. Write the xarray-spatial DataArray with ``.to_zarr(...)``, reopen with
   ``xr.open_zarr(...)``.

The point of this file is to pin the GeoTIFF reader against an external
reference (rasterio) and against a round-trip through a different on-disk
format (Zarr). A regression in header parsing, georef extraction, coord
generation, nodata handling, or Zarr metadata propagation can pass every
existing test and only surface when a user files a bug.

Each input file covers a case that has drifted before:

- single-band float32 with a non-NaN nodata sentinel
- multi-band uint16 with per-band nodata
- north-up and south-up rasters (negative vs positive ``pixel_height``)
- 1xN / Nx1 stripe (#1945)
- tiled COG, no overviews
- no-georef raster with integer coords (#1949)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip('rasterio')
zarr = pytest.importorskip('zarr')

from rasterio.transform import Affine, from_origin  # noqa: E402

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff._crs import _resolve_crs_to_wkt  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

ATOL_FLOAT32 = 0.0          # exact equality for float32 round trips
ATOL_COORD = 1e-9            # coord tolerance (cells are O(1) - O(1e6))
RTOL_TRANSFORM = 1e-9


def _as_xrspatial_layout(rasterio_data: np.ndarray) -> np.ndarray:
    """Convert rasterio ``(band, y, x)`` to xrspatial ``(y, x, band)``.

    Single-band rasterio reads already come back as 2D when the caller uses
    ``ds.read(1)``; this helper is only needed for multi-band ``ds.read()``.
    """
    return np.moveaxis(rasterio_data, 0, -1)


def _apply_nodata_to_float(arr: np.ndarray, nodata) -> np.ndarray:
    """Promote ``arr`` to float64 and replace the nodata sentinel with NaN.

    xrspatial silently promotes integer rasters with a nodata sentinel to
    float64+NaN (see docstring on ``open_geotiff``). The rasterio reference
    keeps the native dtype, so we apply the same promotion before comparing.
    """
    if nodata is None:
        return arr.astype(np.float64, copy=False)
    out = arr.astype(np.float64)
    if np.isnan(nodata):
        return out  # nothing to mask, NaNs already are NaNs
    out[arr == nodata] = np.nan
    return out


def _read_via_zarr(da: xr.DataArray, store_path) -> xr.DataArray:
    """Write ``da`` to a zarr store and reopen it as a DataArray.

    ``to_zarr`` requires the array to carry a name; this helper assigns one
    if missing. The reopened result is materialised with ``.load()`` so the
    caller can pull a plain numpy view without dask complicating the asserts.
    """
    if da.name is None:
        da = da.rename('band_data')
    # Zarr v3 emits a UserWarning about consolidated metadata; that is
    # orthogonal to the parity claims we are making here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        da.to_zarr(str(store_path), mode='w', consolidated=False)
        ds = xr.open_zarr(str(store_path), consolidated=False).load()
    return ds[da.name]


def _assert_pixels_equal(reference: np.ndarray, candidate: np.ndarray) -> None:
    """Compare two pixel arrays with dtype-aware semantics.

    Integer dtypes go through ``array_equal``; floats go through
    ``allclose`` with a NaN-aware mask. NaN positions must coincide.
    """
    assert reference.shape == candidate.shape, (
        f'shape mismatch: {reference.shape} vs {candidate.shape}')
    if np.issubdtype(reference.dtype, np.integer):
        assert np.array_equal(reference, candidate), 'integer pixel mismatch'
        return
    nan_ref = np.isnan(reference)
    nan_can = np.isnan(candidate)
    assert np.array_equal(nan_ref, nan_can), 'NaN positions disagree'
    np.testing.assert_allclose(
        reference[~nan_ref], candidate[~nan_can],
        rtol=0.0, atol=ATOL_FLOAT32, equal_nan=False,
    )


def _assert_transforms_match(rasterio_transform, xrspatial_transform):
    """Compare a rasterio ``Affine`` to xrspatial's ``(pw, 0, ox, 0, ph, oy)``.

    rasterio's ``Affine(a, b, c, d, e, f)`` is laid out as
    ``(pixel_width, 0, origin_x, 0, pixel_height, origin_y)`` for an
    axis-aligned raster, which matches xrspatial's tuple verbatim.
    """
    a, b, c, d, e, f = rasterio_transform.a, rasterio_transform.b, \
        rasterio_transform.c, rasterio_transform.d, \
        rasterio_transform.e, rasterio_transform.f
    expected = (a, b, c, d, e, f)
    np.testing.assert_allclose(
        xrspatial_transform, expected,
        rtol=RTOL_TRANSFORM, atol=0.0,
    )


def _assert_crs_match(rasterio_crs, xrspatial_attrs):
    """Compare CRS through ``_resolve_crs_to_wkt`` normalisation.

    Direct WKT string comparison breaks across PROJ versions and across
    EPSG vs WKT round-trips. Normalising both sides through pyproj's
    ``CRS`` (which is what ``_resolve_crs_to_wkt`` delegates to for ints
    and PROJ strings) gives a stable comparison key.
    """
    from pyproj import CRS

    if 'crs' in xrspatial_attrs:
        cand_wkt = _resolve_crs_to_wkt(xrspatial_attrs['crs'])
    elif 'crs_wkt' in xrspatial_attrs:
        cand_wkt = _resolve_crs_to_wkt(xrspatial_attrs['crs_wkt'])
    else:
        cand_wkt = None

    ref_wkt = rasterio_crs.to_wkt() if rasterio_crs is not None else None

    if ref_wkt is None and cand_wkt is None:
        return
    assert ref_wkt is not None and cand_wkt is not None, (
        f'CRS presence mismatch: rasterio={ref_wkt!r}, xrspatial={cand_wkt!r}')
    assert CRS.from_wkt(ref_wkt) == CRS.from_wkt(cand_wkt), (
        f'CRS objects differ:\n  rasterio: {ref_wkt}\n  xrspatial: {cand_wkt}')


def _build_rasterio_coords(transform: Affine, height: int, width: int):
    """Pixel-centre coords for a north-up or south-up axis-aligned raster.

    Mirrors xrspatial's coord generation: row i centre is at
    ``origin_y + (i + 0.5) * pixel_height``.
    """
    pw, _, ox, _, ph, oy = (transform.a, transform.b, transform.c,
                             transform.d, transform.e, transform.f)
    y = oy + (np.arange(height) + 0.5) * ph
    x = ox + (np.arange(width) + 0.5) * pw
    return y, x


# ---------------------------------------------------------------------------
# parity checks shared across cases
# ---------------------------------------------------------------------------

def _parity_check_single_band(
    path,
    *,
    expected_dtype,
    nodata,
    has_georef,
    zarr_store,
):
    """Run all pairwise checks for a single-band raster."""
    with rasterio.open(path) as ds:
        ras_raw = ds.read(1)
        ras_transform = ds.transform
        ras_crs = ds.crs
        ras_nodata = ds.nodata

    xrs = open_geotiff(path)
    xrs_np = np.asarray(xrs)

    # Pixel parity. For integer rasters with nodata, xrspatial promotes to
    # float64+NaN. Float rasters keep their dtype but the reader still swaps
    # the non-NaN sentinel for NaN, so the rasterio reference needs masking
    # in both cases.
    if np.issubdtype(expected_dtype, np.integer) and nodata is not None:
        ref = _apply_nodata_to_float(ras_raw, ras_nodata)
        assert xrs_np.dtype == np.float64
    elif (nodata is not None
          and not (isinstance(nodata, float) and np.isnan(nodata))):
        ref = ras_raw.astype(expected_dtype, copy=True)
        ref[ras_raw == ras_nodata] = np.nan
        assert xrs_np.dtype == expected_dtype
    else:
        ref = ras_raw.astype(expected_dtype, copy=False)
        assert xrs_np.dtype == expected_dtype

    _assert_pixels_equal(ref, xrs_np)

    # Shape and nodata sentinel match.
    assert xrs.shape == ras_raw.shape
    if nodata is not None:
        assert xrs.attrs.get('nodata') == ras_nodata or (
            np.isnan(xrs.attrs.get('nodata', 0.0))
            and np.isnan(ras_nodata)
        )

    # Coords.
    if has_georef:
        ref_y, ref_x = _build_rasterio_coords(
            ras_transform, ras_raw.shape[0], ras_raw.shape[1])
        np.testing.assert_allclose(xrs.y.values, ref_y,
                                    rtol=0.0, atol=ATOL_COORD)
        np.testing.assert_allclose(xrs.x.values, ref_x,
                                    rtol=0.0, atol=ATOL_COORD)
        _assert_transforms_match(ras_transform, xrs.attrs['transform'])
        _assert_crs_match(ras_crs, xrs.attrs)
    else:
        # The reader emits integer indices when there is no transform.
        assert np.issubdtype(xrs.y.dtype, np.integer)
        assert np.issubdtype(xrs.x.dtype, np.integer)
        np.testing.assert_array_equal(xrs.y.values, np.arange(ras_raw.shape[0]))
        np.testing.assert_array_equal(xrs.x.values, np.arange(ras_raw.shape[1]))
        assert 'transform' not in xrs.attrs
        # ``crs`` and ``crs_wkt`` may or may not be present depending on
        # GeoKey presence; for a truly bare TIFF rasterio sees nothing.
        if ras_crs is None:
            assert 'crs' not in xrs.attrs and 'crs_wkt' not in xrs.attrs

    # Zarr round-trip parity.
    rt = _read_via_zarr(xrs, zarr_store)
    rt_np = np.asarray(rt)
    _assert_pixels_equal(xrs_np, rt_np)
    assert rt.dtype == xrs.dtype
    np.testing.assert_allclose(rt.y.values, xrs.y.values,
                                rtol=0.0, atol=ATOL_COORD)
    np.testing.assert_allclose(rt.x.values, xrs.x.values,
                                rtol=0.0, atol=ATOL_COORD)
    # Critical scalar attrs survive the zarr trip.
    for key in ('crs', 'crs_wkt', 'nodata', 'transform'):
        if key in xrs.attrs:
            stored = rt.attrs.get(key)
            assert stored is not None, f'{key!r} dropped by zarr round-trip'
            if key == 'transform':
                np.testing.assert_allclose(
                    tuple(stored), tuple(xrs.attrs[key]),
                    rtol=RTOL_TRANSFORM, atol=0.0,
                )
            else:
                assert stored == xrs.attrs[key], (
                    f'{key!r} value drifted: {stored!r} vs {xrs.attrs[key]!r}')


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

class TestSingleBandFloat32NodataSentinel:
    def test_round_trip(self, tmp_path):
        path = tmp_path / 'tmp_1961_float32_nodata.tif'
        store = tmp_path / 'tmp_1961_float32_nodata.zarr'
        data = np.arange(24, dtype=np.float32).reshape(4, 6)
        data[1, 2] = -9999.0  # non-NaN nodata sentinel
        with rasterio.open(
            path, 'w', driver='GTiff', dtype='float32',
            height=4, width=6, count=1,
            transform=from_origin(-120.0, 40.0, 0.001, 0.001),
            crs='EPSG:4326', nodata=-9999.0, tiled=False,
        ) as dst:
            dst.write(data, 1)

        _parity_check_single_band(
            str(path),
            expected_dtype=np.float32,
            nodata=-9999.0,
            has_georef=True,
            zarr_store=store,
        )


class TestMultibandUint16PerBandNodata:
    def test_round_trip(self, tmp_path):
        path = tmp_path / 'tmp_1961_uint16_multiband.tif'
        store = tmp_path / 'tmp_1961_uint16_multiband.zarr'
        band1 = np.arange(12, dtype=np.uint16).reshape(3, 4)
        band1[0, 0] = 0  # nodata cell on band 1
        band2 = np.full((3, 4), 7, dtype=np.uint16)
        band2[2, 3] = 0
        data = np.stack([band1, band2])  # (2, 3, 4) for rasterio

        with rasterio.open(
            path, 'w', driver='GTiff', dtype='uint16',
            height=3, width=4, count=2,
            transform=from_origin(0.0, 30.0, 10.0, 10.0),
            crs='EPSG:32633', nodata=0,
        ) as dst:
            dst.write(data)

        with rasterio.open(path) as ds:
            ras_raw = ds.read()                 # (2, 3, 4)
            ras_transform = ds.transform
            ras_crs = ds.crs
            ras_nodata = ds.nodata

        xrs = open_geotiff(str(path))           # dims (y, x, band)
        xrs_np = np.asarray(xrs)

        # xrspatial lays bands on the trailing axis; transpose for compare.
        ref_layout = _as_xrspatial_layout(ras_raw)
        ref_float = _apply_nodata_to_float(ref_layout, ras_nodata)
        assert xrs.dtype == np.float64
        assert xrs_np.shape == ref_float.shape
        _assert_pixels_equal(ref_float, xrs_np)

        # Coord parity.
        ref_y, ref_x = _build_rasterio_coords(
            ras_transform, ras_raw.shape[1], ras_raw.shape[2])
        np.testing.assert_allclose(xrs.y.values, ref_y, atol=ATOL_COORD)
        np.testing.assert_allclose(xrs.x.values, ref_x, atol=ATOL_COORD)
        _assert_transforms_match(ras_transform, xrs.attrs['transform'])
        _assert_crs_match(ras_crs, xrs.attrs)

        # Zarr round-trip.
        rt = _read_via_zarr(xrs, store)
        _assert_pixels_equal(xrs_np, np.asarray(rt))
        assert rt.dtype == xrs.dtype


class TestNorthUpVsSouthUp:
    """Negative vs positive ``pixel_height`` must reach the reader unchanged."""

    @pytest.mark.parametrize('south_up', [False, True])
    def test_round_trip(self, tmp_path, south_up):
        suffix = 'south_up' if south_up else 'north_up'
        path = tmp_path / f'tmp_1961_{suffix}.tif'
        store = tmp_path / f'tmp_1961_{suffix}.zarr'
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        if south_up:
            # origin at the bottom, pixel_height positive (y increases downward
            # in pixel space, increasing geographic y too).
            transform = Affine(1.0, 0.0, 100.0, 0.0, 1.0, 200.0)
        else:
            transform = from_origin(100.0, 200.0, 1.0, 1.0)
        with rasterio.open(
            path, 'w', driver='GTiff', dtype='float32',
            height=4, width=5, count=1, transform=transform,
            crs='EPSG:32633',
        ) as dst:
            dst.write(data, 1)

        _parity_check_single_band(
            str(path),
            expected_dtype=np.float32,
            nodata=None,
            has_georef=True,
            zarr_store=store,
        )


class TestStripeShapes:
    """1xN and Nx1 single-row / single-column rasters (#1945)."""

    @pytest.mark.parametrize('shape', [(1, 8), (8, 1)])
    def test_round_trip(self, tmp_path, shape):
        height, width = shape
        path = tmp_path / f'tmp_1961_stripe_{height}x{width}.tif'
        store = tmp_path / f'tmp_1961_stripe_{height}x{width}.zarr'
        data = np.arange(height * width, dtype=np.float32).reshape(shape)
        with rasterio.open(
            path, 'w', driver='GTiff', dtype='float32',
            height=height, width=width, count=1,
            transform=from_origin(500000.0, 4000000.0, 1.0, 1.0),
            crs='EPSG:32633',
        ) as dst:
            dst.write(data, 1)

        _parity_check_single_band(
            str(path),
            expected_dtype=np.float32,
            nodata=None,
            has_georef=True,
            zarr_store=store,
        )


class TestTiledCogNoOverviews:
    """Tiled layout, no overviews. ``rasterio`` writes a valid COG-style file."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / 'tmp_1961_tiled_cog.tif'
        store = tmp_path / 'tmp_1961_tiled_cog.zarr'
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        with rasterio.open(
            path, 'w', driver='GTiff', dtype='float32',
            height=64, width=64, count=1,
            tiled=True, blockxsize=32, blockysize=32,
            transform=from_origin(500000.0, 4000000.0, 1.0, 1.0),
            crs='EPSG:32633',
        ) as dst:
            dst.write(data, 1)

        _parity_check_single_band(
            str(path),
            expected_dtype=np.float32,
            nodata=None,
            has_georef=True,
            zarr_store=store,
        )


class TestNoGeorefIntegerCoords:
    """No-georef raster -> reader emits integer indices (#1949)."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / 'tmp_1961_no_georef.tif'
        store = tmp_path / 'tmp_1961_no_georef.zarr'
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        # rasterio emits NotGeoreferencedWarning for the identity transform;
        # that's exactly the bare-TIFF case the reader must handle.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(
                path, 'w', driver='GTiff', dtype='float32',
                height=3, width=4, count=1,
            ) as dst:
                dst.write(data, 1)

        _parity_check_single_band(
            str(path),
            expected_dtype=np.float32,
            nodata=None,
            has_georef=False,
            zarr_store=store,
        )
