"""Tests for geodesic aspect computation."""
import numpy as np
import pytest
import xarray as xr

from xrspatial import aspect
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)

try:
    import dask.array as da
except ImportError:
    da = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geo_raster(elev, lat_start, lat_end, lon_start, lon_end,
                     backend='numpy', chunks=(3, 3)):
    H, W = elev.shape
    lat = np.linspace(lat_start, lat_end, H)
    lon = np.linspace(lon_start, lon_end, W)
    raster = xr.DataArray(
        elev.astype(np.float64),
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
    )

    if 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)

    return raster


def _flat_surface(H=6, W=8, elev=500.0):
    return np.full((H, W), elev, dtype=np.float64)


def _east_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                         lon_start=10.0, lon_end=11.0):
    lon = np.linspace(lon_start, lon_end, W)
    elev = base_elev + grade * (lon - lon_start)
    return np.broadcast_to(elev[np.newaxis, :], (H, W)).copy()


def _north_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                          lat_start=40.0, lat_end=41.0):
    lat = np.linspace(lat_start, lat_end, H)
    elev = base_elev + grade * (lat - lat_start)
    return np.broadcast_to(elev[:, np.newaxis], (H, W)).copy()


def _south_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                          lat_start=40.0, lat_end=41.0):
    lat = np.linspace(lat_start, lat_end, H)
    elev = base_elev - grade * (lat - lat_start)
    return np.broadcast_to(elev[:, np.newaxis], (H, W)).copy()


def _west_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                         lon_start=10.0, lon_end=11.0):
    lon = np.linspace(lon_start, lon_end, W)
    elev = base_elev - grade * (lon - lon_start)
    return np.broadcast_to(elev[np.newaxis, :], (H, W)).copy()


# ---------------------------------------------------------------------------
# Tests — compass direction
# ---------------------------------------------------------------------------

class TestGeodesicAspectDirection:

    def test_flat_is_minus_one(self):
        """Flat surface → aspect = -1."""
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        np.testing.assert_allclose(interior, -1.0, atol=1e-4)

    def test_east_facing(self):
        """Surface rising to the east → downslope faces east → aspect ≈ 90."""
        elev = _east_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[2, 4]
        # Downslope is west (270) — wait, rising to east means steepest
        # descent is to the west. Let me reconsider.
        # East-tilted: elevation increases eastward → steepest descent is
        # westward → aspect ≈ 270.
        assert np.isfinite(interior)
        assert abs(interior - 270.0) < 5.0

    def test_north_facing(self):
        """Surface rising to the north → steepest descent is south → aspect ≈ 180."""
        elev = _north_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[2, 4]
        assert np.isfinite(interior)
        assert abs(interior - 180.0) < 5.0

    def test_south_facing(self):
        """Surface rising to the south → steepest descent is north → aspect ≈ 0 or 360."""
        elev = _south_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[2, 4]
        assert np.isfinite(interior)
        # Aspect near 0 or 360
        assert interior < 5.0 or interior > 355.0

    def test_west_facing(self):
        """Surface rising to the west → steepest descent is east → aspect ≈ 90."""
        elev = _west_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[2, 4]
        assert np.isfinite(interior)
        assert abs(interior - 90.0) < 5.0


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestGeodesicAspectEdgeCases:

    def test_nan_handling(self):
        elev = _east_tilted_surface(H=5, W=5)
        elev[2, 2] = np.nan
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        assert np.isnan(result.values[2, 2])

    def test_edges_are_nan(self):
        elev = _east_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        assert np.all(np.isnan(result.values[0, :]))
        assert np.all(np.isnan(result.values[-1, :]))
        assert np.all(np.isnan(result.values[:, 0]))
        assert np.all(np.isnan(result.values[:, -1]))

    def test_aspect_range(self):
        """Non-flat interior cells should have aspect in [0, 360) or == -1."""
        rng = np.random.default_rng(42)
        elev = rng.uniform(100, 1000, size=(10, 10))
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        valid = interior[np.isfinite(interior)]
        flat = valid[valid == -1.0]
        directional = valid[valid != -1.0]
        assert np.all(directional >= 0.0)
        assert np.all(directional < 360.0)


# ---------------------------------------------------------------------------
# Tests — z_unit
# ---------------------------------------------------------------------------

class TestGeodesicAspectZUnit:

    def test_foot_vs_meter(self):
        elev_m = _east_tilted_surface(grade=100.0)
        elev_ft = elev_m / 0.3048

        r_m = _make_geo_raster(elev_m, 40.0, 41.0, 10.0, 11.0)
        r_ft = _make_geo_raster(elev_ft, 40.0, 41.0, 10.0, 11.0)

        a_m = aspect(r_m, method='geodesic', z_unit='meter')
        a_ft = aspect(r_ft, method='geodesic', z_unit='foot')

        np.testing.assert_allclose(
            a_m.values[1:-1, 1:-1],
            a_ft.values[1:-1, 1:-1],
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

class TestGeodesicAspectValidation:

    def test_invalid_method_raises(self):
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(ValueError, match="method"):
            aspect(raster, method='invalid')

    def test_invalid_z_unit_raises(self):
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(ValueError, match="z_unit"):
            aspect(raster, method='geodesic', z_unit='cubit')

    def test_missing_coords_raises(self):
        data = np.ones((5, 5))
        raster = xr.DataArray(data, dims=['dim_0', 'dim_1'])
        with pytest.raises(ValueError, match="coordinates"):
            aspect(raster, method='geodesic')


class TestGeodesicAspectMemoryGuard:
    """``_check_geodesic_memory`` must raise ``MemoryError`` before the
    geodesic backends allocate their ``(3, H, W)`` float64 stacked array
    when the raster is too large for available RAM."""

    def test_oversized_raster_raises_memory_error(self, monkeypatch):
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 1024 * 1024
        )
        elev = _flat_surface(H=200, W=200)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(MemoryError, match="aspect"):
            aspect(raster, method='geodesic')

    def test_normal_size_raster_passes(self, monkeypatch):
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes',
            lambda: 16 * 1024 ** 3,
        )
        elev = _flat_surface(H=8, W=8)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='geodesic')
        assert result.shape == (8, 8)

    def test_planar_method_skips_guard(self, monkeypatch):
        """planar path doesn't touch the geodesic guard."""
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 0
        )
        elev = _flat_surface(H=8, W=8)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = aspect(raster, method='planar')
        assert result.shape == (8, 8)


# ---------------------------------------------------------------------------
# Tests — cross-backend consistency
# ---------------------------------------------------------------------------

@dask_array_available
class TestGeodesicAspectDask:

    def test_numpy_equals_dask(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_da = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                backend='dask+numpy', chunks=(4, 5))
        a_np = aspect(r_np, method='geodesic')
        a_da = aspect(r_da, method='geodesic')
        np.testing.assert_allclose(
            a_np.values, a_da.values, rtol=1e-5, equal_nan=True
        )


@cuda_and_cupy_available
class TestGeodesicAspectCupy:

    def test_numpy_equals_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_cu = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='cupy')
        a_np = aspect(r_np, method='geodesic')
        a_cu = aspect(r_cu, method='geodesic')
        np.testing.assert_allclose(
            a_np.values, a_cu.data.get(), rtol=1e-5, equal_nan=True
        )


@dask_array_available
@cuda_and_cupy_available
class TestGeodesicAspectDaskCupy:

    def test_numpy_equals_dask_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_dc = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                backend='dask+cupy', chunks=(4, 5))
        a_np = aspect(r_np, method='geodesic')
        a_dc = aspect(r_dc, method='geodesic')
        np.testing.assert_allclose(
            a_np.values, a_dc.data.compute().get(), rtol=1e-5, equal_nan=True
        )
