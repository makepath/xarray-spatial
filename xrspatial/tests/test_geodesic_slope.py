"""Tests for geodesic slope computation."""
import re

import numpy as np
import pytest
import xarray as xr

from xrspatial import slope
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
    """Build a DataArray with lat/lon 1-D coords in geographic (degree) space."""
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


def _make_curvilinear_raster(elev, lat_start, lat_end, lon_start, lon_end,
                             backend='numpy', chunks=(3, 3)):
    """Build a curvilinear DataArray: dims ('y', 'x') with numeric y/x index
    coords AND real 2-D lat/lon coords over the same geographic grid.

    This is the layout that exposed the coordinate-resolution bug: the numeric
    y/x pixel-index coords must not be used as lat/lon when real lat/lon coords
    are present.
    """
    H, W = elev.shape
    lat1d = np.linspace(lat_start, lat_end, H)
    lon1d = np.linspace(lon_start, lon_end, W)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    raster = xr.DataArray(
        elev.astype(np.float64),
        dims=['y', 'x'],
        coords={
            'y': np.arange(H, dtype=np.float64),
            'x': np.arange(W, dtype=np.float64),
            'lat': (('y', 'x'), lat2d),
            'lon': (('y', 'x'), lon2d),
        },
    )

    if 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)

    return raster


def _flat_surface(H=6, W=8, elev=500.0):
    """Constant-elevation surface — slope should be 0 everywhere interior."""
    return np.full((H, W), elev, dtype=np.float64)


def _east_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                         lon_start=10.0, lon_end=11.0):
    """Surface that rises linearly to the east.

    grade is elevation increase per degree of longitude.
    """
    lon = np.linspace(lon_start, lon_end, W)
    elev = base_elev + grade * (lon - lon_start)
    return np.broadcast_to(elev[np.newaxis, :], (H, W)).copy()


def _north_tilted_surface(H=6, W=8, base_elev=500.0, grade=100.0,
                          lat_start=40.0, lat_end=41.0):
    """Surface that rises linearly to the north."""
    lat = np.linspace(lat_start, lat_end, H)
    elev = base_elev + grade * (lat - lat_start)
    return np.broadcast_to(elev[:, np.newaxis], (H, W)).copy()


# ---------------------------------------------------------------------------
# Tests — analytical cases
# ---------------------------------------------------------------------------

class TestGeodesicSlopeFlat:
    """Flat surface at various latitudes → slope ≈ 0."""

    @pytest.mark.parametrize("lat_center", [0.0, 30.0, 60.0, -45.0])
    def test_flat_slope_is_zero(self, lat_center):
        elev = _flat_surface()
        raster = _make_geo_raster(
            elev, lat_center - 0.5, lat_center + 0.5, 10.0, 11.0
        )
        result = slope(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        assert np.all(np.isfinite(interior))
        # Small residual (~0.04°) is expected from Earth's curvature
        # over the grid cell spacing; this is negligible for real-world use.
        np.testing.assert_allclose(interior, 0.0, atol=0.1)


class TestGeodesicSlopeTilted:
    """Known tilted surfaces → non-zero slope."""

    def test_east_tilted_has_positive_slope(self):
        elev = _east_tilted_surface()
        raster = _make_geo_raster(elev, 45.0, 46.0, 10.0, 11.0)
        result = slope(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        assert np.all(np.isfinite(interior))
        assert np.all(interior > 0)

    def test_north_tilted_has_positive_slope(self):
        elev = _north_tilted_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = slope(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        assert np.all(np.isfinite(interior))
        assert np.all(interior > 0)


class TestGeodesicSlopeCurvilinear:
    """Curvilinear layout: dims ('y', 'x') with numeric y/x index coords plus
    real 2-D lat/lon coords. The geodesic path must use the lat/lon coords, not
    the pixel indices, so the result must match the equivalent 1-D lat/lon grid.

    Pixel indices (0..N) fall inside the accepted geographic ranges, so the
    range validation does not catch the mistake — only the slope value does.
    """

    def test_curvilinear_matches_1d_latlon(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_curv = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0)
        r_ref = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        s_curv = slope(r_curv, method='geodesic')
        s_ref = slope(r_ref, method='geodesic')
        np.testing.assert_allclose(
            s_curv.values, s_ref.values, rtol=1e-5, equal_nan=True
        )

    def test_curvilinear_ignores_pixel_index_coords(self):
        """Slope must reflect the real geographic grid, not the 0..N indices.

        Using the pixel indices as lat/lon collapses the east tilt to a tiny
        value (~0.007 vs ~0.067), so a correct interior slope is the signal.
        """
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_curv = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0)
        interior = slope(r_curv, method='geodesic').values[1:-1, 1:-1]
        assert np.all(np.isfinite(interior))
        assert np.all(interior > 0.05)


@dask_array_available
class TestGeodesicSlopeCurvilinearDask:

    def test_curvilinear_numpy_equals_dask(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='numpy')
        r_da = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='dask+numpy', chunks=(4, 5))
        s_np = slope(r_np, method='geodesic')
        s_da = slope(r_da, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_da.values, rtol=1e-5, equal_nan=True
        )


@cuda_and_cupy_available
class TestGeodesicSlopeCurvilinearCupy:

    def test_curvilinear_numpy_equals_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='numpy')
        r_cu = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='cupy')
        s_np = slope(r_np, method='geodesic')
        s_cu = slope(r_cu, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_cu.data.get(), rtol=1e-5, equal_nan=True
        )


@dask_array_available
@cuda_and_cupy_available
class TestGeodesicSlopeCurvilinearDaskCupy:

    def test_curvilinear_numpy_equals_dask_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='numpy')
        r_dc = _make_curvilinear_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                        backend='dask+cupy', chunks=(4, 5))
        s_np = slope(r_np, method='geodesic')
        s_dc = slope(r_dc, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_dc.data.compute().get(), rtol=1e-5, equal_nan=True
        )


class TestGeodesicSlopeLatitudeInvariance:
    """Same physical slope at equator vs 60N should give similar geodesic slope."""

    def test_latitude_invariance(self):
        grade = 50.0  # m per degree
        elev_eq = _east_tilted_surface(grade=grade, lon_start=10.0, lon_end=11.0)
        elev_60 = _east_tilted_surface(grade=grade, lon_start=10.0, lon_end=11.0)

        r_eq = _make_geo_raster(elev_eq, -0.5, 0.5, 10.0, 11.0)
        r_60 = _make_geo_raster(elev_60, 59.5, 60.5, 10.0, 11.0)

        s_eq = slope(r_eq, method='geodesic').values[2, 4]
        s_60 = slope(r_60, method='geodesic').values[2, 4]

        # The geodesic slope at 60N should be steeper because 1 degree of
        # longitude is shorter at high latitude. The key point is both are
        # finite and positive — the exact ratio depends on cos(lat).
        assert np.isfinite(s_eq) and s_eq > 0
        assert np.isfinite(s_60) and s_60 > 0
        # At 60N, 1 deg lon ≈ half the distance → slope should be roughly
        # double.  Allow wide tolerance.
        ratio = s_60 / s_eq
        assert 1.5 < ratio < 2.5


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestGeodesicSlopeEdgeCases:

    def test_nan_handling(self):
        """NaN in neighbourhood → NaN output."""
        elev = _flat_surface(H=5, W=5)
        elev[2, 2] = np.nan
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = slope(raster, method='geodesic')
        # The cells adjacent to the NaN should also be NaN
        assert np.isnan(result.values[2, 2])
        # At least the NaN's immediate neighbours should be NaN
        assert np.isnan(result.values[1, 1])
        assert np.isnan(result.values[1, 2])

    def test_edges_are_nan(self):
        """Boundary cells should be NaN."""
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        result = slope(raster, method='geodesic')
        assert np.all(np.isnan(result.values[0, :]))
        assert np.all(np.isnan(result.values[-1, :]))
        assert np.all(np.isnan(result.values[:, 0]))
        assert np.all(np.isnan(result.values[:, -1]))

    def test_near_pole(self):
        """Near-polar latitude should still produce finite results."""
        elev = _north_tilted_surface(H=6, W=6, grade=50.0,
                                     lat_start=88.0, lat_end=89.0)
        raster = _make_geo_raster(elev, 88.0, 89.0, 10.0, 11.0)
        result = slope(raster, method='geodesic')
        interior = result.values[1:-1, 1:-1]
        assert np.all(np.isfinite(interior))
        assert np.all(interior > 0)


# ---------------------------------------------------------------------------
# Tests — z_unit
# ---------------------------------------------------------------------------

class TestGeodesicSlopeZUnit:

    def test_foot_vs_meter(self):
        """Elevation in feet should give consistent slope with proper z_unit."""
        elev_m = _east_tilted_surface(grade=100.0)
        elev_ft = elev_m / 0.3048  # convert to feet

        r_m = _make_geo_raster(elev_m, 40.0, 41.0, 10.0, 11.0)
        r_ft = _make_geo_raster(elev_ft, 40.0, 41.0, 10.0, 11.0)

        s_m = slope(r_m, method='geodesic', z_unit='meter')
        s_ft = slope(r_ft, method='geodesic', z_unit='foot')

        np.testing.assert_allclose(
            s_m.values[1:-1, 1:-1],
            s_ft.values[1:-1, 1:-1],
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

class TestGeodesicSlopeValidation:

    def test_invalid_method_raises(self):
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(ValueError, match="method"):
            slope(raster, method='invalid')

    def test_invalid_z_unit_raises(self):
        elev = _flat_surface()
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(ValueError, match="z_unit") as excinfo:
            slope(raster, method='geodesic', z_unit='cubit')

        # The message must list the accepted unit-name strings (the keys a
        # user is allowed to pass), not the numeric conversion factors.
        msg = str(excinfo.value)
        assert "'meter'" in msg
        assert "'foot'" in msg
        # No bare numeric conversion factor should leak into the message.
        assert not re.search(r"\d+\.\d+", msg)

    def test_missing_coords_raises(self):
        data = np.ones((5, 5))
        raster = xr.DataArray(data, dims=['dim_0', 'dim_1'])
        with pytest.raises(ValueError, match="coordinates"):
            slope(raster, method='geodesic')

    def test_projected_coords_raises(self):
        """Coords outside geographic range should raise."""
        data = np.ones((5, 5))
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={
                'y': np.linspace(4000000, 4100000, 5),
                'x': np.linspace(500000, 600000, 5),
            }
        )
        with pytest.raises(ValueError):
            slope(raster, method='geodesic')


class TestGeodesicSlopeMemoryGuard:
    """The geodesic path allocates a ``(3, H, W)`` float64 stacked array
    plus padded copies and a float32 output. ``_check_geodesic_memory``
    must raise ``MemoryError`` before that allocation if the raster is
    too large for available RAM."""

    def test_oversized_raster_raises_memory_error(self, monkeypatch):
        # Pretend only 1 MB is available; even a tiny raster trips the guard.
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 1024 * 1024
        )
        elev = _flat_surface(H=200, W=200)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(MemoryError, match="slope"):
            slope(raster, method='geodesic')

    def test_normal_size_raster_passes(self, monkeypatch):
        # 16 GB of headroom — plenty for a small raster.
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes',
            lambda: 16 * 1024 ** 3,
        )
        elev = _flat_surface(H=8, W=8)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        # Should not raise.
        result = slope(raster, method='geodesic')
        assert result.shape == (8, 8)

    def test_planar_method_skips_guard(self, monkeypatch):
        """The guard is geodesic-only — planar should still work even
        when ``_available_memory_bytes`` reports zero."""
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 0
        )
        elev = _flat_surface(H=8, W=8)
        raster = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        # planar path doesn't touch the geodesic guard.
        result = slope(raster, method='planar')
        assert result.shape == (8, 8)


@dask_array_available
class TestGeodesicSlopeMemoryGuardDask:
    """The dask geodesic backend streams the raster chunk by chunk, so the
    memory guard must size against the largest chunk, not the full raster.
    A raster that would be rejected eagerly should be allowed once it is
    chunked small enough to fit."""

    def test_chunked_raster_allowed_when_eager_would_reject(self, monkeypatch):
        # 1 MB available. The 200x200 raster needs ~2.2 MB eagerly (56 B/cell)
        # and trips the eager guard, but 20x20 chunks need only ~25 KB each.
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 1024 * 1024
        )
        elev = _flat_surface(H=200, W=200)
        # eager numpy of the same size is rejected (sanity check the budget).
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0)
        with pytest.raises(MemoryError, match="slope"):
            slope(r_np, method='geodesic')
        # same raster, chunked small — guard must let it through.
        r_da = _make_geo_raster(
            elev, 40.0, 41.0, 10.0, 11.0,
            backend='dask+numpy', chunks=(20, 20),
        )
        result = slope(r_da, method='geodesic')
        assert result.compute().shape == (200, 200)

    def test_single_huge_chunk_still_rejected(self, monkeypatch):
        # A dask array whose only chunk spans the whole raster has no memory
        # advantage over eager, so the guard must still reject it.
        monkeypatch.setattr(
            'xrspatial.geodesic._available_memory_bytes', lambda: 1024 * 1024
        )
        elev = _flat_surface(H=200, W=200)
        r_da = _make_geo_raster(
            elev, 40.0, 41.0, 10.0, 11.0,
            backend='dask+numpy', chunks=(200, 200),
        )
        with pytest.raises(MemoryError, match="slope"):
            slope(r_da, method='geodesic')


# ---------------------------------------------------------------------------
# Tests — cross-backend consistency
# ---------------------------------------------------------------------------

@dask_array_available
class TestGeodesicSlopeDask:

    def test_numpy_equals_dask(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_da = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                backend='dask+numpy', chunks=(4, 5))
        s_np = slope(r_np, method='geodesic')
        s_da = slope(r_da, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_da.values, rtol=1e-5, equal_nan=True
        )


@cuda_and_cupy_available
class TestGeodesicSlopeCupy:

    def test_numpy_equals_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_cu = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='cupy')
        s_np = slope(r_np, method='geodesic')
        s_cu = slope(r_cu, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_cu.data.get(), rtol=1e-5, equal_nan=True
        )


@dask_array_available
@cuda_and_cupy_available
class TestGeodesicSlopeDaskCupy:

    def test_numpy_equals_dask_cupy(self):
        elev = _east_tilted_surface(H=8, W=10, grade=100.0)
        r_np = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0, backend='numpy')
        r_dc = _make_geo_raster(elev, 40.0, 41.0, 10.0, 11.0,
                                backend='dask+cupy', chunks=(4, 5))
        s_np = slope(r_np, method='geodesic')
        s_dc = slope(r_dc, method='geodesic')
        np.testing.assert_allclose(
            s_np.values, s_dc.data.compute().get(), rtol=1e-5, equal_nan=True
        )

    def test_latlon_not_materialized_on_gpu_at_graph_build(self):
        """The dask+cupy geodesic path must keep lat/lon chunked.

        Building the graph (no compute) for a large raster must not densify
        the full (H, W) lat/lon grids onto the GPU. Converting the broadcast
        views with ``cupy.asarray`` up front would allocate ~2*H*W*8 bytes of
        GPU memory at graph-construction time and OOM on large rasters.
        """
        import cupy

        H = W = 2048
        elev = cupy.zeros((H, W), dtype=cupy.float64)
        lat = np.linspace(40.0, 41.0, H)
        lon = np.linspace(10.0, 11.0, W)
        raster = xr.DataArray(
            da.from_array(elev, chunks=(256, 256)),
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
        )

        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
        before = pool.used_bytes()
        out = slope(raster, method='geodesic')   # graph construction only
        out.data.__dask_graph__()
        delta = pool.used_bytes() - before

        # A single full lat or lon grid is H*W*8 bytes. If either were
        # densified eagerly the delta would be at least that large.
        one_full_grid = H * W * 8
        assert delta < one_full_grid, (
            f"graph construction allocated {delta} GPU bytes; expected well "
            f"under one full lat/lon grid ({one_full_grid} bytes)"
        )
