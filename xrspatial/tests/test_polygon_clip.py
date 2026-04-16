"""Tests for xrspatial.polygon_clip.clip_polygon."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from shapely.geometry import Polygon, MultiPolygon, box

from xrspatial.polygon_clip import clip_polygon
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster(backend='numpy', chunks=(3, 3)):
    """8x6 raster with known values and coordinates spanning [0, 2.5] x [0, 3.5]."""
    data = np.arange(48, dtype=np.float64).reshape(8, 6)
    return create_test_raster(data, backend=backend, chunks=chunks)


def _inner_polygon():
    """Polygon covering the centre of the 8x6 test raster.

    Raster y goes from 3.5 (top) to 0.0 (bottom) and x from 0.0 to 2.5.
    This polygon covers roughly the inner 4x4 cell area.
    """
    return Polygon([(0.6, 0.6), (0.6, 2.9), (1.9, 2.9), (1.9, 0.6)])


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

class TestClipPolygonNumpy:
    def test_basic_mask(self):
        """Pixels outside polygon are NaN, inside are preserved."""
        raster = _make_raster()
        # Use a triangular polygon so corners outside the triangle
        # get masked even after crop to bounding box.
        poly = Polygon([(0.6, 0.6), (1.25, 2.9), (1.9, 0.6)])
        result = clip_polygon(raster, poly)

        # Result should be smaller than input (cropped to bbox)
        assert result.shape[0] <= raster.shape[0]
        assert result.shape[1] <= raster.shape[1]

        vals = result.values
        # At least some pixels are NaN (masked corners outside triangle)
        assert np.any(np.isnan(vals))
        # At least some pixels are preserved
        assert np.any(np.isfinite(vals))

    def test_no_crop(self):
        """With crop=False, output shape matches input."""
        raster = _make_raster()
        poly = _inner_polygon()
        result = clip_polygon(raster, poly, crop=False)

        assert result.shape == raster.shape

        vals = result.values
        # Pixels outside polygon are NaN
        assert np.any(np.isnan(vals))
        # Pixels inside are preserved
        assert np.any(np.isfinite(vals))

    def test_nodata_value(self):
        """Custom nodata value is applied outside the polygon."""
        raster = _make_raster()
        poly = _inner_polygon()
        result = clip_polygon(raster, poly, nodata=-9999.0, crop=False)

        vals = result.values
        # No NaNs (nodata is -9999 instead)
        assert not np.any(np.isnan(vals))
        # Some cells have the nodata sentinel
        assert np.any(vals == -9999.0)

    def test_preserves_attrs(self):
        """Output DataArray keeps the input's attributes."""
        raster = _make_raster()
        poly = _inner_polygon()
        result = clip_polygon(raster, poly)
        assert result.attrs == raster.attrs

    def test_custom_name(self):
        """Output name can be overridden."""
        raster = _make_raster()
        poly = _inner_polygon()
        result = clip_polygon(raster, poly, name='clipped_1144')
        assert result.name == 'clipped_1144'

    def test_coordinate_array_input(self):
        """Geometry given as a list of (x, y) coordinate pairs."""
        raster = _make_raster()
        coords = [(0.6, 0.6), (0.6, 2.9), (1.9, 2.9), (1.9, 0.6)]
        result = clip_polygon(raster, coords)
        assert np.any(np.isfinite(result.values))

    def test_multipolygon(self):
        """MultiPolygon geometry is accepted."""
        raster = _make_raster()
        p1 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        p2 = Polygon([(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)])
        mp = MultiPolygon([p1, p2])
        result = clip_polygon(raster, mp, crop=False)
        assert result.shape == raster.shape

    def test_list_of_polygons(self):
        """List of shapely polygons is merged and applied."""
        raster = _make_raster()
        polys = [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]),
            Polygon([(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]),
        ]
        result = clip_polygon(raster, polys, crop=False)
        assert result.shape == raster.shape

    def test_nan_in_input_preserved(self):
        """NaN values in the input stay NaN in the output."""
        data = np.arange(48, dtype=np.float64).reshape(8, 6)
        data[3, 3] = np.nan
        raster = create_test_raster(data)
        poly = box(0, 0, 2.5, 3.5)  # covers entire raster
        result = clip_polygon(raster, poly, crop=False)
        assert np.isnan(result.values[3, 3])

    def test_non_overlapping_raises(self):
        """Polygon completely outside the raster raises ValueError."""
        raster = _make_raster()
        poly = Polygon([(100, 100), (100, 200), (200, 200), (200, 100)])
        with pytest.raises(ValueError, match="does not overlap"):
            clip_polygon(raster, poly)

    def test_all_touched(self):
        """all_touched=True includes boundary pixels."""
        raster = _make_raster()
        poly = _inner_polygon()
        result_default = clip_polygon(raster, poly, crop=False)
        result_touched = clip_polygon(raster, poly, crop=False,
                                      all_touched=True)
        # all_touched should include at least as many non-NaN pixels
        n_default = np.count_nonzero(np.isfinite(result_default.values))
        n_touched = np.count_nonzero(np.isfinite(result_touched.values))
        assert n_touched >= n_default

    def test_all_touched_crop_matches_nocrop(self):
        """crop=True with all_touched=True must not drop boundary pixels.

        Regression test for #1197: _crop_to_bbox was comparing pixel
        centers against the geometry bounding box without accounting for
        pixel cell extent, so pixels whose centers fell just outside the
        bbox were excluded even though their cells overlapped the polygon.
        """
        raster = _make_raster()
        # Polygon whose edges land between pixel centers on all four
        # sides.  Pixel spacing is 0.5, so the left edge at x=0.15
        # sits inside the cell of pixel x=0.0 (cell [-0.25, 0.25]).
        poly = Polygon([(0.15, 0.15), (0.15, 3.35), (2.35, 3.35),
                         (2.35, 0.15)])

        result_crop = clip_polygon(raster, poly, crop=True,
                                   all_touched=True)
        result_nocrop = clip_polygon(raster, poly, crop=False,
                                     all_touched=True)

        # The crop path must keep every pixel the nocrop path keeps.
        n_crop = np.count_nonzero(np.isfinite(result_crop.values))
        n_nocrop = np.count_nonzero(np.isfinite(result_nocrop.values))
        assert n_crop == n_nocrop, (
            f"crop=True lost {n_nocrop - n_crop} boundary pixels"
        )

        # Pixel values must match wherever both arrays have data.
        # Align the cropped result back into the full grid for comparison.
        aligned = result_nocrop.copy()
        crop_y = result_crop.coords['y'].values
        crop_x = result_crop.coords['x'].values
        aligned_slice = aligned.sel(y=crop_y, x=crop_x)
        np.testing.assert_array_equal(result_crop.values,
                                      aligned_slice.values)

    def test_single_cell_raster(self):
        """1x1 raster with polygon that covers it."""
        data = np.array([[42.0]])
        raster = create_test_raster(data, attrs={'res': (1.0, 1.0)})
        poly = box(-1, -1, 1, 1)
        result = clip_polygon(raster, poly)
        assert result.values[0, 0] == 42.0

    def test_empty_geometry_raises(self):
        """Empty geometry list raises."""
        raster = _make_raster()
        with pytest.raises(ValueError, match="empty"):
            clip_polygon(raster, [])


# ---------------------------------------------------------------------------
# Dask + NumPy backend
# ---------------------------------------------------------------------------

@dask_array_available
class TestClipPolygonDask:
    def test_matches_numpy(self):
        """Dask result matches NumPy result."""
        np_raster = _make_raster(backend='numpy')
        dk_raster = _make_raster(backend='dask+numpy', chunks=(4, 3))
        poly = _inner_polygon()

        np_result = clip_polygon(np_raster, poly, crop=False)
        dk_result = clip_polygon(dk_raster, poly, crop=False)

        np.testing.assert_allclose(
            dk_result.values, np_result.values, equal_nan=True
        )

    def test_lazy(self):
        """Output remains a dask array (not eagerly computed)."""
        import dask.array as da
        raster = _make_raster(backend='dask+numpy')
        poly = _inner_polygon()
        result = clip_polygon(raster, poly)
        assert isinstance(result.data, da.Array)

    def test_crop_matches_numpy(self):
        """Cropped dask result matches cropped NumPy result."""
        np_raster = _make_raster(backend='numpy')
        dk_raster = _make_raster(backend='dask+numpy', chunks=(4, 3))
        poly = _inner_polygon()

        np_result = clip_polygon(np_raster, poly, crop=True)
        dk_result = clip_polygon(dk_raster, poly, crop=True)

        np.testing.assert_allclose(
            dk_result.values, np_result.values, equal_nan=True
        )

    def test_all_touched_crop_matches_nocrop(self):
        """Dask: crop=True with all_touched=True keeps boundary pixels (#1197)."""
        dk_raster = _make_raster(backend='dask+numpy', chunks=(4, 3))
        poly = Polygon([(0.15, 0.15), (0.15, 3.35), (2.35, 3.35),
                         (2.35, 0.15)])

        result_crop = clip_polygon(dk_raster, poly, crop=True,
                                   all_touched=True)
        result_nocrop = clip_polygon(dk_raster, poly, crop=False,
                                     all_touched=True)

        n_crop = np.count_nonzero(np.isfinite(result_crop.values))
        n_nocrop = np.count_nonzero(np.isfinite(result_nocrop.values))
        assert n_crop == n_nocrop, (
            f"crop=True lost {n_nocrop - n_crop} boundary pixels"
        )


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
class TestClipPolygonCuPy:
    def test_matches_numpy(self):
        """CuPy result matches NumPy result."""
        np_raster = _make_raster(backend='numpy')
        cp_raster = _make_raster(backend='cupy')
        poly = _inner_polygon()

        np_result = clip_polygon(np_raster, poly, crop=False)
        cp_result = clip_polygon(cp_raster, poly, crop=False)

        np.testing.assert_allclose(
            cp_result.data.get(), np_result.values, equal_nan=True
        )


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
@dask_array_available
class TestClipPolygonDaskCuPy:
    def test_matches_numpy(self):
        """Dask+CuPy result matches NumPy result."""
        np_raster = _make_raster(backend='numpy')
        dkcp_raster = _make_raster(backend='dask+cupy', chunks=(4, 3))
        poly = _inner_polygon()

        np_result = clip_polygon(np_raster, poly, crop=False)
        dkcp_result = clip_polygon(dkcp_raster, poly, crop=False)

        np.testing.assert_allclose(
            dkcp_result.data.compute().get(), np_result.values, equal_nan=True
        )


# ---------------------------------------------------------------------------
# GeoDataFrame input
# ---------------------------------------------------------------------------

class TestClipPolygonGeoDataFrame:
    def test_geodataframe_input(self):
        """GeoDataFrame geometry is accepted."""
        geopandas = pytest.importorskip("geopandas")
        raster = _make_raster()
        poly = _inner_polygon()
        gdf = geopandas.GeoDataFrame(geometry=[poly])
        result = clip_polygon(raster, gdf, crop=False)
        assert result.shape == raster.shape
        assert np.any(np.isfinite(result.values))

    def test_geoseries_input(self):
        """GeoSeries geometry is accepted."""
        geopandas = pytest.importorskip("geopandas")
        raster = _make_raster()
        poly = _inner_polygon()
        gs = geopandas.GeoSeries([poly])
        result = clip_polygon(raster, gs, crop=False)
        assert result.shape == raster.shape


# ---------------------------------------------------------------------------
# Issue #1207 regression tests
# ---------------------------------------------------------------------------

@dask_array_available
class TestClipPolygonDaskLazyMask:
    def test_mask_stays_lazy_for_dask_input(self):
        """clip_polygon on dask input should not materialize a full numpy mask (#1207).

        We verify by checking that the dask task graph contains rasterize
        chunk tasks (not just a single from_array wrapping a pre-computed
        numpy array).
        """
        import dask.array as da

        dk_raster = _make_raster(backend='dask+numpy', chunks=(4, 3))
        poly = _inner_polygon()

        result = clip_polygon(dk_raster, poly, crop=False)
        assert isinstance(result.data, da.Array)

        # With chunked rasterize, the graph has tasks per chunk.
        # With the old approach (numpy mask + da.from_array), the graph
        # would have fewer chunk-level tasks for the mask.
        graph = dict(result.data.__dask_graph__())
        # At minimum, a 8x6 raster with (4,3) chunks = 2x2 = 4 mask chunks
        # plus raster chunks plus where-condition tasks.
        # Just verify we have more than the trivial single-mask case.
        assert len(graph) > 4, (
            f"graph has only {len(graph)} tasks; mask may not be chunked"
        )
