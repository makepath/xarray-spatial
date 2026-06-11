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

    def test_nonuniform_chunks_matches_numpy(self):
        """Dask+NumPy with non-uniform chunks matches NumPy (#3186)."""
        np_raster = _make_raster(backend='numpy')
        poly = _inner_polygon()
        np_result = clip_polygon(np_raster, poly, crop=False)

        for chunks in (((3, 5), (6,)), ((3, 2, 3), (2, 1, 3))):
            dk_raster = _make_raster(backend='dask+numpy', chunks=(3, 3))
            dk_raster = dk_raster.copy(data=dk_raster.data.rechunk(chunks))
            dk_result = clip_polygon(dk_raster, poly, crop=False)
            np.testing.assert_allclose(
                dk_result.values, np_result.values, equal_nan=True,
                err_msg=f"mismatch for chunks={chunks}",
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

    def test_nonuniform_chunks_matches_numpy(self):
        """Dask+CuPy with non-uniform chunks matches NumPy (#3186).

        The polygon mask is rasterized with a uniform chunk size taken
        from the raster's first chunk.  When the raster has irregular
        chunks the mask layout differs, and da.map_blocks pairs blocks
        positionally -- so without rechunking the condition to the
        raster's chunks it raised (or stamped the mask onto the wrong
        cells).
        """
        np_raster = _make_raster(backend='numpy')
        poly = _inner_polygon()
        np_result = clip_polygon(np_raster, poly, crop=False)

        for chunks in (((3, 5), (6,)), ((3, 2, 3), (2, 1, 3))):
            dkcp_raster = _make_raster(backend='dask+cupy', chunks=(3, 3))
            dkcp_raster = dkcp_raster.copy(
                data=dkcp_raster.data.rechunk(chunks)
            )
            dkcp_result = clip_polygon(dkcp_raster, poly, crop=False)
            np.testing.assert_allclose(
                dkcp_result.data.compute().get(),
                np_result.values,
                equal_nan=True,
                err_msg=f"mismatch for chunks={chunks}",
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

# ---------------------------------------------------------------------------
# Issue #3197 coverage: NaN / Inf / integer-dtype edge cases (numpy)
# ---------------------------------------------------------------------------

class TestClipPolygonEdgeInputs:
    def test_inf_input_preserved(self):
        """+Inf / -Inf inside the polygon survive the clip (#3197)."""
        data = np.arange(48, dtype=np.float64).reshape(8, 6)
        data[2, 2] = np.inf
        data[3, 3] = -np.inf
        raster = create_test_raster(data)
        result = clip_polygon(raster, box(0, 0, 2.5, 3.5), crop=False)
        assert np.isposinf(result.values[2, 2])
        assert np.isneginf(result.values[3, 3])

    def test_all_nan_input(self):
        """All-NaN input clips to all-NaN without error (#3197)."""
        data = np.full((8, 6), np.nan)
        raster = create_test_raster(data)
        result = clip_polygon(raster, _inner_polygon(), crop=False)
        assert np.all(np.isnan(result.values))

    def test_integer_dtype_sentinel_nodata(self):
        """Integer raster with a sentinel nodata value (#3197).

        Outside-polygon cells take the sentinel; inside cells keep their
        original integer values.
        """
        data = np.arange(48, dtype=np.int32).reshape(8, 6)
        raster = create_test_raster(data)
        result = clip_polygon(raster, _inner_polygon(), nodata=-1, crop=False)
        # Output keeps the input integer dtype (docstring contract).
        assert result.dtype == np.int32
        # Sentinel appears in the masked region.
        assert np.any(result.values == -1)
        # Interior cells keep their original values (no sentinel collision:
        # original data is in [0, 47], sentinel is -1).
        kept = result.values[result.values != -1]
        assert kept.size > 0
        assert np.all(np.isin(kept, data))


# ---------------------------------------------------------------------------
# Issue #3197 coverage: degenerate strip rasters (numpy)
# ---------------------------------------------------------------------------

class TestClipPolygonStripRasters:
    def test_single_column_strip(self):
        """Nx1 single-column raster (#3197)."""
        data = np.arange(8, dtype=np.float64).reshape(8, 1)
        raster = create_test_raster(data)
        # Box spans the full x extent and the middle of the y extent.
        poly = box(-1.0, 0.5, 1.0, 3.0)
        result = clip_polygon(raster, poly, crop=False)
        assert result.shape == (8, 1)
        assert np.any(np.isfinite(result.values))
        assert np.any(np.isnan(result.values))

    def test_single_row_strip(self):
        """1xN single-row raster (#3197)."""
        data = np.arange(6, dtype=np.float64).reshape(1, 6)
        raster = create_test_raster(data)
        poly = box(0.5, -1.0, 2.0, 1.0)
        result = clip_polygon(raster, poly, crop=False)
        assert result.shape == (1, 6)
        assert np.any(np.isfinite(result.values))
        assert np.any(np.isnan(result.values))


# ---------------------------------------------------------------------------
# Issue #3197 coverage: coordinate / metadata preservation (numpy)
# ---------------------------------------------------------------------------

class TestClipPolygonCoords:
    def test_coords_preserved_no_crop(self):
        """crop=False keeps the input coordinates unchanged (#3197)."""
        raster = _make_raster()
        result = clip_polygon(raster, _inner_polygon(), crop=False)
        np.testing.assert_array_equal(
            result.coords['y'].values, raster.coords['y'].values
        )
        np.testing.assert_array_equal(
            result.coords['x'].values, raster.coords['x'].values
        )

    def test_crop_coords_are_contiguous_subset(self):
        """crop=True coords are a contiguous slice of the input coords (#3197)."""
        raster = _make_raster()
        result = clip_polygon(raster, _inner_polygon(), crop=True)

        in_y = raster.coords['y'].values
        in_x = raster.coords['x'].values
        out_y = result.coords['y'].values
        out_x = result.coords['x'].values

        # Every output coordinate is one of the input coordinates.
        assert np.all(np.isin(out_y, in_y))
        assert np.all(np.isin(out_x, in_x))

        # The output coords are a contiguous run of the input coords.
        y0 = int(np.where(in_y == out_y[0])[0][0])
        x0 = int(np.where(in_x == out_x[0])[0][0])
        np.testing.assert_array_equal(in_y[y0:y0 + len(out_y)], out_y)
        np.testing.assert_array_equal(in_x[x0:x0 + len(out_x)], out_x)


# ---------------------------------------------------------------------------
# Issue #3197 coverage: GPU backend parameter / NaN coverage
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
class TestClipPolygonCuPyCoverage:
    def test_custom_nodata_matches_numpy(self):
        """CuPy custom nodata matches numpy (#3197)."""
        poly = _inner_polygon()
        np_result = clip_polygon(_make_raster(backend='numpy'), poly,
                                 nodata=-9999.0, crop=False)
        cp_result = clip_polygon(_make_raster(backend='cupy'), poly,
                                 nodata=-9999.0, crop=False)
        np.testing.assert_allclose(
            cp_result.data.get(), np_result.values, equal_nan=True
        )

    def test_all_touched_matches_numpy(self):
        """CuPy all_touched=True matches numpy (#3197)."""
        poly = _inner_polygon()
        np_result = clip_polygon(_make_raster(backend='numpy'), poly,
                                 all_touched=True, crop=False)
        cp_result = clip_polygon(_make_raster(backend='cupy'), poly,
                                 all_touched=True, crop=False)
        np.testing.assert_allclose(
            cp_result.data.get(), np_result.values, equal_nan=True
        )

    def test_nan_input_preserved_matches_numpy(self):
        """CuPy preserves input NaN like numpy (#3197)."""
        data = np.arange(48, dtype=np.float64).reshape(8, 6)
        data[3, 3] = np.nan
        poly = box(0, 0, 2.5, 3.5)
        np_result = clip_polygon(create_test_raster(data, backend='numpy'),
                                 poly, crop=False)
        cp_result = clip_polygon(create_test_raster(data, backend='cupy'),
                                 poly, crop=False)
        np.testing.assert_allclose(
            cp_result.data.get(), np_result.values, equal_nan=True
        )
        assert np.isnan(cp_result.data.get()[3, 3])


@cuda_and_cupy_available
@dask_array_available
class TestClipPolygonDaskCuPyCoverage:
    def test_custom_nodata_matches_numpy(self):
        """Dask+CuPy custom nodata matches numpy (#3197)."""
        poly = _inner_polygon()
        np_result = clip_polygon(_make_raster(backend='numpy'), poly,
                                 nodata=-9999.0, crop=False)
        dkcp_result = clip_polygon(
            _make_raster(backend='dask+cupy', chunks=(4, 3)), poly,
            nodata=-9999.0, crop=False)
        np.testing.assert_allclose(
            dkcp_result.data.compute().get(), np_result.values,
            equal_nan=True
        )

    def test_all_touched_matches_numpy(self):
        """Dask+CuPy all_touched=True matches numpy (#3197)."""
        poly = _inner_polygon()
        np_result = clip_polygon(_make_raster(backend='numpy'), poly,
                                 all_touched=True, crop=False)
        dkcp_result = clip_polygon(
            _make_raster(backend='dask+cupy', chunks=(4, 3)), poly,
            all_touched=True, crop=False)
        np.testing.assert_allclose(
            dkcp_result.data.compute().get(), np_result.values,
            equal_nan=True
        )

    def test_nan_input_preserved_matches_numpy(self):
        """Dask+CuPy preserves input NaN like numpy (#3197)."""
        data = np.arange(48, dtype=np.float64).reshape(8, 6)
        data[3, 3] = np.nan
        poly = box(0, 0, 2.5, 3.5)
        np_result = clip_polygon(create_test_raster(data, backend='numpy'),
                                 poly, crop=False)
        dkcp_result = clip_polygon(
            create_test_raster(data, backend='dask+cupy', chunks=(4, 3)),
            poly, crop=False)
        np.testing.assert_allclose(
            dkcp_result.data.compute().get(), np_result.values,
            equal_nan=True
        )


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


# ---------------------------------------------------------------------------
# Issue #3191 regression test
# ---------------------------------------------------------------------------

@dask_array_available
class TestClipPolygonCropGraphSize:
    """crop=True must not fragment the mask into tiny edge-chunk-sized blocks.

    Regression test for #3191: _crop_to_bbox slices the dask raster, leaving
    a tiny partial chunk at the leading edge. clip_polygon picked that first
    chunk (chunks[-1][0]) as the rasterize mask chunk size, so a wide output
    got hundreds of narrow mask chunks and xarray.where blew the task graph
    up. The fix uses the largest chunk per axis instead.
    """

    def _big_dask_raster(self):
        import dask.array as da
        # 2560x2560 raster, uniform 256-px interior chunks.
        arr = da.zeros((2560, 2560), chunks=(256, 256), dtype='float64')
        ys = np.linspace(2560.0, 0.0, 2560)
        xs = np.linspace(0.0, 2560.0, 2560)
        return xr.DataArray(arr, dims=['y', 'x'],
                            coords={'y': ys, 'x': xs})

    def test_crop_graph_stays_bounded(self):
        """A mid-chunk crop must not explode the task graph (no .compute())."""
        raster = self._big_dask_raster()
        # box(500, 500, 2000, 2000) starts and ends mid-chunk, so the
        # cropped raster gets tiny partial edge chunks on both axes.
        poly = box(500.0, 500.0, 2000.0, 2000.0)

        result = clip_polygon(raster, poly, crop=True)

        # Mask/result chunks should track the 256-px interior size, not a
        # ~12-px partial edge chunk. With the bug the x-axis fragmented into
        # ~125 chunks; the fix keeps it in single digits.
        assert max(result.data.numblocks) < 20, (
            f"result over-fragmented: numblocks={result.data.numblocks}"
        )

        # Graph-construction-only size check. The buggy path produced
        # ~13169 tasks for this case; the fix lands near ~1000. Assert well
        # under the explosion threshold without calling .compute().
        graph = result.data.__dask_graph__()
        assert len(graph) < 4000, (
            f"task graph not bounded: {len(graph)} tasks"
        )

    def test_crop_matches_nocrop_values(self):
        """The coarser mask chunking must not change output values."""
        raster = self._big_dask_raster()
        poly = box(500.0, 500.0, 2000.0, 2000.0)

        result_crop = clip_polygon(raster, poly, crop=True)
        result_nocrop = clip_polygon(raster, poly, crop=False)

        # Align the cropped window back into the full grid and compare the
        # overlapping region. Both are all-zeros inside the polygon and NaN
        # outside, so equal_nan parity is the right check.
        crop_y = result_crop.coords['y'].values
        crop_x = result_crop.coords['x'].values
        aligned = result_nocrop.sel(y=crop_y, x=crop_x)
        np.testing.assert_allclose(
            result_crop.values, aligned.values, equal_nan=True
        )


# ---------------------------------------------------------------------------
# Issue #3190: integer raster nodata dtype consistency across backends
# ---------------------------------------------------------------------------

def _int_raster(backend='numpy', chunks=(4, 3)):
    """8x6 integer raster aligned to the same grid as ``_make_raster``."""
    data = np.arange(48, dtype=np.int32).reshape(8, 6)
    return create_test_raster(data, backend=backend, chunks=chunks)


def _to_numpy(arr):
    arr = arr.compute() if hasattr(arr, 'compute') else arr
    return arr.get() if hasattr(arr, 'get') else arr


class TestClipPolygonIntegerNodata:
    """An integer raster clipped with a NaN nodata used to raise on the GPU
    backends while silently upcasting on the CPU backends (#3190).  Every
    backend must now agree on both dtype and values.
    """

    def test_int_raster_default_nan_upcasts_numpy(self):
        """NaN nodata on an int raster promotes to float on numpy."""
        result = clip_polygon(_int_raster(), _inner_polygon(), crop=False)
        assert np.issubdtype(result.dtype, np.floating)
        assert np.isnan(result.values).any()

    def test_int_raster_finite_nodata_stays_integer(self):
        """A finite integer nodata keeps the integer dtype."""
        result = clip_polygon(
            _int_raster(), _inner_polygon(), nodata=-1, crop=False
        )
        assert result.dtype == np.int32
        assert (result.values == -1).any()

    @pytest.mark.parametrize('nodata', [np.nan, -1, 2.5])
    @dask_array_available
    def test_dask_numpy_matches_numpy(self, nodata):
        poly = _inner_polygon()
        ref = clip_polygon(_int_raster(), poly, nodata=nodata, crop=False)
        got = clip_polygon(
            _int_raster(backend='dask+numpy'), poly, nodata=nodata, crop=False
        )
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(
            _to_numpy(got.data), ref.values, equal_nan=True
        )

    @pytest.mark.parametrize('nodata', [np.nan, -1, 2.5])
    @cuda_and_cupy_available
    def test_cupy_matches_numpy(self, nodata):
        poly = _inner_polygon()
        ref = clip_polygon(_int_raster(), poly, nodata=nodata, crop=False)
        got = clip_polygon(
            _int_raster(backend='cupy'), poly, nodata=nodata, crop=False
        )
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(
            _to_numpy(got.data), ref.values, equal_nan=True
        )

    @pytest.mark.parametrize('nodata', [np.nan, -1, 2.5])
    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_matches_numpy(self, nodata):
        poly = _inner_polygon()
        ref = clip_polygon(_int_raster(), poly, nodata=nodata, crop=False)
        got = clip_polygon(
            _int_raster(backend='dask+cupy'), poly, nodata=nodata, crop=False
        )
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(
            _to_numpy(got.data), ref.values, equal_nan=True
        )
