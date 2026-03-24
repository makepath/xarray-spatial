"""Tests for multi_overlap."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import multi_overlap

da = pytest.importorskip("dask.array")


def _triple_kernel(chunk):
    """Return 3 bands from a padded (H+2, W+2) chunk."""
    interior = chunk[1:-1, 1:-1]
    return np.stack([interior + 1, interior * 2, interior - 1], axis=0)


def _make_dask_raster(shape=(64, 64), chunks=16, dtype=np.float32):
    data = da.from_array(
        np.random.RandomState(99).rand(*shape).astype(dtype), chunks=chunks
    )
    return xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.arange(shape[0]), 'x': np.arange(shape[1])},
        attrs={'crs': 'EPSG:4326'},
    )


class TestMultiOverlapDask:
    def test_matches_sequential_stack(self):
        raster = _make_dask_raster()
        multi = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

        from functools import partial
        def _band_i(chunk, i=0):
            return _triple_kernel(chunk)[i]

        bands = []
        for i in range(3):
            b = raster.data.map_overlap(
                partial(_band_i, i=i), depth=1, boundary=np.nan,
                trim=False, meta=np.array(()),
            )
            bands.append(b)
        ref = da.stack(bands, axis=0).compute()
        np.testing.assert_array_equal(multi.values, ref)

    def test_output_shape(self):
        raster = _make_dask_raster(shape=(32, 32), chunks=16)
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.shape == (3, 32, 32)

    def test_returns_dataarray_with_band_dim(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert isinstance(result, xr.DataArray)
        assert result.dims[0] == 'band'
        assert result.dims[1] == 'y'
        assert result.dims[2] == 'x'

    def test_preserves_coords_and_attrs(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.attrs == raster.attrs
        xr.testing.assert_equal(result.coords['x'], raster.coords['x'])

    def test_explicit_dtype(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1, dtype=np.float64)
        assert result.dtype == np.float64

    def test_fewer_graph_tasks_than_sequential(self):
        raster = _make_dask_raster()
        multi = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

        from functools import partial
        def _band_i(chunk, i=0):
            return _triple_kernel(chunk)[i]
        bands = []
        for i in range(3):
            b = raster.data.map_overlap(
                partial(_band_i, i=i), depth=1, boundary=np.nan,
                trim=False, meta=np.array(()),
            )
            bands.append(b)
        sequential = da.stack(bands, axis=0)
        assert len(dict(multi.data.__dask_graph__())) < len(dict(sequential.__dask_graph__()))

    def test_single_output_matches_map_overlap(self):
        def _single_kernel(chunk):
            return (chunk[1:-1, 1:-1] + 1)[np.newaxis, :]
        raster = _make_dask_raster()
        multi = multi_overlap(raster, _single_kernel, n_outputs=1, depth=1)
        def _ref_func(chunk):
            return chunk[1:-1, 1:-1] + 1
        ref = raster.data.map_overlap(
            _ref_func, depth=1, boundary=np.nan, trim=False, meta=np.array(()),
        )
        np.testing.assert_array_equal(multi.values[0], ref.compute())

    def test_dtype_inference_defaults_to_input(self):
        raster = _make_dask_raster(dtype=np.float32)
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.dtype == np.float32

    def test_non_nan_boundary(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1, boundary='nearest')
        assert result.shape == (3, 64, 64)
        assert not np.any(np.isnan(result.values))


class TestMultiOverlapNumpy:
    def test_numpy_fallback(self):
        raster = xr.DataArray(
            np.random.RandomState(99).rand(32, 32).astype(np.float32), dims=['y', 'x'],
        )
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3, 32, 32)


class TestMultiOverlapValidation:
    def test_rejects_non_2d(self):
        raster = xr.DataArray(da.zeros((4, 32, 32), chunks=16), dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2-D"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

    def test_rejects_n_outputs_zero(self):
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="n_outputs.*>= 1"):
            multi_overlap(raster, _triple_kernel, n_outputs=0, depth=1)

    def test_rejects_depth_zero(self):
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="depth.*>= 1"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=0)

    def test_rejects_chunks_smaller_than_depth(self):
        raster = _make_dask_raster(shape=(32, 32), chunks=4)
        with pytest.raises(ValueError, match="[Cc]hunk size"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=5)

    def test_rejects_non_dataarray(self):
        with pytest.raises(TypeError):
            multi_overlap(np.zeros((10, 10)), _triple_kernel, 3, 1)


class TestMultiOverlapAccessor:
    def test_accessor_delegates(self):
        import xrspatial  # noqa: F401
        raster = _make_dask_raster()
        direct = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        via_acc = raster.xrs.multi_overlap(_triple_kernel, n_outputs=3, depth=1)
        np.testing.assert_array_equal(direct.values, via_acc.values)
